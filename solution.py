"""
Prescient Coding Challenge 2026 -- your submission.

THIS IS THE ONLY FILE YOU MAY CHANGE.

You implement one function. The harness calls it once per trading day and hands
you a `hist` object holding every observation STRICTLY BEFORE that day. You
return the weights you want to hold for that day.

    generate_weights(hist, prev_weights, params) -> weights

What you get
------------
hist.date                 the day you are allocating for (no data for it yet)
hist.returns              DataFrame [date x asset] of daily returns, decimals
hist.prices               DataFrame [date x asset] of total-return index levels
hist.macro                DataFrame [date x macro feature]
hist.assets               list of the six asset codes, in order
hist.benchmark            Series of benchmark weights
hist.active_weight(w)     total active weight of w -- the number rule 3 tests

prev_weights              what you held yesterday. Trading away from it costs
                          money, so look at it.
params                    the PARAMS dict below, passed straight through

Optional extras, in case you want them: hist.cov() gives an EWMA covariance
matrix and hist.te(w) an ex-ante tracking error. No rule depends on either.

What you must return
--------------------
Six weights (dict, Series or array in hist.assets order) that sum to 1, are all
non-negative, sit within 10% of their benchmark weight, have a total active
weight of no more than 40%, keep total equity at or below 75% and gold at or
below 10%. `make_legal()` below already does all of that -- you can leave it
alone.

Declare every tuneable number in PARAMS. Parameter count is part of the score.

Run `python harness.py` to test on the practice window (calendar 2025), then
`python validate.py` before you submit.

--------------------------------------------------------------------------- #
Model, in one paragraph
--------------------------------------------------------------------------- #
build_signal() replaces the naive inverse-volatility placeholder with a
learned view: raw lag-return / volatility / macro features are fed both
directly, and through a small supervised neural net, into a Ridge (L2-
regularised linear) regression that predicts each asset's forward return.
The two feature sets are concatenated (raw ++ NN-learned) before Ridge sees
them -- the NN's job is to hand Ridge a few nonlinear combinations of the raw
inputs it couldn't otherwise form on its own, while Ridge's job is to combine
everything with a heavily shrunk, low-variance linear fit rather than a
high-variance nonparametric one (an Extra Trees version of this was tried
first and swapped out for exactly this reason -- on daily return data this
noisy, a few dozen unconstrained tree splits per refit is an easy way to fit
noise that Ridge's shrinkage resists by construction). Everything downstream
of the signal (make_legal, the daily trade_speed step-through) is exactly
the shipped scaffold and is untouched.

Refitting the whole pipeline on every one of ~250 calls per window would blow
the 10-minute budget, so the fitted model is cached at module level and
refit only every `refit_every` trading days. See `_maybe_refit()` for the
one correctness subtlety this introduces (guarding against being handed an
earlier window's history after having already trained on a later one).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

# --------------------------------------------------------------------------- #
# Every tuneable number lives here. Fewer is better.
# --------------------------------------------------------------------------- #

PARAMS = {
    "tilt_size":       0.06,   # how far a 1-sigma signal moves a weight
    "trade_speed":     0.10,   # fraction of the gap to yesterday closed per day
    "forward_horizon": 5,      # days-ahead return the model is trained to predict
    "min_train_rows":  500,    # trading days of history required before trusting the model
    "refit_every":     21,     # trading days between refits (~monthly)
    "hidden_units":    16,     # width of the NN feature-extractor's hidden layer
    "ridge_alpha":     10.0,   # L2 penalty on the final Ridge regression
    "train_lookback":  2000,   # trading days (~8yrs) of history used to fit each refit
    "random_seed":     7,      # seeds every source of randomness -> deterministic
}

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10

# --------------------------------------------------------------------------- #
# <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>
# --------------------------------------------------------------------------- #

RET_LAGS = (1, 5, 10, 20, 60)          # trailing return windows, in trading days
VOL_WINDOWS = (20, 60)                 # trailing realised-vol windows
MACRO_COLS = [
    "usdzar", "dxy", "vix", "brent", "us_2y", "us_10y",
    "sa_10y", "jibar_3m", "sa_repo", "em_equity",
]

# Fitted model persists here across daily calls within one process, so we
# refit on a schedule instead of on every single trading day.
_CACHE: dict = {"model": None, "trained_at_n": None}


def _asset_features(returns_col: pd.Series, prices_col: pd.Series) -> pd.DataFrame:
    """Raw per-asset features: trailing returns and realised vol."""
    feats = {}
    for lag in RET_LAGS:
        feats[f"ret_{lag}d"] = prices_col.pct_change(lag)
    for w in VOL_WINDOWS:
        feats[f"vol_{w}d"] = returns_col.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(feats, index=returns_col.index)


def _macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Macro levels (raw) plus two curve spreads.

    No z-scoring here on purpose: the final pipeline standardises the whole
    feature vector (`x_mean`/`x_std`, fit once at refit time) anyway, and
    z-scoring macro afresh from full history would mean recomputing a
    rolling mean/std over ~20 years every single day just to read off one
    row -- exactly the daily-cost mistake `_latest_asset_row` below is
    written to avoid.
    """
    cols = [c for c in MACRO_COLS if c in macro.columns]
    feats = macro[cols].copy()
    if "us_10y" in macro.columns and "us_2y" in macro.columns:
        feats["us_curve"] = macro["us_10y"] - macro["us_2y"]
    if "sa_10y" in macro.columns and "jibar_3m" in macro.columns:
        feats["sa_curve"] = macro["sa_10y"] - macro["jibar_3m"]
    return feats


def _build_panel(hist, forward_horizon: int) -> pd.DataFrame:
    """Stack all six assets into one training panel (rows = date x asset).

    Only used at refit time (every `refit_every` days), never on every
    daily call -- it walks the full history with vectorised pandas ops,
    which is fine occasionally but far too slow to redo every single day.

    Each row carries that asset's lag/vol features, the day's shared macro
    features, a one-hot asset id (so one pooled model can still tell assets
    apart), and the forward return used only as the training target.
    Pooling assets into one panel multiplies the effective training sample
    six-fold, which matters a lot with a small NN and only ~20 years of data.
    """
    macro_feats = _macro_features(hist.macro)
    frames = []
    for asset in hist.assets:
        panel = _asset_features(hist.returns[asset], hist.prices[asset])
        panel = panel.join(macro_feats, how="left")
        panel["fwd_return"] = (
            hist.prices[asset].pct_change(forward_horizon).shift(-forward_horizon)
        )
        for a2 in hist.assets:
            panel[f"is_{a2}"] = 1.0 if a2 == asset else 0.0
        panel["_asset"] = asset
        frames.append(panel)
    return pd.concat(frames, axis=0)


def _latest_asset_row(hist, asset: str) -> dict:
    """Today's raw features for one asset, computed off only the tail of the
    history -- O(lookback), not O(full history). This is what runs on every
    single daily call, so it has to stay cheap regardless of how many years
    of data have piled up by the time we're deep into the 2025 or 2026
    windows.
    """
    max_lookback = max(max(RET_LAGS), max(VOL_WINDOWS)) + 1
    prices_tail = hist.prices[asset].tail(max_lookback)
    returns_tail = hist.returns[asset].tail(max_lookback)

    row = {}
    for lag in RET_LAGS:
        if len(prices_tail) > lag:
            row[f"ret_{lag}d"] = prices_tail.iloc[-1] / prices_tail.iloc[-1 - lag] - 1.0
        else:
            row[f"ret_{lag}d"] = np.nan
    for w in VOL_WINDOWS:
        row[f"vol_{w}d"] = returns_tail.tail(w).std() * np.sqrt(252)
    return row


def _latest_features(hist) -> pd.DataFrame:
    """Today's feature row for every asset, in the same columns/order that
    `_build_panel` produces -- but built from tail slices only, so it is
    cheap enough to call once per day.
    """
    macro_row = hist.macro.iloc[-1]
    macro_cols = [c for c in MACRO_COLS if c in hist.macro.columns]
    macro_feats = {c: macro_row[c] for c in macro_cols}
    if "us_10y" in macro_row and "us_2y" in macro_row:
        macro_feats["us_curve"] = macro_row["us_10y"] - macro_row["us_2y"]
    if "sa_10y" in macro_row and "jibar_3m" in macro_row:
        macro_feats["sa_curve"] = macro_row["sa_10y"] - macro_row["jibar_3m"]

    rows = {}
    for asset in hist.assets:
        row = _latest_asset_row(hist, asset)
        row.update(macro_feats)
        for a2 in hist.assets:
            row[f"is_{a2}"] = 1.0 if a2 == asset else 0.0
        rows[asset] = row
    return pd.DataFrame(rows).T.reindex(hist.assets)


def _mlp_hidden(nn: MLPRegressor, X: np.ndarray) -> np.ndarray:
    """Hidden-layer activations of the fitted 1-hidden-layer, ReLU MLP.

    This is the "NN learned features" half of the pipeline: rather than only
    handing Ridge the raw inputs, we also hand it the nonlinear
    combinations of those inputs that the NN learned while training toward
    the same forward-return target.
    """
    z = X @ nn.coefs_[0] + nn.intercepts_[0]
    return np.maximum(z, 0.0)


def _fit_pipeline(hist, params: dict):
    """Fit the NN feature-extractor + Ridge pipeline on all history strictly
    before hist.date. Returns None if there isn't enough clean data yet
    (only possible right at the start of the full history).
    """
    seed = int(params["random_seed"])
    np.random.seed(seed)

    panel = _build_panel(hist, int(params["forward_horizon"]))
    feature_cols = [c for c in panel.columns if c not in ("fwd_return", "_asset")]

    # Fit on a rolling lookback rather than the full ~20-year history: it's
    # cheaper to refit, and it keeps the model reading current regimes
    # rather than diluting them against, e.g., 2004-08 pre-crisis data.
    lookback = int(params["train_lookback"])
    cutoff = hist.returns.index[-lookback] if len(hist.returns) > lookback else hist.returns.index[0]
    panel = panel.loc[panel.index >= cutoff]

    train = panel.dropna(subset=feature_cols + ["fwd_return"])
    if len(train) < 200:
        return None

    X = train[feature_cols].to_numpy(dtype=float)
    y = train["fwd_return"].to_numpy(dtype=float)

    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    Xs = (X - x_mean) / x_std

    nn = MLPRegressor(
        hidden_layer_sizes=(int(params["hidden_units"]),),
        activation="relu",
        solver="lbfgs",          # full-batch, deterministic given random_state
        max_iter=300,
        random_state=seed,
    )
    nn.fit(Xs, y)
    hidden = _mlp_hidden(nn, Xs)

    ridge = Ridge(alpha=float(params["ridge_alpha"]))   # closed-form solve;
                                                          # no randomness at all
    ridge.fit(np.hstack([Xs, hidden]), y)

    return {
        "nn": nn, "ridge": ridge,
        "x_mean": x_mean, "x_std": x_std,
        "feature_cols": feature_cols,
    }


def _maybe_refit(hist, params: dict):
    """Return a cached, up-to-date model; refit only when the schedule says to.

    The one subtlety: `validate.py` scores several windows (2015, 2016, 2020,
    2022, 2025, ...) and there is no guarantee they run in chronological
    order within the process. If the amount of history we've been handed
    ever *shrinks* relative to what the cached model was trained on, that
    means we've been handed an earlier window after a later one -- the
    cached model would effectively be trained on the future relative to
    this window, which is exactly the look-ahead bias the harness is built
    to prevent. So a shrink forces an immediate refit, not just a stale one.
    """
    n_rows = len(hist.returns)
    if n_rows < int(params["min_train_rows"]):
        return None

    trained_at_n = _CACHE["trained_at_n"]
    needs_refit = (
        _CACHE["model"] is None
        or trained_at_n is None
        or n_rows < trained_at_n
        or n_rows - trained_at_n >= int(params["refit_every"])
    )
    if needs_refit:
        model = _fit_pipeline(hist, params)
        if model is not None:
            _CACHE["model"] = model
            _CACHE["trained_at_n"] = n_rows

    return _CACHE["model"]


def build_signal(hist, params) -> pd.Series:
    """Score per asset. Positive means overweight, negative means underweight.

    Predicts each asset's forward return with the cached NN+Ridge pipeline,
    then z-scores across the six assets so the signal scale is stable
    through time (matching the shipped placeholder's convention). Falls
    back to a flat zero signal (sit on the benchmark) whenever there isn't
    yet enough history to trust a fitted model.
    """
    model = _maybe_refit(hist, params)
    if model is None:
        return pd.Series(0.0, index=hist.assets)

    latest = _latest_features(hist)[model["feature_cols"]]
    if latest.isna().any().any():
        return pd.Series(0.0, index=hist.assets)

    Xs = (latest.to_numpy(dtype=float) - model["x_mean"]) / model["x_std"]
    hidden = _mlp_hidden(model["nn"], Xs)
    pred = model["ridge"].predict(np.hstack([Xs, hidden]))

    score = pd.Series(pred, index=hist.assets)
    if score.std() > 0:
        score = (score - score.mean()) / score.std()
    return score


def make_legal(weights: pd.Series, hist) -> pd.Series:
    """Force `weights` to satisfy every rule. You can leave this alone.

    Everything happens in active space -- how far each asset sits from its
    benchmark weight -- because that is how the rules are written.

    The loop is there because the steps interfere: forcing the active weights
    to net to zero (so the portfolio sums to 1) can push an asset back outside
    its band. A few passes settles it. The budget scaling goes last and is safe
    there: shrinking every active weight toward zero cannot breach a band, a
    cap, or non-negativity.
    """
    bm = hist.benchmark
    active = weights.reindex(hist.assets).astype(float) - bm

    for _ in range(50):
        active = active.clip(lower=-ACTIVE_BAND, upper=ACTIVE_BAND)  # rule 2
        active = active.clip(lower=-bm)                              # keeps weights >= 0
        # rule 4: total equity cap. Trim the equity block back, sharing the
        # cut over whichever equity assets still have room to come down.
        eq_excess = (bm[EQUITY] + active[EQUITY]).sum() - EQUITY_CAP
        eq_full = eq_excess > -1e-12
        if eq_excess > 0:
            floor = np.maximum(-ACTIVE_BAND, -bm[EQUITY])
            down = (active[EQUITY] - floor).clip(lower=0)
            if down.sum() > 1e-15:
                active[EQUITY] = active[EQUITY] - eq_excess * down / down.sum()

        active["GOLD"] = min(active["GOLD"], GOLD_CAP - bm["GOLD"])  # rule 5

        excess = active.sum()          # must be zero for weights to sum to 1
        if abs(excess) < 1e-12:
            break
        # give the correction to the assets that have room to absorb it
        room = (ACTIVE_BAND - active) if excess < 0 else (active + bm).clip(lower=0)
        room = room.clip(lower=0)
        if excess < 0 and eq_full:
            room[EQUITY] = 0.0   # equity is at its cap -- top up elsewhere
        if room.sum() <= 1e-15:
            break
        active = active - excess * room / room.sum()

    total = active.abs().sum()                                       # rule 3
    if total > ACTIVE_BUDGET:
        active = active * (ACTIVE_BUDGET / total)

    return bm + active


def generate_weights(hist, prev_weights, params):
    """Return the six portfolio weights to hold on hist.date."""
    bm = hist.benchmark

    # not enough history to trust the model yet: sit on the benchmark
    if len(hist.returns) < int(params["min_train_rows"]):
        return bm.to_dict()

    # 1. signal -> target weights around the benchmark
    signal = build_signal(hist, params)
    target = make_legal(bm + float(params["tilt_size"]) * signal, hist)

    # 2. trade gradually toward the target rather than jumping to it
    prev = prev_weights.reindex(hist.assets)
    w = prev + float(params["trade_speed"]) * (target - prev)

    return make_legal(w, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>