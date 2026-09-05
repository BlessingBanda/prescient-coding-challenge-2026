# -----------------------------------------------------------------------------
# Prescient Coding Challenge 2026 -- your submission (R).
#
# THIS IS THE ONLY FILE YOU MAY CHANGE.
#
# You implement one function. The harness calls it once per trading day and
# hands you a `hist` list holding every observation STRICTLY BEFORE that day.
# You return the weights you want to hold for that day.
#
#     generate_weights(hist, prev_weights, params) -> named numeric vector
#
# What you get
# ------------
# hist$date                 the day you are allocating for (no data for it yet)
# hist$returns              matrix [date x asset] of daily returns, decimals
# hist$prices               matrix [date x asset] of total-return index levels
# hist$macro                matrix [date x macro feature]
# hist$assets               the six asset codes, in order
# hist$benchmark            named vector of benchmark weights
# hist$active_weight(w)     total active weight of w -- the number rule 3 tests
#
# prev_weights              what you held yesterday. Trading away from it costs
#                           money, so look at it.
# params                    the PARAMS list below, passed straight through
#
# Optional extras, in case you want them: hist$cov() gives an EWMA covariance
# matrix and hist$te(w) an ex-ante tracking error. No rule depends on either.
#
# What you must return
# --------------------
# Six weights (named numeric vector) that sum to 1, are all non-negative, sit
# within 10% of their benchmark weight, have a total active weight of no more
# than 40%, keep total equity at or below 75% and gold at or below 10%.
# make_legal() below already does all
# of that -- you can leave it alone.
#
# Declare every tuneable number in PARAMS. Parameter count is part of the score.
#
# Run `Rscript harness.R` to test on the practice window (calendar 2025), then
# `Rscript validate.R` before you submit.
# -----------------------------------------------------------------------------

# ---- Every tuneable number lives here. Fewer is better. ---------------------

PARAMS <- list(
  vol_days    = 250,     # lookback for the volatility estimate
  tilt_size   = 0.06,    # how far a 1-sigma signal moves a weight
  trade_speed = 0.10     # fraction of the gap to yesterday we close per day
)

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND   <- 0.10    # per asset, distance from benchmark
ACTIVE_BUDGET <- 0.40    # total, summed over assets
EQUITY        <- c("SA_EQUITY", "GLOBAL_EQUITY")
EQUITY_CAP    <- 0.75    # total equity, whatever the bands allow
GOLD_CAP      <- 0.10


# -----------------------------------------------------------------------------
# YOUR CODE GOES BELOW THIS LINE ----------------------------------------------
#
# This is your playground. Delete or rewrite anything here. What follows is a
# deliberately naive starting point so you can see the shape of a working
# answer. It is NOT a good answer -- on the practice window it loses to the
# benchmark. Your job is to do better.
#
# Three steps:
#   1. build a signal (here: a plain inverse-volatility tilt, which knows
#      nothing at all about expected return),
#   2. make the weights legal,
#   3. move only part of the way from yesterday, so you do not pay the full
#      trading cost every day.
#
# Steps 2 and 3 are plumbing. Keep them. Step 1 is the actual question, and
# inverse volatility is a poor answer to it: it will always prefer cash and
# bonds, whatever is happening in the world.
#
# Things worth thinking about. Which of these six assets actually diversifies
# the other five? Gold and global equity are both priced in rands -- what does
# that mean when the currency moves? The macro file has a term spread and a
# policy rate in it; what should a steepening curve do to your bond weight? And
# look at the cost table in the README before you trade property daily.
# -----------------------------------------------------------------------------


#' Score per asset. Positive means overweight, negative means underweight.
#'
#' Naive placeholder: inverse volatility. Lower-volatility assets score higher.
#' That is a statement about risk, not about return -- replace it.
build_signal <- function(hist, params) {
  n <- nrow(hist$returns)
  lookback <- as.integer(params$vol_days)
  window <- hist$returns[max(1, n - lookback + 1):n, , drop = FALSE]

  vol <- apply(window, 2, sd) * sqrt(252)
  score <- ifelse(vol > 0, 1 / vol, 0)
  names(score) <- hist$assets

  if (sd(score) > 0) score <- (score - mean(score)) / sd(score)
  score
}


#' Force `weights` to satisfy every rule. You can leave this alone.
#'
#' Everything happens in active space -- how far each asset sits from its
#' benchmark weight -- because that is how the rules are written.
#'
#' The loop is there because the steps interfere: forcing the active weights to
#' net to zero (so the portfolio sums to 1) can push an asset back outside its
#' band. A few passes settles it. The budget scaling goes last and is safe
#' there: shrinking every active weight toward zero cannot breach a band, a
#' cap, or non-negativity.
make_legal <- function(weights, hist) {
  bm <- hist$benchmark
  active <- weights[hist$assets] - bm

  for (i in 1:50) {
    active <- pmin(pmax(active, -ACTIVE_BAND), ACTIVE_BAND)   # rule 2
    active <- pmax(active, -bm)                               # keeps weights >= 0
    # rule 4: total equity cap. Trim the equity block back, sharing the cut
    # over whichever equity assets still have room to come down.
    eq_excess <- sum(bm[EQUITY] + active[EQUITY]) - EQUITY_CAP
    eq_full   <- eq_excess > -1e-12
    if (eq_excess > 0) {
      floor <- pmax(-ACTIVE_BAND, -bm[EQUITY])
      down  <- pmax(active[EQUITY] - floor, 0)
      if (sum(down) > 1e-15)
        active[EQUITY] <- active[EQUITY] - eq_excess * down / sum(down)
    }

    active[["GOLD"]] <- min(active[["GOLD"]], GOLD_CAP - bm[["GOLD"]])  # rule 5

    excess <- sum(active)        # must be zero for weights to sum to 1
    if (abs(excess) < 1e-12) break
    # give the correction to the assets that have room to absorb it
    room <- if (excess < 0) ACTIVE_BAND - active else active + bm
    room <- pmax(room, 0)
    if (excess < 0 && eq_full) room[EQUITY] <- 0  # equity full -- top up elsewhere
    if (sum(room) <= 1e-15) break
    active <- active - excess * room / sum(room)
  }

  total <- sum(abs(active))                                   # rule 3
  if (total > ACTIVE_BUDGET) active <- active * (ACTIVE_BUDGET / total)

  bm + active
}


#' Return the six portfolio weights to hold on hist$date.
generate_weights <- function(hist, prev_weights, params) {
  bm <- hist$benchmark

  # not enough history to estimate anything: sit on the benchmark
  if (nrow(hist$returns) < 260) return(bm)

  # 1. signal -> target weights around the benchmark
  signal <- build_signal(hist, params)
  target <- make_legal(bm + as.numeric(params$tilt_size) * signal, hist)

  # 2. trade gradually toward the target rather than jumping to it
  prev <- prev_weights[hist$assets]
  w <- prev + as.numeric(params$trade_speed) * (target - prev)

  make_legal(w, hist)
}


# YOUR CODE GOES ABOVE THIS LINE ----------------------------------------------
