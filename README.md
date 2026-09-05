![img](img/header.png)

# Welcome!

Welcome to the Prescient Coding Challenge 2026. You have three hours. We think
the grand prize is worth it.

## The Job

You are running a South African balanced fund. Your benchmark is a fixed blend of
six asset classes. Every trading day you choose your weights, and every day you
are measured against that benchmark after trading costs.

Beat the benchmark, risk-adjusted, over a period **you do not have the data for**.

| Asset code | What it is | Benchmark weight |
|---|---|---|
| `SA_EQUITY` | FTSE/JSE All Share, total return | 40.0% |
| `GLOBAL_EQUITY` | MSCI World net total return, in rands | 20.0% |
| `SA_BONDS` | FTSE/JSE All Bond Index (ALBI) | 25.0% |
| `SA_CASH` | STeFI Composite | 7.5% |
| `SA_PROPERTY` | FTSE/JSE SA Listed Property, total return | 5.0% |
| `GOLD` | Gold spot per ounce, in rands | 2.5% |

The benchmark rebalances daily at no cost, because it is an index. You are a
portfolio, so you pay to trade.

**As in previous years, you are marked on data you have never seen.** Your files
stop at 31 December 2025. We score you on 2026, which we hold back and merge in at
grading.

**What is new this year is how you submit.** You write one function, not a
script. The backtest loop belongs to us, and it only ever hands you data from
before the day you are allocating for. So look-ahead bias is not something you
have to be careful about - it is not available to you.

## The Data

Two files in `data/`, January 2004 to 31 December 2025.

`data_assets.csv`, one row per asset per day: `datestamp`, `asset_code`,
`asset_name`, `price` (total return index level), `return` (daily total return
**as a decimal**, so `0.01` is 1%).

`data_macro.csv`, one row per day: `usdzar`, `dxy` (dollar index), `vix`,
`brent`, `us_2y`, `us_10y`, `sa_10y`, `jibar_3m`, `sa_repo`, `em_equity`. Yields
are in percent.

## The Rules

Your six weights must, **every day**:

1. **Sum to 1** and be **non-negative**. Fully invested, long only.
2. **Sit within 10% of the benchmark.** `SA_EQUITY` can run from 30% to 50%,
   `SA_BONDS` from 15% to 35%, and so on.
3. **Use no more than 40% of active weight in total.** Add up how far every
   asset sits from its benchmark weight; that total is your risk budget. A 10%
   overweight to global equity funded by a 10% underweight to SA equity uses 20%
   of it - so the budget funds two bets of that size, not one.
4. **Keep total equity at or below 75%.** `SA_EQUITY` plus `GLOBAL_EQUITY`.
   The benchmark holds 60%, and the bands alone would let you reach 80% - this
   is a balanced fund, so 75% is the ceiling.
5. **Keep gold at or below 10%.**

And over the whole run:

6. **Actually take a view.** Your average total active weight must be at least
   5%, so an eighth of the risk budget in rule 3. Hugging the benchmark is not
   an answer.
7. **No external data.** No extra files, downloads, API calls, or hard-coded
   market history. The two CSVs are all you get.
8. **Run in under 10 minutes**, and be **deterministic** - same input, same
   output. Seed anything random.
9. **Only change `solution.py` or `solution.R`.** We diff the harness against
   ours. Changing it is disqualification.

`make_legal()` in the sample solution already enforces rules 1 to 5. You can
leave it exactly as it is and never think about them again.

### Trading costs

Charged one-way on the amount you trade, every day, per asset:

| `SA_CASH` | `SA_BONDS` | `SA_EQUITY` | `GLOBAL_EQUITY` | `GOLD` | `SA_PROPERTY` |
|---|---|---|---|---|---|
| 1 bp | 8 bp | 15 bp | 20 bp | 25 bp | 35 bp |

Move 10% into and out of listed property every day for a year and you burn about
8% of the fund. Read the table before you build something that churns.

## How You Are Scored

**Stage 1, the gates.** These are pass/fail, and failing one means you score
nothing: submitted before 14:00 on 5 September; only the solution file changed;
runs unattended; under 10 minutes; identical on consecutive runs; average active
weight at least 5%. `validate.py` checks the last three for you.

Breaking rules 1 to 5 on a given day is **not** one of those gates. If your
weights are illegal on a day, we hold the benchmark for you that day and count
it. You forfeit whatever view you had, and you pay to trade back in, so it is a
real cost - but one bad day does not throw away your whole submission. Getting
this right still matters: `validate.py` will fail loudly on any illegal day, and
a high violation count is something we look at directly.

**Stage 2, the numbers.** The metric is simple: **how much did you beat the
benchmark by, after costs.** We measure it as **mean net excess return per day,
in basis points**. Per-day rather than annualised, because the live window is
under a year and annualising a part-year return overstates it; a daily rate also
makes a shorter window and a longer one directly comparable. We do not adjust for
risk, because the rules above have already done that - the 10% bands and the 40%
budget cap how much risk you can take, so within those limits the best portfolio
is just the one that makes the most money.

For scale: our sample solution scores about **+0.08 bps/day**. A genuinely good
entry is somewhere around **+1 bps/day**. Perfect foresight, one week ahead,
would get about +6.

We run your function over five windows: the **live window** (calendar 2026, which
you have never seen) and four **historical regimes** - 2015 (the rand shock), 2016
(the post-Nenegate rebound), 2020 (covid) and 2022 (the inflation shock).

```
score = 0.50 x excess(live) + 0.50 x median excess(2015, 2016, 2020, 2022)
```

A strategy that only works in one regime gets found out, because the median across
four regimes is half the score. And you cannot tune to the live window at all.
**Curve-fitting is the failure mode we are testing for.** The four historical
windows are in your data, so run them yourself.

One warning from our own testing: a fixed tilt earns almost nothing in the live
window, whichever direction you pick. Guessing which asset wins the year is worth
close to zero here. Deciding *when* to hold it is worth a great deal.


## Using AI

**AI tools are allowed. All of them.** ChatGPT, Claude, Copilot, Gemini,
whatever you have. We are not going to police it and we are not going to pretend
you would not use these tools on the job, because you would - we do.

Two asks:

1. **Tell us what you used** in your pull request. One line, no penalty.
2. **Use it to think faster, not instead of thinking.** The scoring window is
   hidden, we test you across four other regimes, and we count your parameters.
   A model can write you 300 lines in ninety seconds. It cannot tell you which of
   three plausible signals survives a regime break, and that judgement is what
   the scoring is built to reward.

The problem is built so that judgement is the bottleneck, not typing speed. If
having a paid subscription turns out to matter much, we designed this badly - say
so on the feedback form.

## Getting Started

```bash
# Python
python -m pip install -r requirements.txt
python harness.py       # your solution on the practice window (2025)
python validate.py      # every window we will score you on

# R  (base R only, nothing to install)
Rscript harness.R
Rscript validate.R
```

Open `solution.py` or `solution.R`. You are editing one function:

```python
def generate_weights(hist, prev_weights, params):
    """Return the six weights to hold on hist.date."""
```

| Python | R | What it is |
|---|---|---|
| `hist.date` | `hist$date` | the day you are allocating for |
| `hist.returns` | `hist$returns` | daily returns, every day **before** today |
| `hist.prices` | `hist$prices` | total return index levels |
| `hist.macro` | `hist$macro` | the macro features |
| `hist.benchmark` | `hist$benchmark` | the benchmark weights |
| `hist.active_weight(w)` | `hist$active_weight(w)` | total active weight of `w` |
| `prev_weights` | `prev_weights` | what you held yesterday |

Two optional extras if you want to reason about risk rather than weights:
`hist.cov()` gives an exponentially weighted covariance matrix and `hist.te(w)`
an ex-ante tracking error. No rule depends on either.

The shipped solution is a deliberately naive inverse-volatility tilt. It **loses
to the benchmark on the practice window**. It is a starting point, not a baseline
worth defending. Change `build_signal()` and leave the rest alone if you like -
that is the intended shape of a first answer.

## Before You Submit

```bash
python validate.py      # or: Rscript validate.R
```

If it does not say **"All windows passed"**, we cannot score you. This is the
most common way to throw away three hours of good work.

## Getting The Project On Your Computer (GitHub)

1. Sign in or sign up to GitHub.
2. **Fork the repo**, as below. Fork before you clone or you cannot submit a
   pull request.

![alt text](img/image2.png)

3. Once it shows on your GitHub profile, clone it.

![alt text](img/image3.png)

4. Open a terminal where you want to work and run `git clone "your https url"`.

![alt text](img/image4.png)

5. It is your own profile, so work on `main`.

## How To Submit Your Answer

1. `git add .`
2. `git commit -m 'Your Team Name'`
3. `git push origin main`
4. Check your changes are only in `solution.py` **or** `solution.R`.
5. On the "Pull Requests" tab, select "New pull request".

![alt text](img/image5.png)

6. The summary should mention only 1 file change. Select "Create pull request".
7. In the description put **your team name**, a short note on how you solved it
   and **why**, and **which AI tools you used**. Confirm.

![alt text](img/image6.png)

8. Your pull request should now appear on our repository's list.

![alt text](img/image7.png)

# Download Links

[Git](https://git-scm.com/downloads) |
[Python](https://www.python.org/downloads/) |
[VS Code](https://code.visualstudio.com/download) |
[R Base](https://cran.r-project.org/) |
[R Studio](https://posit.co/downloads/)

Good luck. Come find us at lunch - we are happy to talk about the problem, about
Prescient, and about what working here actually looks like.
