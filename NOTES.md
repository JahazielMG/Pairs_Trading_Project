# Findings

Running log of things discovered while building, kept so they can be folded into
the README's limitations section later.

## Resolved

- **Cointegration gate.** `run_backtest` now takes `require_cointegration`,
  default `False`. Off, it trades and prints a loud warning banner; on, it
  stops before tuning and returns `traded=False` with `metrics=None` (beta and
  the p-value are still reported). The warning fires whenever the gate fails
  regardless of the flag, so a failure is never silent.
- **`MIN_ROWS`** raised from 250 to 1000 in `data.py`.
- **Default seed** changed from 3 to 9 (see below). Seed 3 is retained as the
  documented counter-example.

---

## Simulator seed sensitivity: individual legs can test stationary

**Found while:** verifying `data.py` standalone (spec section 8, step 1).

**What happened.** The simulated pair is reliably cointegrated, as intended —
Engle-Granger returns p between 0.00004 and 0.0018 across every seed tested. But
on the spec's original default of `seed=7`, an augmented Dickey-Fuller test on
each leg *individually* rejects the unit root: p = 0.0021 for A and p = 0.0037
for B. That contradicts the premise the project is built to demonstrate, which
is that each stock is individually non-stationary and only the spread reverts.

**Why.** This is a finite-sample artifact of the particular random draw, not a
flaw in the construction. The shared common factor is a true random walk
(`cumsum` of IID normal shocks), but any given 1500-day path can happen to trace
a shape — here a decay from ~100 to ~45 followed by a long flat stretch — that
ADF, regressing against a constant, reads as mean reversion. Roughly a quarter
of draws do this, which is in line with ADF's known finite-sample behavior.

**Seed sweep** (n_days=1500):

| seed | ADF p, A | ADF p, B | coint p |
|-----:|---------:|---------:|--------:|
| 0 | 0.5416 | 0.5475 | 0.000037 |
| 1 | 0.2902 | 0.3644 | 0.000046 |
| 2 | 0.9032 | 0.9327 | 0.000953 |
| **3** | **0.6814** | **0.5738** | **0.000248** |
| 4 | 0.4072 | 0.4021 | 0.000054 |
| 5 | 0.8926 | 0.8584 | 0.000161 |
| 6 | 0.6300 | 0.5728 | 0.001445 |
| 7 | 0.0021 | 0.0037 | 0.000217 |
| 8 | 0.9602 | 0.9636 | 0.000053 |
| 9 | 0.4916 | 0.6334 | 0.000046 |
| 10 | 0.0029 | 0.0012 | 0.000226 |
| 11 | 0.0291 | 0.0227 | 0.001787 |

Seeds 7, 10, and 11 produce legs that test stationary at the 5% level.

**Resolution.** Changed the simulator default from `seed=7` to `seed=3`, where
both legs are comfortably non-stationary (p = 0.68 and 0.57) and the pair is
still strongly cointegrated (p = 0.00025). *Superseded:* seed 3 later turned
out to fail the cointegration gate on the training slice, so the default is now
**seed 9**, which clears both criteria — ADF legs at p = 0.49 / 0.63, train
slice cointegration at p = 0.00029. Of the first 25 seeds, 15 satisfy both
constraints; 7 fail on the train gate and 3 have stationary legs.

**Why it's worth writing up.** The strategy's edge is supposed to come from
trading a mean-reverting spread between two non-mean-reverting assets. If the
legs themselves revert, a profitable backtest no longer isolates the spread as
the source of the edge — you can't tell whether the pairs logic did the work or
whether you just bought a dip in a mean-reverting asset. Reporting the seed
sweep is also a concrete instance of the broader caveat that single-path results
are draw-dependent.

---

## The headline result: real KO/PEP is not cointegrated, and loses money

**Found while:** first run of `main.py` against live yfinance data
(2015-01-01 to 2025-12-31, 2765 rows).

**What happened.** KO and PEP — the textbook pairs-trading example — **fail the
cointegration gate on the training slice at p = 0.2932**, nowhere near the 0.05
threshold, and this is on a 1382-day train slice where the test has good power
(37/40 detection on known-cointegrated data at 1000 days). This is not the
marginal Type II error seen on the simulated seed 3; it is a clear rejection.

Traded anyway, the strategy loses money out of sample:

| metric | value |
|---|---:|
| train cointegration p | 0.2932 (FAIL) |
| hedge ratio (beta) | 0.3325 |
| tuned params | window 30, entry 1.5, exit 0.25 |
| validation Sharpe | 0.362 |
| **test Sharpe** | **-0.912** |
| total return | -6.53% |
| max drawdown | -8.93% |
| trades | 27 |
| win rate | 51.9% |
| avg P&L per trade | -$242.69 |

**Why it happened.** The price chart shows it plainly: PEP roughly doubles from
~65 to ~145 over the period while KO moves from ~30 to ~68. The two are
strongly *correlated* — they trend together — but the spread between them has
no fixed level to revert to, so it drifts. The fitted beta of 0.3325 is the
regression's attempt to scale KO up to PEP, and the residual from that fit is
not stationary. Every mean-reversion entry is a bet that a drifting series will
come back, and the equity curve is the result: a grind from $100k down to
roughly $93.5k with a 51.9% win rate, meaning the trades are close to a coin
flip while the losers are substantially larger than the winners.

**Why this is the most valuable result in the project.** It is a direct,
empirical demonstration of the distinction the whole project is built around —
correlation is not cointegration — using the exact pair most people cite as the
canonical example. The gate did its job: it said don't trade this, and the
out-of-sample P&L confirms that was the right call. Reporting a losing backtest
that the methodology correctly predicted is far more credible than reporting a
winning one, and it makes the cointegration test the point of the project
rather than a box-ticking preliminary.

**Suggested framing for the README:** lead with this rather than the simulated
result. The simulated pair shows the machinery works when the premise holds;
KO/PEP shows the premise usually doesn't hold, and that the gate catches it.

---

## Engle-Granger has weak power on a half-sample — the train gate fails

**Found while:** first end-to-end `run_backtest` (spec section 8, step 3).

**What happened.** On seed 3, the cointegration test *fails on the training
slice* (p = 0.1127, above the 0.05 threshold) even though the pair is
cointegrated by construction and tests at p = 0.000248 on the full sample. The
spec's own logic says this is the gate on whether to trade the pair at all, so
a failing gate is not a cosmetic problem.

**Why.** It isn't a levels-versus-logs issue — the data is built in log space,
but testing log prices gives nearly identical p-values (0.1124 on the same
slice). It's statistical power. The train slice is half the data, and
Engle-Granger needs a long sample to reject its null. Detection rate on
known-cointegrated data, 40 seeds:

| sample length | detected at p<0.05 |
|---|---:|
| 250 days | 1/40 |
| 500 days | 17/40 |
| 750 days (the train slice) | 30/40 |
| 1000 days | 37/40 |
| 1500 days (full sample) | 40/40 |

So roughly a quarter of draws fail the gate on a 750-day train slice purely by
Type II error, and seed 3 is one of them.

**Two consequences.**

1. **`MIN_ROWS = 250` in `data.py` is far too low to support the gating
   decision.** At 250 observations the test detects known cointegration 1 time
   in 40 — it has essentially no power, so a "pass" would be close to
   meaningless and a "fail" would carry no information either. Any real-data
   run near that floor is not actually testing anything.
2. **The train/validation/test split trades statistical power for honesty.**
   Reserving half the data for training is what makes the gate weak. This is a
   genuine tension worth stating in the README rather than hiding: the same
   discipline that prevents lookahead also leaves less data for the test that
   justifies the trade.

**Note for the README.** A failed gate here is most likely a Type II error, not
evidence the pair is unrelated — we know it's cointegrated because we built it
that way. On real data that distinction is unobservable, which is the honest
version of the point: you cannot tell a genuinely non-cointegrated pair from a
cointegrated one your sample was too short to confirm.

---

## Silent boolean-dtype bug in trade attribution

**Found while:** unit-testing `_trade_ids` against a hand-built position series.

**What happened.** Trade grouping assigned every single day its own trade ID,
so a 10-day trade counted as 10 trades and the win rate became the fraction of
*days* that were profitable rather than the fraction of *trades*.

**Why.** The entry mask was written as
`in_trade & ~in_trade.shift(1).fillna(False)`. On a boolean Series, pandas
`.shift(1)` promotes to **object** dtype to hold the NaN, and `.fillna(False)`
leaves it as object. Applying `~` to an object-dtype Series of Python bools
falls back to integer bitwise inversion: `~True` is `-2` and `~False` is `-1`.
Both are truthy, so the negation silently became a no-op and the mask reduced
to `in_trade`.

**Fix.** `in_trade.shift(1, fill_value=False)`, which preserves bool dtype.

**Why it's worth writing up.** This produced no error, no warning, and a
plausible-looking win rate. It was caught only because the trade-attribution
test asserted exact row membership against a hand-built series rather than
eyeballing summary statistics. Good example of why the hand-built fixtures in
the verification step earn their keep.

---

## The rolling z-score is not N(0,1) — "2 sigma" is not a 2-sigma event

**Found while:** verifying `strategy.py` standalone (spec section 8, step 2).

**What happened.** The rolling z-score has a standard deviation of about 1.26,
not 1.0, and `|z| > 2` occurs on roughly 10% of days rather than the ~4.6% a
standard normal would imply. The effect grows with the window:

| z-score | std | P(\|z\| > 2) | trades opened |
|---|---:|---:|---:|
| full-sample | 1.0000 | 0.0507 | — |
| rolling, window=15 | 1.2243 | 0.0774 | 68 |
| rolling, window=20 | 1.2496 | 0.0932 | 65 |
| rolling, window=30 | 1.2627 | 0.1027 | 52 |
| rolling, window=45 | 1.2911 | 0.1099 | 43 |
| rolling, window=60 | 1.2955 | 0.1152 | 33 |
| N(0,1) reference | 1.0000 | 0.0455 | — |

**Why.** Standardizing against a *trailing* window rather than the full sample
divides by a standard deviation estimated from a short, autocorrelated stretch
of an Ornstein-Uhlenbeck process. That window sees only part of the spread's
full range, so it systematically understates the stationary standard deviation,
and the resulting ratio is inflated. Longer windows capture more of the range in
the numerator while the denominator saturates, so dispersion rises with window
length even as the number of trades falls. This is expected behavior for a
rolling standardization, not a defect — the full-sample z-score in the first row
does come out at exactly std 1.0, confirming the arithmetic is right.

**Why it matters.** The entry threshold should not be described as a
"2-standard-deviation move," because it isn't one: it fires more than twice as
often as that phrasing implies. Two practical consequences:

1. The `entry_grid` of (1.5, 2.0, 2.5) in the tuning step spans a much wider
   range of trade frequencies than it looks like it does — worth checking that
   `min_trades` is binding at the high end rather than the grid silently
   collapsing to one viable config.
2. Window and entry threshold are not independent knobs. Increasing the window
   raises the dispersion of the z-score, which partially offsets the reduction
   in trade count you would otherwise expect. Tuning them jointly on a grid is
   the right call, but the grid is not exploring two orthogonal dimensions.

---

## Spread amplitude is heteroskedastic by construction

**Found while:** eyeballing the simulated price/spread plot.

The dollar spread `A - B` swings roughly ±6 early in the sample and roughly ±2
late in it. The cause is that the OU spread lives in log space, so it represents
a roughly constant *percentage* dislocation, while the price level in this draw
falls by about half over the sample. A constant percentage of a smaller price is
a smaller dollar amount.

This is a point in favor of the spec's insistence on a rolling z-score rather
than a full-sample one, for a reason beyond the lookahead argument: a
full-sample standard deviation would be dominated by the high-volatility early
period, so late-sample dislocations would rarely clear a fixed entry threshold
and the strategy would mostly stop trading in the back half of the data. The
rolling window renormalizes as the scale drifts. Worth mentioning in the README
alongside the lookahead point.

---

## Validation-vs-test Sharpe gap persists across seeds

**Found while:** first end-to-end `run_backtest`.

On seed 3, tuning selected window=45, entry=2.0, exit=0.5 with a validation
Sharpe of 1.852. The same configuration scores 0.856 on test — a gap of almost
exactly 1.0, with +2.35% total return, -0.82% max drawdown, 12 trades and a
66.7% win rate.

On the new default seed 9 the same pattern holds: tuning picks window=45,
entry=2.0, exit=0.75 at validation Sharpe 1.274, which delivers 0.448 on test —
a gap of 0.826, with +1.31% total return, -1.66% max drawdown, 14 trades and a
78.6% win rate. Two independent draws both losing roughly half to two-thirds of
the validation Sharpe out of sample is a much stronger README point than a
single instance, since it rules out "unlucky test period" as the explanation.

This is the overfitting signature the spec asks the README to discuss, and it
showed up on the first honest run without needing to be manufactured. The
validation number is the maximum over 36 configurations, so it is biased upward
by construction; the test number is a single unconditional evaluation. Roughly
half the headline performance was selection effect.

**On `min_trades`:** on the seed-3 grid the filter did *not* bind — the
top-Sharpe config had 10 validation trades and would have been selected either
way, though 6 of 36 configs were excluded including two tied at Sharpe 1.818 on
just 4 trades each.

*Updated after the full checklist sweep:* on the **default seed 9 it does
bind**. The highest Sharpe across all 36 configs was 1.495 on **2 trades**;
without the floor that config would have been selected and reported. Across 12
seeds the filter changed the selection on 3 of them — seed 1 (Sharpe 2.150 on 3
trades), seed 2 (1.668 on 4), and seed 9 (1.495 on 2). So roughly one grid
search in four would otherwise report a headline number produced by two or three
lucky trades. The earlier "doesn't bind, but does real work" framing was drawn
from seed 3 alone and understates the filter's importance.

---

## Removing the one-day lag makes this strategy look WORSE

**Found while:** the section 7 checklist sweep, item 5.

Comparing the correct `position.shift(1)` P&L against the unlagged
`position * spread.diff()` variant on the same positions gives a gross Sharpe of
**0.638 lagged versus −1.217 unlagged**. The cheating version is dramatically
*worse*.

**Why.** Lookahead bias is normally assumed to flatter results, and for a
momentum strategy it would: you'd be credited with the move that generated your
signal. Mean reversion inverts that. Entries fire *because* the spread has just
moved sharply against the direction you're about to take, so attributing that
same-day move to the new position books the adverse leg of every entry.

**Why it matters.** The instinct "the numbers look plausible, so there's
probably no lookahead" is unreliable — the sign of the distortion depends on
whether the strategy is trend-following or contrarian. Worth stating in the
README, since it inverts a common assumption.
