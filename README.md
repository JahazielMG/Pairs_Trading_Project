# Pairs Trading: a cointegration-gated statistical arbitrage backtest

A mean-reversion strategy on a pair of related stocks, built to be defensible
rather than impressive. The headline result is a **losing** backtest — and the
point of the project is that the methodology predicted the loss before the P&L
was computed.

---

## Hypothesis

Two economically related stocks share a long-run price relationship. If the pair
is genuinely **cointegrated**, the spread `A − β·B` is stationary: it has a level
it returns to, even though neither stock's price does. Standardise that spread
into a rolling z-score, sell it when it is stretched high, buy it when stretched
low, and close as it reverts.

The hypothesis this project actually tests is narrower and more interesting:
**correlation is not cointegration, and the difference decides whether the
strategy has any reason to work.** Two stocks that trend together are correlated;
that says nothing about whether the gap between them reverts. Only a
cointegration test speaks to the spread itself. So the test is not a
preliminary — it is the entire thesis, and it runs *before* any trading logic.

---

## Methodology

1. **Load data** — real close prices via `yfinance`, or a simulated
   cointegrated pair as a fallback (see [Data note](#data-note)).
2. **Split chronologically** into train (50%), validation (20%), test (30%).
   Never shuffled.
3. **Test for cointegration on train only** (Engle-Granger, p < 0.05). This is
   the gate: it decides whether the pair is worth trading at all. Running it on
   data that includes the test period would leak the future into that decision.
4. **Fit the hedge ratio β on train only** by OLS of A on B.
5. **Build the signal** — spread `A − β·B`, standardised by a *rolling* mean and
   standard deviation, not a full-sample one. A full-sample z-score normalises
   past values using volatility that hadn't happened yet.
6. **Tune on validation only** — grid search over z-score window, entry and exit
   thresholds (36 configurations), selecting the best Sharpe among configs with
   at least 5 trades.
7. **Evaluate once on test**, in dollar terms, net of 5bps costs, with signals
   lagged one day so that today's signal trades tomorrow.

Test data is touched exactly once, at step 7.

---

## Results

### Primary: KO / PEP, real data, 2015–2025

The canonical textbook pair. **It fails the cointegration gate on the training
slice, and loses money out of sample — in that order.**

| | |
|---|---:|
| Train cointegration p-value | **0.2932 — FAIL** (threshold 0.05) |
| Hedge ratio β | 0.3325 |
| Tuned params (validation) | window 30, entry ±1.5, exit ±0.25 |
| Validation Sharpe | 0.362 |
| **Test Sharpe** | **−0.912** |
| Test total return | −6.53% |
| Max drawdown | −8.93% |
| Trades | 27 |
| Win rate | 51.9% |
| Average P&L per trade | −$242.69 |

The p-value of 0.2932 is not a marginal miss, and it is not a power problem: the
training slice runs to 1,382 observations, where the test detects known
cointegration in 37 of 40 simulated trials. This is a clear rejection.

Why it fails is visible in the prices. Over the period PEP roughly doubles, from
about \$65 to \$145, while KO moves from about \$30 to \$68. The two are strongly
*correlated* — they trend together — but the gap between them has no fixed level
to return to. The fitted β of 0.3325 is the regression scaling KO up toward PEP,
and the residual from that fit drifts rather than reverts. Every entry is
therefore a bet that a drifting series will come back. The 51.9% win rate tells
the story: the trades are close to a coin flip, and the losers are larger than
the winners.

**This is the result worth reporting.** The gate said don't trade this pair, and
the out-of-sample P&L confirms that was correct. A backtest that loses money in
exactly the way the methodology predicted is stronger evidence that the
methodology works than a profitable one would be — and it demonstrates the
correlation-versus-cointegration distinction empirically, on the very pair most
often cited as the textbook example.

Reproduce with:

```bash
python main.py --tickers KO PEP
```

### Secondary: a controlled illustration where cointegration does hold

To show the machinery works when the premise is satisfied, the simulator builds
a pair that is cointegrated by construction — a shared random-walk factor plus a
mean-reverting Ornstein-Uhlenbeck spread — so the gate passes and the strategy
has something real to trade.

| | |
|---|---:|
| Train cointegration p-value | **0.000291 — PASS** |
| Hedge ratio β | 1.0040 |
| Tuned params (validation) | window 45, entry ±2.0, exit ±0.75 |
| Validation Sharpe | 1.274 |
| **Test Sharpe** | **0.448** |
| Test total return | +1.31% |
| Max drawdown | −1.66% |
| Trades | 14 |
| Win rate | 78.6% |
| Average P&L per trade | +$92.20 |

This is a sanity check on the implementation, not evidence about markets. The
data was built to be cointegrated, so a positive Sharpe here confirms only that
the pipeline can extract a signal that is known to exist. Treated as a market
result it would be circular; treated as a unit test on the whole system it is
informative.

```bash
python main.py --seed 9   # the default
```

---

## Why the validation-to-test Sharpe gap is the most interesting number

Both runs lose a large fraction of their validation Sharpe out of sample:

| run | validation Sharpe | test Sharpe | gap |
|---|---:|---:|---:|
| KO/PEP (real) | 0.362 | −0.912 | 1.273 |
| simulated, seed 9 | 1.274 | 0.448 | 0.826 |
| simulated, seed 3 | 1.852 | 0.856 | 0.996 |

The gap is not bad luck; it is arithmetic. The validation Sharpe is the
**maximum over 36 configurations**, so it is biased upward by construction —
selecting the best of 36 noisy estimates returns a number inflated by however
much noise the luckiest configuration happened to catch. The test Sharpe is a
single unconditional evaluation of one pre-committed configuration, with no
selection applied. The difference between them is a direct estimate of how much
of the tuned performance was selection effect rather than signal.

Three independent draws each surrendering half to two-thirds of the validation
figure rules out "unlucky test period" as an explanation. If this project
reported only the tuned validation Sharpe of 1.274, that number would be roughly
three times the honest out-of-sample result — which is precisely how backtests
come to be published with Sharpe ratios that never survive contact with live
trading.

**The `min_trades` filter is not decoration.** On the default seed-9 grid the
highest validation Sharpe of all 36 configurations was **1.495, achieved on 2
trades**. Without the 5-trade floor that configuration would have been selected
and its Sharpe reported as the tuned result; with it, the strategy instead takes
the best config with a real track record, at 1.274 on 10 trades. Across 12
simulated draws the filter changed the selected configuration on 3 — seeds 1
(a Sharpe of 2.150 on 3 trades), 2 (1.668 on 4), and 9 (1.495 on 2).

That is roughly a one-in-four chance that an unfiltered grid search would have
reported a headline number generated by two or three lucky trades. On the seed-3
grid it happened not to bind: the winner there had 10 trades and would have won
either way, though 6 of 36 configs were still excluded, including two tied at a
Sharpe of 1.818 on 4 trades each.

---

## Limitations

- **Single pair, single period.** One pair over one decade is one observation.
  Nothing here establishes that the approach generalises.
- **Small trade counts.** 14 to 27 trades out of sample. A Sharpe ratio computed
  from that many trades has enormous error bars, and the 78.6% win rate on the
  simulated run rests on 14 outcomes — a couple of trades flipping sign would
  move it substantially.
- **Static hedge ratio.** β is fitted once on train and held fixed for the whole
  test period. Real relationships drift, and a β estimated on 2015–2020 data is
  a strong assumption about 2022–2025.
- **No regime detection.** The strategy trades identically through calm and
  crisis. A spread that mean-reverts normally can trend hard during a structural
  break, which is when pairs strategies typically take their worst losses.
- **Simplified costs.** A flat 5bps on traded notional, with no slippage, no
  market impact, no borrow cost on the short leg, and no bid-ask modelling. On
  the simulated run costs already consume about 16% of gross P&L, so this
  assumption is load-bearing rather than cosmetic.
- **The split trades statistical power for honesty.** Reserving half the data
  for training is what makes the cointegration gate weak on shorter samples —
  the same discipline that prevents lookahead leaves less data for the test that
  justifies the trade. This is a real tension, not an oversight.
- **A failed gate is not always proof of no relationship.** On short samples,
  Engle-Granger fails to detect cointegration that genuinely exists (see the
  power table in `NOTES.md`: 1 detection in 40 at 250 observations). On
  simulated data we know the ground truth; on real data that distinction is
  unobservable. You cannot tell a truly unrelated pair from a related one your
  sample was too short to confirm.
- **The entry threshold is not a sigma level.** Because the z-score is
  standardised against a trailing window, its dispersion runs about 1.26 rather
  than 1.0, and `|z| > 2` fires on roughly 10% of days instead of 4.6%.
  Describing entries as "2-standard-deviation moves" would be wrong by more than
  a factor of two.

---

## Data note

`load_data` attempts real close prices from `yfinance` and falls back to a
simulated cointegrated pair on any failure — no network, bad ticker, or fewer
than `MIN_ROWS = 1000` observations. The report always states which source was
used, and simulated runs are labelled as not being live-market evidence.

The floor of 1,000 rows is set by the power of the cointegration test, not by
the rolling window. At 250 observations Engle-Granger detects known
cointegration roughly 1 time in 40; a gate evaluated on that little data is not
testing anything.

The simulator does not use two random walks — those are neither correlated nor
cointegrated. It combines a shared stochastic common factor (making the legs
move together and each individually non-stationary) with a mean-reverting
Ornstein-Uhlenbeck spread added to one leg and subtracted from the other (making
the combination stationary), plus independent per-stock noise.

Results on simulated data are draw-dependent, and `--seed` exposes that
honestly:

```bash
python main.py                         # seed 9: gate passes
python main.py --seed 3                # gate fails on train; warns, then trades
python main.py --seed 3 --require-cointegration   # gate fails; stops, no trades
```

Seed 3 is a deliberately retained counter-example: a pair that is cointegrated
by construction but fails the gate on a 750-day training slice through Type II
error alone. It is the cleanest available illustration that a failed gate and a
non-existent relationship are not the same thing.

---

## Extensions

- **Multi-pair universe.** Screen a sector for cointegration rather than
  assuming one pair, with a multiple-testing correction — searching 100 pairs at
  p < 0.05 yields about 5 false positives by construction, which would quietly
  undo the discipline the gate is meant to enforce.
- **Kalman-filtered dynamic hedge ratio.** Let β evolve over time instead of
  fixing it on train, which directly addresses the static-β limitation.
- **Walk-forward re-fitting.** Roll the train/validation/test split forward
  repeatedly instead of splitting once, producing many out-of-sample windows.
  This would substantially help the small-sample problem: the current test
  Sharpe rests on a single period.
- **Regime filter.** Suspend trading when the spread's rolling half-life of
  mean reversion lengthens beyond a threshold, as an early warning that the
  relationship is breaking down.
- **Cost sensitivity curve.** Report Sharpe as a function of assumed cost rather
  than at a single 5bps point, showing where the strategy breaks even.

---

## Verification

Every item in the project's own validation checklist is checked empirically by
`verify_checklist.py`, which tests behaviour rather than reading the source:

```bash
python verify_checklist.py      # 8/8 checks passed
```

It scales the test slice by 1.37x and confirms the cointegration p-value, β and
the tuned parameters are all bit-identical (no leakage) while the test metrics
do move; perturbs a mid-sample spread value and confirms **zero** earlier
rolling z-scores change, against 800 for a full-sample z-score; asserts the
dollar scale of `daily_pnl`; reproduces the lagged P&L formula exactly and
contrasts it with the unlagged variant; confirms costs are charged and material;
reads the threshold values **off the rendered chart object** and compares them to
the tuned parameters; and sweeps seeds to measure how often `min_trades`
actually changes the selection.

One incidental finding from that last check: removing the one-day lag makes this
strategy look *worse*, not better (Sharpe −1.217 against 0.638). Lookahead bias
is usually assumed to flatter results, but for mean reversion the unlagged
version credits each position with the adverse move that triggered the entry.
The lesson is that a lookahead bug will not always announce itself with
implausibly good numbers.

---

## Running it

```bash
pip install -r requirements.txt
python main.py                  # KO/PEP, falling back to simulation offline
python main.py --help           # tickers, dates, seed, costs, capital, output
```

Outputs a text report and a three-panel chart (`backtest_results.png`): full
price history with the split boundaries marked, the out-of-sample z-score with
the *actual tuned* thresholds drawn, and the out-of-sample equity curve net of
costs.

`NOTES.md` is the working log from building this — including two genuine bugs
caught during verification and the numerical evidence behind each decision.
