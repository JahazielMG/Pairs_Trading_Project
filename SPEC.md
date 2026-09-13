# Pairs Trading Backtest — Build Spec

**Purpose:** A cointegration-based statistical arbitrage backtest, built with honest methodology (no lookahead bias, no test-set overfitting) so the results and the code are both genuinely defensible in a job interview.

Give this whole document to Cursor as context, then build it module by module in the order listed. Each module includes the *why*, not just the *what* — keep that reasoning in code comments and the README, since being able to explain these choices matters more than the code itself in interviews.

---

## 0. Project structure

```
pairs_trading_project/
├── data.py          # data loading: real market data or simulated fallback
├── strategy.py       # cointegration test, hedge ratio, z-score signals
├── backtest.py       # train/val/test split, tuning, P&L, metrics
├── main.py            # CLI entry point, orchestration, chart output
├── README.md          # research-note style write-up
└── requirements.txt
```

Dependencies: `pandas`, `numpy`, `statsmodels`, `matplotlib`, and optionally `yfinance` for real data.

---

## 1. Core concept (for your own reference while building)

Two economically related stocks (A and B) share a long-run price relationship. The **spread** = `price_A − beta × price_B`, where beta is a hedge ratio from regressing A on B. If the pair is genuinely **cointegrated**, this spread is mean-reverting even though each stock's raw price is not. Standardize the spread into a rolling **z-score**; trade extreme z-score values betting on reversion to zero; exit as it reverts.

Key distinction to bake into the code and comments: **correlation ≠ cointegration**. Two trending stocks can be highly correlated while their spread drifts forever. Only cointegration confirms the spread itself reverts — this is why the cointegration test comes before any trading logic, not after.

---

## 2. `data.py` — data loading

**Function: `simulate_cointegrated_pair(n_days=1500, seed=7, start_price=100.0)`**

Don't just generate two random walks and call them "cointegrated" — that's not actually cointegrated and defeats the purpose of practicing on realistic data. Build it properly:

- A **shared stochastic common factor** (cumulative sum of small daily shocks) that both stock's log-prices include — this is what makes them move together.
- A **mean-reverting idiosyncratic spread** component using an Ornstein-Uhlenbeck process: `spread[t] = spread[t-1] + theta*(mu - spread[t-1]) + sigma*noise`, with `theta≈0.03`, `mu=0`, `sigma≈0.006`. This is the actual cointegrating relationship — it's what creates a spread that reverts.
- Small independent idiosyncratic noise per stock (each stock has some noise the other doesn't share).
- `log_price_A = log(start_price) + common_factor + spread/2 + small_noise_A`
- `log_price_B = log(start_price) + common_factor − spread/2 + small_noise_B`
- Exponentiate back to price levels. Return a DataFrame with columns `A`, `B`, indexed by business days.

**Function: `load_real_data(ticker_a, ticker_b, start, end)`**
- Use `yfinance` to pull real close prices for both tickers, drop NaNs, rename columns to `A`/`B`.

**Function: `load_data(ticker_a="KO", ticker_b="PEP", start=..., end=...)`**
- Try `load_real_data`. On any exception (no internet, bad ticker, insufficient rows — require at least 250), fall back to `simulate_cointegrated_pair` and print clearly which one was used. Return `(dataframe, is_real: bool)`.

---

## 3. `strategy.py` — signal logic

**`test_cointegration(price_a, price_b)`**
- Use `statsmodels.tsa.stattools.coint`. Return `(is_cointegrated: bool, p_value: float)` using `p_value < 0.05` as the threshold.
- **Critical:** this must only ever be called on the training slice of data, never the full dataset — calling it on data that includes the test period leaks future information into a decision (whether to trade this pair at all) that should only depend on the past.

**`estimate_hedge_ratio(price_a, price_b)`**
- OLS regression: `price_a ~ const + price_b` via `statsmodels.api.OLS`. Return `(beta, alpha)`. Also train-only.

**`compute_spread(price_a, price_b, beta)`** → `price_a - beta * price_b`

**`rolling_zscore(spread, window=30)`**
- `(spread - spread.rolling(window).mean()) / spread.rolling(window).std()`
- **Critical:** must be a *rolling* window, not the full-sample mean/std. A full-sample z-score uses future data to normalize past values — a subtle but real lookahead bug.

**`generate_signals(zscore, entry_z=2.0, exit_z=0.5)`**
- Stateful loop over the z-score series producing a position in `{-1, 0, +1}`:
  - From flat (0): enter **short the spread** (`-1`) if `z > entry_z`; enter **long the spread** (`+1`) if `z < -entry_z`.
  - From long (+1): exit to flat if `z > -exit_z`.
  - From short (-1): exit to flat if `z < exit_z`.
  - Carry forward the previous position on NaN z-score values (early rows before the rolling window fills).

---

## 4. `backtest.py` — the part that's easy to get subtly wrong

This is where the real methodology lives. Build these pieces in order and sanity-check each before moving on.

**`split_data(df, train_frac=0.5, val_frac=0.2)`**
- Three-way chronological split: train / validation / test. Never shuffle — this is time series.
- **Why three-way, not two:** if you tune your z-score window and entry/exit thresholds by checking performance on the same data you report final results on, you've overfit to that specific window — the reported Sharpe ratio becomes meaningless. Validation exists specifically to separate "choosing parameters" from "reporting results."

**`tune_parameters(df, beta, val, window_grid=(15,20,30,45), entry_grid=(1.5,2.0,2.5), exit_grid=(0.25,0.5,0.75), min_trades=5)`**
- Grid search over window/entry/exit combinations, evaluated **only on the validation slice**, selecting the config with the best Sharpe ratio among configs that produced at least `min_trades` (filters out configs that "won" on 1–2 lucky trades — a common false-positive trap in backtesting).

**`_simulate_pnl(price_df, spread, position, beta, notional_per_trade, cost_bps, capital=100_000)`** — a shared P&L engine used both inside tuning and for the final test evaluation.

- **The bug to avoid (I made this mistake building it the first time, catch it now):** don't compute P&L directly in raw spread-price units — `position * spread.diff()` looks reasonable but represents holding exactly *one share* of the spread, which produces P&L in the range of cents on a $100,000 portfolio. It's not wrong math, it's wrong *scale* — it doesn't represent a realistic trade size. Fix: size each trade in dollar terms.
- **Position sizing:** when a new trade opens (position flips from 0 to ±1), compute `shares = notional_per_trade / price_A_at_that_moment` and hold that share count constant for the life of the trade (reset to 0 when flat). `notional_per_trade` around $20,000 is a reasonable default.
- **P&L:** `daily_pnl = position.shift(1) * shares.shift(1) * spread.diff()` — shifted one day so today's signal trades tomorrow, not today (avoiding a look-ahead where you'd trade on the same day's closing price that generated the signal).
- **Transaction costs:** on every day the position changes, charge `cost_bps/10000 * shares * (price_A + beta*price_B)` — a round-trip cost assumption (~5bps is reasonable for liquid large caps) applied to the notional actually traded.
- Return daily P&L, cumulative equity curve (`capital + cumsum(daily_pnl)`), and daily returns (`daily_pnl / capital`).

**`compute_metrics(daily_returns, equity_curve, position, daily_pnl)`**
- **Annualized Sharpe:** `(mean(daily_returns) / std(daily_returns)) * sqrt(252)`
- **Max drawdown:** `((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min()`
- **Total return:** `equity_curve.iloc[-1] / equity_curve.iloc[0] - 1`
- **Trade-level win rate:** group contiguous non-zero position blocks into individual trades, sum `daily_pnl` (dollar terms, not return terms — return-based trade sums are what "average trade P&L" reads as $0.00 due to floating point display, not because they're actually zero) within each block, then compute the fraction with positive total P&L. Also report number of trades and average P&L per trade.

**`run_backtest(df, train_frac=0.5, val_frac=0.2, cost_bps=5, capital=100_000, notional_per_trade=20_000)`**
- Orchestrates: split → fit cointegration + hedge ratio on train → tune params on validation → apply final params to compute spread/z-score/positions on the full series → simulate P&L on the test slice only → compute metrics.
- Return a dict with: `is_cointegrated`, `coint_pvalue`, `beta`, `alpha`, `tuned_params`, `test_index`, `spread`, `zscore`, `position`, `daily_pnl`, `equity_curve`, `daily_returns`, `metrics`.

---

## 5. `main.py` — CLI and output

- `argparse` with `--tickers A B` (default `KO PEP`), `--start`, `--end`.
- Call `load_data`, then `run_backtest`.
- **Print a report** showing: data source (real vs. simulated), cointegration p-value and pass/fail, hedge ratio, tuned parameters and their validation Sharpe, then the test-period metrics (Sharpe, total return, max drawdown, trade count, win rate, avg trade P&L).
- **Plot three stacked panels** (matplotlib, `Agg` backend, saved to PNG):
  1. Full price history for both tickers with a vertical line marking the train/test split.
  2. Out-of-sample z-score with horizontal lines at the *actual tuned* entry/exit thresholds (not hardcoded ±2.0/±0.5 — pull these from the tuned params dict, since hardcoding will silently mislabel the chart if tuning picks different values).
  3. Out-of-sample equity curve, net of costs.

---

## 6. README.md — write like a research note, not a school report

Sections: Hypothesis → Methodology (the 7 steps above, briefly) → Results table → **a short section on why the validation-Sharpe-vs-test-Sharpe gap matters** (if tuning found a config with much higher validation Sharpe than test Sharpe, that gap itself is the most interesting finding — it's the signature of overfitting, and noting it explicitly is more credible than only reporting the headline number) → Limitations (single pair/period, no regime detection, simplified flat cost assumption, small sample size caveats on trade count) → Data note (sandbox/simulated fallback and how to switch to real data) → Extensions (multi-pair universe testing, Kalman-filtered dynamic hedge ratio, walk-forward re-fitting instead of a single static split).

---

## 7. Validation checklist before calling it done

Run through these explicitly — they're the exact mistakes that are easy to make on a first pass:

- [ ] Cointegration test and hedge ratio are computed on **train only**, never on validation or test data.
- [ ] Z-score uses a **rolling**, not full-sample, mean/std.
- [ ] Parameter tuning (window/entry/exit) is selected on **validation**, evaluated finally on **test** — the two are never the same data.
- [ ] Position sizing produces dollar P&L in a realistic range for the portfolio size (thousands, not fractions of a cent) — sanity check by printing `daily_pnl.describe()`.
- [ ] Signals are shifted forward one day before computing P&L (trade tomorrow on today's signal, not today on today's signal).
- [ ] Transaction costs are actually subtracted, and are large enough to matter (check that they're not accidentally zero or negligible).
- [ ] Chart entry/exit threshold lines match the actual tuned parameters, not hardcoded defaults.
- [ ] `min_trades` filter in tuning is actually preventing the selection of high-Sharpe/low-trade-count noise.

---

## 8. Suggested Cursor workflow

To avoid burning credits on rework, build and verify in this order rather than generating everything at once:

1. `data.py` — build and run standalone, print `.describe()` on both price series and eyeball the plot to confirm the two series actually look related but not identical.
2. `strategy.py` — build and test standalone: verify cointegration p-value comes back significant on the simulated pair, inspect a few z-score values manually.
3. `backtest.py` — build incrementally: first `split_data` and confirm sizes, then `_simulate_pnl` on a hardcoded position series and manually sanity-check the dollar P&L magnitude before wiring up tuning, then add `tune_parameters`, then `run_backtest` end to end.
4. `main.py` last, once the pieces are individually verified.

This mirrors how the first version of this project was actually debugged — the position-sizing scale bug above was caught exactly by printing `daily_pnl.describe()` after wiring up P&L for the first time, before trusting the final Sharpe ratio.
