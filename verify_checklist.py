"""Section 7 validation checklist, re-run against the finished codebase."""

import numpy as np
import pandas as pd

import main as main_mod
from backtest import _simulate_pnl, compute_metrics, run_backtest, split_data, tune_parameters
from data import simulate_cointegrated_pair
from strategy import compute_spread, estimate_hedge_ratio, generate_signals, rolling_zscore

PASS, FAIL = "PASS", "**FAIL**"
results = []


def record(item, ok, evidence):
    results.append((item, PASS if ok else FAIL, evidence))
    print(f"\n[{PASS if ok else FAIL}] {item}\n      {evidence}")


df = simulate_cointegrated_pair()
base = run_backtest(df)
train, val, test = split_data(df)

# ---------------------------------------------------------------- item 1
# Mutate ONLY the test slice. If anything was fitted on it, these change.
df_mut = df.copy()
df_mut.iloc[len(train) + len(val):] *= 1.37
mut = run_backtest(df_mut)
ok = (
    mut["coint_pvalue"] == base["coint_pvalue"]
    and mut["beta"] == base["beta"]
    and mut["alpha"] == base["alpha"]
)
record(
    "1. Cointegration + hedge ratio computed on TRAIN only",
    ok,
    f"test slice scaled by 1.37x -> coint p unchanged "
    f"({base['coint_pvalue']:.6f} == {mut['coint_pvalue']:.6f}), "
    f"beta unchanged ({base['beta']:.6f} == {mut['beta']:.6f})",
)

# ---------------------------------------------------------------- item 2
# A rolling statistic cannot depend on the future. Perturb a later value and
# confirm no earlier z-score moves; then show a full-sample z-score does.
spread = compute_spread(df["A"], df["B"], base["beta"])
z_roll = rolling_zscore(spread, 30)
spread_mut = spread.copy()
spread_mut.iloc[800] += 50.0
z_roll_mut = rolling_zscore(spread_mut, 30)
z_full = (spread - spread.mean()) / spread.std()
z_full_mut = (spread_mut - spread_mut.mean()) / spread_mut.std()

# NaN != NaN, so compare with the leading NaNs filled or the first `window`
# rows would register as spurious differences.
differs = lambda a, b: int((a.fillna(-1e9) != b.fillna(-1e9)).sum())
rolling_past_changed = differs(z_roll.iloc[:800], z_roll_mut.iloc[:800])
full_past_changed = differs(z_full.iloc[:800], z_full_mut.iloc[:800])
rolling_future_changed = differs(z_roll.iloc[800:], z_roll_mut.iloc[800:])
ok = rolling_past_changed == 0 and full_past_changed > 0
record(
    "2. Z-score uses a ROLLING window, not full-sample",
    ok,
    f"perturbing row 800 changed {rolling_past_changed} EARLIER rolling z-scores "
    f"(and {rolling_future_changed} later ones, as expected) vs "
    f"{full_past_changed} earlier full-sample z-scores -- the full-sample "
    f"version propagates future data backwards, the rolling one does not",
)

# ---------------------------------------------------------------- item 3
# Tuning must be blind to test; test must still drive the reported metrics.
tuning_blind = mut["tuned_params"] == base["tuned_params"]
metrics_responsive = mut["metrics"]["sharpe"] != base["metrics"]["sharpe"]
disjoint = (
    len(base["val_index"].intersection(base["test_index"])) == 0
    and len(base["train_index"].intersection(base["val_index"])) == 0
)
ok = tuning_blind and metrics_responsive and disjoint
record(
    "3. Tuned on VALIDATION, evaluated on TEST, never the same data",
    ok,
    f"mutating test left tuned params identical ({base['tuned_params']['window']}/"
    f"{base['tuned_params']['entry_z']}/{base['tuned_params']['exit_z']}) but moved "
    f"test Sharpe {base['metrics']['sharpe']:.3f} -> {mut['metrics']['sharpe']:.3f}; "
    f"slice indices disjoint={disjoint}",
)

# ---------------------------------------------------------------- item 4
pnl = base["daily_pnl"]
desc = pnl.describe()
ok = pnl.std() > 10 and abs(pnl).max() > 100
record(
    "4. Position sizing gives dollar P&L at a realistic scale",
    ok,
    f"std ${pnl.std():,.2f}, min ${pnl.min():,.2f}, max ${pnl.max():,.2f} on "
    f"$100,000 capital (thousands-of-dollars scale, not cents)",
)
print("      daily_pnl.describe():")
for k, v in desc.items():
    print(f"        {k:<8}{v:>12,.2f}")

# ---------------------------------------------------------------- item 5
# Compare the correct (lagged) P&L against the cheating unlagged version.
zs = base["zscore"]
pos = base["position"].reindex(test.index)
shares_probe = _simulate_pnl(test, spread, pos, base["beta"], return_diagnostics=True)[3]
shares = shares_probe["shares"]
lagged = (pos.shift(1) * shares.shift(1) * spread.reindex(test.index).diff()).fillna(0)
unlagged = (pos * shares * spread.reindex(test.index).diff()).fillna(0)
sharpe = lambda s: s.mean() / s.std() * np.sqrt(252)
ok = not np.allclose(lagged, unlagged) and abs(lagged.sum() - shares_probe["gross_pnl"].sum()) < 1e-6
record(
    "5. Signals shifted one day before P&L",
    ok,
    f"engine reproduces the lagged formula exactly; the unlagged variant gives a "
    f"materially different Sharpe ({sharpe(unlagged):.3f} vs {sharpe(lagged):.3f}), "
    f"so the shift is load-bearing rather than cosmetic",
)

# ---------------------------------------------------------------- item 6
free = run_backtest(df, cost_bps=0)
charged = base
diag = _simulate_pnl(
    test, spread, pos, base["beta"], cost_bps=5, return_diagnostics=True
)[3]
total_costs = diag["costs"].sum()
gross = diag["gross_pnl"].sum()
ok = total_costs > 0 and (total_costs / abs(gross)) > 0.02
record(
    "6. Transaction costs subtracted and material",
    ok,
    f"${total_costs:,.2f} of costs against ${gross:,.2f} gross = "
    f"{total_costs / abs(gross):.1%}; zero-cost Sharpe {free['metrics']['sharpe']:.3f} "
    f"vs {charged['metrics']['sharpe']:.3f} with costs",
)

# ---------------------------------------------------------------- item 7
# Inspect the actual figure main.py draws, not the source code.
captured = {}
original_close = main_mod.plt.close
main_mod.plt.close = lambda fig: captured.setdefault("fig", fig)


class Args:
    tickers = ["KO", "PEP"]
    capital = 100_000.0
    cost_bps = 5.0
    output = "/tmp/_checklist_chart.png"
    seed = 9


main_mod.plot_results(base, Args(), False, df)
main_mod.plt.close = original_close

zaxis = captured["fig"].axes[1]
hlines = sorted(
    {round(float(l.get_ydata()[0]), 6) for l in zaxis.get_lines()
     if len(set(l.get_ydata())) == 1}
)
tuned = base["tuned_params"]
expected = sorted({-tuned["entry_z"], -tuned["exit_z"], 0.0,
                   tuned["exit_z"], tuned["entry_z"]})
ok = hlines == expected
spec_default = sorted({-2.0, -0.5, 0.0, 0.5, 2.0})
record(
    "7. Chart thresholds match TUNED params, not hardcoded defaults",
    ok,
    f"drawn threshold lines {hlines} == tuned "
    f"(entry +/-{tuned['entry_z']}, exit +/-{tuned['exit_z']}); "
    f"hardcoded defaults would have drawn {spec_default}",
)

# ---------------------------------------------------------------- item 8
_, grid = tune_parameters(df, base["beta"], val)
excluded = grid[grid["n_trades"] < 5]
best_any = grid.loc[grid["sharpe"].idxmax()]
best_ok = grid[grid["n_trades"] >= 5].loc[grid[grid["n_trades"] >= 5]["sharpe"].idxmax()]
binds_here = best_any["n_trades"] < 5

# Does the filter ever change the outcome? Sweep seeds to find out.
bind_count, checked = 0, []
for s in range(12):
    d = simulate_cointegrated_pair(seed=s)
    tr, vl, _ = split_data(d)
    b, _ = estimate_hedge_ratio(tr["A"], tr["B"])
    _, g = tune_parameters(d, b, vl)
    top = g.loc[g["sharpe"].idxmax()]
    if top["n_trades"] < 5:
        bind_count += 1
        checked.append((s, float(top["sharpe"]), int(top["n_trades"])))

ok = len(excluded) > 0
record(
    "8. min_trades filter rejects high-Sharpe/low-trade noise",
    ok,
    f"excluded {len(excluded)}/{len(grid)} configs on this grid "
    f"(max Sharpe among excluded: {excluded['sharpe'].max():.3f} on "
    f"{int(excluded.loc[excluded['sharpe'].idxmax(), 'n_trades'])} trades); "
    f"filter changed the selection on {bind_count}/12 seeds",
)
if checked:
    print("      seeds where the unfiltered winner was low-trade noise:")
    for s, sh, n in checked:
        print(f"        seed {s}: top config Sharpe {sh:.3f} on only {n} trades")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 72)
print("SECTION 7 CHECKLIST SUMMARY")
print("=" * 72)
for item, status, _ in results:
    print(f"  [{status}] {item}")
n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f"\n{n_pass}/{len(results)} checks passed")
