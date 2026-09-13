"""Splitting, P&L simulation, parameter tuning, and metrics.

This is where the methodology lives, and where the mistakes that quietly
inflate a backtest are easiest to make. Two of them are structural and are
handled here rather than in strategy.py:

1. Anything fitted to data -- the cointegration test, the hedge ratio, the
   z-score window and entry/exit thresholds -- is fitted on train or selected
   on validation, and never on test.
2. Signals are acted on one day late, because a signal generated from today's
   close cannot be traded at today's close.
"""

import numpy as np
import pandas as pd

from strategy import (
    COINT_PVALUE_THRESHOLD as COINT_GATE,
    compute_spread,
    estimate_hedge_ratio,
    generate_signals,
    rolling_zscore,
    test_cointegration,
)

TRADING_DAYS = 252


def split_data(df, train_frac=0.5, val_frac=0.2):
    """Chronological three-way split into (train, validation, test).

    Never shuffled -- this is a time series, and a shuffled split would let the
    model learn from data that comes after what it's tested on.

    Three slices rather than two because tuning and reporting are different
    jobs. If the z-score window and entry/exit thresholds are chosen by
    checking performance on the same slice the final Sharpe is reported on,
    that Sharpe is the best of many tries rather than an estimate of future
    performance, and it will be too high by an amount nobody can quantify.
    Validation absorbs the search; test stays untouched until the end.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _simulate_pnl(
    price_df,
    spread,
    position,
    beta,
    notional_per_trade=20_000,
    cost_bps=5,
    capital=100_000,
    return_diagnostics=False,
):
    """Convert a position series into dollar P&L, net of transaction costs.

    Position sizing is the subtle part. The naive `position * spread.diff()`
    is arithmetically fine but represents holding exactly one share of the
    spread, which on a $100k portfolio produces P&L measured in cents -- not
    wrong math, wrong scale, and a Sharpe computed from it is meaningless as a
    portfolio statistic. Instead each trade is sized in dollar terms at entry:
    `shares = notional_per_trade / price_A`, held constant for the life of the
    trade so that intra-trade price moves don't silently resize the position.

    P&L is lagged one day (`position.shift(1)`): today's signal is computed
    from today's close, so the earliest it can be acted on is tomorrow.
    Skipping the shift lets the strategy trade on information it wouldn't have
    had, which is the single easiest way to manufacture a good backtest.
    """
    price_a = price_df["A"].astype(float)
    price_b = price_df["B"].astype(float)
    pos = position.reindex(price_df.index).fillna(0).astype(float)
    spread = spread.reindex(price_df.index).astype(float)

    # Fix the share count when a trade opens and hold it until flat. A position
    # already open at the first row is treated as opening here, since the
    # entry that created it happened outside this slice.
    pos_values = pos.to_numpy()
    price_a_values = price_a.to_numpy()
    shares = np.zeros(len(pos_values))
    held = 0.0
    previous = 0.0
    for i, p in enumerate(pos_values):
        if p == 0:
            held = 0.0
        elif previous == 0 or np.sign(p) != np.sign(previous):
            held = notional_per_trade / price_a_values[i]
        shares[i] = held
        previous = p
    shares = pd.Series(shares, index=price_df.index, name="shares")

    gross_pnl = (pos.shift(1) * shares.shift(1) * spread.diff()).fillna(0.0)

    # Charge costs on the notional that actually changes hands. Using the
    # current day's share count instead would charge nothing on exits, because
    # shares is already 0 there -- that would halve the true cost.
    exposure = pos * shares
    traded_shares = exposure.diff()
    traded_shares.iloc[0] = exposure.iloc[0]
    traded_shares = traded_shares.abs()
    costs = (cost_bps / 10_000.0) * traded_shares * (price_a + beta * price_b)

    daily_pnl = gross_pnl - costs
    equity_curve = capital + daily_pnl.cumsum()
    daily_returns = daily_pnl / capital

    if return_diagnostics:
        return daily_pnl, equity_curve, daily_returns, {
            "shares": shares,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "traded_shares": traded_shares,
        }
    return daily_pnl, equity_curve, daily_returns


def _trade_ids(position):
    """Label each trade's full economic footprint, for trade-level attribution.

    A trade that is open in `position` on days [s, e] earns P&L on days
    [s+1, e+1], because of the one-day execution lag. Day s isn't empty
    though -- it carries the entry cost, since gross P&L there is zero by
    construction (the prior position was flat). So the days belonging to a
    trade are [s, e+1]: entry cost, the P&L itself, and the exit cost.

    Grouping on the raw position blocks instead would drop the entry cost from
    every trade and count one day of P&L that belongs to the next trade,
    which flatters the win rate.
    """
    in_trade = position.fillna(0) != 0
    # shift(fill_value=...) keeps bool dtype. Plain .shift(1).fillna(False)
    # yields object dtype, where `~True` is the integer -2 rather than False --
    # truthy, so the entry mask would silently match every in-trade day.
    was_in_trade = in_trade.shift(1, fill_value=False)
    opened = in_trade & ~was_in_trade
    block_id = opened.cumsum()
    return block_id.where(in_trade | was_in_trade)


def compute_metrics(daily_returns, equity_curve, position, daily_pnl):
    """Portfolio and trade-level statistics for the out-of-sample period."""
    std = daily_returns.std()
    sharpe = float(daily_returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
    max_drawdown = float(drawdown.min())
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    # Trade P&L is summed in dollars, not returns. Summing daily returns per
    # trade produces numbers so small they display as 0.00, which reads as a
    # broken calculation when it's really just a formatting artifact.
    trade_pnl = daily_pnl.groupby(_trade_ids(position)).sum()
    n_trades = int(len(trade_pnl))
    win_rate = float((trade_pnl > 0).mean()) if n_trades else 0.0
    avg_trade_pnl = float(trade_pnl.mean()) if n_trades else 0.0

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "total_pnl": float(daily_pnl.sum()),
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_trade_pnl": avg_trade_pnl,
    }


def tune_parameters(
    df,
    beta,
    val,
    window_grid=(15, 20, 30, 45),
    entry_grid=(1.5, 2.0, 2.5),
    exit_grid=(0.25, 0.5, 0.75),
    min_trades=5,
    cost_bps=5,
    capital=100_000,
    notional_per_trade=20_000,
):
    """Grid search over (window, entry_z, exit_z), scored on validation only.

    The z-score and signals are computed on the full series and then sliced to
    validation, so the rolling window is already warm at the start of the
    validation period. That uses only trailing data, so it's not a lookahead --
    it just avoids throwing away the first `window` days of validation.

    The `min_trades` filter is the part that earns its keep. Across a grid this
    size, some config will always post a spectacular Sharpe off two or three
    lucky trades, and without a floor on trade count the search reliably picks
    exactly that config -- it is selecting noise, and the selection looks like
    a result.
    """
    spread = compute_spread(df["A"], df["B"], beta)
    rows = []

    for window in window_grid:
        zscore = rolling_zscore(spread, window)
        for entry_z in entry_grid:
            for exit_z in exit_grid:
                position = generate_signals(zscore, entry_z, exit_z)
                val_position = position.reindex(val.index)
                daily_pnl, equity_curve, daily_returns = _simulate_pnl(
                    val,
                    spread,
                    val_position,
                    beta,
                    notional_per_trade=notional_per_trade,
                    cost_bps=cost_bps,
                    capital=capital,
                )
                metrics = compute_metrics(
                    daily_returns, equity_curve, val_position, daily_pnl
                )
                rows.append(
                    {
                        "window": window,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "sharpe": metrics["sharpe"],
                        "n_trades": metrics["n_trades"],
                        "total_pnl": metrics["total_pnl"],
                    }
                )

    results = pd.DataFrame(rows)
    eligible = results[results["n_trades"] >= min_trades]

    if eligible.empty:
        # Don't silently fall back to the unfiltered winner -- that is exactly
        # the low-trade-count noise the filter exists to reject.
        raise ValueError(
            f"no parameter configuration produced at least {min_trades} trades on "
            f"the validation slice (best was {int(results['n_trades'].max())}). "
            "The validation window is likely too short to tune on."
        )

    best = eligible.loc[eligible["sharpe"].idxmax()]
    best_params = {
        "window": int(best["window"]),
        "entry_z": float(best["entry_z"]),
        "exit_z": float(best["exit_z"]),
        "val_sharpe": float(best["sharpe"]),
        "val_trades": int(best["n_trades"]),
    }
    return best_params, results


def _warn_not_cointegrated(coint_pvalue, n_train):
    """Loud banner whenever the strategy trades a pair that failed its gate."""
    print("\n" + "!" * 72)
    print("!! WARNING: trading a pair that FAILED the cointegration gate on train")
    print(f"!! train p-value = {coint_pvalue:.4f} (threshold {COINT_GATE:.2f}), "
          f"n_train = {n_train}")
    print("!! The premise of this strategy is a stationary spread. Without it,")
    print("!! there is no reason to expect mean reversion and the results below")
    print("!! should not be read as evidence the strategy works.")
    print("!! Note this may be a Type II error rather than a genuinely")
    print("!! non-cointegrated pair -- Engle-Granger has weak power on short")
    print("!! samples (see NOTES.md). Either way the gate did not pass.")
    print("!" * 72 + "\n")


def run_backtest(
    df,
    train_frac=0.5,
    val_frac=0.2,
    cost_bps=5,
    capital=100_000,
    notional_per_trade=20_000,
    require_cointegration=False,
):
    """Full pipeline: split, fit on train, tune on validation, report on test.

    `require_cointegration` controls what happens when the train slice fails
    the gate. Off by default, the backtest proceeds and prints a warning --
    useful because a failed gate is often a power problem rather than proof the
    pair is unrelated, and seeing the P&L is informative. Switched on, it stops
    before trading, which is the behaviour the gate's rationale actually
    implies. Either way the failure is never silent.
    """
    train, val, test = split_data(df, train_frac, val_frac)

    # Fitted on train only. Both of these are decisions informed by data, and
    # both would leak the future if they saw validation or test.
    is_cointegrated, coint_pvalue = test_cointegration(train["A"], train["B"])
    beta, alpha = estimate_hedge_ratio(train["A"], train["B"])

    if not is_cointegrated:
        if require_cointegration:
            print(
                f"\nSTOPPED: train-slice cointegration p-value {coint_pvalue:.4f} "
                f"exceeds {COINT_GATE:.2f} and require_cointegration=True.\n"
                "No trades simulated.\n"
            )
            return {
                "is_cointegrated": False,
                "coint_pvalue": coint_pvalue,
                "beta": beta,
                "alpha": alpha,
                "traded": False,
                "train_index": train.index,
                "val_index": val.index,
                "test_index": test.index,
                "tuned_params": None,
                "tuning_results": None,
                "spread": None,
                "zscore": None,
                "position": None,
                "daily_pnl": None,
                "equity_curve": None,
                "daily_returns": None,
                "metrics": None,
            }
        _warn_not_cointegrated(coint_pvalue, len(train))

    tuned_params, tuning_results = tune_parameters(
        df,
        beta,
        val,
        min_trades=5,
        cost_bps=cost_bps,
        capital=capital,
        notional_per_trade=notional_per_trade,
    )

    spread = compute_spread(df["A"], df["B"], beta)
    zscore = rolling_zscore(spread, tuned_params["window"])
    position = generate_signals(
        zscore, tuned_params["entry_z"], tuned_params["exit_z"]
    )

    # Test is touched here for the first time and only here.
    test_position = position.reindex(test.index)
    daily_pnl, equity_curve, daily_returns = _simulate_pnl(
        test,
        spread,
        test_position,
        beta,
        notional_per_trade=notional_per_trade,
        cost_bps=cost_bps,
        capital=capital,
    )
    metrics = compute_metrics(daily_returns, equity_curve, test_position, daily_pnl)

    return {
        "is_cointegrated": is_cointegrated,
        "coint_pvalue": coint_pvalue,
        "beta": beta,
        "alpha": alpha,
        "traded": True,
        "tuned_params": tuned_params,
        "tuning_results": tuning_results,
        "train_index": train.index,
        "val_index": val.index,
        "test_index": test.index,
        "spread": spread,
        "zscore": zscore,
        "position": position,
        "daily_pnl": daily_pnl,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "metrics": metrics,
    }
