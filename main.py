"""CLI entry point: load data, run the backtest, print a report, save charts."""

import argparse

import matplotlib

matplotlib.use("Agg")  # no display in this environment; write PNGs directly
import matplotlib.pyplot as plt

from backtest import run_backtest
from data import DEFAULT_END, DEFAULT_START, load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cointegration-based pairs trading backtest."
    )
    parser.add_argument(
        "--tickers", nargs=2, metavar=("A", "B"), default=["KO", "PEP"],
        help="the two tickers to trade as a pair (default: KO PEP)",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="end date YYYY-MM-DD")
    parser.add_argument(
        "--seed", type=int, default=9,
        help="RNG seed for the simulated fallback. Seed 3 is a documented "
             "edge case whose training slice fails the cointegration gate.",
    )
    parser.add_argument(
        "--require-cointegration", action="store_true",
        help="stop instead of trading when the train slice fails the gate",
    )
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--notional-per-trade", type=float, default=20_000.0)
    parser.add_argument("--output", default="backtest_results.png")
    return parser.parse_args()


def print_report(results, args, is_real):
    """Text report. Everything fitted is labelled with the slice it was fit on."""
    ticker_a, ticker_b = args.tickers
    label_a, label_b = (ticker_a, ticker_b) if is_real else ("A (sim)", "B (sim)")

    print("\n" + "=" * 64)
    print(f"  PAIRS TRADING BACKTEST — {label_a} / {label_b}")
    print("=" * 64)

    source = "real market data" if is_real else f"SIMULATED pair (seed={args.seed})"
    print(f"\nData source                 {source}")
    if not is_real:
        print("                            results are not live-market evidence")

    print("\n--- Fitted on TRAIN only " + "-" * 39)
    verdict = "PASS" if results["is_cointegrated"] else "FAIL"
    print(f"Cointegration p-value       {results['coint_pvalue']:.6f}   [{verdict}]")
    print(f"Hedge ratio (beta)          {results['beta']:.4f}")
    print(f"Intercept (alpha)           {results['alpha']:.4f}")

    if not results["traded"]:
        print("\nNo trades simulated: the pair failed its cointegration gate and")
        print("--require-cointegration was set. Nothing further to report.")
        print("=" * 64 + "\n")
        return

    params = results["tuned_params"]
    print("\n--- Tuned on VALIDATION only " + "-" * 35)
    print(f"Z-score window              {params['window']}")
    print(f"Entry threshold             +/- {params['entry_z']}")
    print(f"Exit threshold              +/- {params['exit_z']}")
    print(f"Validation Sharpe           {params['val_sharpe']:.3f}"
          f"  ({params['val_trades']} trades)")

    metrics = results["metrics"]
    print("\n--- Evaluated on TEST only " + "-" * 37)
    print(f"Sharpe ratio (annualised)   {metrics['sharpe']:.3f}")
    print(f"Total return                {metrics['total_return']:.2%}")
    print(f"Max drawdown                {metrics['max_drawdown']:.2%}")
    print(f"Total P&L                   ${metrics['total_pnl']:,.2f}")
    print(f"Number of trades            {metrics['n_trades']}")
    print(f"Win rate                    {metrics['win_rate']:.1%}")
    print(f"Average P&L per trade       ${metrics['avg_trade_pnl']:,.2f}")

    # The validation-to-test gap is the most informative number here, so it
    # gets stated outright rather than left for the reader to subtract.
    gap = params["val_sharpe"] - metrics["sharpe"]
    print(f"\nValidation -> test Sharpe    {params['val_sharpe']:.3f} -> "
          f"{metrics['sharpe']:.3f}  (gap {gap:+.3f})")
    if gap > 0.3:
        print("The validation figure is the best of a 36-config grid search, so it")
        print("is biased upward; the test figure is a single unconditional")
        print("evaluation. A gap of this size is the signature of that selection.")

    pnl = results["daily_pnl"]
    print(f"\nDaily P&L sanity            mean ${pnl.mean():,.2f}, "
          f"std ${pnl.std():,.2f}, range ${pnl.min():,.2f} to ${pnl.max():,.2f}")
    print("=" * 64 + "\n")


def plot_results(results, args, is_real, df):
    """Build three stacked panels: prices with splits, OOS z-score, OOS equity.

    Returns the figure rather than writing it. The caller decides what to do
    with it -- the CLI saves a PNG, the Streamlit app renders it in the page.
    Saving to a fixed path in here would mean concurrent users of the deployed
    app overwriting each other's chart between render and read.
    """
    ticker_a, ticker_b = args.tickers
    label_a, label_b = (ticker_a, ticker_b) if is_real else ("A (sim)", "B (sim)")
    params = results["tuned_params"]
    test_index = results["test_index"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # Panel 1: full price history, with the slice boundaries marked so it is
    # visually obvious how little of the data the reported metrics come from.
    axes[0].plot(df.index, df["A"], lw=1, label=label_a)
    axes[0].plot(df.index, df["B"], lw=1, label=label_b)
    axes[0].axvline(results["val_index"][0], color="black", ls="--", lw=1.2,
                    label="train/validation split")
    axes[0].axvline(test_index[0], color="red", ls="--", lw=1.2,
                    label="validation/test split")
    axes[0].set_title(f"Price history — {label_a} vs {label_b}")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="best", fontsize=9)

    # Panel 2: thresholds come from the tuned params, never hardcoded --
    # hardcoding +/-2.0 and +/-0.5 would mislabel the chart whenever tuning
    # picked anything else, which it did here.
    zscore = results["zscore"].reindex(test_index)
    entry_z, exit_z = params["entry_z"], params["exit_z"]
    axes[1].plot(zscore.index, zscore, lw=1, color="tab:blue")
    axes[1].axhline(entry_z, color="red", ls="--", lw=1,
                    label=f"entry +/-{entry_z}")
    axes[1].axhline(-entry_z, color="red", ls="--", lw=1)
    axes[1].axhline(exit_z, color="green", ls="--", lw=1,
                    label=f"exit +/-{exit_z}")
    axes[1].axhline(-exit_z, color="green", ls="--", lw=1)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title(
        f"Out-of-sample z-score (window={params['window']}) with tuned thresholds"
    )
    axes[1].set_ylabel("z-score")
    axes[1].legend(loc="best", fontsize=9)

    # Panel 3: equity is already net of transaction costs.
    equity = results["equity_curve"]
    axes[2].plot(equity.index, equity, lw=1.2, color="tab:green")
    axes[2].axhline(args.capital, color="black", ls=":", lw=1,
                    label=f"starting capital ${args.capital:,.0f}")
    axes[2].set_title(
        f"Out-of-sample equity curve, net of {args.cost_bps:g}bps costs "
        f"(Sharpe {results['metrics']['sharpe']:.2f})"
    )
    axes[2].set_ylabel("Equity ($)")
    axes[2].legend(loc="best", fontsize=9)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    df, is_real = load_data(
        args.tickers[0], args.tickers[1], args.start, args.end, seed=args.seed
    )
    results = run_backtest(
        df,
        cost_bps=args.cost_bps,
        capital=args.capital,
        notional_per_trade=args.notional_per_trade,
        require_cointegration=args.require_cointegration,
    )
    print_report(results, args, is_real)
    if results["traded"]:
        fig = plot_results(results, args, is_real, df)
        fig.savefig(args.output, dpi=120)
        plt.close(fig)
        print(f"Chart saved to {args.output}\n")


if __name__ == "__main__":
    main()
