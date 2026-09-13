"""Streamlit front end for the pairs trading backtest.

This module is UI only. Every number it shows comes from `run_backtest`, the
same call `main.py` makes, and the chart comes from `main.plot_results` -- so
there is no second implementation of the methodology that could drift away
from the command line version and start reporting something different.

Run locally with:

    streamlit run app.py
"""

from types import SimpleNamespace

import matplotlib

# Set before pyplot is imported. Streamlit serves requests off the main thread,
# where the default macOS backend raises.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backtest import run_backtest
from data import (
    DEFAULT_END,
    DEFAULT_START,
    MIN_ROWS,
    load_real_data,
    simulate_cointegrated_pair,
)
from main import plot_results
from strategy import COINT_PVALUE_THRESHOLD

SIM_SEED = 9
PRESET_KOPEP = "kopep"
PRESET_SIM = "sim"

# Liquid large caps with long, unbroken histories, so a run is unlikely to fall
# below MIN_ROWS and drop to the simulator. Alphabetical rather than grouped by
# sector: the point of the dropdown is scanning for a known symbol, and pairing
# hints belong in COMMON_PAIRS instead.
CUSTOM_TICKER = "Custom..."
TICKERS = [
    "AAPL", "AMZN", "BAC", "CL", "CVX", "GOOGL", "HD", "JNJ", "JPM", "KO",
    "LOW", "MA", "MSFT", "PEP", "PG", "TGT", "UNH", "V", "WMT", "XOM",
]
TICKER_OPTIONS = TICKERS + [CUSTOM_TICKER]

# Pairs that get cited as natural candidates because the two businesses are
# economically similar. That similarity is an argument about correlation, and
# whether any of them is actually cointegrated is the question this app exists
# to answer -- KO / PEP is on the list precisely because the answer is no.
CUSTOM_PAIR = "Custom"
COMMON_PAIRS = {
    "KO / PEP": ("KO", "PEP"),
    "XOM / CVX": ("XOM", "CVX"),
    "JPM / BAC": ("JPM", "BAC"),
    "GOOGL / MSFT": ("GOOGL", "MSFT"),
    "HD / LOW": ("HD", "LOW"),
    "MA / V": ("MA", "V"),
    "PG / CL": ("PG", "CL"),
    "WMT / TGT": ("WMT", "TGT"),
}
PAIR_OPTIONS = list(COMMON_PAIRS) + [CUSTOM_PAIR]


# ----------------------------------------------------------------- data access


def _load_with_reason(ticker_a, ticker_b, start, end, seed):
    """Load real prices, falling back to simulation, keeping the reason why.

    Mirrors `data.load_data` deliberately rather than calling it. `load_data`
    prints its fallback reason to stdout, which is invisible in a browser, and
    the reason matters here: the fallback fires on a bad ticker or too little
    history just as readily as on a missing network, and a badge that always
    blamed the network would be telling the visitor something untrue.

    Returns `(df, is_real, reason)` where reason is None on a real-data load.
    """
    try:
        df = load_real_data(ticker_a, ticker_b, start, end)
        if len(df) < MIN_ROWS:
            raise ValueError(
                f"only {len(df)} rows of overlapping history, and the "
                f"cointegration gate needs at least {MIN_ROWS} to mean anything"
            )
        return df, True, None
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        return simulate_cointegrated_pair(seed=seed), False, reason


@st.cache_data(show_spinner=False)
def run_real_pair(ticker_a, ticker_b, start, end, require_cointegration, fixed):
    """Backtest a named pair, falling back to simulated data if it won't load.

    Cached on the arguments, all of them scalars, so re-rendering the page or
    switching tabs doesn't refetch prices or rerun a grid search. A genuinely
    new pair, date range or parameter set misses the cache and runs fresh.
    `fixed` is a `(window, entry_z, exit_z)` tuple or None; tuples hash cleanly
    as cache keys where a dict is awkward.
    """
    df, is_real, reason = _load_with_reason(
        ticker_a, ticker_b, start, end, SIM_SEED
    )
    result = run_backtest(
        df,
        require_cointegration=require_cointegration,
        fixed_params=_fixed_dict(fixed),
    )
    return df, is_real, reason, result


@st.cache_data(show_spinner=False)
def run_simulated_pair(seed, require_cointegration, fixed):
    """Backtest the simulated pair, bypassing the real-data path entirely.

    Tab 1's "a pair that works" preset is *about* the simulator, so it calls
    `simulate_cointegrated_pair` directly instead of going through a loader
    that would return real KO/PEP prices whenever the network happens to be up.
    """
    df = simulate_cointegrated_pair(seed=seed)
    result = run_backtest(
        df,
        require_cointegration=require_cointegration,
        fixed_params=_fixed_dict(fixed),
    )
    reason = f"this preset is built on the simulator by design (seed {seed})"
    return df, False, reason, result


def _fixed_dict(fixed):
    """Turn the cache-friendly tuple back into `run_backtest`'s dict."""
    if fixed is None:
        return None
    window, entry_z, exit_z = fixed
    return {"window": window, "entry_z": entry_z, "exit_z": exit_z}


# -------------------------------------------------------- shared result render


def render_data_source_badge(is_real, reason, ticker_a, ticker_b, n_rows):
    """Say plainly where the numbers came from, on every single result.

    The real-versus-simulated distinction is load-bearing for this project's
    credibility, so it survives the move to a GUI as the first thing on screen
    rather than a footnote under the charts.
    """
    if is_real:
        st.success(
            f"**REAL MARKET DATA** — {ticker_a} / {ticker_b}, "
            f"{n_rows:,} daily closes from yfinance."
        )
    else:
        st.warning(
            f"**SIMULATED DATA — not live-market evidence.** "
            f"{n_rows:,} rows from the built-in cointegrated simulator, "
            f"because real prices were unavailable: {reason}. "
            "The pair is cointegrated by construction, so a positive result "
            "here tests the pipeline, not the market."
        )


def render_cointegration_gate(result, ticker_a, ticker_b):
    """The gate verdict and what it means, in words rather than just a number."""
    p_value = result["coint_pvalue"]
    passed = result["is_cointegrated"]

    left, right = st.columns([1, 2])
    left.metric("Train cointegration p-value", f"{p_value:.4f}")
    # A metric `delta` would render an up/down arrow next to the verdict, which
    # reads as "the p-value rose" rather than "the gate failed".
    left.markdown(
        ":green-background[**PASS**]" if passed else ":red-background[**FAIL**]"
    )

    if passed:
        right.markdown(
            f"**The gate passed** (p = {p_value:.4f} < "
            f"{COINT_PVALUE_THRESHOLD}). On the training slice the spread "
            f"between {ticker_a} and {ticker_b} tests as stationary — it has a "
            "level it returns to — so mean reversion has a reason to work here."
        )
    else:
        right.markdown(
            f"**The gate failed** (p = {p_value:.4f}, above the "
            f"{COINT_PVALUE_THRESHOLD} threshold). The spread between "
            f"{ticker_a} and {ticker_b} shows no stationary level to revert "
            "to on the training data. The two can still be strongly "
            "*correlated* — trending together says nothing about whether the "
            "gap between them closes. Every entry below is therefore a bet "
            "that a drifting series will come back.\n\n"
            "One honest caveat: on short samples Engle-Granger also fails to "
            "detect cointegration that genuinely exists, so a failure is not "
            "proof the pair is unrelated. On real data those two cases are "
            "indistinguishable."
        )


def render_gate_stop(result, ticker_a, ticker_b):
    """The require_cointegration stop path: explain it, don't show empty charts."""
    st.error(
        f"**No trades were simulated.** {ticker_a} / {ticker_b} failed the "
        f"cointegration gate on the training slice at p = "
        f"{result['coint_pvalue']:.4f}, and *Require cointegration* is on, so "
        "the backtest stopped before tuning or trading."
    )
    st.markdown(
        "This is the gate doing its job. The strategy's entire premise is a "
        "stationary spread; without one there is no reason to expect reversion, "
        "so the honest response is not to trade the pair at all. The hedge "
        "ratio below was still fitted on train and is shown for reference."
    )
    st.metric("Hedge ratio (beta)", f"{result['beta']:.4f}")
    st.caption(
        "Turn off *Require cointegration* in the sidebar to trade it anyway "
        "and see the out-of-sample P&L — which is exactly how the KO/PEP "
        "headline result on the other tab is produced."
    )


def render_parameters(result):
    """Hedge ratio plus the parameters, labelled with how they were chosen."""
    params = result["tuned_params"]
    is_tuned = params.get("is_tuned", True)

    st.markdown(
        "##### Parameters "
        + ("(tuned on validation only)" if is_tuned else "(set manually)")
    )
    cols = st.columns(4)
    cols[0].metric("Hedge ratio (beta)", f"{result['beta']:.4f}")
    cols[1].metric("Z-score window", params["window"])
    cols[2].metric("Entry threshold", f"±{params['entry_z']:g}")
    cols[3].metric("Exit threshold", f"±{params['exit_z']:g}")

    if is_tuned:
        st.caption(
            f"Chosen by grid search over 36 configurations, scored on the "
            f"validation slice only: validation Sharpe "
            f"{params['val_sharpe']:.3f} on {params['val_trades']} trades."
        )
    else:
        st.caption(
            "Chosen by you, so no grid search ran and the validation slice "
            "went unused. The cointegration gate above still ran on train."
        )


def render_metrics(result):
    """The five headline out-of-sample numbers as metric cards."""
    metrics = result["metrics"]
    st.markdown("##### Out-of-sample results (test slice, touched once)")
    cols = st.columns(5)
    cols[0].metric("Sharpe ratio", f"{metrics['sharpe']:.3f}")
    cols[1].metric("Total return", f"{metrics['total_return']:.2%}")
    cols[2].metric("Max drawdown", f"{metrics['max_drawdown']:.2%}")
    cols[3].metric("Trades", metrics["n_trades"])
    cols[4].metric("Win rate", f"{metrics['win_rate']:.1%}")

    if metrics["n_trades"] == 0:
        st.info(
            "**No trades opened**, so every figure above is zero by default "
            "rather than measured. The z-score never crossed this entry "
            "threshold during the test period — which is itself the answer to "
            "how sensitive the strategy is to that one number."
        )
        return

    # Dollar signs are escaped throughout: Streamlit's markdown reads a pair of
    # unescaped `$` as LaTeX delimiters and swallows both signs and formatting.
    st.caption(
        f"Net of costs. Total P&L \\${metrics['total_pnl']:,.2f}, averaging "
        f"\\${metrics['avg_trade_pnl']:,.2f} per trade."
    )


def render_sharpe_gap(result):
    """The validation-to-test gap, stated outright.

    This is the most informative number the project produces, so it gets its
    own section instead of being left for the reader to subtract out of two
    metric cards on opposite sides of the page.
    """
    params = result["tuned_params"]
    test_sharpe = result["metrics"]["sharpe"]

    st.markdown("##### Validation → test Sharpe")

    if not params.get("is_tuned", True):
        st.info(
            f"Not applicable to a manual run: nothing was selected on "
            f"validation, so there is no selection effect to measure. The "
            f"test Sharpe of {test_sharpe:.3f} stands on its own. Switch to "
            "auto-tune to see the gap."
        )
        return

    val_sharpe = params["val_sharpe"]
    gap = val_sharpe - test_sharpe

    cols = st.columns(3)
    cols[0].metric("Validation Sharpe", f"{val_sharpe:.3f}")
    cols[1].metric("Test Sharpe", f"{test_sharpe:.3f}")
    cols[2].metric("Gap", f"{gap:+.3f}")

    if gap > 0.3:
        st.markdown(
            f"**The gap is the finding.** The validation figure of "
            f"{val_sharpe:.3f} is the *maximum* over 36 configurations, so it "
            "is biased upward by construction — picking the best of 36 noisy "
            "estimates returns a number inflated by whatever noise the "
            "luckiest configuration caught. The test figure of "
            f"{test_sharpe:.3f} is a single unconditional evaluation of one "
            "pre-committed configuration. The difference between them is a "
            "direct estimate of how much of the tuned performance was "
            "selection effect rather than signal."
        )


def render_chart(result, is_real, df, ticker_a, ticker_b, capital, cost_bps):
    """The same three-panel figure the CLI saves, rendered into the page."""
    fig = plot_results(
        result,
        SimpleNamespace(
            tickers=[ticker_a, ticker_b], capital=capital, cost_bps=cost_bps
        ),
        is_real,
        df,
    )
    st.pyplot(fig)
    plt.close(fig)


def render_backtest_result(
    result,
    is_real,
    df,
    reason=None,
    ticker_a="A",
    ticker_b="B",
    capital=100_000.0,
    cost_bps=5.0,
):
    """Render one backtest result. The single code path for both tabs.

    Called identically from Key Findings and Try It Yourself, so "what a
    result looks like" is defined once. Handles the no-trade case, where most
    of the result dict is None because the gate stopped the run early.
    """
    render_data_source_badge(is_real, reason, ticker_a, ticker_b, len(df))
    render_cointegration_gate(result, ticker_a, ticker_b)
    st.divider()

    if not result["traded"]:
        render_gate_stop(result, ticker_a, ticker_b)
        return

    render_parameters(result)
    st.divider()
    render_metrics(result)
    st.divider()
    render_sharpe_gap(result)
    st.divider()
    render_chart(result, is_real, df, ticker_a, ticker_b, capital, cost_bps)


# ------------------------------------------------------------------ tab bodies


def render_key_findings_tab():
    """Tab 1: the headline result, pre-loaded so nobody has to hunt for it."""
    st.markdown(
        "### Correlation is not cointegration — and the difference costs money"
    )
    st.markdown(
        "This app runs a mean-reversion pairs trade behind a **cointegration "
        "gate**: a statistical test, run on training data only, of whether the "
        "spread between two stocks actually has a level it reverts to. The "
        "headline result is a *losing* backtest, and that is the point — the "
        "gate rejected KO / PEP, the most commonly cited textbook pairs trade, "
        "before any money was lost, and the out-of-sample P&L confirms the "
        "rejection was right. A backtest that loses money exactly the way the "
        "methodology predicted is better evidence the methodology works than a "
        "profitable one would be."
    )

    left, right = st.columns(2)
    if left.button(
        "The textbook example that fails (KO / PEP)",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.preset = PRESET_KOPEP
    if right.button(
        f"A pair that works (simulated, seed {SIM_SEED})", use_container_width=True
    ):
        st.session_state.preset = PRESET_SIM

    preset = st.session_state.get("preset", PRESET_KOPEP)

    with st.spinner("Running backtest..."):
        if preset == PRESET_KOPEP:
            df, is_real, reason, result = run_real_pair(
                "KO", "PEP", DEFAULT_START, DEFAULT_END, False, None
            )
            ticker_a, ticker_b = ("KO", "PEP") if is_real else ("A (sim)", "B (sim)")
        else:
            df, is_real, reason, result = run_simulated_pair(SIM_SEED, False, None)
            ticker_a, ticker_b = "A (sim)", "B (sim)"

    if preset == PRESET_SIM:
        st.info(
            "**A controlled illustration, not a market result.** This pair is "
            "cointegrated by construction — a shared random-walk factor plus a "
            "mean-reverting Ornstein-Uhlenbeck spread — so the gate passes and "
            "the strategy has something real to trade. Treated as evidence "
            "about markets it would be circular; treated as a test of the "
            "whole pipeline it is informative."
        )

    render_backtest_result(
        result, is_real, df, reason=reason, ticker_a=ticker_a, ticker_b=ticker_b
    )


def render_try_it_tab():
    """Tab 2: the same machinery, pointed at whatever the visitor chooses."""
    st.markdown("### Run it yourself")
    st.markdown(
        "Pick a pair in the sidebar and press **Run backtest**. Everything is "
        "fitted the same way as above: the gate and hedge ratio on train, "
        "parameters on validation, and the test slice touched exactly once."
    )

    config = st.session_state.get("run_config")
    if config is None:
        st.info("Configure a run in the sidebar, then press **Run backtest**.")
        return

    try:
        with st.spinner("Running backtest..."):
            df, is_real, reason, result = run_real_pair(
                config["ticker_a"],
                config["ticker_b"],
                config["start"],
                config["end"],
                config["require_cointegration"],
                config["fixed"],
            )
    except ValueError as exc:
        # The likeliest cause is tune_parameters finding no configuration that
        # clears its 5-trade floor, which a short date range easily produces.
        st.error(f"**The backtest could not complete.** {exc}")
        st.caption(
            "Try a longer date range, or switch to manual parameters to "
            "evaluate a single configuration without a grid search."
        )
        return

    ticker_a, ticker_b = (
        (config["ticker_a"], config["ticker_b"]) if is_real else ("A (sim)", "B (sim)")
    )
    render_backtest_result(
        result, is_real, df, reason=reason, ticker_a=ticker_a, ticker_b=ticker_b
    )


def _apply_pair_preset():
    """Push a chosen preset down into the two ticker dropdowns.

    Widget callbacks run before the rerun, so writing to the dropdowns' own
    session_state keys here is what makes them show the new values. Assigning
    to those keys does not re-fire their own callbacks, so this cannot bounce
    back and forth with `_clear_pair_preset`.
    """
    pair = st.session_state.pair_preset
    if pair != CUSTOM_PAIR:
        st.session_state.ticker_a_choice, st.session_state.ticker_b_choice = (
            COMMON_PAIRS[pair]
        )


def _clear_pair_preset():
    """Drop back to "Custom" once a hand-picked ticker leaves the preset behind.

    Without this the preset would keep claiming "KO / PEP" while the dropdowns
    read XOM and PEP, and the label a visitor trusts would be describing a run
    that isn't happening.
    """
    pair = st.session_state.get("pair_preset", CUSTOM_PAIR)
    if pair == CUSTOM_PAIR:
        return
    chosen = (st.session_state.ticker_a_choice, st.session_state.ticker_b_choice)
    if chosen != COMMON_PAIRS[pair]:
        st.session_state.pair_preset = CUSTOM_PAIR


def _pick_ticker(label, choice_key, custom_key):
    """A dropdown of known symbols, with an escape hatch for anything else."""
    choice = st.sidebar.selectbox(
        label, TICKER_OPTIONS, key=choice_key, on_change=_clear_pair_preset
    )
    if choice == CUSTOM_TICKER:
        return (
            st.sidebar.text_input(
                f"{label} symbol", key=custom_key, placeholder="e.g. NKE"
            )
            .strip()
            .upper()
        )
    return choice


def render_sidebar():
    """Inputs for tab 2. Nothing runs until the button is pressed."""
    st.sidebar.markdown("## Try it yourself")
    st.sidebar.caption("These controls drive the second tab.")

    # Seeded before the widgets exist so the dropdowns can be driven purely
    # through session_state, without also juggling an `index` argument.
    st.session_state.setdefault("pair_preset", "KO / PEP")
    st.session_state.setdefault("ticker_a_choice", "KO")
    st.session_state.setdefault("ticker_b_choice", "PEP")

    st.sidebar.selectbox(
        "Common pairs",
        PAIR_OPTIONS,
        key="pair_preset",
        on_change=_apply_pair_preset,
        help="Fills both dropdowns below. Commonly cited candidates — which is "
             "not the same as cointegrated.",
    )

    ticker_a = _pick_ticker("Ticker A", "ticker_a_choice", "ticker_a_custom")
    ticker_b = _pick_ticker("Ticker B", "ticker_b_choice", "ticker_b_custom")

    start = st.sidebar.date_input(
        "Start date", value=pd.Timestamp(DEFAULT_START).date()
    )
    end = st.sidebar.date_input("End date", value=pd.Timestamp(DEFAULT_END).date())

    mode = st.sidebar.radio(
        "Parameters", ["Auto-tune (recommended)", "Manual"], index=0
    )

    fixed = None
    if mode == "Manual":
        st.sidebar.caption(
            "Skips the grid search and evaluates your values on test directly. "
            "Worth trying a deliberately tight entry threshold to see how "
            "sensitive the results are to this choice."
        )
        window = st.sidebar.slider("Z-score window (days)", 5, 90, 30)
        entry_z = st.sidebar.slider("Entry threshold |z|", 0.5, 4.0, 2.0, 0.1)
        exit_z = st.sidebar.slider("Exit threshold |z|", 0.0, 2.0, 0.5, 0.05)
        if exit_z >= entry_z:
            st.sidebar.warning(
                "Exit should be tighter than entry, or every trade closes the "
                "day after it opens."
            )
        fixed = (window, entry_z, exit_z)

    require_cointegration = st.sidebar.checkbox(
        "Require cointegration", value=False,
        help="Stop before trading when the training slice fails the gate, "
             "instead of trading anyway with a warning.",
    )

    # An explicit click, so that typing a ticker one character at a time
    # doesn't fire a price fetch and a grid search on every keystroke.
    if st.sidebar.button("Run backtest", type="primary", use_container_width=True):
        if not ticker_a or not ticker_b:
            st.sidebar.error("Both tickers are required.")
        elif ticker_a == ticker_b:
            # A series against itself has a spread of exactly zero, so the
            # z-score divides by zero and the gate has nothing to test.
            st.sidebar.error("Pick two different tickers.")
        elif start >= end:
            st.sidebar.error("The start date must come before the end date.")
        else:
            st.session_state.run_config = {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "require_cointegration": require_cointegration,
                "fixed": fixed,
            }


def main():
    st.set_page_config(page_title="Cointegration-gated pairs trading", layout="wide")
    st.title("Pairs trading, behind a cointegration gate")
    st.caption(
        "A statistical arbitrage backtest built to be defensible rather than "
        "impressive."
    )

    render_sidebar()

    key_findings, try_it = st.tabs(["Key Findings", "Try It Yourself"])
    with key_findings:
        render_key_findings_tab()
    with try_it:
        render_try_it_tab()


if __name__ == "__main__":
    main()
