"""Data loading for the pairs trading backtest.

Two sources: real close prices via yfinance, or a simulated pair when that
isn't available. The simulator is deliberately not two independent random
walks -- those are neither correlated nor cointegrated, so a backtest built
on them would be testing nothing. See simulate_cointegrated_pair for the
construction that actually produces a mean-reverting spread.
"""

import numpy as np
import pandas as pd

# Minimum usable history, set by the power of the cointegration test rather
# than by the rolling window. The train slice is half of whatever we load, and
# Engle-Granger detects known cointegration on a 250-day slice roughly 1 time
# in 40 -- at that length neither a pass nor a fail carries information. 1000
# rows gives a ~500-day train slice at the default split; see NOTES.md for the
# power curve behind this number.
MIN_ROWS = 1000

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"


def simulate_cointegrated_pair(n_days=1500, seed=9, start_price=100.0):
    """Simulate two stocks that are genuinely cointegrated.

    Three components, each doing a different job:

    1. A shared stochastic common factor (a random walk both log-prices
       include). This is what makes the two stocks move together -- it's the
       "market/sector" term, and it is non-stationary by construction.
    2. A mean-reverting idiosyncratic spread from an Ornstein-Uhlenbeck
       process, added to A and subtracted from B. This is the actual
       cointegrating relationship: the common factor cancels in A - B, leaving
       only the OU term, which reverts to zero.
    3. Small independent noise per stock, so neither is a deterministic
       function of the other.

    The point of (1) + (2) together is that each price series is individually
    non-stationary (it inherits the random walk) while a linear combination of
    them is stationary. That is the definition of cointegration, and it's why
    this is a fair test bed for the strategy rather than a rigged one.

    On the default seed: cointegration holds on every draw, but individual
    draws vary in two ways that matter. Some produce a common factor whose path
    ADF reads as mean-reverting, so the legs test stationary and the setup no
    longer demonstrates what it's meant to. Others fail the cointegration test
    on the shorter training slice through simple lack of power. Seed 9 clears
    both: legs at ADF p = 0.49/0.63, train-slice cointegration at p = 0.0003.
    Seed 3 is a useful counter-example where the train gate fails. See NOTES.md
    for the sweep.
    """
    rng = np.random.default_rng(seed)

    # Non-stationary shared driver: ~1% daily log-return shocks.
    common_factor = np.cumsum(rng.normal(0.0, 0.010, n_days))

    # OU process: spread[t] = spread[t-1] + theta*(mu - spread[t-1]) + sigma*noise.
    # theta is the pull toward mu per day, so 1/theta ~ 33 days is the rough
    # half-life of a dislocation -- slow enough to trade, fast enough to revert
    # within the test window.
    theta, mu, sigma = 0.03, 0.0, 0.006
    spread = np.zeros(n_days)
    shocks = rng.normal(0.0, 1.0, n_days)
    for t in range(1, n_days):
        spread[t] = spread[t - 1] + theta * (mu - spread[t - 1]) + sigma * shocks[t]

    noise_a = rng.normal(0.0, 0.002, n_days)
    noise_b = rng.normal(0.0, 0.002, n_days)

    # Split the spread symmetrically so it shows up in A - B, not in the level
    # of either stock alone.
    log_price_a = np.log(start_price) + common_factor + spread / 2 + noise_a
    log_price_b = np.log(start_price) + common_factor - spread / 2 + noise_b

    index = pd.bdate_range(start=DEFAULT_START, periods=n_days, name="Date")
    return pd.DataFrame(
        {"A": np.exp(log_price_a), "B": np.exp(log_price_b)},
        index=index,
    )


def load_real_data(ticker_a, ticker_b, start, end):
    """Pull real close prices for both tickers and return them as columns A/B."""
    import yfinance as yf

    raw = yf.download(
        [ticker_a, ticker_b],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    df = close[[ticker_a, ticker_b]].dropna()
    df.columns = ["A", "B"]
    df.index.name = "Date"
    return df


def load_data(
    ticker_a="KO", ticker_b="PEP", start=DEFAULT_START, end=DEFAULT_END, seed=9
):
    """Load real prices, falling back to simulation on any failure.

    Returns (df, is_real). The flag is carried through to the final report so
    results are never presented as live-market evidence when they came from the
    simulator. `seed` only affects the simulated fallback.
    """
    try:
        df = load_real_data(ticker_a, ticker_b, start, end)
        if len(df) < MIN_ROWS:
            raise ValueError(
                f"only {len(df)} rows for {ticker_a}/{ticker_b}, need {MIN_ROWS}"
            )
        print(f"Data source: real market data ({ticker_a}, {ticker_b}), {len(df)} rows")
        return df, True
    except Exception as exc:
        print(f"Real data unavailable ({type(exc).__name__}: {exc})")
        df = simulate_cointegrated_pair(seed=seed)
        print(f"Data source: SIMULATED cointegrated pair (seed={seed}), {len(df)} rows")
        return df, False


if __name__ == "__main__":
    df = simulate_cointegrated_pair()
    print(df.describe())
    print()
    print("Correlation of prices:", round(df["A"].corr(df["B"]), 4))
    print("Spread (A - B) mean:", round((df["A"] - df["B"]).mean(), 4))
    print("Spread (A - B) std: ", round((df["A"] - df["B"]).std(), 4))
