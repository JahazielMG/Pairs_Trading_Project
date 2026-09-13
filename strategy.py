"""Signal logic: cointegration test, hedge ratio, spread, z-score, positions.

Order matters here. The cointegration test comes first and gates everything
else: correlation only says two series move together, which two trending
stocks can do while their spread drifts apart forever. Cointegration is the
stronger claim that the spread itself is stationary, and that claim is the
only reason to expect a mean-reversion trade to work at all.

Every function in this module that *fits* anything to data -- the cointegration
test and the hedge ratio -- must be called on the training slice only. These
functions don't enforce that themselves; backtest.py is responsible for
passing the right slice. See the note in test_cointegration for why.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

COINT_PVALUE_THRESHOLD = 0.05


def test_cointegration(price_a, price_b):
    """Engle-Granger cointegration test. Returns (is_cointegrated, p_value).

    Null hypothesis is *no* cointegration, so a small p-value is evidence the
    spread is stationary and worth trading.

    Train-only, always. The decision this drives -- whether to trade this pair
    at all -- is a real decision, and making it with data from the test period
    leaks the future into the past just as surely as peeking at tomorrow's
    price. A pair that looks cointegrated over 2015-2025 in hindsight may not
    have looked that way to someone standing in 2020 with only the prior data,
    and the backtest is supposed to simulate that person.
    """
    _, p_value, _ = coint(price_a, price_b)
    return bool(p_value < COINT_PVALUE_THRESHOLD), float(p_value)


def estimate_hedge_ratio(price_a, price_b):
    """OLS of A on B. Returns (beta, alpha).

    beta is how many shares of B offset one share of A, so that the combination
    A - beta*B strips out the shared factor driving both and leaves the
    mean-reverting residual. Train-only for the same reason as above: fitting
    beta on data that includes the test period hands the strategy the exact
    hedge that happened to work out of sample.
    """
    x = sm.add_constant(np.asarray(price_b, dtype=float))
    model = sm.OLS(np.asarray(price_a, dtype=float), x).fit()
    alpha, beta = model.params[0], model.params[1]
    return float(beta), float(alpha)


def compute_spread(price_a, price_b, beta):
    """The traded quantity: long 1 unit of A, short beta units of B."""
    return price_a - beta * price_b


def rolling_zscore(spread, window=30):
    """Standardize the spread against its own trailing window.

    Rolling, not full-sample. A full-sample mean and standard deviation are
    computed from the entire history including the future, so an early z-score
    would be normalized by volatility that hadn't happened yet -- a quiet
    lookahead bug that inflates results without ever looking obviously wrong.

    A trailing window also adapts when the spread's scale drifts over time,
    which it does here: the dislocation is roughly constant in percentage terms
    while the price level moves, so a fixed full-sample sigma would misjudge
    the tails at both ends of the sample.

    The first `window - 1` values are NaN by construction; generate_signals
    handles those rather than filling them with fabricated numbers.
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def generate_signals(zscore, entry_z=2.0, exit_z=0.5):
    """Turn z-scores into positions in {-1, 0, +1} via a stateful loop.

    The position is the position *in the spread*: +1 means long the spread
    (long A, short beta*B), betting it rises back toward zero from below; -1
    means short the spread.

    This has to be a loop rather than vectorized comparisons because the rule
    is path-dependent -- whether a z-score of 1.0 means "hold" or "flat"
    depends on whether a trade is already open. Separate entry and exit bands
    (enter at |z| > 2, exit at |z| < 0.5) create a deadband that avoids
    thrashing in and out on noise around the threshold.
    """
    position = np.zeros(len(zscore), dtype=int)
    current = 0

    for i, z in enumerate(zscore.to_numpy()):
        if np.isnan(z):
            # Early rows before the rolling window fills. Carry the previous
            # position forward rather than forcing a flat, so a NaN can't
            # silently close an open trade.
            position[i] = current
            continue

        if current == 0:
            if z > entry_z:
                current = -1  # spread stretched high, bet on it falling
            elif z < -entry_z:
                current = 1  # spread stretched low, bet on it rising
        elif current == 1:
            if z > -exit_z:
                current = 0  # reverted far enough toward the mean
        elif current == -1:
            if z < exit_z:
                current = 0

        position[i] = current

    return pd.Series(position, index=zscore.index, name="position")
