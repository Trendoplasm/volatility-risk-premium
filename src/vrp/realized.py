"""Realised variance: what the underlying actually did.

The variance risk premium is the difference between the variance the option market implied and the
variance the underlying subsequently delivered. This module supplies the second half, from real
closing prices.

Realised variance is the annualised sum of squared close-to-close log returns over a window. Log
returns are used because they are additive across days, and the annualisation factor is trading
days per year to match the horizon convention used everywhere else in the study.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vrp.config import MIN_REALIZED_OBSERVATIONS, TRADING_DAYS_PER_YEAR


def log_returns(prices: Sequence[float]) -> np.ndarray:
    """Return close-to-close log returns.

    Args:
        prices: Consecutive closing prices.

    Returns:
        One fewer return than there were prices; an empty array for fewer than two prices.

    Raises:
        ValueError: If any price is not positive, which would make a log return undefined.
    """
    values = np.asarray(prices, dtype=float)
    if values.size < 2:
        return np.empty(0, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Log returns require strictly positive prices")
    return np.diff(np.log(values))


def realized_variance(prices: Sequence[float]) -> float | None:
    """Return annualised realised variance over a window of prices.

    Args:
        prices: Consecutive closing prices spanning the window.

    Returns:
        Annualised realised variance, or None when the window holds too few returns to estimate
        one. A short window is reported as missing rather than as a noisy number.
    """
    returns = log_returns(prices)
    if returns.size == 0:
        return None
    return float(TRADING_DAYS_PER_YEAR * np.mean(returns**2))


def realized_volatility(prices: Sequence[float]) -> float | None:
    """Return annualised realised volatility over a window of prices."""
    variance = realized_variance(prices)
    return None if variance is None else float(np.sqrt(variance))


def trailing_realized_volatility(prices: Sequence[float], index: int, window: int) -> float | None:
    """Return realised volatility over the ``window`` trading days ending at ``index``.

    This is a backward-looking signal input, so it deliberately uses only information available at
    ``index``.

    Args:
        prices: Full price history.
        index: Position of the observation date.
        window: Number of trailing returns to use.

    Returns:
        Annualised realised volatility, or None if the history is too short.
    """
    start = index - window
    if start < 0:
        return None
    segment = prices[start : index + 1]
    if len(segment) - 1 < MIN_REALIZED_OBSERVATIONS:
        return None
    return realized_volatility(segment)


def forward_realized_variance(prices: Sequence[float], index: int, horizon: int) -> float | None:
    """Return annualised realised variance over the ``horizon`` days *after* ``index``.

    This is the quantity an implied variance observed at ``index`` is a forecast of, so it is
    strictly forward-looking and must never be used as a signal.

    Args:
        prices: Full price history.
        index: Position of the observation date.
        horizon: Number of forward returns to use.

    Returns:
        Annualised realised variance, or None if the history ends before the horizon does.
    """
    end = index + horizon
    if end >= len(prices):
        return None
    return realized_variance(prices[index : end + 1])
