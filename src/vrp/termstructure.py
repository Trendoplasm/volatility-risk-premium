"""Matching implied volatility to the study's horizons using the observed term structure.

The study compares implied variance at a 21-, 42-, or 63-trading-day horizon against the variance
the underlying subsequently realised over exactly that horizon. Cboe's single-name indexes only
publish a 30-day horizon, so the other two need a term-structure adjustment.

Rather than assume a shape, this module measures one. Cboe publishes four points on the S&P 500
volatility curve every day -- 9, 30, 93, and 365 calendar days -- and the study interpolates that
observed curve. The resulting *relative* shape, expressed against the 30-day point, is then
applied to a single name's own 30-day index.

That last step is an approximation and the study says so: a single name's term structure is not
observable free of charge, and it is generally flatter than the index's. The approximation is
also mild in practice, because the study's horizons land close to points Cboe already publishes --
21 trading days is about 30 calendar days, and 63 trading days is about 91.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date

import numpy as np

from vrp.config import (
    CALENDAR_DAYS_PER_YEAR,
    INDEX_TERM_STRUCTURE,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)

#: The 30-day point, against which relative shape is expressed.
ANCHOR_HORIZON_DAYS = 30.0


def trading_days_to_calendar_days(trading_days: float) -> float:
    """Convert a trading-day horizon to the calendar-day horizon Cboe quotes against.

    Args:
        trading_days: Horizon in trading days.

    Returns:
        The equivalent horizon in calendar days.
    """
    return trading_days * CALENDAR_DAYS_PER_YEAR / TRADING_DAYS_PER_YEAR


def interpolate_curve(curve: Mapping[float, float], target_days: float) -> float:
    """Interpolate an implied-volatility curve to a target horizon.

    Interpolation is linear in *total variance* against calendar time, which is the convention VIX
    itself uses and the only one that keeps the implied term structure free of calendar arbitrage.
    Targets outside the observed range are held flat at the nearest observed point rather than
    extrapolated, since extrapolating a variance curve invents information.

    Args:
        curve: Observed decimal implied volatility keyed by calendar-day horizon.
        target_days: Horizon to interpolate to, in calendar days.

    Returns:
        Decimal implied volatility at the target horizon.

    Raises:
        ValueError: If the curve is empty or the target horizon is not positive.
    """
    if not curve:
        raise ValueError("Cannot interpolate an empty volatility curve")
    if target_days <= 0:
        raise ValueError(f"Target horizon must be positive, got {target_days}")

    horizons = sorted(curve)
    if target_days <= horizons[0]:
        return curve[horizons[0]]
    if target_days >= horizons[-1]:
        return curve[horizons[-1]]

    upper_index = int(np.searchsorted(horizons, target_days, side="left"))
    lower, upper = horizons[upper_index - 1], horizons[upper_index]
    if target_days == lower:
        return curve[lower]

    lower_variance = curve[lower] ** 2 * lower
    upper_variance = curve[upper] ** 2 * upper
    weight = (target_days - lower) / (upper - lower)
    total_variance = lower_variance + weight * (upper_variance - lower_variance)
    return float(np.sqrt(max(total_variance, 0.0) / target_days))


def shape_multiplier(curve: Mapping[float, float], target_trading_days: float) -> float:
    """Return the observed curve's level at a horizon relative to its 30-day level.

    Args:
        curve: Observed decimal implied volatility keyed by calendar-day horizon.
        target_trading_days: Horizon in trading days.

    Returns:
        The ratio of interpolated implied volatility at the target horizon to the 30-day point.
        A value above one means the curve slopes upward out to that horizon.

    Raises:
        ValueError: If the curve has no positive 30-day level.
    """
    anchor = curve.get(ANCHOR_HORIZON_DAYS)
    if anchor is None or anchor <= 0:
        raise ValueError("The observed curve needs a positive 30-day level to normalise against")
    target = interpolate_curve(curve, trading_days_to_calendar_days(target_trading_days))
    return target / anchor


def index_curve(
    volatility_series: Mapping[str, Mapping[date, float]], trading_date: date
) -> dict[float, float]:
    """Assemble the observed S&P 500 volatility curve for one date.

    Args:
        volatility_series: Loaded Cboe series keyed by index name.
        trading_date: Date to assemble the curve for.

    Returns:
        Decimal implied volatility keyed by calendar-day horizon, containing only the points that
        actually traded on the date.
    """
    curve: dict[float, float] = {}
    for name, horizon in INDEX_TERM_STRUCTURE.items():
        level = volatility_series[name].get(trading_date)
        if level is not None:
            curve[horizon] = level
    return curve


def matched_implied_volatility(
    anchor_iv: float, curve: Mapping[float, float], target_trading_days: float
) -> float | None:
    """Scale a 30-day implied volatility to the study's horizon.

    Args:
        anchor_iv: The security's own observed 30-day implied volatility, as a decimal.
        curve: Observed S&P 500 curve for the same date.
        target_trading_days: Horizon in trading days.

    Returns:
        Implied volatility at the target horizon, or None if the curve lacks its 30-day anchor
        and the adjustment therefore cannot be measured.
    """
    if ANCHOR_HORIZON_DAYS not in curve:
        return None
    return anchor_iv * shape_multiplier(curve, target_trading_days)
