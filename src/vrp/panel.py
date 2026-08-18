"""The matched implied-versus-realised variance panel.

This is the study's most direct measurement and the one that rests entirely on observed data: for
every trading day and every horizon, what variance did the option market imply, and what variance
did the underlying then actually deliver over exactly that horizon?

The difference is the variance risk premium. It is strictly forward-looking by construction, so no
row can be used as a trading signal -- the realised leg is not knowable at the observation date.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from vrp.config import IV_ANCHOR, StudyConfig
from vrp.models import LevelByDate, Row, Security, Table
from vrp.realized import forward_realized_variance, trailing_realized_volatility
from vrp.termstructure import index_curve, matched_implied_volatility

logger = logging.getLogger(__name__)


def aligned_calendar(
    prices: LevelByDate, anchor: LevelByDate, start: date, end: date | None = None
) -> tuple[list[date], list[float]]:
    """Return the price history and its dates from ``start`` onward.

    The price series defines the calendar, because realised variance must be computed from
    consecutive trading closes. Days where implied volatility is missing are skipped as
    *observations* later, but they still contribute their return.

    Args:
        prices: Closing prices keyed by date.
        anchor: The security's 30-day implied-volatility index, used only to bound the range.
        start: First date of the study period.
        end: Last date of the study period; None uses all available history.

    Returns:
        Ascending dates and their closes, beginning early enough to support trailing windows.
    """
    if not anchor:
        raise ValueError("Cannot align a calendar against an empty volatility series")
    last = min(max(prices), max(anchor))
    if end is not None:
        last = min(last, end)
    dates = sorted(day for day in prices if day <= last)
    if not dates:
        raise ValueError("Price and volatility histories do not overlap")
    # Keep a run-up of history before the study start so trailing windows are populated on day one.
    first_index = max(0, sum(day < start for day in dates) - 2 * 63)
    dates = dates[first_index:]
    return dates, [prices[day] for day in dates]


def build_panel(
    securities: Sequence[Security],
    volatility_series: Mapping[str, LevelByDate],
    price_series: Mapping[str, LevelByDate],
    config: StudyConfig,
) -> Table:
    """Build the implied-versus-realised variance panel.

    Args:
        securities: Study universe; only measured securities contribute rows.
        volatility_series: Loaded Cboe series keyed by index name.
        price_series: Loaded price histories keyed by study ticker.
        config: Study horizons and period.

    Returns:
        One row per security, date, and horizon for which both legs are observable.
    """
    rows: Table = []
    start = config.start()

    for security in securities:
        if not security.measured:
            continue
        anchor_name = IV_ANCHOR[security.ticker]
        anchor = volatility_series[anchor_name]
        dates, prices = aligned_calendar(price_series[security.ticker], anchor, start, config.end())

        for index, trading_date in enumerate(dates):
            if trading_date < start:
                continue
            anchor_iv = anchor.get(trading_date)
            if anchor_iv is None:
                continue
            curve = index_curve(volatility_series, trading_date)
            vix = curve.get(30.0)
            if vix is None:
                continue

            for horizon in config.all_horizons():
                matched_iv = matched_implied_volatility(anchor_iv, curve, horizon)
                realized = forward_realized_variance(prices, index, horizon)
                if matched_iv is None or realized is None:
                    continue
                implied_variance = matched_iv**2
                rows.append(
                    {
                        "date": trading_date,
                        "ticker": security.ticker,
                        "horizon_days": horizon,
                        "spot": prices[index],
                        "anchor_iv": anchor_iv,
                        "matched_iv": matched_iv,
                        "term_multiplier": matched_iv / anchor_iv,
                        "implied_variance": implied_variance,
                        "realized_variance": realized,
                        "realized_vol": float(np.sqrt(realized)),
                        "vrp_variance": implied_variance - realized,
                        "vrp_vol_points": matched_iv - float(np.sqrt(realized)),
                        "vrp_ratio": implied_variance / realized if realized > 0 else None,
                        "trailing_realized_vol": trailing_realized_volatility(
                            prices, index, config.realized_window_days
                        ),
                        "vix": vix,
                        "in_sample": trading_date <= config.train_cutoff(),
                    }
                )

    rows.sort(key=lambda row: (row["ticker"], row["horizon_days"], row["date"]))
    logger.info("Built variance panel: %d observations", len(rows))
    return rows


def panel_summary(rows: Sequence[Row], label: str, horizon: int) -> Row:
    """Summarise one group of panel rows at one horizon.

    Args:
        rows: Panel rows already filtered to the group and horizon.
        label: Value written to the ``group`` column.
        horizon: Horizon the rows belong to.

    Returns:
        Central tendencies of the premium plus the frequency with which it was positive, which is
        the most legible statement of whether implied variance usually exceeded realised.

    Raises:
        ValueError: If the group is empty.
    """
    if not rows:
        raise ValueError(f"Cannot summarise an empty panel group: {label} at {horizon}d")

    variance_premium = np.array([row["vrp_variance"] for row in rows], dtype=float)
    volatility_premium = np.array([row["vrp_vol_points"] for row in rows], dtype=float)
    implied = np.array([row["implied_variance"] for row in rows], dtype=float)
    realized = np.array([row["realized_variance"] for row in rows], dtype=float)

    return {
        "group": label,
        "horizon_days": horizon,
        "n": len(rows),
        "first_date": min(row["date"] for row in rows),
        "last_date": max(row["date"] for row in rows),
        "mean_implied_vol": float(np.mean(np.sqrt(implied))),
        "mean_realized_vol": float(np.mean(np.sqrt(realized))),
        "mean_vrp_variance": float(np.mean(variance_premium)),
        "median_vrp_variance": float(np.median(variance_premium)),
        "mean_vrp_vol_points": float(np.mean(volatility_premium)),
        "median_vrp_vol_points": float(np.median(volatility_premium)),
        "sd_vrp_vol_points": float(np.std(volatility_premium, ddof=1)),
        "pct_positive_vrp": float(np.mean(variance_premium > 0)),
        "mean_variance_ratio": float(np.mean(implied / realized)),
    }
