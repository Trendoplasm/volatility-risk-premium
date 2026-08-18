"""Synthetic fixtures with analytically known properties.

Tests build their own price and volatility histories rather than reading market data, so the suite
runs anywhere and every assertion has a derivable answer.

The standard price path is constructed so its realised volatility is *exact*: log returns
alternate between plus and minus a fixed daily size, so the annualised realised variance is
``252 * daily**2`` with no sampling error at all. That makes it possible to test the variance
premium and the delta-hedged gain against known numbers instead of against recorded output.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pytest

from vrp.config import TRADING_DAYS_PER_YEAR, CostModel, StudyConfig
from vrp.models import LevelByDate, Security

#: Realised volatility the standard price path delivers, exactly.
KNOWN_REALIZED_VOL = 0.20

#: Daily log-return size that produces it.
DAILY_RETURN = KNOWN_REALIZED_VOL / math.sqrt(TRADING_DAYS_PER_YEAR)

#: Implied volatility of the standard volatility history, above the realised level so that the
#: variance premium is positive by construction.
KNOWN_IMPLIED_VOL = 0.25

START_SPOT = 100.0


def trading_dates(start: date, count: int) -> list[date]:
    """Generate ascending weekday dates."""
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def alternating_price_path(count: int, daily: float = DAILY_RETURN) -> list[float]:
    """Build a price path whose realised volatility is exactly ``daily * sqrt(252)``.

    Args:
        count: Number of closes.
        daily: Absolute size of each daily log return.

    Returns:
        Closes beginning at :data:`START_SPOT`.
    """
    prices = [START_SPOT]
    for step in range(1, count):
        direction = 1.0 if step % 2 == 1 else -1.0
        prices.append(prices[-1] * math.exp(direction * daily))
    return prices


def flat_volatility(dates: Sequence[date], level: float = KNOWN_IMPLIED_VOL) -> LevelByDate:
    """Return a constant implied-volatility history."""
    return dict.fromkeys(dates, level)


def volatility_series(
    dates: Sequence[date], level: float = KNOWN_IMPLIED_VOL
) -> dict[str, LevelByDate]:
    """Return a full set of Cboe series with a flat term structure.

    A flat term structure means the matched-maturity adjustment is exactly one, which keeps tests
    of downstream stages independent of the term-structure logic.
    """
    return {
        name: flat_volatility(dates, level)
        for name in ("VIX9D", "VIX", "VIX3M", "VIX1Y", "VXAPL", "VXAZN")
    }


def security(
    ticker: str = "INDEX", *, measured: bool = True, dividend_yield: float = 0.0
) -> Security:
    """Return one study security."""
    return Security(
        ticker=ticker,
        underlying_symbol=ticker,
        name=f"{ticker} test security",
        security_type="Index" if ticker == "INDEX" else "Single name",
        group="Test",
        size_bucket="Test",
        dividend_yield=dividend_yield,
        iv_evidence="Synthetic test fixture",
        return_evidence="Synthetic test fixture",
        measured=measured,
    )


def write_cboe_csv(path: Path, dates: Sequence[date], levels: Sequence[float]) -> None:
    """Write a Cboe-format history file, byte-order mark included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
        for day, level in zip(dates, levels, strict=True):
            writer.writerow([day.strftime("%m/%d/%Y"), level, level, level, level])


def write_price_csv(path: Path, dates: Sequence[date], prices: Sequence[float]) -> None:
    """Write a two-column price history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "close"])
        for day, price in zip(dates, prices, strict=True):
            writer.writerow([day.isoformat(), price])


def write_universe_csv(path: Path, securities: Sequence[Security]) -> None:
    """Write a universe reference file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticker",
        "underlying_symbol",
        "name",
        "security_type",
        "group",
        "size_bucket",
        "dividend_yield",
        "iv_evidence",
        "return_evidence",
        "measured",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in securities:
            writer.writerow(
                {
                    "ticker": item.ticker,
                    "underlying_symbol": item.underlying_symbol,
                    "name": item.name,
                    "security_type": item.security_type,
                    "group": item.group,
                    "size_bucket": item.size_bucket,
                    "dividend_yield": item.dividend_yield,
                    "iv_evidence": item.iv_evidence,
                    "return_evidence": item.return_evidence,
                    "measured": "TRUE" if item.measured else "FALSE",
                }
            )


@pytest.fixture
def dates() -> list[date]:
    """Return 400 weekday trading dates from 2011."""
    return trading_dates(date(2011, 1, 3), 400)


@pytest.fixture
def prices(dates: list[date]) -> list[float]:
    """Return the standard exact-volatility price path."""
    return alternating_price_path(len(dates))


@pytest.fixture
def config() -> StudyConfig:
    """Return a study configuration with a cheap bootstrap."""
    return StudyConfig(start_date="2011-01-03", bootstrap_iterations=200)


@pytest.fixture
def costs() -> CostModel:
    """Return the default cost model."""
    return CostModel()


@pytest.fixture
def free_costs() -> CostModel:
    """Return a cost model with every cost switched off, for testing gross mechanics."""
    return CostModel(
        option_commission=0.0,
        option_half_spread=0.0,
        stock_hedge_cost_bps=0.0,
        index_hedge_cost_bps=0.0,
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent
