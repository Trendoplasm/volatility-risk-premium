"""Readers for the study's three inputs: Cboe volatility history, price history, and the universe.

Every reader fails loudly. A silently dropped observation or a coerced date would change a
published statistic without changing anything visible.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from vrp.config import (
    REQUIRED_CBOE_FILES,
    REQUIRED_PRICE_FILES,
    VOLATILITY_POINTS_PER_UNIT,
)
from vrp.models import LevelByDate, Security

logger = logging.getLogger(__name__)

#: Columns every Cboe history file must provide.
CBOE_COLUMNS = frozenset({"DATE", "OPEN", "HIGH", "LOW", "CLOSE"})

#: Columns a price history file must provide.
PRICE_COLUMNS = frozenset({"date", "close"})

#: Columns the universe reference file must provide.
UNIVERSE_COLUMNS = frozenset(
    {
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
    }
)

CBOE_DATE_FORMAT = "%m/%d/%Y"
ISO_DATE_FORMAT = "%Y-%m-%d"
TRUE_STRINGS = frozenset({"true", "1", "yes", "y"})


def parse_bool(value: str | None) -> bool:
    """Interpret a spreadsheet-style truthy value."""
    return (value or "").strip().lower() in TRUE_STRINGS


def load_cboe_history(path: Path) -> LevelByDate:
    """Load a Cboe volatility-index history as decimal volatility by trading date.

    Cboe quotes these indexes in percentage points. They are converted to decimals here, once, so
    that no downstream formula has to remember which convention it is in.

    Args:
        path: Path to a ``*_History.csv`` file as published by Cboe.

    Returns:
        Decimal implied volatility keyed by trading date.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is wrong, a row will not parse, a level is not positive, or the
            file holds no data rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing Cboe input: {path}")

    levels: LevelByDate = {}
    # utf-8-sig: Cboe's files carry a byte-order mark that would corrupt the "DATE" header.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not CBOE_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks expected Cboe columns: {sorted(CBOE_COLUMNS)}")
        for raw in reader:
            try:
                trading_date = datetime.strptime(raw["DATE"], CBOE_DATE_FORMAT).date()
                level = float(raw["CLOSE"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {path}: {raw}") from exc
            if level <= 0:
                raise ValueError(f"Nonpositive volatility level in {path} on {trading_date}")
            levels[trading_date] = level / VOLATILITY_POINTS_PER_UNIT

    if not levels:
        raise ValueError(f"No data rows found in {path}")
    return levels


def load_price_history(path: Path) -> LevelByDate:
    """Load a daily close series.

    Args:
        path: Path to a ``date,close`` CSV.

    Returns:
        Closing price keyed by trading date.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is wrong, a row will not parse, a price is not positive, or the
            file holds no data rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing price input: {path}")

    prices: LevelByDate = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not PRICE_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks expected columns: {sorted(PRICE_COLUMNS)}")
        for raw in reader:
            try:
                trading_date = datetime.strptime(raw["date"], ISO_DATE_FORMAT).date()
                close = float(raw["close"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {path}: {raw}") from exc
            if close <= 0:
                raise ValueError(f"Nonpositive close in {path} on {trading_date}")
            prices[trading_date] = close

    if not prices:
        raise ValueError(f"No data rows found in {path}")
    return prices


def load_volatility_series(data_dir: Path) -> dict[str, LevelByDate]:
    """Load every Cboe volatility series the study requires."""
    return {
        name: load_cboe_history(data_dir / filename)
        for name, filename in REQUIRED_CBOE_FILES.items()
    }


def load_price_series(data_dir: Path) -> dict[str, LevelByDate]:
    """Load every price series the study requires."""
    return {
        ticker: load_price_history(data_dir / filename)
        for ticker, filename in REQUIRED_PRICE_FILES.items()
    }


def load_universe(path: Path) -> list[Security]:
    """Load the study universe and each security's evidence status.

    Args:
        path: Path to the universe reference CSV.

    Returns:
        Every security, in file order.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is wrong or no securities are present.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing universe reference: {path}")

    securities: list[Security] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not UNIVERSE_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(UNIVERSE_COLUMNS)}")
        for raw in reader:
            securities.append(
                Security(
                    ticker=raw["ticker"].strip().upper(),
                    underlying_symbol=raw["underlying_symbol"].strip(),
                    name=raw["name"].strip(),
                    security_type=raw["security_type"].strip(),
                    group=raw["group"].strip(),
                    size_bucket=raw["size_bucket"].strip(),
                    dividend_yield=float(raw["dividend_yield"] or 0.0),
                    iv_evidence=raw["iv_evidence"].strip(),
                    return_evidence=raw["return_evidence"].strip(),
                    measured=parse_bool(raw["measured"]),
                )
            )

    if not securities:
        raise ValueError(f"No securities found in {path}")
    logger.info(
        "Loaded %d securities; %d measured from observed data",
        len(securities),
        sum(security.measured for security in securities),
    )
    return securities
