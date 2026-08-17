#!/usr/bin/env python3
"""Download the Cboe volatility-index histories used as implied-volatility anchors.

Cboe's data is published under Cboe's terms of use and is not redistributed with this
repository, so a fresh clone fetches it from the source:

    python scripts/fetch_cboe_data.py

Four of the series form an observed volatility *term structure* for the S&P 500 -- 9-day,
30-day, 93-day, and 1-year expected volatility. The study interpolates that curve to its 21-,
42-, and 63-trading-day study horizons rather than assuming a shape.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices"

#: Series required by the study, with the role each one plays.
SERIES: dict[str, str] = {
    "VIX9D": "S&P 500 9-day expected volatility (short end of the term structure)",
    "VIX": "S&P 500 30-day expected volatility (the index implied-volatility anchor)",
    "VIX3M": "S&P 500 93-day expected volatility (long end of the term structure)",
    "VIX1Y": "S&P 500 1-year expected volatility (term-structure context)",
    "VXAPL": "Apple 30-day expected volatility",
    "VXAZN": "Amazon 30-day expected volatility",
}

DEFAULT_DEST = Path("data/raw")
TIMEOUT_SECONDS = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Cboe volatility-index histories.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Download directory.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    return parser.parse_args(argv)


def fetch(series: str, dest: Path, *, force: bool) -> bool:
    """Download one series, returning True if a file was written."""
    filename = f"{series}_History.csv"
    target = dest / filename
    if target.exists() and not force:
        print(f"  {filename}: already present, skipping")
        return False

    url = f"{BASE_URL}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    if b"DATE" not in payload[:200].upper():
        raise RuntimeError(f"{url} did not return a Cboe history file")

    dest.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f"  {filename}: {len(payload) / 1024:.0f} KiB written")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Fetching {len(SERIES)} Cboe volatility-index histories into {args.dest}")
    try:
        written = sum(fetch(series, args.dest, force=args.force) for series in SERIES)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {written} written, {len(SERIES) - written} already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
