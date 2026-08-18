"""Input parsing, and the loud failures that protect the published statistics."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from vrp.loaders import (
    load_cboe_history,
    load_price_history,
    load_price_series,
    load_universe,
    load_volatility_series,
    parse_bool,
)

from .conftest import (
    alternating_price_path,
    security,
    trading_dates,
    write_cboe_csv,
    write_price_csv,
    write_universe_csv,
)


class TestParseBool:
    @pytest.mark.parametrize("value", ["TRUE", "true", "1", "yes", "Y", " true "])
    def test_truthy_spellings(self, value: str) -> None:
        assert parse_bool(value) is True

    @pytest.mark.parametrize("value", ["FALSE", "false", "0", "no", "", None])
    def test_everything_else_is_false(self, value: str | None) -> None:
        assert parse_bool(value) is False


class TestLoadCboeHistory:
    def test_converts_percentage_points_to_decimals(self, tmp_path: Path) -> None:
        # Cboe quotes 20 for 20% volatility. Converting once, on the way in, means no formula
        # downstream has to remember which convention it is in.
        write_cboe_csv(
            tmp_path / "VIX_History.csv", trading_dates(date(2011, 1, 3), 2), [20.0, 25.0]
        )
        levels = load_cboe_history(tmp_path / "VIX_History.csv")
        assert list(levels.values()) == pytest.approx([0.20, 0.25])

    def test_tolerates_a_byte_order_mark(self, tmp_path: Path) -> None:
        path = tmp_path / "VIX_History.csv"
        path.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n01/03/2011,20,20,20,20\n", encoding="utf-8-sig")
        assert load_cboe_history(path)[date(2011, 1, 3)] == pytest.approx(0.20)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing Cboe input"):
            load_cboe_history(tmp_path / "absent.csv")

    def test_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "VIX_History.csv"
        path.write_text("DATE,CLOSE\n01/03/2011,20\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected Cboe columns"):
            load_cboe_history(path)

    def test_unparseable_date(self, tmp_path: Path) -> None:
        path = tmp_path / "VIX_History.csv"
        path.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n2011-01-03,20,20,20,20\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid row"):
            load_cboe_history(path)

    @pytest.mark.parametrize("level", ["0", "-3"])
    def test_nonpositive_level_is_rejected(self, tmp_path: Path, level: str) -> None:
        path = tmp_path / "VIX_History.csv"
        path.write_text(f"DATE,OPEN,HIGH,LOW,CLOSE\n01/03/2011,1,1,1,{level}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Nonpositive volatility"):
            load_cboe_history(path)

    def test_header_only_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "VIX_History.csv"
        path.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No data rows"):
            load_cboe_history(path)


class TestLoadPriceHistory:
    def test_reads_a_two_column_file(self, tmp_path: Path) -> None:
        calendar = trading_dates(date(2011, 1, 3), 3)
        write_price_csv(tmp_path / "INDEX_prices.csv", calendar, [100.0, 101.0, 102.0])
        prices = load_price_history(tmp_path / "INDEX_prices.csv")
        assert prices == {calendar[0]: 100.0, calendar[1]: 101.0, calendar[2]: 102.0}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing price input"):
            load_price_history(tmp_path / "absent.csv")

    def test_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "p.csv"
        path.write_text("date,open\n2011-01-03,100\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected columns"):
            load_price_history(path)

    def test_nonpositive_close_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.csv"
        path.write_text("date,close\n2011-01-03,0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Nonpositive close"):
            load_price_history(path)

    def test_unparseable_date(self, tmp_path: Path) -> None:
        path = tmp_path / "p.csv"
        path.write_text("date,close\n01/03/2011,100\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid row"):
            load_price_history(path)


class TestLoadSeriesSets:
    def test_loads_every_required_volatility_series(self, tmp_path: Path) -> None:
        calendar = trading_dates(date(2011, 1, 3), 5)
        for name in ("VIX9D", "VIX", "VIX3M", "VIX1Y", "VXAPL", "VXAZN"):
            write_cboe_csv(tmp_path / f"{name}_History.csv", calendar, [20.0] * 5)
        assert set(load_volatility_series(tmp_path)) == {
            "VIX9D",
            "VIX",
            "VIX3M",
            "VIX1Y",
            "VXAPL",
            "VXAZN",
        }

    def test_reports_a_missing_volatility_series(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_volatility_series(tmp_path)

    def test_loads_every_required_price_series(self, tmp_path: Path) -> None:
        calendar = trading_dates(date(2011, 1, 3), 5)
        prices = alternating_price_path(5)
        for ticker in ("INDEX", "AAPL", "AMZN"):
            write_price_csv(tmp_path / f"{ticker}_prices.csv", calendar, prices)
        assert set(load_price_series(tmp_path)) == {"INDEX", "AAPL", "AMZN"}


class TestLoadUniverse:
    def test_reads_securities_and_their_evidence_status(self, tmp_path: Path) -> None:
        path = tmp_path / "universe.csv"
        write_universe_csv(path, [security("INDEX"), security("MSFT", measured=False)])
        loaded = load_universe(path)
        assert [s.ticker for s in loaded] == ["INDEX", "MSFT"]
        assert loaded[0].measured is True
        assert loaded[1].measured is False

    def test_keeps_the_dividend_yield_assumption(self, tmp_path: Path) -> None:
        path = tmp_path / "universe.csv"
        write_universe_csv(path, [security("AAPL", dividend_yield=0.005)])
        assert load_universe(path)[0].dividend_yield == pytest.approx(0.005)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing universe reference"):
            load_universe(tmp_path / "absent.csv")

    def test_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "universe.csv"
        path.write_text("ticker,name\nINDEX,S&P 500\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain"):
            load_universe(path)

    def test_empty_universe_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "universe.csv"
        write_universe_csv(path, [])
        with pytest.raises(ValueError, match="No securities found"):
            load_universe(path)
