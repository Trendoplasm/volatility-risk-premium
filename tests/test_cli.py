"""Command-line behaviour, including how failures are reported."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from vrp import __version__
from vrp.cli import main

from .conftest import (
    alternating_price_path,
    security,
    trading_dates,
    write_cboe_csv,
    write_price_csv,
    write_universe_csv,
)

EXPECTED_TABLES = 15


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Build a complete synthetic input tree."""
    calendar = trading_dates(date(2011, 1, 3), 400)
    prices = alternating_price_path(len(calendar))
    raw = tmp_path / "raw"
    for name in ("VIX9D", "VIX", "VIX3M", "VIX1Y", "VXAPL", "VXAZN"):
        write_cboe_csv(raw / f"{name}_History.csv", calendar, [25.0] * len(calendar))
    for ticker in ("INDEX", "AAPL", "AMZN"):
        write_price_csv(raw / f"{ticker}_prices.csv", calendar, prices)
    write_universe_csv(
        tmp_path / "universe.csv",
        [security(t) for t in ("INDEX", "AAPL", "AMZN")] + [security("MSFT", measured=False)],
    )
    return tmp_path


def run(workspace: Path, *extra: str) -> int:
    return main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--universe",
            str(workspace / "universe.csv"),
            "--output-dir",
            str(workspace / "out"),
            "--start-date",
            "2011-01-03",
            "--bootstrap-iterations",
            "50",
            *extra,
        ]
    )


def test_writes_every_table_and_figure(workspace: Path) -> None:
    assert run(workspace) == 0
    out = workspace / "out"
    assert len(list((out / "tables").glob("*.csv"))) == EXPECTED_TABLES
    assert {p.name for p in (out / "plots").glob("*.png")} == {
        "implied_vs_realized.png",
        "pnl_attribution.png",
        "regime_premium.png",
        "term_structure.png",
    }
    assert (out / "summary.json").exists()


def test_reports_the_headline_result(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(workspace)
    output = capsys.readouterr().out
    assert "delta-hedged trades" in output
    assert "variance risk premium" in output


def test_summary_states_the_scope(workspace: Path) -> None:
    import json

    run(workspace, "--no-plots")
    payload = json.loads((workspace / "out" / "summary.json").read_text())
    # Every export restates what is and is not measured, so a downstream reader cannot lose it.
    assert "not measured" in payload["scope_note"]
    assert payload["securities_measured"] == ["INDEX", "AAPL", "AMZN"]
    assert payload["securities_not_measured"] == ["MSFT"]


def test_no_plots_skips_figures(workspace: Path) -> None:
    assert run(workspace, "--no-plots") == 0
    assert not (workspace / "out" / "plots").exists()


def test_quiet_suppresses_progress_output(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(workspace, "--quiet", "--no-plots")
    assert capsys.readouterr().out == ""


def test_hedge_interval_is_configurable(workspace: Path) -> None:
    import csv

    assert run(workspace, "--no-plots", "--hedge-interval-days", "5") == 0
    rows = list(csv.DictReader((workspace / "out" / "tables" / "hedged_trades.csv").open()))
    assert {row["hedge_interval_days"] for row in rows} == {"5"}


def test_missing_input_reports_a_message_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        ["--data-dir", str(tmp_path / "absent"), "--universe", str(tmp_path / "absent.csv")]
    )
    captured = capsys.readouterr()
    assert status == 1
    assert captured.err.lower().startswith("error:")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("flag", ["--bootstrap-iterations", "--hedge-interval-days"])
def test_nonpositive_settings_are_usage_errors(
    workspace: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    status = main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--universe",
            str(workspace / "universe.csv"),
            "--output-dir",
            str(workspace / "out"),
            flag,
            "0",
        ]
    )
    assert status == 2
    assert "must be positive" in capsys.readouterr().err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
