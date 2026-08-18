"""Conditioning signals, and the guarantee that selection cannot peek at the test period.

The single most important property in this module is negative: nothing about the out-of-sample
period may influence which rule gets chosen. A study that violates that reports a discovery when
all it has done is describe its own sample.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from vrp.config import CostModel, StudyConfig
from vrp.signals import (
    MIN_IN_SAMPLE_TRADES,
    SCAN_SIGNAL,
    build_signal_rows,
    select_threshold,
    selection_summary,
    signal_regression,
    threshold_scan,
)
from vrp.strategy import SecurityInputs, run_protocol

from .conftest import alternating_price_path, security, trading_dates, volatility_series


def make_rows(
    count: int, start: date, *, in_sample: bool, signal_scale: float
) -> list[dict[str, Any]]:
    """Build signal rows whose outcome depends on the signal by a controllable amount."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        signal = -0.10 + 0.25 * index / max(count - 1, 1)
        rows.append(
            {
                "ticker": "INDEX",
                "entry_date": start + timedelta(days=index),
                "in_sample": in_sample,
                SCAN_SIGNAL: signal,
                "term_slope": 0.01 * (index % 5),
                "vix_percentile": (index % 10) / 10,
                "entry_iv": 0.22,
                "trailing_realized_vol": 0.20,
                "realized_vol": 0.19,
                "short_return_on_capital": signal_scale * signal,
                "short_net_pnl": signal_scale * signal * 1000,
            }
        )
    return rows


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    """In-sample rows where the signal helps, out-of-sample rows where it hurts."""
    return make_rows(120, date(2012, 1, 2), in_sample=True, signal_scale=0.5) + make_rows(
        120, date(2020, 1, 2), in_sample=False, signal_scale=-0.5
    )


class TestThresholdScan:
    def test_reports_both_periods_for_every_threshold(self, rows: list[dict[str, Any]]) -> None:
        scan = threshold_scan(rows)
        assert scan
        for entry in scan:
            assert "in_sample_mean_roc" in entry
            assert "out_of_sample_mean_roc" in entry

    def test_tighter_thresholds_keep_fewer_trades(self, rows: list[dict[str, Any]]) -> None:
        counts = [entry["in_sample_n"] for entry in threshold_scan(rows)]
        assert counts == sorted(counts, reverse=True)

    def test_marks_thresholds_with_too_little_evidence(self, rows: list[dict[str, Any]]) -> None:
        for entry in threshold_scan(rows):
            assert entry["eligible"] == (entry["in_sample_n"] >= MIN_IN_SAMPLE_TRADES)

    def test_empty_selection_reports_missing_rather_than_zero(self) -> None:
        # Every signal here sits below the tightest threshold, so that threshold selects nothing.
        # An empty selection must report a missing mean, not a zero one that reads as a result.
        rows = make_rows(20, date(2012, 1, 2), in_sample=True, signal_scale=0.5)
        for row in rows:
            row[SCAN_SIGNAL] = -0.20
        tightest = threshold_scan(rows)[-1]
        assert tightest["in_sample_n"] == 0
        assert tightest["in_sample_mean_roc"] is None
        assert tightest["in_sample_win_rate"] is None


class TestSelectionCannotPeek:
    def test_selection_ignores_the_out_of_sample_period(self, rows: list[dict[str, Any]]) -> None:
        # The decisive test. Replace every out-of-sample outcome with its opposite: if selection
        # were influenced by the test period at all, the chosen threshold would move.
        chosen = select_threshold(threshold_scan(rows))
        assert chosen is not None

        tampered = [dict(row) for row in rows]
        for row in tampered:
            if not row["in_sample"]:
                row["short_return_on_capital"] *= -100.0
                row["short_net_pnl"] *= -100.0
        after = select_threshold(threshold_scan(tampered))
        assert after is not None
        assert after["threshold"] == chosen["threshold"]

    def test_selection_responds_to_the_in_sample_period(self, rows: list[dict[str, Any]]) -> None:
        # The mirror image: it must react to the data it is allowed to see, or it is not selecting.
        chosen = select_threshold(threshold_scan(rows))
        assert chosen is not None
        reversed_rows = [dict(row) for row in rows]
        for row in reversed_rows:
            if row["in_sample"]:
                row["short_return_on_capital"] *= -1.0
        after = select_threshold(threshold_scan(reversed_rows))
        assert after is not None
        assert after["threshold"] != chosen["threshold"]

    def test_picks_the_best_eligible_in_sample_threshold(self, rows: list[dict[str, Any]]) -> None:
        scan = threshold_scan(rows)
        chosen = select_threshold(scan)
        assert chosen is not None
        eligible = [e for e in scan if e["eligible"] and e["in_sample_mean_roc"] is not None]
        assert chosen["in_sample_mean_roc"] == max(e["in_sample_mean_roc"] for e in eligible)

    def test_returns_none_without_enough_in_sample_evidence(self) -> None:
        thin = make_rows(3, date(2012, 1, 2), in_sample=True, signal_scale=0.5)
        assert select_threshold(threshold_scan(thin)) is None


class TestSelectionSummary:
    def test_covers_both_rules_in_both_periods(self, rows: list[dict[str, Any]]) -> None:
        summary = selection_summary(rows, threshold_scan(rows))
        assert len(summary) == 4
        assert {row["period"] for row in summary} == {
            "In sample (to 2018)",
            "Out of sample (2019+)",
        }

    def test_unconditional_row_keeps_every_trade(self, rows: list[dict[str, Any]]) -> None:
        summary = selection_summary(rows, threshold_scan(rows))
        unconditional = [r for r in summary if r["rule"] == "Take every trade"]
        assert sum(r["n"] for r in unconditional) == len(rows)

    def test_filtered_row_keeps_no_more_than_the_unconditional_row(
        self, rows: list[dict[str, Any]]
    ) -> None:
        summary = selection_summary(rows, threshold_scan(rows))
        by_key = {(r["rule"].startswith("Filter"), r["period"]): r["n"] for r in summary}
        for period in ("In sample (to 2018)", "Out of sample (2019+)"):
            assert by_key[(True, period)] <= by_key[(False, period)]


class TestSignalRegression:
    def test_fits_on_the_in_sample_period_only(self, rows: list[dict[str, Any]]) -> None:
        coefficients, meta = signal_regression(rows)
        assert meta["n"] == sum(row["in_sample"] for row in rows)
        assert "In sample only" in meta["period"]
        assert next(row["term"] for row in coefficients) == "Intercept"

    def test_recovers_a_planted_relationship(self, rows: list[dict[str, Any]]) -> None:
        # The in-sample outcome is 0.5 times the signal by construction.
        coefficients, _ = signal_regression(rows)
        slope = next(c for c in coefficients if c["term"] == "IV minus trailing realized vol")
        assert slope["coefficient"] == pytest.approx(0.5, abs=0.05)

    def test_too_few_observations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Too few in-sample observations"):
            signal_regression(make_rows(3, date(2012, 1, 2), in_sample=True, signal_scale=0.5))


class TestBuildSignalRows:
    def test_signals_use_only_pre_entry_information(self, config: StudyConfig) -> None:
        calendar = trading_dates(date(2011, 1, 3), 300)
        prices = alternating_price_path(len(calendar))
        volatility = volatility_series(calendar)
        item = security("INDEX")
        inputs = {
            "INDEX": SecurityInputs(
                item, volatility, {"INDEX": dict(zip(calendar, prices, strict=True))}, config
            )
        }
        trades = run_protocol(inputs["INDEX"], config.core_horizon_days, config, CostModel())
        rows = build_signal_rows(inputs, trades, config)
        assert rows
        for row in rows:
            # The trailing realised volatility is exact for this path, so a signal contaminated by
            # forward information would not equal the constructed value.
            assert row["trailing_realized_vol"] == pytest.approx(0.20)
            assert row["iv_minus_trailing_rv"] == pytest.approx(row["entry_iv"] - 0.20)

    def test_flags_the_in_sample_period_by_the_configured_cutoff(self, config: StudyConfig) -> None:
        calendar = trading_dates(date(2011, 1, 3), 300)
        prices = alternating_price_path(len(calendar))
        volatility = volatility_series(calendar)
        item = security("INDEX")
        inputs = {
            "INDEX": SecurityInputs(
                item, volatility, {"INDEX": dict(zip(calendar, prices, strict=True))}, config
            )
        }
        trades = run_protocol(inputs["INDEX"], config.core_horizon_days, config, CostModel())
        for row in build_signal_rows(inputs, trades, config):
            assert row["in_sample"] == (row["entry_date"] <= config.train_cutoff())
