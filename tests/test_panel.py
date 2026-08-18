"""The matched implied-versus-realised variance panel."""

from __future__ import annotations

from datetime import date

import pytest

from vrp.config import StudyConfig
from vrp.models import Security
from vrp.panel import aligned_calendar, build_panel, panel_summary

from .conftest import (
    KNOWN_IMPLIED_VOL,
    KNOWN_REALIZED_VOL,
    alternating_price_path,
    security,
    trading_dates,
    volatility_series,
)


@pytest.fixture
def calendar() -> list[date]:
    """Return 300 weekday trading dates."""
    return trading_dates(date(2011, 1, 3), 300)


@pytest.fixture
def universe() -> list[Security]:
    """Return a single measured security."""
    return [security("INDEX")]


@pytest.fixture
def volatility(calendar: list[date]) -> dict[str, dict[date, float]]:
    """Return a flat volatility term structure over the calendar."""
    return volatility_series(calendar)


@pytest.fixture
def price_map(calendar: list[date]) -> dict[str, dict[date, float]]:
    """Return the exact-volatility price path keyed by date."""
    return {"INDEX": dict(zip(calendar, alternating_price_path(len(calendar)), strict=True))}


class TestAlignedCalendar:
    def test_stops_at_the_last_common_date(self, config: StudyConfig) -> None:
        calendar = trading_dates(date(2011, 1, 3), 100)
        prices = dict(zip(calendar, alternating_price_path(len(calendar)), strict=True))
        anchor = dict.fromkeys(calendar[:80], 0.2)
        dates, _ = aligned_calendar(prices, anchor, config.start())
        assert max(dates) == calendar[79]

    def test_keeps_history_before_the_study_start(self) -> None:
        # Trailing windows must be populated on the study's first day rather than empty.
        calendar = trading_dates(date(2011, 1, 3), 400)
        prices = dict(zip(calendar, alternating_price_path(len(calendar)), strict=True))
        anchor = dict.fromkeys(calendar, 0.2)
        dates, _ = aligned_calendar(prices, anchor, calendar[200])
        assert min(dates) < calendar[200]

    def test_empty_volatility_history_is_rejected(self, config: StudyConfig) -> None:
        calendar = trading_dates(date(2011, 1, 3), 10)
        prices = dict(zip(calendar, alternating_price_path(len(calendar)), strict=True))
        with pytest.raises(ValueError, match="empty volatility series"):
            aligned_calendar(prices, {}, config.start())

    def test_non_overlapping_histories_are_rejected(self, config: StudyConfig) -> None:
        prices = {date(2020, 1, 2): 100.0}
        anchor = {date(2011, 1, 3): 0.2}
        with pytest.raises(ValueError, match="do not overlap"):
            aligned_calendar(prices, anchor, config.start())


class TestBuildPanel:
    def test_measures_the_known_premium_exactly(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        # Implied volatility is flat at 25% and the path delivers exactly 20%, so every row's
        # premium is a known constant. A flat term structure means no maturity adjustment either.
        panel = build_panel(universe, volatility, price_map, config)
        assert panel
        for row in panel:
            assert row["matched_iv"] == pytest.approx(KNOWN_IMPLIED_VOL)
            assert row["realized_vol"] == pytest.approx(KNOWN_REALIZED_VOL)
            assert row["vrp_vol_points"] == pytest.approx(KNOWN_IMPLIED_VOL - KNOWN_REALIZED_VOL)
            assert row["vrp_variance"] == pytest.approx(
                KNOWN_IMPLIED_VOL**2 - KNOWN_REALIZED_VOL**2
            )

    def test_covers_every_configured_horizon(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        panel = build_panel(universe, volatility, price_map, config)
        assert {row["horizon_days"] for row in panel} == set(config.all_horizons())

    def test_flat_term_structure_leaves_the_multiplier_at_one(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        for row in build_panel(universe, volatility, price_map, config):
            assert row["term_multiplier"] == pytest.approx(1.0)

    def test_skips_securities_that_are_not_measured(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        panel = build_panel([security("INDEX", measured=False)], volatility, price_map, config)
        assert panel == []

    def test_marks_the_in_sample_period(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        for row in build_panel(universe, volatility, price_map, config):
            assert row["in_sample"] == (row["date"] <= config.train_cutoff())

    def test_longer_horizons_have_fewer_observations(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        # A longer forward window runs off the end of the data sooner.
        panel = build_panel(universe, volatility, price_map, config)
        counts = {
            horizon: sum(row["horizon_days"] == horizon for row in panel)
            for horizon in config.all_horizons()
        }
        assert counts[21] > counts[63]

    def test_rows_are_sorted(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        panel = build_panel(universe, volatility, price_map, config)
        keys = [(row["ticker"], row["horizon_days"], row["date"]) for row in panel]
        assert keys == sorted(keys)


class TestPanelSummary:
    def test_summarises_a_known_panel(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        panel = [
            row
            for row in build_panel(universe, volatility, price_map, config)
            if row["horizon_days"] == 21
        ]
        summary = panel_summary(panel, "INDEX", 21)
        assert summary["n"] == len(panel)
        assert summary["mean_implied_vol"] == pytest.approx(KNOWN_IMPLIED_VOL)
        assert summary["mean_realized_vol"] == pytest.approx(KNOWN_REALIZED_VOL)
        # Implied exceeds realised on every single observation of this fixture.
        assert summary["pct_positive_vrp"] == pytest.approx(1.0)

    def test_variance_ratio_is_reported(
        self,
        universe: list[Security],
        volatility: dict[str, dict[date, float]],
        price_map: dict[str, dict[date, float]],
        config: StudyConfig,
    ) -> None:
        panel = [
            row
            for row in build_panel(universe, volatility, price_map, config)
            if row["horizon_days"] == 21
        ]
        expected = (KNOWN_IMPLIED_VOL / KNOWN_REALIZED_VOL) ** 2
        assert panel_summary(panel, "INDEX", 21)["mean_variance_ratio"] == pytest.approx(expected)

    def test_empty_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty panel group"):
            panel_summary([], "INDEX", 21)
