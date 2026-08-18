"""Matched-maturity implied volatility from the observed term structure."""

from __future__ import annotations

import math

import pytest

from vrp.termstructure import (
    interpolate_curve,
    matched_implied_volatility,
    shape_multiplier,
    trading_days_to_calendar_days,
)

FLAT_CURVE = {9.0: 0.20, 30.0: 0.20, 93.0: 0.20, 365.0: 0.20}
UPWARD_CURVE = {9.0: 0.12, 30.0: 0.15, 93.0: 0.19, 365.0: 0.23}


class TestCalendarConversion:
    def test_a_month_of_trading_is_about_a_calendar_month(self) -> None:
        # 21 trading days is close to 30 calendar days, which is why the study's core horizon
        # lands almost exactly on the 30-day index Cboe already publishes.
        assert trading_days_to_calendar_days(21) == pytest.approx(30.4, abs=0.1)

    def test_a_quarter_of_trading_is_about_ninety_days(self) -> None:
        assert trading_days_to_calendar_days(63) == pytest.approx(91.25, abs=0.1)

    def test_a_full_year_of_trading_is_a_calendar_year(self) -> None:
        assert trading_days_to_calendar_days(252) == pytest.approx(365.0)


class TestInterpolateCurve:
    @pytest.mark.parametrize("horizon", [9.0, 30.0, 93.0, 365.0])
    def test_returns_observed_points_exactly(self, horizon: float) -> None:
        assert interpolate_curve(UPWARD_CURVE, horizon) == pytest.approx(UPWARD_CURVE[horizon])

    def test_flat_curve_stays_flat_everywhere(self) -> None:
        for horizon in (10.0, 21.0, 45.0, 200.0):
            assert interpolate_curve(FLAT_CURVE, horizon) == pytest.approx(0.20)

    def test_interpolates_linearly_in_total_variance(self) -> None:
        # Total variance, not volatility, is what must be linear in time for the curve to be free
        # of calendar arbitrage.
        target = 60.0
        lower, upper = 30.0, 93.0
        weight = (target - lower) / (upper - lower)
        blended = UPWARD_CURVE[lower] ** 2 * lower + weight * (
            UPWARD_CURVE[upper] ** 2 * upper - UPWARD_CURVE[lower] ** 2 * lower
        )
        assert interpolate_curve(UPWARD_CURVE, target) == pytest.approx(math.sqrt(blended / target))

    def test_interpolated_value_lies_between_its_neighbours(self) -> None:
        value = interpolate_curve(UPWARD_CURVE, 60.0)
        assert UPWARD_CURVE[30.0] < value < UPWARD_CURVE[93.0]

    def test_holds_flat_beyond_the_observed_range(self) -> None:
        # Extrapolating a variance curve would invent information the market never quoted.
        assert interpolate_curve(UPWARD_CURVE, 1.0) == pytest.approx(UPWARD_CURVE[9.0])
        assert interpolate_curve(UPWARD_CURVE, 5000.0) == pytest.approx(UPWARD_CURVE[365.0])

    def test_single_point_curve_is_flat(self) -> None:
        assert interpolate_curve({30.0: 0.18}, 60.0) == pytest.approx(0.18)

    def test_empty_curve_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty volatility curve"):
            interpolate_curve({}, 30.0)

    @pytest.mark.parametrize("horizon", [0.0, -5.0])
    def test_nonpositive_horizon_is_rejected(self, horizon: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            interpolate_curve(UPWARD_CURVE, horizon)


class TestShapeMultiplier:
    def test_flat_curve_gives_a_multiplier_of_one(self) -> None:
        for horizon in (21, 42, 63):
            assert shape_multiplier(FLAT_CURVE, horizon) == pytest.approx(1.0)

    def test_upward_curve_lifts_longer_horizons(self) -> None:
        assert shape_multiplier(UPWARD_CURVE, 21) > 1.0
        assert shape_multiplier(UPWARD_CURVE, 63) > shape_multiplier(UPWARD_CURVE, 21)

    def test_core_horizon_multiplier_is_close_to_one(self) -> None:
        # The core horizon sits almost on the 30-day point, so the adjustment is small even when
        # the curve is steep. That is what keeps the single-name approximation mild.
        assert shape_multiplier(UPWARD_CURVE, 21) == pytest.approx(1.0, abs=0.02)

    def test_curve_without_a_thirty_day_point_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive 30-day level"):
            shape_multiplier({9.0: 0.2, 93.0: 0.25}, 21)


class TestMatchedImpliedVolatility:
    def test_flat_curve_leaves_the_anchor_untouched(self) -> None:
        assert matched_implied_volatility(0.30, FLAT_CURVE, 63) == pytest.approx(0.30)

    def test_scales_the_anchor_by_the_observed_shape(self) -> None:
        result = matched_implied_volatility(0.30, UPWARD_CURVE, 63)
        assert result == pytest.approx(0.30 * shape_multiplier(UPWARD_CURVE, 63))

    def test_returns_none_without_the_thirty_day_anchor(self) -> None:
        assert matched_implied_volatility(0.30, {9.0: 0.2, 93.0: 0.25}, 21) is None
