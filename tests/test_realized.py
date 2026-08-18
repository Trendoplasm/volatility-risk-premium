"""Realised variance, and the guarantee that it never looks the wrong way in time."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vrp.config import TRADING_DAYS_PER_YEAR
from vrp.realized import (
    forward_realized_variance,
    log_returns,
    realized_variance,
    realized_volatility,
    trailing_realized_volatility,
)

from .conftest import DAILY_RETURN, KNOWN_REALIZED_VOL, alternating_price_path


class TestLogReturns:
    def test_computes_log_differences(self) -> None:
        assert log_returns([100.0, 110.0]) == pytest.approx([math.log(1.1)])

    def test_returns_one_fewer_value_than_prices(self) -> None:
        assert log_returns([1.0, 2.0, 3.0, 4.0]).size == 3

    @pytest.mark.parametrize("prices", [[], [100.0]])
    def test_too_short_a_window_yields_nothing(self, prices: list[float]) -> None:
        assert log_returns(prices).size == 0

    @pytest.mark.parametrize("prices", [[100.0, 0.0], [100.0, -5.0]])
    def test_nonpositive_prices_are_rejected(self, prices: list[float]) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            log_returns(prices)

    def test_log_returns_are_additive(self) -> None:
        prices = [100.0, 105.0, 99.0, 120.0]
        assert float(np.sum(log_returns(prices))) == pytest.approx(math.log(120.0 / 100.0))


class TestRealizedVariance:
    def test_recovers_the_constructed_volatility_exactly(self) -> None:
        # The fixture path moves by a fixed log amount every day, so its realised variance is
        # 252 * daily^2 with no sampling error whatsoever.
        prices = alternating_price_path(50)
        assert realized_variance(prices) == pytest.approx(KNOWN_REALIZED_VOL**2)
        assert realized_volatility(prices) == pytest.approx(KNOWN_REALIZED_VOL)

    def test_annualises_by_trading_days(self) -> None:
        prices = alternating_price_path(30, daily=0.01)
        assert realized_variance(prices) == pytest.approx(TRADING_DAYS_PER_YEAR * 0.01**2)

    def test_a_flat_path_has_no_variance(self) -> None:
        assert realized_variance([100.0] * 30) == pytest.approx(0.0)

    def test_too_short_a_window_is_reported_as_missing(self) -> None:
        assert realized_variance([100.0]) is None
        assert realized_volatility([100.0]) is None

    def test_doubling_the_move_quadruples_the_variance(self) -> None:
        single = realized_variance(alternating_price_path(40, daily=0.01))
        double = realized_variance(alternating_price_path(40, daily=0.02))
        assert single is not None and double is not None
        assert double / single == pytest.approx(4.0)


class TestTrailingRealizedVolatility:
    def test_uses_the_window_ending_at_the_index(self) -> None:
        prices = alternating_price_path(100)
        assert trailing_realized_volatility(prices, 60, 21) == pytest.approx(KNOWN_REALIZED_VOL)

    def test_returns_none_before_enough_history_exists(self) -> None:
        prices = alternating_price_path(100)
        assert trailing_realized_volatility(prices, 5, 21) is None

    def test_short_windows_are_refused(self) -> None:
        # Fewer observations than the configured minimum would give a noisy estimate that looks
        # like a real one.
        prices = alternating_price_path(100)
        assert trailing_realized_volatility(prices, 50, 5) is None

    def test_ignores_the_future(self) -> None:
        # The decisive property for a signal input: changing prices after the observation date must
        # not change its value, or the study would be trading on information it did not have.
        prices = alternating_price_path(100)
        before = trailing_realized_volatility(prices, 50, 21)
        tampered = list(prices)
        for index in range(51, len(tampered)):
            tampered[index] *= 3.0
        assert trailing_realized_volatility(tampered, 50, 21) == pytest.approx(before)


class TestForwardRealizedVariance:
    def test_uses_the_window_after_the_index(self) -> None:
        prices = alternating_price_path(100)
        assert forward_realized_variance(prices, 20, 21) == pytest.approx(KNOWN_REALIZED_VOL**2)

    def test_returns_none_when_the_horizon_runs_past_the_data(self) -> None:
        prices = alternating_price_path(30)
        assert forward_realized_variance(prices, 20, 21) is None

    def test_ignores_the_past(self) -> None:
        prices = alternating_price_path(100)
        before = forward_realized_variance(prices, 50, 21)
        tampered = list(prices)
        for index in range(50):
            tampered[index] *= 3.0
        assert forward_realized_variance(tampered, 50, 21) == pytest.approx(before)

    def test_reacts_to_the_window_it_covers(self) -> None:
        prices = alternating_price_path(100)
        tampered = list(prices)
        # A single large jump inside the forward window must raise the measured variance.
        tampered[55] *= 1.20
        raised = forward_realized_variance(tampered, 50, 21)
        base = forward_realized_variance(prices, 50, 21)
        assert raised is not None and base is not None and raised > base


def test_realized_volatility_is_the_square_root_of_variance() -> None:
    prices = alternating_price_path(60, daily=0.013)
    variance = realized_variance(prices)
    volatility = realized_volatility(prices)
    assert variance is not None and volatility is not None
    assert volatility == pytest.approx(math.sqrt(variance))


def test_fixture_daily_size_matches_its_documented_volatility() -> None:
    assert DAILY_RETURN * math.sqrt(TRADING_DAYS_PER_YEAR) == pytest.approx(KNOWN_REALIZED_VOL)
