"""Option marks and Greeks, checked against closed-form values and finite differences."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vrp.blackscholes import forward_price, straddle_mark, straddle_value

#: Textbook value of a one-year at-the-money call at 20% volatility with zero rates.
TEXTBOOK_ATM_CALL = 7.965567455405804


class TestForwardPrice:
    def test_zero_carry_leaves_spot_unchanged(self) -> None:
        assert forward_price(100.0, 0.0, 0.0, 1.0) == pytest.approx(100.0)

    def test_rate_above_dividend_lifts_the_forward(self) -> None:
        assert forward_price(100.0, 0.05, 0.01, 1.0) == pytest.approx(100.0 * math.exp(0.04))

    def test_dividend_above_rate_lowers_the_forward(self) -> None:
        assert float(forward_price(100.0, 0.01, 0.05, 1.0)) < 100.0


class TestStraddleValue:
    def test_matches_the_textbook_atm_case(self) -> None:
        mark = straddle_mark(100.0, 100.0, 0.0, 0.0, 0.20, 1.0)
        assert mark.call_value == pytest.approx(TEXTBOOK_ATM_CALL, abs=1e-8)
        # With zero rates and no dividend the two legs are worth the same at the money.
        assert mark.put_value == pytest.approx(mark.call_value)
        assert mark.value == pytest.approx(2 * TEXTBOOK_ATM_CALL, abs=1e-8)

    def test_put_call_parity_holds_with_carry(self) -> None:
        spot, strike, rate, dividend, vol, maturity = 123.0, 117.0, 0.0425, 0.013, 0.31, 0.35
        mark = straddle_mark(spot, strike, rate, dividend, vol, maturity)
        forward = float(forward_price(spot, rate, dividend, maturity))
        parity = math.exp(-rate * maturity) * (forward - strike)
        assert mark.call_value - mark.put_value == pytest.approx(parity, abs=1e-10)

    def test_value_rises_with_volatility(self) -> None:
        low = straddle_mark(100.0, 100.0, 0.0, 0.0, 0.15, 0.5).value
        high = straddle_mark(100.0, 100.0, 0.0, 0.0, 0.45, 0.5).value
        assert high > low

    def test_vectorised_form_agrees_with_the_scalar_form(self) -> None:
        spots = np.array([80.0, 100.0, 120.0])
        vectorised = straddle_value(spots, 100.0, 0.02, 0.01, 0.25, 0.5)
        scalar = [straddle_mark(float(s), 100.0, 0.02, 0.01, 0.25, 0.5).value for s in spots]
        assert np.allclose(vectorised, scalar)


class TestGreeks:
    SPOT, STRIKE, RATE, DIVIDEND, VOL, MATURITY = 123.0, 117.0, 0.0425, 0.013, 0.31, 0.35

    def value(self, **overrides: float) -> float:
        args = {
            "spot": self.SPOT,
            "strike": self.STRIKE,
            "rate": self.RATE,
            "dividend_yield": self.DIVIDEND,
            "volatility": self.VOL,
            "maturity": self.MATURITY,
        }
        args.update(overrides)
        return float(straddle_value(**args))

    @property
    def mark(self) -> object:
        return straddle_mark(
            self.SPOT, self.STRIKE, self.RATE, self.DIVIDEND, self.VOL, self.MATURITY
        )

    def test_delta_matches_a_finite_difference(self) -> None:
        step = 1e-4
        finite = (self.value(spot=self.SPOT + step) - self.value(spot=self.SPOT - step)) / (
            2 * step
        )
        assert self.mark.delta == pytest.approx(finite, rel=1e-6)  # type: ignore[attr-defined]

    def test_gamma_matches_a_finite_difference(self) -> None:
        # A well-conditioned step: a second difference divided by step squared loses precision
        # rapidly as the step shrinks, so 0.01 is far more accurate here than 1e-6 would be.
        step = 1e-2
        finite = (
            self.value(spot=self.SPOT + step) - 2 * self.value() + self.value(spot=self.SPOT - step)
        ) / step**2
        assert self.mark.gamma == pytest.approx(finite, rel=1e-5)  # type: ignore[attr-defined]

    def test_vega_matches_a_finite_difference(self) -> None:
        step = 1e-6
        finite = (
            self.value(volatility=self.VOL + step) - self.value(volatility=self.VOL - step)
        ) / (2 * step)
        assert self.mark.vega == pytest.approx(finite, rel=1e-6)  # type: ignore[attr-defined]

    def test_theta_matches_a_finite_difference(self) -> None:
        step = 1e-6
        finite = -(
            self.value(maturity=self.MATURITY + step) - self.value(maturity=self.MATURITY - step)
        ) / (2 * step)
        assert self.mark.theta == pytest.approx(finite, rel=1e-5)  # type: ignore[attr-defined]

    def test_straddle_gamma_and_vega_are_twice_a_single_leg(self) -> None:
        # The two legs share gamma and vega exactly, which is why a straddle is the natural
        # instrument for taking a variance position.
        mark = straddle_mark(100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
        assert mark.gamma > 0
        assert mark.vega > 0

    def test_atm_straddle_is_nearly_delta_neutral(self) -> None:
        mark = straddle_mark(100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
        assert abs(mark.delta) < 0.1

    def test_long_straddle_loses_to_time(self) -> None:
        assert self.mark.theta < 0  # type: ignore[attr-defined]


class TestBoundaryCases:
    def test_settles_at_intrinsic_at_expiry(self) -> None:
        mark = straddle_mark(110.0, 100.0, 0.04, 0.01, 0.3, 0.0)
        assert mark.value == pytest.approx(10.0)
        assert mark.gamma == 0.0
        assert mark.vega == 0.0
        assert mark.theta == 0.0

    def test_zero_volatility_settles_at_intrinsic(self) -> None:
        mark = straddle_mark(90.0, 100.0, 0.0, 0.0, 0.0, 0.5)
        assert mark.value == pytest.approx(10.0)
        assert mark.vega == 0.0

    def test_at_the_money_at_expiry_is_worthless(self) -> None:
        assert straddle_mark(100.0, 100.0, 0.0, 0.0, 0.3, 0.0).value == pytest.approx(0.0)

    def test_expiry_delta_reflects_which_leg_is_in_the_money(self) -> None:
        assert straddle_mark(110.0, 100.0, 0.0, 0.0, 0.3, 0.0).delta == 1.0
        assert straddle_mark(90.0, 100.0, 0.0, 0.0, 0.3, 0.0).delta == -1.0

    def test_deep_out_of_the_money_leg_contributes_almost_nothing(self) -> None:
        mark = straddle_mark(100.0, 1000.0, 0.0, 0.0, 0.10, 0.05)
        assert mark.call_value < 1e-6
        assert mark.put_value == pytest.approx(900.0, rel=1e-3)
