"""The delta-hedged straddle simulation and its profit decomposition.

The tests here pin down the property the whole study rests on: a delta-hedged option position
earns or loses in proportion to the gap between the volatility it was priced at and the volatility
the underlying went on to deliver. If that relationship did not hold in the simulation, no
downstream result would mean anything.
"""

from __future__ import annotations

from datetime import date

import pytest

from vrp.config import CostModel, StudyConfig
from vrp.hedged import Trade, attribution_error, simulate_trade, trade_row

from .conftest import KNOWN_REALIZED_VOL, alternating_price_path, trading_dates

ENTRY = 20
HORIZON = 21

#: The simulation is a daily discretisation of a continuous hedge, so break-even is approached
#: rather than hit exactly. Half a percent of the premium is far tighter than any effect the study
#: reports, and comfortably distinguishes break-even from the 30%-plus swings a real mispricing
#: produces.
BREAK_EVEN_TOLERANCE = 0.005


@pytest.fixture
def calendar() -> list[date]:
    return trading_dates(date(2011, 1, 3), 100)


@pytest.fixture
def path(calendar: list[date]) -> list[float]:
    return alternating_price_path(len(calendar))


def run(
    calendar: list[date],
    path: list[float],
    implied: float,
    config: StudyConfig,
    costs: CostModel,
    **kwargs: object,
) -> Trade:
    """Simulate one trade at a constant implied volatility."""
    trade = simulate_trade(
        "INDEX",
        calendar,
        path,
        [implied] * len(calendar),
        ENTRY,
        HORIZON,
        dividend_yield=0.0,
        entry_vix=implied,
        config=config,
        costs=costs,
        **kwargs,  # type: ignore[arg-type]
    )
    assert trade is not None
    return trade


class TestEconomics:
    def test_breaks_even_when_implied_equals_delivered(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, KNOWN_REALIZED_VOL, config, free_costs)
        assert abs(trade.gross_pnl / trade.entry_premium) < BREAK_EVEN_TOLERANCE

    def test_long_loses_when_implied_exceeds_delivered(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        # This is the volatility risk premium seen from the buyer's side.
        trade = run(calendar, path, 0.25, config, free_costs)
        assert trade.gross_pnl < 0
        assert trade.short_net_pnl > 0

    def test_long_gains_when_delivered_exceeds_implied(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.15, config, free_costs)
        assert trade.gross_pnl > 0
        assert trade.short_net_pnl < 0

    def test_profit_is_monotone_in_the_mispricing(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        results = [
            run(calendar, path, implied, config, free_costs).gross_pnl
            for implied in (0.14, 0.18, 0.22, 0.26)
        ]
        assert results == sorted(results, reverse=True)

    def test_seller_collects_time_decay_and_pays_for_movement(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        # For the long holder theta is a cost and gamma is a benefit; the seller's position is the
        # mirror image, which is the economic story the study is decomposing.
        trade = run(calendar, path, 0.25, config, free_costs)
        assert trade.theta_pnl < 0
        assert trade.gamma_pnl > 0

    def test_richer_premium_buys_more_time_decay(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        cheap = run(calendar, path, 0.15, config, free_costs)
        rich = run(calendar, path, 0.30, config, free_costs)
        assert rich.entry_premium > cheap.entry_premium
        assert rich.theta_pnl < cheap.theta_pnl


class TestAttribution:
    def test_components_sum_to_the_total(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        # The decomposition is an identity, not an approximation, and the residual term is what
        # makes it one. This is the check that keeps the attribution honest.
        for implied in (0.12, 0.20, 0.35):
            trade = run(calendar, path, implied, config, free_costs)
            assert attribution_error(trade) < 1e-9

    def test_constant_implied_volatility_leaves_no_vega_profit(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.22, config, free_costs)
        assert trade.vega_pnl == pytest.approx(0.0)

    def test_a_volatility_repricing_shows_up_in_vega(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        # Step implied volatility up part way through: the long holder should profit on vega.
        implied = [0.20] * len(calendar)
        for index in range(ENTRY + 10, len(calendar)):
            implied[index] = 0.30
        trade = simulate_trade(
            "INDEX",
            calendar,
            path,
            implied,
            ENTRY,
            HORIZON,
            dividend_yield=0.0,
            entry_vix=0.20,
            config=config,
            costs=free_costs,
        )
        assert trade is not None
        assert trade.vega_pnl > 0
        assert attribution_error(trade) < 1e-9


class TestCostsAndCapital:
    def test_costs_reduce_the_seller_s_profit(
        self,
        calendar: list[date],
        path: list[float],
        config: StudyConfig,
        costs: CostModel,
        free_costs: CostModel,
    ) -> None:
        charged = run(calendar, path, 0.25, config, costs)
        free = run(calendar, path, 0.25, config, free_costs)
        assert charged.cost > 0
        assert free.cost == 0
        assert charged.short_net_pnl < free.short_net_pnl

    def test_short_profit_is_the_negated_gross_less_costs(
        self, calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, costs)
        expected = -trade.gross_pnl * costs.contract_multiplier - trade.cost
        assert trade.short_net_pnl == pytest.approx(expected)

    def test_capital_is_the_greater_of_the_two_proxies(
        self, calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, costs)
        notional = config.short_capital_spot_fraction * trade.entry_spot * costs.contract_multiplier
        premium = (
            config.short_capital_premium_multiple * trade.entry_premium * costs.contract_multiplier
        )
        assert trade.capital == pytest.approx(max(notional, premium))

    def test_return_on_capital_is_profit_over_capital(
        self, calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, costs)
        assert trade.short_return_on_capital == pytest.approx(trade.short_net_pnl / trade.capital)

    def test_index_hedging_is_cheaper_than_single_name_hedging(
        self, calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
    ) -> None:
        index = run(calendar, path, 0.25, config, costs)
        single = simulate_trade(
            "AAPL",
            calendar,
            path,
            [0.25] * len(calendar),
            ENTRY,
            HORIZON,
            dividend_yield=0.0,
            entry_vix=0.25,
            config=config,
            costs=costs,
        )
        assert single is not None
        assert index.cost < single.cost


class TestProtocol:
    def test_hedges_every_day_by_default(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, free_costs)
        # One rebalance per intervening day; the final day settles instead of rebalancing.
        assert trade.rebalances == HORIZON - 1

    def test_a_longer_interval_hedges_less_often(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, free_costs, hedge_interval_days=5)
        assert trade.rebalances == 4
        assert trade.hedge_interval_days == 5

    def test_less_frequent_hedging_costs_less_to_run(
        self, calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
    ) -> None:
        daily = run(calendar, path, 0.25, config, costs)
        weekly = run(calendar, path, 0.25, config, costs, hedge_interval_days=5)
        assert weekly.cost < daily.cost

    def test_strike_is_set_from_the_entry_forward(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, free_costs)
        assert trade.strike == pytest.approx(trade.entry_spot, rel=0.01)

    @pytest.mark.parametrize("moneyness", [0.95, 1.05])
    def test_moneyness_shifts_the_strike(
        self,
        calendar: list[date],
        path: list[float],
        config: StudyConfig,
        free_costs: CostModel,
        moneyness: float,
    ) -> None:
        trade = run(calendar, path, 0.25, config, free_costs, moneyness=moneyness)
        assert trade.strike == pytest.approx(moneyness * trade.entry_spot, rel=0.01)
        assert trade.moneyness == moneyness

    def test_holds_for_exactly_the_stated_horizon(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, free_costs)
        assert trade.entry_date == calendar[ENTRY]
        assert trade.exit_date == calendar[ENTRY + HORIZON]
        assert trade.horizon_days == HORIZON

    def test_settles_at_intrinsic_value(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, free_costs)
        intrinsic = abs(trade.exit_spot - trade.strike)
        assert trade.exit_value == pytest.approx(intrinsic)

    def test_records_the_delivered_volatility(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        trade = run(calendar, path, 0.25, config, free_costs)
        assert trade.realized_vol == pytest.approx(KNOWN_REALIZED_VOL)


class TestGuards:
    def test_returns_none_when_the_window_runs_past_the_data(
        self, calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
    ) -> None:
        assert (
            simulate_trade(
                "INDEX",
                calendar,
                path,
                [0.25] * len(calendar),
                len(calendar) - 5,
                HORIZON,
                dividend_yield=0.0,
                entry_vix=0.25,
                config=config,
                costs=costs,
            )
            is None
        )

    def test_returns_none_without_an_entry_volatility(
        self, calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
    ) -> None:
        implied: list[float | None] = [0.25] * len(calendar)
        implied[ENTRY] = None
        assert (
            simulate_trade(
                "INDEX",
                calendar,
                path,
                implied,
                ENTRY,
                HORIZON,
                dividend_yield=0.0,
                entry_vix=0.25,
                config=config,
                costs=costs,
            )
            is None
        )

    def test_a_missing_mid_trade_quote_carries_forward(
        self, calendar: list[date], path: list[float], config: StudyConfig, free_costs: CostModel
    ) -> None:
        # The position still exists on a day the index does not publish; truncating the trade would
        # quietly drop it from the sample.
        implied: list[float | None] = [0.25] * len(calendar)
        implied[ENTRY + 5] = None
        trade = simulate_trade(
            "INDEX",
            calendar,
            path,
            implied,
            ENTRY,
            HORIZON,
            dividend_yield=0.0,
            entry_vix=0.25,
            config=config,
            costs=free_costs,
        )
        assert trade is not None
        assert trade.exit_date == calendar[ENTRY + HORIZON]
        assert attribution_error(trade) < 1e-9


def test_trade_row_is_flat_and_self_consistent(
    calendar: list[date], path: list[float], config: StudyConfig, costs: CostModel
) -> None:
    trade = run(calendar, path, 0.25, config, costs)
    row = trade_row(trade)
    assert row["ticker"] == "INDEX"
    assert row["gross_pnl_pct_premium"] == pytest.approx(trade.gross_pnl / trade.entry_premium)
    error = row["attribution_error"]
    assert isinstance(error, float)
    assert error < 1e-9
    assert all(not isinstance(value, (list, dict)) for value in row.values())
