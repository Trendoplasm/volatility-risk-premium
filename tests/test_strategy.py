"""Selecting which trades to run."""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from vrp.config import CostModel, StudyConfig
from vrp.strategy import SecurityInputs, core_trades, run_protocol, variant_trades

from .conftest import alternating_price_path, security, trading_dates, volatility_series


@pytest.fixture
def single_input(config: StudyConfig) -> SecurityInputs:
    calendar = trading_dates(date(2011, 1, 3), 300)
    prices = alternating_price_path(len(calendar))
    return SecurityInputs(
        security("INDEX"),
        volatility_series(calendar),
        {"INDEX": dict(zip(calendar, prices, strict=True))},
        config,
    )


class TestRunProtocol:
    def test_entries_do_not_overlap(
        self, single_input: SecurityInputs, config: StudyConfig, costs: CostModel
    ) -> None:
        # Overlapping trades would share most of one price path, inflating the sample size without
        # adding independent information.
        trades = run_protocol(single_input, config.core_horizon_days, config, costs)
        assert len(trades) > 1
        for earlier, later in pairwise(trades):
            assert later.entry_date >= earlier.exit_date

    def test_spacing_equals_the_holding_period(
        self, single_input: SecurityInputs, config: StudyConfig, costs: CostModel
    ) -> None:
        trades = run_protocol(single_input, config.core_horizon_days, config, costs)
        for earlier, later in pairwise(trades):
            assert later.entry_date == earlier.exit_date

    def test_a_longer_horizon_produces_fewer_trades(
        self, single_input: SecurityInputs, config: StudyConfig, costs: CostModel
    ) -> None:
        short = run_protocol(single_input, 21, config, costs)
        long = run_protocol(single_input, 63, config, costs)
        assert len(long) < len(short)

    def test_passes_the_variant_settings_through(
        self, single_input: SecurityInputs, config: StudyConfig, costs: CostModel
    ) -> None:
        trades = run_protocol(
            single_input, 21, config, costs, moneyness=1.05, hedge_interval_days=5
        )
        assert trades
        assert all(t.moneyness == 1.05 and t.hedge_interval_days == 5 for t in trades)

    def test_every_trade_starts_inside_the_study_period(
        self, single_input: SecurityInputs, config: StudyConfig, costs: CostModel
    ) -> None:
        for trade in run_protocol(single_input, 21, config, costs):
            assert trade.entry_date >= config.start()


class TestCoreAndVariants:
    def test_core_covers_every_measured_security(
        self, config: StudyConfig, costs: CostModel
    ) -> None:
        calendar = trading_dates(date(2011, 1, 3), 300)
        prices = alternating_price_path(len(calendar))
        price_series = {
            t: dict(zip(calendar, prices, strict=True)) for t in ("INDEX", "AAPL", "AMZN")
        }
        universe = [security(t) for t in ("INDEX", "AAPL", "AMZN")] + [
            security("MSFT", measured=False)
        ]
        trades, inputs = core_trades(
            universe, volatility_series(calendar), price_series, config, costs
        )
        assert set(inputs) == {"INDEX", "AAPL", "AMZN"}
        assert {t.ticker for t in trades} == {"INDEX", "AAPL", "AMZN"}

    def test_variants_move_one_dimension_at_a_time(
        self, single_input: SecurityInputs, config: StudyConfig, costs: CostModel
    ) -> None:
        trades = variant_trades({"INDEX": single_input}, config, costs)
        keys = {(t.horizon_days, t.moneyness, t.hedge_interval_days) for t in trades}
        core = (config.core_horizon_days, config.core_moneyness, config.hedge_interval_days)
        assert core in keys
        for key in keys:
            differences = sum(a != b for a, b in zip(key, core, strict=True))
            assert differences <= 1

    def test_variants_include_every_horizon_and_moneyness(
        self, single_input: SecurityInputs, config: StudyConfig, costs: CostModel
    ) -> None:
        trades = variant_trades({"INDEX": single_input}, config, costs)
        assert {t.horizon_days for t in trades} >= set(config.all_horizons())
        assert {t.moneyness for t in trades} >= set(config.moneyness_grid)
