"""Summaries, tail statistics, and the resampled confidence intervals."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from vrp.aggregate import (
    attribution_table,
    bootstrap_interval,
    expected_shortfall,
    group_summaries,
    regime_table,
    summarize_trades,
    tail_tables,
    variant_table,
)
from vrp.config import CostModel, StudyConfig
from vrp.hedged import Trade


def trade(
    ticker: str = "INDEX",
    *,
    roc: float = 0.04,
    entry_vix: float = 0.15,
    horizon: int = 21,
    moneyness: float = 1.00,
    hedge_interval: int = 1,
    theta: float = -3.0,
    gamma: float = 2.0,
    vega: float = 0.2,
    residual: float = -0.1,
) -> Trade:
    """Build a trade with directly specified outcomes."""
    gross = theta + gamma + vega + residual
    return Trade(
        ticker=ticker,
        horizon_days=horizon,
        moneyness=moneyness,
        hedge_interval_days=hedge_interval,
        entry_date=date(2012, 1, 3),
        exit_date=date(2012, 2, 3),
        entry_spot=100.0,
        exit_spot=101.0,
        strike=100.0,
        entry_iv=0.25,
        realized_vol=0.20,
        entry_premium=5.0,
        exit_value=1.0,
        gross_pnl=gross,
        theta_pnl=theta,
        gamma_pnl=gamma,
        vega_pnl=vega,
        residual_pnl=residual,
        hedge_pnl=0.5,
        financing_pnl=-0.01,
        cost=2.0,
        contracts=1.0,
        capital=2000.0,
        short_net_pnl=roc * 2000.0,
        short_return_on_capital=roc,
        rebalances=20,
        entry_vix=entry_vix,
        in_sample=True,
    )


class TestExpectedShortfall:
    def test_averages_the_worst_tail(self) -> None:
        values = list(range(100))
        # The worst 5% of 0..99 is 0..4, whose mean is 2.
        assert expected_shortfall(values, 0.05) == pytest.approx(2.0)

    def test_always_includes_at_least_one_observation(self) -> None:
        assert expected_shortfall([5.0, 9.0], 0.001) == pytest.approx(5.0)

    def test_is_no_greater_than_the_mean(self) -> None:
        values = [0.05, -0.30, 0.04, 0.06, -0.10]
        shortfall = expected_shortfall(values)
        assert shortfall is not None and shortfall <= float(np.mean(values))

    def test_empty_sample_reports_missing(self) -> None:
        assert expected_shortfall([]) is None


class TestSummarizeTrades:
    def test_reports_return_risk_and_tail(self) -> None:
        trades = [trade(roc=r) for r in (0.05, -0.20, 0.04, 0.06)]
        summary = summarize_trades(trades, "Pooled")
        assert summary["n"] == 4
        assert summary["mean_return_on_capital"] == pytest.approx(-0.0125)
        assert summary["win_rate"] == pytest.approx(0.75)
        assert summary["worst_trade_roc"] == pytest.approx(-0.20)
        assert summary["best_trade_roc"] == pytest.approx(0.06)

    def test_annualises_from_the_holding_period(self) -> None:
        # Twelve non-overlapping 21-day periods make a year.
        summary = summarize_trades([trade(roc=0.01, horizon=21)], "Pooled")
        assert summary["annualized_mean_roc"] == pytest.approx(0.01 * 252 / 21)

    def test_a_single_trade_has_no_dispersion(self) -> None:
        summary = summarize_trades([trade()], "Pooled")
        assert summary["sd_return_on_capital"] is None
        assert summary["sharpe_like_ratio"] is None

    def test_empty_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty trade group"):
            summarize_trades([], "Pooled")


class TestAttributionTable:
    def test_components_sum_to_the_total(self) -> None:
        rows = attribution_table([trade(), trade(ticker="AAPL")])
        assert rows
        for row in rows:
            assert row["check"] == pytest.approx(0.0, abs=1e-12)

    def test_scales_components_by_entry_premium(self) -> None:
        row = attribution_table([trade(theta=-3.0)])[0]
        assert row["theta_pct_premium"] == pytest.approx(-3.0 / 5.0)

    def test_covers_the_pooled_sample_and_each_security(self) -> None:
        rows = attribution_table([trade("INDEX"), trade("AAPL"), trade("AMZN")])
        assert [row["group"] for row in rows] == ["Pooled", "INDEX", "AAPL", "AMZN"]


class TestVariantTable:
    def test_keeps_only_single_dimension_variants(self) -> None:
        config = StudyConfig()
        trades = [
            trade(),
            trade(horizon=63),
            trade(moneyness=0.95),
            trade(hedge_interval=5),
            trade(horizon=63, moneyness=0.95),  # two dimensions moved at once
        ]
        rows = variant_table(trades, config)
        keys = {(r["horizon_days"], r["moneyness"], r["hedge_interval_days"]) for r in rows}
        assert (63, 0.95, 1) not in keys
        assert len(rows) == 4

    def test_marks_the_core_protocol_and_lists_it_first(self) -> None:
        config = StudyConfig()
        rows = variant_table([trade(), trade(horizon=63)], config)
        assert rows[0]["is_core_protocol"] is True
        assert sum(row["is_core_protocol"] for row in rows) == 1


class TestRegimeTable:
    def test_splits_into_three_populated_bands(self) -> None:
        trades = [trade(entry_vix=0.10 + 0.01 * index) for index in range(30)]
        rows = regime_table(trades)
        assert [row["group"] for row in rows] == ["Low VIX", "Middle VIX", "High VIX"]
        assert sum(row["n"] for row in rows) == len(trades)

    def test_bands_are_ordered_by_volatility(self) -> None:
        trades = [trade(entry_vix=0.10 + 0.01 * index) for index in range(30)]
        means = [row["mean_entry_vix"] for row in regime_table(trades)]
        assert means == sorted(means)


class TestTailTables:
    def test_lists_the_worst_and_best_individual_trades(self) -> None:
        trades = [trade(roc=r) for r in (-0.5, 0.1, 0.2, -0.3, 0.4)]
        worst, best = tail_tables(trades)
        assert worst[0]["short_return_on_capital"] == pytest.approx(-0.5)
        assert best[0]["short_return_on_capital"] == pytest.approx(0.4)

    def test_orders_each_table_from_most_extreme(self) -> None:
        trades = [trade(roc=r) for r in (-0.5, 0.1, 0.2, -0.3, 0.4)]
        worst, best = tail_tables(trades)
        assert [r["short_return_on_capital"] for r in worst] == sorted(
            r["short_return_on_capital"] for r in worst
        )
        assert [r["short_return_on_capital"] for r in best] == sorted(
            (r["short_return_on_capital"] for r in best), reverse=True
        )


class TestBootstrapInterval:
    def test_brackets_the_sample_mean(self) -> None:
        values = list(np.random.default_rng(0).normal(0.04, 0.10, 200))
        low, high, mean = bootstrap_interval(values, np.random.default_rng(1), 500)
        assert low < mean < high
        assert mean == pytest.approx(float(np.mean(values)), abs=0.01)

    def test_is_reproducible_under_a_fixed_seed(self) -> None:
        values = [0.05, -0.2, 0.03, 0.07, 0.01]
        first = bootstrap_interval(values, np.random.default_rng(7), 300)
        second = bootstrap_interval(values, np.random.default_rng(7), 300)
        assert first == second

    def test_a_constant_sample_has_a_degenerate_interval(self) -> None:
        low, high, _ = bootstrap_interval([0.04] * 20, np.random.default_rng(3), 200)
        assert low == pytest.approx(0.04)
        assert high == pytest.approx(0.04)


def test_group_summaries_cover_pooled_and_each_security() -> None:
    trades = [trade("INDEX"), trade("AAPL"), trade("AMZN")]
    assert [row["group"] for row in group_summaries(trades)] == ["Pooled", "INDEX", "AAPL", "AMZN"]


def test_cost_share_uses_the_contract_multiplier() -> None:
    costs = CostModel()
    summary = summarize_trades([trade()], "Pooled")
    assert summary["mean_cost_share_of_premium"] == pytest.approx(
        2.0 / (5.0 * costs.contract_multiplier)
    )
