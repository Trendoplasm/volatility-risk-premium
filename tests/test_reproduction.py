"""End-to-end reproduction of the published results.

This is the test that guards the numbers. It runs the complete study against the downloaded
histories and compares every exported table with the committed results.

It skips itself when the inputs are absent, which is the case on a fresh clone and in continuous
integration, because neither Cboe's volatility history nor the price history is redistributed here.
Populate them with ``python scripts/fetch_cboe_data.py`` and ``python scripts/fetch_price_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vrp.config import CostModel, StudyConfig
from vrp.hedged import attribution_error
from vrp.pipeline import StudyResults, headline, run_study, write_outputs
from vrp.verify import compare_output_dirs

EXPECTED_HEADLINE = (
    "Completed 555 delta-hedged trades over 11602 matched observations: mean variance risk "
    "premium 3.84 volatility points (positive 78.1%), mean short return on capital 4.06% per "
    "trade, win rate 74.2%."
)

EXPECTED_PANEL_OBSERVATIONS = 34617
EXPECTED_CORE_TRADES = 555
EXPECTED_MEASURED = ["INDEX", "AAPL", "AMZN"]
EXPECTED_TABLE_COUNT = 15

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "raw"
    universe = root / "data" / "reference" / "study_universe.csv"
    if not data_dir.is_dir() or not universe.exists():
        pytest.skip("downloaded inputs absent; run the fetch scripts to enable")
    return data_dir, universe


@pytest.fixture(scope="module")
def results(paths: tuple[Path, Path]) -> StudyResults:
    data_dir, universe = paths
    return run_study(data_dir, universe, StudyConfig(), CostModel())


def test_headline_result_is_unchanged(results: StudyResults) -> None:
    assert headline(results) == EXPECTED_HEADLINE


def test_sample_sizes(results: StudyResults) -> None:
    assert len(results.panel) == EXPECTED_PANEL_OBSERVATIONS
    assert len(results.trades) == EXPECTED_CORE_TRADES


def test_only_securities_with_observed_inputs_are_measured(results: StudyResults) -> None:
    measured = [s.ticker for s in results.universe if s.measured]
    assert measured == EXPECTED_MEASURED
    assert len(results.universe) == 21
    assert {trade.ticker for trade in results.trades} == set(EXPECTED_MEASURED)


def test_implied_variance_exceeds_realised_at_every_horizon(results: StudyResults) -> None:
    # The study's central empirical claim.
    pooled = [row for row in results.panel_summaries if row["group"] == "Pooled"]
    assert len(pooled) == 3
    for row in pooled:
        assert row["mean_vrp_vol_points"] > 0
        assert row["pct_positive_vrp"] > 0.7


def test_index_premium_matches_the_published_literature(results: StudyResults) -> None:
    # The S&P 500 one-month premium is a widely replicated figure of roughly 3 to 4 volatility
    # points. Landing in that band is the study's external sanity check.
    index = next(
        row
        for row in results.panel_summaries
        if row["group"] == "INDEX" and row["horizon_days"] == 21
    )
    assert 0.02 < index["mean_vrp_vol_points"] < 0.05


def test_the_seller_earns_and_the_buyer_pays(results: StudyResults) -> None:
    pooled = results.pooled_strategy
    assert pooled["mean_return_on_capital"] > 0
    assert pooled["win_rate"] > 0.5
    # And the tail is genuinely worse than the average: this is not a free lunch.
    assert pooled["expected_shortfall_roc"] < pooled["mean_return_on_capital"]
    assert pooled["worst_trade_roc"] < 0


def test_profit_comes_from_theta_and_is_partly_returned_by_gamma(results: StudyResults) -> None:
    pooled = results.attribution[0]
    assert pooled["theta_pct_premium"] < 0
    assert pooled["gamma_pct_premium"] > 0
    assert abs(pooled["gamma_pct_premium"]) < abs(pooled["theta_pct_premium"])
    assert pooled["check"] == pytest.approx(0.0, abs=1e-9)


def test_every_trade_satisfies_the_attribution_identity(results: StudyResults) -> None:
    assert max(attribution_error(trade) for trade in results.trades) < 1e-9


def test_bootstrap_interval_excludes_zero(results: StudyResults) -> None:
    low, high, _ = results.bootstrap["vrp_vol_points"]
    assert 0 < low < high


def test_selected_threshold_is_reported_for_both_periods(results: StudyResults) -> None:
    periods = {row["period"] for row in results.signal_selection}
    assert periods == {"In sample (to 2018)", "Out of sample (2019+)"}
    # The scan must report the test period without having used it to choose.
    assert all(row["out_of_sample_n"] is not None for row in results.threshold_scan)


def test_the_frozen_end_date_bounds_the_sample(results: StudyResults) -> None:
    # Every downloaded series extends past this date; the frozen end is what makes the study
    # reproduce identically from a download taken at any later time.
    cutoff = results.config.end()
    assert max(row["date"] for row in results.panel) <= cutoff
    assert max(trade.exit_date for trade in results.trades) <= cutoff


def test_all_tables_match_the_committed_results(results: StudyResults, tmp_path: Path) -> None:
    committed = Path(__file__).resolve().parent.parent / "outputs"
    if not (committed / "tables").is_dir():
        pytest.skip("no committed outputs to compare against")

    write_outputs(results, tmp_path, with_plots=False)
    comparison = compare_output_dirs(committed, tmp_path)
    assert comparison.matches, "\n".join(comparison.discrepancies[:20])
    assert comparison.compared_files == EXPECTED_TABLE_COUNT
