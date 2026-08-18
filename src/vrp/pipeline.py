"""End-to-end orchestration: load, measure, simulate, summarise, export."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vrp.aggregate import (
    attribution_table,
    bootstrap_interval,
    group_summaries,
    regime_table,
    tail_tables,
    variant_table,
)
from vrp.config import INDEX_TERM_STRUCTURE, CostModel, StudyConfig
from vrp.figures import (
    plot_attribution,
    plot_implied_vs_realized,
    plot_regime_premium,
    plot_term_structure,
)
from vrp.hedged import Trade, trade_row
from vrp.loaders import load_price_series, load_universe, load_volatility_series
from vrp.models import LevelByDate, Row, Security, Table
from vrp.panel import build_panel, panel_summary
from vrp.signals import build_signal_rows, selection_summary, signal_regression, threshold_scan
from vrp.strategy import core_trades, variant_trades
from vrp.writers import write_csv, write_json

logger = logging.getLogger(__name__)

#: Restated in every export so no downstream reader can mistake the study's scope.
SCOPE_NOTE = (
    "Implied volatility is observed Cboe index history and returns are observed closing prices, "
    "for three securities only: the S&P 500, Apple, and Amazon. Option prices are modelled from "
    "those observed inputs with Black-Scholes rather than taken from historical quotes, so "
    "reported profits exclude the bid-ask spread of a real option chain beyond the modelled entry "
    "half-spread. The capital denominator is a research normalisation, not broker margin. The "
    "remaining eighteen securities in the universe have no free implied-volatility history and "
    "are therefore not measured."
)


@dataclass(frozen=True)
class StudyResults:
    """Everything one run produces."""

    config: StudyConfig
    costs: CostModel
    universe: list[Security] = field(repr=False)
    panel: Table = field(repr=False)
    panel_summaries: Table = field(repr=False)
    term_structure: Table = field(repr=False)
    trades: list[Trade] = field(repr=False)
    variants: list[Trade] = field(repr=False)
    strategy_summaries: Table = field(repr=False)
    attribution: Table = field(repr=False)
    variant_comparison: Table = field(repr=False)
    regimes: Table = field(repr=False)
    worst_trades: Table = field(repr=False)
    best_trades: Table = field(repr=False)
    signals: Table = field(repr=False)
    threshold_scan: Table = field(repr=False)
    signal_selection: Table = field(repr=False)
    signal_regression: Table = field(repr=False)
    signal_regression_meta: Row = field(repr=False)
    bootstrap: dict[str, tuple[float, float, float]] = field(repr=False)

    @property
    def pooled_strategy(self) -> Row:
        """Return the pooled strategy summary."""
        return self.strategy_summaries[0]

    @property
    def pooled_panel(self) -> Row:
        """Return the pooled core-horizon panel summary."""
        return next(
            row
            for row in self.panel_summaries
            if row["group"] == "Pooled" and row["horizon_days"] == self.config.core_horizon_days
        )


def _term_structure_table(volatility_series: dict[str, LevelByDate], config: StudyConfig) -> Table:
    """Summarise the observed S&P 500 volatility curve over the study period."""
    start, end = config.start(), config.end()
    rows: Table = []
    for name, horizon in INDEX_TERM_STRUCTURE.items():
        levels = np.array(
            [level for day, level in volatility_series[name].items() if start <= day <= end],
            dtype=float,
        )
        rows.append(
            {
                "index": name,
                "horizon_days": horizon,
                "n": int(levels.size),
                "mean_iv": float(np.mean(levels)),
                "median_iv": float(np.median(levels)),
                "p10_iv": float(np.quantile(levels, 0.10)),
                "p90_iv": float(np.quantile(levels, 0.90)),
                "min_iv": float(np.min(levels)),
                "max_iv": float(np.max(levels)),
            }
        )
    rows.sort(key=lambda row: row["horizon_days"])
    return rows


def run_study(
    data_dir: Path, universe_path: Path, config: StudyConfig, costs: CostModel
) -> StudyResults:
    """Load the inputs and run the complete study.

    Args:
        data_dir: Directory holding the downloaded volatility and price histories.
        universe_path: Path to the universe reference CSV.
        config: Study windows, horizons, and inference settings.
        costs: Transaction-cost model.

    Returns:
        Every measurement, simulation, and summary the study produces.
    """
    volatility_series = load_volatility_series(data_dir)
    price_series = load_price_series(data_dir)
    universe = load_universe(universe_path)

    panel = build_panel(universe, volatility_series, price_series, config)
    panel_summaries: Table = []
    for horizon in config.all_horizons():
        at_horizon = [row for row in panel if row["horizon_days"] == horizon]
        if at_horizon:
            panel_summaries.append(panel_summary(at_horizon, "Pooled", horizon))
        for ticker in sorted({row["ticker"] for row in at_horizon}):
            rows = [row for row in at_horizon if row["ticker"] == ticker]
            panel_summaries.append(panel_summary(rows, ticker, horizon))

    trades, inputs = core_trades(universe, volatility_series, price_series, config, costs)
    variants = variant_trades(inputs, config, costs)

    signals = build_signal_rows(inputs, trades, config)
    scan = threshold_scan(signals)
    regression_rows, regression_meta = signal_regression(signals)

    rng = np.random.default_rng(config.random_seed)
    bootstrap = {
        "vrp_vol_points": bootstrap_interval(
            [
                row["vrp_vol_points"]
                for row in panel
                if row["horizon_days"] == config.core_horizon_days
            ],
            rng,
            config.bootstrap_iterations,
        ),
        "short_return_on_capital": bootstrap_interval(
            [trade.short_return_on_capital for trade in trades], rng, config.bootstrap_iterations
        ),
    }

    worst, best = tail_tables(trades)
    return StudyResults(
        config=config,
        costs=costs,
        universe=universe,
        panel=panel,
        panel_summaries=panel_summaries,
        term_structure=_term_structure_table(volatility_series, config),
        trades=trades,
        variants=variants,
        strategy_summaries=group_summaries(trades),
        attribution=attribution_table(trades),
        variant_comparison=variant_table(variants, config),
        regimes=regime_table(trades),
        worst_trades=worst,
        best_trades=best,
        signals=signals,
        threshold_scan=scan,
        signal_selection=selection_summary(signals, scan),
        signal_regression=regression_rows,
        signal_regression_meta=regression_meta,
        bootstrap=bootstrap,
    )


def summary_payload(results: StudyResults) -> dict[str, Any]:
    """Assemble the machine-readable run summary."""
    return {
        "configuration": asdict(results.config),
        "cost_model": asdict(results.costs),
        "securities_measured": [s.ticker for s in results.universe if s.measured],
        "securities_not_measured": [s.ticker for s in results.universe if not s.measured],
        "panel_observations": len(results.panel),
        "core_trades": len(results.trades),
        "variant_trades": len(results.variants),
        "panel_summaries": results.panel_summaries,
        "term_structure": results.term_structure,
        "strategy_summaries": results.strategy_summaries,
        "attribution": results.attribution,
        "regimes": results.regimes,
        "signal_selection": results.signal_selection,
        "signal_regression": results.signal_regression,
        "signal_regression_meta": results.signal_regression_meta,
        "bootstrap": results.bootstrap,
        "scope_note": SCOPE_NOTE,
    }


def write_outputs(results: StudyResults, output_dir: Path, *, with_plots: bool = True) -> None:
    """Write every table, figure, and summary of a completed run."""
    tables_dir = output_dir / "tables"
    tables: dict[str, Table] = {
        "variance_panel": results.panel,
        "variance_panel_summary": results.panel_summaries,
        "observed_term_structure": results.term_structure,
        "hedged_trades": [trade_row(trade) for trade in results.trades],
        "strategy_summary": results.strategy_summaries,
        "pnl_attribution": results.attribution,
        "protocol_variants": results.variant_comparison,
        "volatility_regimes": results.regimes,
        "worst_trades": results.worst_trades,
        "best_trades": results.best_trades,
        "signal_observations": results.signals,
        "signal_threshold_scan": results.threshold_scan,
        "signal_selection": results.signal_selection,
        "signal_regression": results.signal_regression,
        "study_universe": [
            {
                "ticker": s.ticker,
                "name": s.name,
                "security_type": s.security_type,
                "group": s.group,
                "size_bucket": s.size_bucket,
                "dividend_yield": s.dividend_yield,
                "iv_evidence": s.iv_evidence,
                "return_evidence": s.return_evidence,
                "measured": s.measured,
            }
            for s in results.universe
        ],
    }
    for name, rows in tables.items():
        write_csv(tables_dir / f"{name}.csv", rows)

    if with_plots:
        plots_dir = output_dir / "plots"
        plot_implied_vs_realized(
            results.panel, results.config.core_horizon_days, plots_dir / "implied_vs_realized.png"
        )
        plot_attribution(results.attribution, plots_dir / "pnl_attribution.png")
        plot_regime_premium(results.regimes, plots_dir / "regime_premium.png")
        plot_term_structure(results.term_structure, plots_dir / "term_structure.png")

    write_json(output_dir / "summary.json", summary_payload(results))
    logger.info("Wrote %d tables to %s", len(tables), output_dir)


def headline(results: StudyResults) -> str:
    """Render the one-line result used to confirm a successful reproduction."""
    panel = results.pooled_panel
    strategy = results.pooled_strategy
    return (
        f"Completed {len(results.trades)} delta-hedged trades over "
        f"{panel['n']} matched observations: mean variance risk premium "
        f"{panel['mean_vrp_vol_points'] * 100:.2f} volatility points "
        f"(positive {panel['pct_positive_vrp']:.1%}), mean short return on capital "
        f"{strategy['mean_return_on_capital']:.2%} per trade, win rate "
        f"{strategy['win_rate']:.1%}."
    )
