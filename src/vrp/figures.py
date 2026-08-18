"""The study's four figures.

Each answers one question: is implied variance above realised, where does the profit come from,
how does the premium depend on the volatility regime, and what does the observed term structure
look like.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib

# A non-interactive backend keeps the study runnable headless, in CI and over SSH.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from vrp.config import MEASURED_TICKERS
from vrp.models import Row

FIGURE_DPI = 180


def _finish(
    figure: Figure,
    axis: Axes,
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    xlabel: str | None = None,
    grid_axis: Literal["both", "x", "y"] = "both",
    legend: bool = True,
) -> None:
    """Apply the shared styling and write the file."""
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if legend:
        axis.legend()
    axis.grid(axis=grid_axis, alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_implied_vs_realized(panel: Sequence[Row], horizon: int, output_path: Path) -> None:
    """Scatter implied against subsequently realised volatility, one point per observation.

    Points above the diagonal are observations where the option market implied more variance than
    the underlying delivered. That the cloud sits above the line *is* the volatility risk premium.
    """
    figure, axis = plt.subplots(figsize=(8, 8))
    for ticker, marker in zip(MEASURED_TICKERS, ("o", "s", "^"), strict=False):
        rows = [r for r in panel if r["ticker"] == ticker and r["horizon_days"] == horizon]
        if not rows:
            continue
        axis.scatter(
            [row["realized_vol"] for row in rows],
            [row["matched_iv"] for row in rows],
            s=4,
            alpha=0.25,
            marker=marker,
            label=ticker,
        )
    limit = max(
        [row["matched_iv"] for row in panel if row["horizon_days"] == horizon]
        + [row["realized_vol"] for row in panel if row["horizon_days"] == horizon]
    )
    axis.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
        linewidth=1,
        color="black",
        label="implied = realised",
    )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    _finish(
        figure,
        axis,
        output_path,
        title=f"Implied Versus Subsequently Realised Volatility ({horizon} trading days)",
        xlabel="Realised volatility over the following window",
        ylabel="Matched implied volatility at observation",
    )


def plot_attribution(attribution: Sequence[Row], output_path: Path) -> None:
    """Stack the Greek components of average profit, per security.

    Theta is the source of the carry; gamma is what gives most of it back.
    """
    groups = [row["group"] for row in attribution]
    positions = np.arange(len(groups))
    components = [
        ("theta_pct_premium", "Theta (time decay)"),
        ("gamma_pct_premium", "Gamma (realised moves)"),
        ("vega_pct_premium", "Vega (IV repricing)"),
        ("residual_pct_premium", "Residual (higher order)"),
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    positive_base = np.zeros(len(groups))
    negative_base = np.zeros(len(groups))
    for key, label in components:
        values = np.array([row[key] for row in attribution], dtype=float)
        base = np.where(values >= 0, positive_base, negative_base)
        axis.bar(positions, values, 0.55, bottom=base, label=label)
        positive_base = positive_base + np.where(values >= 0, values, 0.0)
        negative_base = negative_base + np.where(values < 0, values, 0.0)
    totals = np.array([row["total_pct_premium"] for row in attribution], dtype=float)
    axis.plot(positions, totals, "kD", markersize=7, label="Net (long straddle)")
    axis.axhline(0, linewidth=0.8, color="black")
    axis.set_xticks(positions, groups)
    _finish(
        figure,
        axis,
        output_path,
        title="Where Delta-Hedged Profit Comes From",
        ylabel="Share of entry premium",
        grid_axis="y",
    )


def plot_regime_premium(regimes: Sequence[Row], output_path: Path) -> None:
    """Show the seller's average return and tail loss by volatility regime."""
    labels = [row["group"] for row in regimes]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(
        positions - width / 2,
        [100 * row["mean_return_on_capital"] for row in regimes],
        width,
        label="Mean return on capital",
    )
    axis.bar(
        positions + width / 2,
        [100 * row["expected_shortfall_roc"] for row in regimes],
        width,
        label="Worst 5% average (expected shortfall)",
    )
    axis.axhline(0, linewidth=0.8, color="black")
    axis.set_xticks(positions, labels)
    _finish(
        figure,
        axis,
        output_path,
        title="Short-Volatility Outcome by Entry Volatility Regime",
        ylabel="Per-trade return on research capital (%)",
        grid_axis="y",
    )


def plot_term_structure(term_rows: Sequence[Row], output_path: Path) -> None:
    """Plot the average observed S&P 500 volatility term structure."""
    figure, axis = plt.subplots(figsize=(8, 5))
    horizons = [row["horizon_days"] for row in term_rows]
    axis.plot(
        horizons,
        [100 * row["mean_iv"] for row in term_rows],
        marker="o",
        label="Mean observed level",
    )
    axis.fill_between(
        horizons,
        [100 * row["p10_iv"] for row in term_rows],
        [100 * row["p90_iv"] for row in term_rows],
        alpha=0.2,
        label="10th to 90th percentile",
    )
    axis.set_xscale("log")
    axis.set_xticks(horizons, [str(int(h)) for h in horizons])
    _finish(
        figure,
        axis,
        output_path,
        title="Observed S&P 500 Volatility Term Structure",
        xlabel="Horizon (calendar days, log scale)",
        ylabel="Implied volatility (%)",
    )
