"""Summaries of the simulated trades: overall, by variant, by regime, and in the tails.

A mean is not enough to describe a short-volatility strategy. The distribution is deliberately
asymmetric -- many small gains against occasional large losses -- so every summary here reports a
tail statistic alongside the average, and the tail tables list the individual worst outcomes rather
than only their average.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from vrp.config import (
    BOOTSTRAP_LOWER_QUANTILE,
    BOOTSTRAP_UPPER_QUANTILE,
    EXTREME_TRADE_COUNT,
    GROUP_ORDER,
    TRADING_DAYS_PER_YEAR,
    StudyConfig,
)
from vrp.hedged import Trade, trade_row
from vrp.models import Row, Table
from vrp.stats_utils import Samples, mean_or_none, std_dev_or_none

logger = logging.getLogger(__name__)

#: Fraction of the worst outcomes averaged to form the expected-shortfall statistic.
TAIL_FRACTION = 0.05


def expected_shortfall(values: Samples, fraction: float = TAIL_FRACTION) -> float | None:
    """Return the mean of the worst ``fraction`` of outcomes.

    This is the number that matters for a strategy whose losses cluster in a short list of days.

    Args:
        values: Outcomes, higher being better.
        fraction: Share of the distribution treated as the tail.

    Returns:
        The tail mean, or None for an empty sample.
    """
    if len(values) == 0:
        return None
    ordered = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil(fraction * ordered.size)))
    return float(np.mean(ordered[:count]))


def summarize_trades(trades: Sequence[Trade], label: str) -> Row:
    """Summarise a group of trades.

    Args:
        trades: Trades in the group.
        label: Value written to the ``group`` column.

    Returns:
        Return, risk, attribution, and tail statistics for the short side of the trade, which is
        the side that earns the premium.

    Raises:
        ValueError: If the group is empty.
    """
    if not trades:
        raise ValueError(f"Cannot summarise an empty trade group: {label}")

    net = np.array([trade.short_net_pnl for trade in trades], dtype=float)
    roc = np.array([trade.short_return_on_capital for trade in trades], dtype=float)
    premium = np.array([trade.entry_premium for trade in trades], dtype=float)
    horizon = float(np.mean([trade.horizon_days for trade in trades]))
    periods_per_year = TRADING_DAYS_PER_YEAR / horizon if horizon else float("nan")

    return {
        "group": label,
        "n": len(trades),
        "first_entry": min(trade.entry_date for trade in trades),
        "last_entry": max(trade.exit_date for trade in trades),
        "mean_entry_iv": mean_or_none([trade.entry_iv for trade in trades]),
        "mean_realized_vol": mean_or_none([trade.realized_vol for trade in trades]),
        "mean_iv_minus_realized": mean_or_none(
            [trade.entry_iv - trade.realized_vol for trade in trades]
        ),
        "mean_short_net_pnl": float(np.mean(net)),
        "median_short_net_pnl": float(np.median(net)),
        "mean_return_on_capital": float(np.mean(roc)),
        "median_return_on_capital": float(np.median(roc)),
        "sd_return_on_capital": std_dev_or_none(roc),
        "win_rate": float(np.mean(net > 0)),
        # Annualised from non-overlapping periods, so the scaling is honest about frequency.
        "annualized_mean_roc": float(np.mean(roc) * periods_per_year),
        "sharpe_like_ratio": (
            float(np.mean(roc) / np.std(roc, ddof=1) * np.sqrt(periods_per_year))
            if len(roc) > 1 and np.std(roc, ddof=1) > 0
            else None
        ),
        "worst_trade_roc": float(np.min(roc)),
        "best_trade_roc": float(np.max(roc)),
        "expected_shortfall_roc": expected_shortfall(roc),
        "mean_cost_share_of_premium": float(
            np.mean([trade.cost for trade in trades] / (premium * 100.0))
        ),
        "mean_theta_pct_premium": mean_or_none(
            [trade.theta_pnl / trade.entry_premium for trade in trades]
        ),
        "mean_gamma_pct_premium": mean_or_none(
            [trade.gamma_pnl / trade.entry_premium for trade in trades]
        ),
        "mean_vega_pct_premium": mean_or_none(
            [trade.vega_pnl / trade.entry_premium for trade in trades]
        ),
        "mean_residual_pct_premium": mean_or_none(
            [trade.residual_pnl / trade.entry_premium for trade in trades]
        ),
    }


def group_summaries(trades: Sequence[Trade]) -> Table:
    """Summarise the pooled sample and each measured security."""
    summaries: Table = []
    for label in GROUP_ORDER:
        subset = list(trades) if label == "Pooled" else [t for t in trades if t.ticker == label]
        if subset:
            summaries.append(summarize_trades(subset, label))
    return summaries


def attribution_table(trades: Sequence[Trade]) -> Table:
    """Decompose average profit into its Greek components, per security.

    Every row's components sum to its total by construction; the ``check`` column reports the
    floating-point residual of that identity so the reader can confirm it rather than trust it.
    """
    rows: Table = []
    for label in GROUP_ORDER:
        subset = list(trades) if label == "Pooled" else [t for t in trades if t.ticker == label]
        if not subset:
            continue
        scale = np.array([t.entry_premium for t in subset], dtype=float)

        def component(values: Sequence[float], premium: np.ndarray = scale) -> float:
            """Average one P&L component as a share of the premium paid for it."""
            return float(np.mean(np.asarray(values, dtype=float) / premium))

        theta = component([t.theta_pnl for t in subset])
        gamma = component([t.gamma_pnl for t in subset])
        vega = component([t.vega_pnl for t in subset])
        residual = component([t.residual_pnl for t in subset])
        total = component([t.gross_pnl for t in subset])
        rows.append(
            {
                "group": label,
                "n": len(subset),
                "theta_pct_premium": theta,
                "gamma_pct_premium": gamma,
                "vega_pct_premium": vega,
                "residual_pct_premium": residual,
                "total_pct_premium": total,
                "check": theta + gamma + vega + residual - total,
                "days_theta_positive_for_seller": float(np.mean([t.theta_pnl < 0 for t in subset])),
            }
        )
    return rows


def variant_table(trades: Sequence[Trade], config: StudyConfig) -> Table:
    """Compare protocol variants: maturity, moneyness, and hedge frequency.

    Each row holds every dimension fixed at its core value except one, so a difference between
    rows is attributable to the dimension that moved.
    """
    rows: Table = []
    seen: set[tuple[int, float, int]] = set()
    for trade in trades:
        key = (trade.horizon_days, trade.moneyness, trade.hedge_interval_days)
        if key in seen:
            continue
        varied = sum(
            (
                key[0] != config.core_horizon_days,
                key[1] != config.core_moneyness,
                key[2] != config.hedge_interval_days,
            )
        )
        if varied > 1:
            continue
        seen.add(key)
        subset = [t for t in trades if (t.horizon_days, t.moneyness, t.hedge_interval_days) == key]
        summary = summarize_trades(subset, "Pooled")
        rows.append(
            {
                "horizon_days": key[0],
                "moneyness": key[1],
                "hedge_interval_days": key[2],
                "is_core_protocol": varied == 0,
                "n": summary["n"],
                "mean_return_on_capital": summary["mean_return_on_capital"],
                "win_rate": summary["win_rate"],
                "sd_return_on_capital": summary["sd_return_on_capital"],
                "worst_trade_roc": summary["worst_trade_roc"],
                "expected_shortfall_roc": summary["expected_shortfall_roc"],
                "mean_cost_share_of_premium": summary["mean_cost_share_of_premium"],
                "mean_theta_pct_premium": summary["mean_theta_pct_premium"],
                "mean_gamma_pct_premium": summary["mean_gamma_pct_premium"],
            }
        )
    rows.sort(key=lambda row: (not row["is_core_protocol"], row["horizon_days"], row["moneyness"]))
    return rows


def regime_table(trades: Sequence[Trade]) -> Table:
    """Split trades by the level of market-wide implied volatility at entry.

    The premium is not a constant. Selling volatility when volatility is already high is a
    different trade from selling it when volatility is low, and the split is where that shows.
    """
    entry_vix = np.array([trade.entry_vix for trade in trades], dtype=float)
    lower, upper = np.quantile(entry_vix, [1 / 3, 2 / 3])
    bands = (
        ("Low VIX", -np.inf, lower),
        ("Middle VIX", lower, upper),
        ("High VIX", upper, np.inf),
    )
    rows: Table = []
    for label, low, high in bands:
        subset = [
            t
            for t in trades
            if low < t.entry_vix <= high or (low == -np.inf and t.entry_vix <= high)
        ]
        if not subset:
            logger.warning("Regime %s is empty", label)
            continue
        summary = summarize_trades(subset, label)
        summary["vix_lower_bound"] = None if low == -np.inf else float(low)
        summary["vix_upper_bound"] = None if high == np.inf else float(high)
        summary["mean_entry_vix"] = float(np.mean([t.entry_vix for t in subset]))
        rows.append(summary)
    return rows


def tail_tables(trades: Sequence[Trade]) -> tuple[Table, Table]:
    """Return the worst and best individual trades for the short seller."""
    by_outcome = sorted(trades, key=lambda trade: trade.short_return_on_capital)
    worst = [trade_row(trade) for trade in by_outcome[:EXTREME_TRADE_COUNT]]
    best = [trade_row(trade) for trade in reversed(by_outcome[-EXTREME_TRADE_COUNT:])]
    return worst, best


def bootstrap_interval(
    values: Sequence[float], rng: np.random.Generator, iterations: int
) -> tuple[float, float, float]:
    """Bootstrap a mean and its 95% interval.

    Args:
        values: Outcomes to resample.
        rng: Seeded generator, so a run is reproducible.
        iterations: Number of resamples.

    Returns:
        The lower bound, upper bound, and bootstrap mean.
    """
    sample = np.asarray(values, dtype=float)
    draws = rng.integers(0, sample.size, size=(iterations, sample.size))
    means = sample[draws].mean(axis=1)
    return (
        float(np.quantile(means, BOOTSTRAP_LOWER_QUANTILE)),
        float(np.quantile(means, BOOTSTRAP_UPPER_QUANTILE)),
        float(np.mean(means)),
    )
