"""Conditioning signals, and an honest test of whether they help.

Knowing that the premium exists on average is not the same as knowing when to collect it. This
module builds three signals from information available *before* entry, selects a threshold using
only the in-sample period, and then reports what that threshold went on to do out of sample.

The separation is the whole point. A threshold chosen on the full history will always look good,
because it was chosen to. Reporting the in-sample and out-of-sample columns side by side is what
makes the difference visible instead of flattering.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import statsmodels.api as sm

from vrp.config import StudyConfig
from vrp.hedged import Trade
from vrp.models import Row, Table
from vrp.realized import trailing_realized_volatility
from vrp.strategy import SecurityInputs

logger = logging.getLogger(__name__)

#: Signals scanned for a usable entry threshold, with the direction that should help a seller.
#: A seller wants implied volatility rich relative to what the underlying has been delivering.
SCAN_SIGNAL = "iv_minus_trailing_rv"

#: Candidate thresholds, in volatility points.
THRESHOLD_GRID: tuple[float, ...] = (
    -0.10,
    -0.05,
    -0.02,
    0.0,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.15,
)

#: A threshold that keeps too few in-sample trades is not evidence of anything.
MIN_IN_SAMPLE_TRADES = 40

#: Trailing window, in trading days, over which the VIX percentile is ranked.
VIX_PERCENTILE_WINDOW = 504

#: Regressors used in the predictive regression.
REGRESSION_TERMS: tuple[str, ...] = (
    "Intercept",
    "IV minus trailing realized vol",
    "Term slope (21d minus 63d)",
    "VIX percentile",
)


def build_signal_rows(
    inputs: Mapping[str, SecurityInputs], trades: Sequence[Trade], config: StudyConfig
) -> Table:
    """Attach pre-entry signal values to each trade's outcome.

    Args:
        inputs: Aligned inputs per security.
        trades: Core trades to annotate.
        config: Study configuration.

    Returns:
        One row per trade for which every signal is computable.
    """
    rows: Table = []
    for ticker, security_inputs in inputs.items():
        positions = {day: index for index, day in enumerate(security_inputs.dates)}
        short_horizon = security_inputs.matched_volatility(config.core_horizon_days)
        long_horizon = security_inputs.matched_volatility(max(config.all_horizons()))
        index_vol = list(security_inputs.index_volatility)

        for trade in (t for t in trades if t.ticker == ticker):
            index = positions.get(trade.entry_date)
            if index is None:
                continue
            trailing = trailing_realized_volatility(
                security_inputs.prices, index, config.realized_window_days
            )
            near, far = short_horizon[index], long_horizon[index]
            if trailing is None or near is None or far is None:
                continue

            window_start = max(0, index - VIX_PERCENTILE_WINDOW)
            history = [v for v in index_vol[window_start : index + 1] if v is not None]
            percentile = float(np.mean(np.asarray(history) <= trade.entry_vix)) if history else None
            if percentile is None:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "entry_date": trade.entry_date,
                    "in_sample": trade.in_sample,
                    "iv_minus_trailing_rv": trade.entry_iv - trailing,
                    "term_slope": near - far,
                    "vix_percentile": percentile,
                    "entry_iv": trade.entry_iv,
                    "trailing_realized_vol": trailing,
                    "realized_vol": trade.realized_vol,
                    "short_return_on_capital": trade.short_return_on_capital,
                    "short_net_pnl": trade.short_net_pnl,
                }
            )
    rows.sort(key=lambda row: (row["ticker"], row["entry_date"]))
    logger.info("Built %d signal observations", len(rows))
    return rows


def _outcome_stats(rows: Sequence[Row]) -> tuple[int, float | None, float | None]:
    """Return trade count, mean return on capital, and win rate for a subset."""
    if not rows:
        return 0, None, None
    values = np.array([row["short_return_on_capital"] for row in rows], dtype=float)
    return len(rows), float(np.mean(values)), float(np.mean(values > 0))


def threshold_scan(rows: Sequence[Row]) -> Table:
    """Evaluate every candidate threshold in sample and out of sample.

    Args:
        rows: Signal rows from :func:`build_signal_rows`.

    Returns:
        One row per threshold, reporting both periods so the honest comparison is unavoidable.
    """
    scan: Table = []
    for threshold in THRESHOLD_GRID:
        selected = [row for row in rows if row[SCAN_SIGNAL] >= threshold]
        in_sample = [row for row in selected if row["in_sample"]]
        out_sample = [row for row in selected if not row["in_sample"]]
        in_n, in_mean, in_win = _outcome_stats(in_sample)
        out_n, out_mean, out_win = _outcome_stats(out_sample)
        scan.append(
            {
                "signal": SCAN_SIGNAL,
                "threshold": threshold,
                "in_sample_n": in_n,
                "in_sample_mean_roc": in_mean,
                "in_sample_win_rate": in_win,
                "out_of_sample_n": out_n,
                "out_of_sample_mean_roc": out_mean,
                "out_of_sample_win_rate": out_win,
                "eligible": in_n >= MIN_IN_SAMPLE_TRADES,
            }
        )
    return scan


def select_threshold(scan: Sequence[Row]) -> Row | None:
    """Choose a threshold using in-sample results only.

    Args:
        scan: Output of :func:`threshold_scan`.

    Returns:
        The selected row, or None if no threshold retains enough in-sample trades. Selection looks
        exclusively at the in-sample columns; the out-of-sample columns exist to be reported, not
        to be optimised against.
    """
    eligible = [row for row in scan if row["eligible"] and row["in_sample_mean_roc"] is not None]
    if not eligible:
        logger.warning("No threshold retained at least %d in-sample trades", MIN_IN_SAMPLE_TRADES)
        return None
    return max(eligible, key=lambda row: row["in_sample_mean_roc"])


def selection_summary(rows: Sequence[Row], scan: Sequence[Row]) -> Table:
    """Compare the selected rule against taking every trade, in both periods.

    Returns:
        Four rows: the unconditional strategy and the filtered strategy, each split into the
        in-sample and out-of-sample periods.
    """
    chosen = select_threshold(scan)
    threshold = None if chosen is None else chosen["threshold"]
    summary: Table = []
    for rule, keep in (
        ("Take every trade", lambda row: True),
        (
            f"Filter: {SCAN_SIGNAL} >= {threshold}" if threshold is not None else "Filter: none",
            (lambda row: True)
            if threshold is None
            else (lambda row: row[SCAN_SIGNAL] >= threshold),
        ),
    ):
        for period, in_sample in (("In sample (to 2018)", True), ("Out of sample (2019+)", False)):
            subset = [row for row in rows if keep(row) and row["in_sample"] is in_sample]
            count, mean, win = _outcome_stats(subset)
            summary.append(
                {
                    "rule": rule,
                    "period": period,
                    "n": count,
                    "mean_return_on_capital": mean,
                    "win_rate": win,
                    "threshold_selected_in_sample_only": threshold is not None,
                }
            )
    return summary


def signal_regression(rows: Sequence[Row]) -> tuple[Table, Row]:
    """Regress the short seller's return on the three pre-entry signals, in sample only.

    Standard errors are heteroskedasticity-robust. The regression is fitted on the in-sample
    period alone, for the same reason the threshold is chosen there.

    Args:
        rows: Signal rows.

    Returns:
        Coefficient rows and a metadata row.

    Raises:
        ValueError: If the in-sample period holds too few observations to fit.
    """
    in_sample = [row for row in rows if row["in_sample"]]
    if len(in_sample) <= len(REGRESSION_TERMS):
        raise ValueError("Too few in-sample observations for the signal regression")

    outcome = np.array([row["short_return_on_capital"] for row in in_sample], dtype=float)
    # has_constant="add" is required: if a regressor happens to be constant -- a perfectly flat
    # term structure makes the slope identically zero -- the default would silently decline to add
    # an intercept, leaving one fewer coefficient than there are named terms.
    design = sm.add_constant(
        np.column_stack(
            [
                np.array([row["iv_minus_trailing_rv"] for row in in_sample], dtype=float),
                np.array([row["term_slope"] for row in in_sample], dtype=float),
                np.array([row["vix_percentile"] for row in in_sample], dtype=float),
            ]
        ),
        has_constant="add",
    )
    fit = sm.OLS(outcome, design).fit(cov_type="HC1")
    interval = fit.conf_int()
    coefficients: Table = [
        {
            "term": term,
            "coefficient": float(fit.params[position]),
            "std_error_robust": float(fit.bse[position]),
            "t_stat": float(fit.tvalues[position]),
            "p_value": float(fit.pvalues[position]),
            "ci_low": float(interval[position, 0]),
            "ci_high": float(interval[position, 1]),
        }
        for position, term in enumerate(REGRESSION_TERMS)
    ]
    meta: Row = {
        "n": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "adj_r_squared": float(fit.rsquared_adj),
        "covariance": "Heteroskedasticity-robust (HC1)",
        "dependent_variable": "Short-seller return on research capital",
        "period": "In sample only (through 2018-12-31)",
    }
    return coefficients, meta
