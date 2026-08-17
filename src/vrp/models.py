"""Typed records passed between the stages of the study.

Output tables stay as ordered dictionaries because their key order is the CSV column order.
Everything that never reaches a file is a dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, TypeAlias

#: One output record; key order is the exported column order.
Row: TypeAlias = dict[str, Any]

#: An output table.
Table: TypeAlias = list[Row]

#: A daily level series keyed by trading date.
LevelByDate: TypeAlias = dict[date, float]


@dataclass(frozen=True)
class Security:
    """One security in the study universe.

    Attributes:
        ticker: Study identifier.
        underlying_symbol: Market symbol of the price series, empty when not downloaded.
        name: Human-readable name.
        security_type: Index, single name, or ETF.
        group: Sector or role grouping.
        size_bucket: Size classification used in the cross-section.
        dividend_yield: Annual continuous dividend yield assumption.
        iv_evidence: What the implied-volatility input actually is.
        return_evidence: What the return input actually is.
        measured: Whether this security's variance risk premium is measured from observed data.
    """

    ticker: str
    underlying_symbol: str
    name: str
    security_type: str
    group: str
    size_bucket: str
    dividend_yield: float
    iv_evidence: str
    return_evidence: str
    measured: bool


@dataclass(frozen=True)
class DailyMark:
    """One day of a delta-hedged straddle's life.

    Attributes:
        date: Trading date.
        spot: Underlying close.
        implied_volatility: Implied volatility used to mark the package.
        maturity: Year fraction remaining to expiry.
        value: Straddle present value per share.
        delta: Package delta before any rebalance.
        gamma: Package gamma.
        vega: Package vega.
        theta: Package theta, per year.
        hedge_shares: Underlying position held into the next session.
        theta_pnl: Value change attributed to the passage of time.
        gamma_pnl: Value change attributed to underlying convexity.
        vega_pnl: Value change attributed to the change in implied volatility.
        residual_pnl: Value change the first-order terms do not explain.
        hedge_pnl: Profit on the underlying hedge position.
        financing_pnl: Carry on the hedge position.
        cost: Transaction cost incurred on the day.
        hedged_pnl: Total delta-hedged profit for the day, before cost.
    """

    date: date
    spot: float
    implied_volatility: float
    maturity: float
    value: float
    delta: float
    gamma: float
    vega: float
    theta: float
    hedge_shares: float
    theta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    residual_pnl: float
    hedge_pnl: float
    financing_pnl: float
    cost: float
    hedged_pnl: float
