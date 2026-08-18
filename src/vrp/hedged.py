"""Delta-hedged straddle simulation and profit decomposition.

Measuring the variance risk premium as implied minus realised variance says the premium exists. It
does not say what an investor would have *earned*, or where that money came from. This module
answers both by simulating the trade.

The protocol
------------
Open one call plus one put at a common strike set from the entry forward. Mark the package every
day with the security's observed implied volatility and its observed close. Rebalance the
Black-Scholes delta hedge, finance the resulting position, and hold to cash settlement. Entries do
not overlap: the next trade begins when the previous one expires.

The decomposition
-----------------
Each day's hedged profit is attributed to the Greeks that generated it:

    dV - Delta * dS + financing  =  Theta * dt  +  0.5 * Gamma * dS^2  +  Vega * dIV  +  residual

Read left to right, that is the definition of a delta-hedged gain. Read right to left, it is the
economics: theta is collected almost every day, while gamma and vega losses arrive in bursts. The
residual is what the first-order terms miss -- mostly higher-order convexity on large moves -- and
reporting it rather than hiding it is what makes the attribution checkable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from vrp.blackscholes import forward_price, straddle_mark
from vrp.config import TRADING_DAYS_PER_YEAR, CostModel, StudyConfig
from vrp.models import DailyMark
from vrp.realized import realized_volatility

logger = logging.getLogger(__name__)

#: One trading day as a year fraction.
DAY_FRACTION = 1.0 / TRADING_DAYS_PER_YEAR

#: Basis points per unit.
BASIS_POINTS = 10_000.0


@dataclass(frozen=True)
class Trade:
    """One completed delta-hedged straddle.

    Sign convention: every ``*_pnl`` field is the profit of a **long** straddle position, which is
    the academic convention and the one in which the attribution identity holds. A short seller
    earns the negative of ``gross_pnl``, and pays costs either way; :attr:`short_net_pnl` states
    that directly.

    Attributes:
        ticker: Security traded.
        horizon_days: Trading days from entry to expiry.
        moneyness: Strike divided by the entry forward.
        hedge_interval_days: Trading days between delta rebalances.
        entry_date: Date the position was opened.
        exit_date: Date the position settled.
        entry_spot: Underlying close at entry.
        exit_spot: Underlying close at settlement.
        strike: Common strike of the two legs.
        entry_iv: Implied volatility used to price the package at entry.
        realized_vol: Volatility the underlying delivered over the holding period.
        entry_premium: Straddle premium per share at entry.
        exit_value: Intrinsic value per share at settlement.
        gross_pnl: Delta-hedged profit per share, long the straddle, before costs.
        theta_pnl: Portion attributed to the passage of time.
        gamma_pnl: Portion attributed to underlying convexity.
        vega_pnl: Portion attributed to changes in implied volatility.
        residual_pnl: Portion the first-order terms do not explain.
        hedge_pnl: Profit on the underlying hedge, per share.
        financing_pnl: Financing carry on the hedged position, per share.
        cost: Transaction costs in dollars for the modelled contract.
        contracts: Contracts traded.
        capital: Research capital proxy in dollars.
        short_net_pnl: Dollars earned by the short seller after costs.
        short_return_on_capital: :attr:`short_net_pnl` divided by :attr:`capital`.
        rebalances: Number of delta rebalances performed.
        entry_vix: 30-day S&P 500 implied volatility at entry, for regime conditioning.
        in_sample: Whether entry falls in the period used to choose signal thresholds.
    """

    ticker: str
    horizon_days: int
    moneyness: float
    hedge_interval_days: int
    entry_date: date
    exit_date: date
    entry_spot: float
    exit_spot: float
    strike: float
    entry_iv: float
    realized_vol: float
    entry_premium: float
    exit_value: float
    gross_pnl: float
    theta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    residual_pnl: float
    hedge_pnl: float
    financing_pnl: float
    cost: float
    contracts: float
    capital: float
    short_net_pnl: float
    short_return_on_capital: float
    rebalances: int
    entry_vix: float
    in_sample: bool


def _capital_requirement(
    entry_spot: float, entry_premium: float, costs: CostModel, config: StudyConfig
) -> float:
    """Return the research capital proxy for one short straddle contract.

    Deliberately transparent and conservative rather than accurate: real portfolio margin depends
    on offsets, concentration, stress scans, liquidity, assignment risk, and house rules. This is a
    normalisation constant for comparing returns, not a broker requirement.
    """
    notional = entry_spot * costs.contract_multiplier
    premium = entry_premium * costs.contract_multiplier
    return max(
        config.short_capital_spot_fraction * notional,
        config.short_capital_premium_multiple * premium,
    )


def simulate_trade(
    ticker: str,
    dates: Sequence[date],
    prices: Sequence[float],
    implied_volatilities: Sequence[float | None],
    entry_index: int,
    horizon_days: int,
    *,
    dividend_yield: float,
    entry_vix: float,
    config: StudyConfig,
    costs: CostModel,
    moneyness: float | None = None,
    hedge_interval_days: int | None = None,
) -> Trade | None:
    """Simulate one delta-hedged straddle from entry to settlement.

    Args:
        ticker: Security being traded.
        dates: Trading calendar.
        prices: Closes aligned to ``dates``.
        implied_volatilities: Matched implied volatility aligned to ``dates``; None where the
            security's volatility index did not publish.
        entry_index: Position in ``dates`` at which to open.
        horizon_days: Trading days to expiry.
        dividend_yield: Annual continuous dividend yield.
        entry_vix: 30-day S&P 500 implied volatility at entry.
        config: Rates and capital conventions.
        costs: Transaction-cost model.
        moneyness: Strike over entry forward; defaults to the configured core moneyness.
        hedge_interval_days: Rebalance interval; defaults to the configured core interval.

    Returns:
        The completed trade, or None if the window runs past the end of the data or the entry day
        has no implied volatility to price against.
    """
    exit_index = entry_index + horizon_days
    if exit_index >= len(dates):
        return None
    entry_iv = implied_volatilities[entry_index]
    if entry_iv is None or entry_iv <= 0:
        return None

    moneyness = config.core_moneyness if moneyness is None else moneyness
    interval = config.hedge_interval_days if hedge_interval_days is None else hedge_interval_days
    rate = config.risk_free_rate

    entry_spot = prices[entry_index]
    maturity = config.horizon_years(horizon_days)
    strike = moneyness * float(forward_price(entry_spot, rate, dividend_yield, maturity))

    mark = straddle_mark(entry_spot, strike, rate, dividend_yield, entry_iv, maturity)
    entry_premium = mark.value

    # Entry costs: a half-spread on the package plus one commission for each of the two legs.
    cost = (
        costs.option_half_spread * entry_premium * costs.contract_multiplier
        + 2.0 * costs.option_commission
    )
    hedge_cost_rate = costs.hedge_cost_bps(ticker) / BASIS_POINTS

    hedge_shares = -mark.delta
    cost += abs(hedge_shares) * entry_spot * costs.contract_multiplier * hedge_cost_rate

    totals = {
        "theta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "residual": 0.0,
        "hedge": 0.0,
        "financing": 0.0,
        "gross": 0.0,
    }
    marks: list[DailyMark] = []
    rebalances = 0
    previous = mark
    previous_spot = entry_spot
    previous_iv = entry_iv

    for step in range(1, horizon_days + 1):
        index = entry_index + step
        remaining = config.horizon_years(horizon_days - step)
        spot = prices[index]
        current_iv = implied_volatilities[index]
        # A missing volatility print is carried forward: the position still exists, and the
        # alternative would be to silently truncate the trade.
        if current_iv is None or current_iv <= 0:
            current_iv = previous_iv

        current = straddle_mark(spot, strike, rate, dividend_yield, current_iv, remaining)

        change_in_value = current.value - previous.value
        change_in_spot = spot - previous_spot
        change_in_iv = current_iv - previous_iv

        hedge_pnl = hedge_shares * change_in_spot
        # Financing on the net cash position: long the option, short the hedge.
        financing_pnl = -rate * (previous.value + hedge_shares * previous_spot) * DAY_FRACTION
        hedged_pnl = change_in_value + hedge_pnl + financing_pnl

        theta_pnl = previous.theta * DAY_FRACTION
        gamma_pnl = 0.5 * previous.gamma * change_in_spot**2
        vega_pnl = previous.vega * change_in_iv
        residual_pnl = hedged_pnl - (theta_pnl + gamma_pnl + vega_pnl)

        totals["theta"] += theta_pnl
        totals["gamma"] += gamma_pnl
        totals["vega"] += vega_pnl
        totals["residual"] += residual_pnl
        totals["hedge"] += hedge_pnl
        totals["financing"] += financing_pnl
        totals["gross"] += hedged_pnl

        if step < horizon_days and step % interval == 0:
            target_shares = -current.delta
            traded = abs(target_shares - hedge_shares)
            cost += traded * spot * costs.contract_multiplier * hedge_cost_rate
            hedge_shares = target_shares
            rebalances += 1

        marks.append(
            DailyMark(
                date=dates[index],
                spot=spot,
                implied_volatility=current_iv,
                maturity=remaining,
                value=current.value,
                delta=current.delta,
                gamma=current.gamma,
                vega=current.vega,
                theta=current.theta,
                hedge_shares=hedge_shares,
                theta_pnl=theta_pnl,
                gamma_pnl=gamma_pnl,
                vega_pnl=vega_pnl,
                residual_pnl=residual_pnl,
                hedge_pnl=hedge_pnl,
                financing_pnl=financing_pnl,
                cost=0.0,
                hedged_pnl=hedged_pnl,
            )
        )
        previous, previous_spot, previous_iv = current, spot, current_iv

    window = prices[entry_index : exit_index + 1]
    delivered = realized_volatility(window)
    capital = _capital_requirement(entry_spot, entry_premium, costs, config)
    gross_dollars = totals["gross"] * costs.contract_multiplier
    short_net = -gross_dollars - cost

    return Trade(
        ticker=ticker,
        horizon_days=horizon_days,
        moneyness=moneyness,
        hedge_interval_days=interval,
        entry_date=dates[entry_index],
        exit_date=dates[exit_index],
        entry_spot=entry_spot,
        exit_spot=prices[exit_index],
        strike=strike,
        entry_iv=entry_iv,
        realized_vol=delivered if delivered is not None else float("nan"),
        entry_premium=entry_premium,
        exit_value=previous.value,
        gross_pnl=totals["gross"],
        theta_pnl=totals["theta"],
        gamma_pnl=totals["gamma"],
        vega_pnl=totals["vega"],
        residual_pnl=totals["residual"],
        hedge_pnl=totals["hedge"],
        financing_pnl=totals["financing"],
        cost=cost,
        contracts=1.0,
        capital=capital,
        short_net_pnl=short_net,
        short_return_on_capital=short_net / capital if capital > 0 else float("nan"),
        rebalances=rebalances,
        entry_vix=entry_vix,
        in_sample=dates[entry_index] <= config.train_cutoff(),
    )


def attribution_error(trade: Trade) -> float:
    """Return how far the Greek attribution is from the trade's actual hedged profit.

    The four components are defined to sum to the total, so this should be zero to floating-point
    precision. It is reported so that the identity is checked rather than assumed.
    """
    parts = trade.theta_pnl + trade.gamma_pnl + trade.vega_pnl + trade.residual_pnl
    return float(abs(parts - trade.gross_pnl))


def trade_row(trade: Trade) -> dict[str, object]:
    """Flatten a trade into an export row."""
    return {
        "ticker": trade.ticker,
        "horizon_days": trade.horizon_days,
        "moneyness": trade.moneyness,
        "hedge_interval_days": trade.hedge_interval_days,
        "entry_date": trade.entry_date,
        "exit_date": trade.exit_date,
        "entry_spot": trade.entry_spot,
        "exit_spot": trade.exit_spot,
        "strike": trade.strike,
        "entry_iv": trade.entry_iv,
        "realized_vol": trade.realized_vol,
        "iv_minus_realized": trade.entry_iv - trade.realized_vol,
        "entry_premium": trade.entry_premium,
        "exit_value": trade.exit_value,
        "gross_pnl_per_share": trade.gross_pnl,
        # Normalised so securities at very different price levels are comparable: the S&P 500
        # trades near 7,000 and Apple near 300, so per-share figures are not on one scale.
        "gross_pnl_pct_spot": trade.gross_pnl / trade.entry_spot,
        "gross_pnl_pct_premium": trade.gross_pnl / trade.entry_premium,
        "theta_pct_premium": trade.theta_pnl / trade.entry_premium,
        "gamma_pct_premium": trade.gamma_pnl / trade.entry_premium,
        "vega_pct_premium": trade.vega_pnl / trade.entry_premium,
        "residual_pct_premium": trade.residual_pnl / trade.entry_premium,
        "theta_pnl": trade.theta_pnl,
        "gamma_pnl": trade.gamma_pnl,
        "vega_pnl": trade.vega_pnl,
        "residual_pnl": trade.residual_pnl,
        "hedge_pnl": trade.hedge_pnl,
        "financing_pnl": trade.financing_pnl,
        "cost_dollars": trade.cost,
        "capital_dollars": trade.capital,
        "short_net_pnl": trade.short_net_pnl,
        "short_return_on_capital": trade.short_return_on_capital,
        "rebalances": trade.rebalances,
        "entry_vix": trade.entry_vix,
        "in_sample": trade.in_sample,
        "attribution_error": attribution_error(trade),
    }
