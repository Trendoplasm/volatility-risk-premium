"""Running the delta-hedged protocol across securities, horizons, and variants.

The simulation in :mod:`vrp.hedged` handles one trade. This module decides which trades to run:
non-overlapping entries every ``horizon_days`` trading days, for each measured security, plus the
robustness variants that vary maturity, moneyness, and hedge frequency one dimension at a time.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from vrp.config import IV_ANCHOR, CostModel, StudyConfig
from vrp.hedged import Trade, simulate_trade
from vrp.models import LevelByDate, Security
from vrp.panel import aligned_calendar
from vrp.termstructure import index_curve, matched_implied_volatility

logger = logging.getLogger(__name__)


class SecurityInputs:
    """Aligned inputs for one security, ready to simulate against.

    Attributes:
        security: The security itself.
        dates: Trading calendar taken from the price history.
        prices: Closes aligned to :attr:`dates`.
        index_volatility: 30-day S&P 500 implied volatility aligned to :attr:`dates`.
        first_study_index: Position of the first date inside the study period.
    """

    def __init__(
        self,
        security: Security,
        volatility_series: Mapping[str, LevelByDate],
        price_series: Mapping[str, LevelByDate],
        config: StudyConfig,
    ) -> None:
        """Align one security's price and volatility histories onto a single calendar.

        Args:
            security: The security to prepare.
            volatility_series: Loaded Cboe series keyed by index name.
            price_series: Loaded price histories keyed by study ticker.
            config: Study period and horizons.
        """
        anchor = volatility_series[IV_ANCHOR[security.ticker]]
        self.security = security
        self.dates, self.prices = aligned_calendar(
            price_series[security.ticker], anchor, config.start(), config.end()
        )
        self._anchor = anchor
        self._curves = {day: index_curve(volatility_series, day) for day in self.dates}
        self.index_volatility: list[float | None] = [
            self._curves[day].get(30.0) for day in self.dates
        ]
        start = config.start()
        self.first_study_index = sum(day < start for day in self.dates)

    def matched_volatility(self, horizon_days: int) -> list[float | None]:
        """Return implied volatility matched to one horizon, aligned to :attr:`dates`.

        Args:
            horizon_days: Horizon in trading days.

        Returns:
            Decimal implied volatility per date, None where the security's index did not publish
            or the S&P 500 curve lacked its 30-day anchor.
        """
        matched: list[float | None] = []
        for day in self.dates:
            anchor_iv = self._anchor.get(day)
            if anchor_iv is None:
                matched.append(None)
                continue
            matched.append(matched_implied_volatility(anchor_iv, self._curves[day], horizon_days))
        return matched


def run_protocol(
    inputs: SecurityInputs,
    horizon_days: int,
    config: StudyConfig,
    costs: CostModel,
    *,
    moneyness: float | None = None,
    hedge_interval_days: int | None = None,
) -> list[Trade]:
    """Run non-overlapping entries for one security under one protocol variant.

    Entries are spaced by the holding period so that no two trades are open at once. Overlapping
    entries would inflate the apparent sample size while adding almost no independent information,
    since neighbouring trades would share most of the same price path.

    Args:
        inputs: Aligned inputs for the security.
        horizon_days: Trading days to expiry.
        config: Study configuration.
        costs: Transaction-cost model.
        moneyness: Strike over entry forward; defaults to the core value.
        hedge_interval_days: Rebalance interval; defaults to the core value.

    Returns:
        Completed trades in entry order.
    """
    matched = inputs.matched_volatility(horizon_days)
    trades: list[Trade] = []
    index = inputs.first_study_index
    while index + horizon_days < len(inputs.dates):
        entry_vix = inputs.index_volatility[index]
        if entry_vix is None:
            index += 1
            continue
        trade = simulate_trade(
            inputs.security.ticker,
            inputs.dates,
            inputs.prices,
            matched,
            index,
            horizon_days,
            dividend_yield=inputs.security.dividend_yield,
            entry_vix=entry_vix,
            config=config,
            costs=costs,
            moneyness=moneyness,
            hedge_interval_days=hedge_interval_days,
        )
        if trade is None:
            index += 1
            continue
        trades.append(trade)
        index += horizon_days
    return trades


def core_trades(
    securities: Sequence[Security],
    volatility_series: Mapping[str, LevelByDate],
    price_series: Mapping[str, LevelByDate],
    config: StudyConfig,
    costs: CostModel,
) -> tuple[list[Trade], dict[str, SecurityInputs]]:
    """Run the core protocol for every measured security.

    Returns:
        The core trades and the aligned inputs, so that variant runs need not rebuild them.
    """
    inputs = {
        security.ticker: SecurityInputs(security, volatility_series, price_series, config)
        for security in securities
        if security.measured
    }
    trades: list[Trade] = []
    for security_inputs in inputs.values():
        trades.extend(run_protocol(security_inputs, config.core_horizon_days, config, costs))
    logger.info("Simulated %d core delta-hedged trades", len(trades))
    return trades, inputs


def variant_trades(
    inputs: Mapping[str, SecurityInputs], config: StudyConfig, costs: CostModel
) -> list[Trade]:
    """Run every robustness variant: maturity, moneyness, and hedge frequency.

    Each variant moves one dimension away from the core protocol and leaves the rest fixed, so a
    difference in outcome is attributable to that dimension.

    Returns:
        Trades from every variant, including a re-run of the core protocol so that the comparison
        table has a like-for-like baseline row.
    """
    trades: list[Trade] = []
    for security_inputs in inputs.values():
        for horizon in config.all_horizons():
            trades.extend(run_protocol(security_inputs, horizon, config, costs))
        for moneyness in config.moneyness_grid:
            if moneyness == config.core_moneyness:
                continue
            trades.extend(
                run_protocol(
                    security_inputs, config.core_horizon_days, config, costs, moneyness=moneyness
                )
            )
        trades.extend(
            run_protocol(
                security_inputs,
                config.core_horizon_days,
                config,
                costs,
                hedge_interval_days=config.supplementary_hedge_interval,
            )
        )
    logger.info("Simulated %d variant delta-hedged trades", len(trades))
    return trades
