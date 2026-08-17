"""Study parameters and the fixed data contract.

Values here are research choices carried over from the original study's Assumptions sheet, or
external constraints on what data exists. Anything that appears as a bare number in a formula
elsewhere is named here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

# --- External data contract -------------------------------------------------------------

#: Observed S&P 500 volatility term structure: Cboe index name -> nominal calendar-day horizon.
#: These four points are what let the study interpolate to its study horizons instead of
#: assuming a term-structure shape.
INDEX_TERM_STRUCTURE: Final[dict[str, float]] = {
    "VIX9D": 9.0,
    "VIX": 30.0,
    "VIX3M": 93.0,
    "VIX1Y": 365.0,
}

#: 30-day implied-volatility index for each measured security.
IV_ANCHOR: Final[dict[str, str]] = {"INDEX": "VIX", "AAPL": "VXAPL", "AMZN": "VXAZN"}

#: Cboe history files required, keyed by series name.
REQUIRED_CBOE_FILES: Final[dict[str, str]] = {
    name: f"{name}_History.csv" for name in (*INDEX_TERM_STRUCTURE, "VXAPL", "VXAZN")
}

#: Price history files required, keyed by study ticker.
REQUIRED_PRICE_FILES: Final[dict[str, str]] = {
    ticker: f"{ticker}_prices.csv" for ticker in IV_ANCHOR
}

#: Securities whose variance risk premium this study measures, in reporting order.
MEASURED_TICKERS: Final[tuple[str, ...]] = ("INDEX", "AAPL", "AMZN")

#: Reporting order for pooled and per-security summaries.
GROUP_ORDER: Final[tuple[str, ...]] = ("Pooled", *MEASURED_TICKERS)

# --- Conventions ------------------------------------------------------------------------

#: Trading days per year, used to annualise variance and to convert horizons to year fractions.
TRADING_DAYS_PER_YEAR: Final[float] = 252.0

#: Calendar days per year, used for the Cboe term structure, which is quoted in calendar days.
CALENDAR_DAYS_PER_YEAR: Final[float] = 365.0

#: Cboe quotes volatility in percentage points; the models work in decimals.
VOLATILITY_POINTS_PER_UNIT: Final[float] = 100.0

#: Minimum observations before a trailing realised-volatility estimate is reported.
MIN_REALIZED_OBSERVATIONS: Final[int] = 15

#: Quantile bounds of a two-sided 95% bootstrap interval.
BOOTSTRAP_LOWER_QUANTILE: Final[float] = 0.025
BOOTSTRAP_UPPER_QUANTILE: Final[float] = 0.975

#: Number of trades listed in the best- and worst-outcome tables.
EXTREME_TRADE_COUNT: Final[int] = 15


@dataclass(frozen=True)
class CostModel:
    """Transaction costs applied to a delta-hedged option position.

    Attributes:
        option_commission: Dollars per contract per leg.
        option_half_spread: Entry half-spread as a fraction of option premium.
        stock_hedge_cost_bps: Basis points of underlying turnover for single names.
        index_hedge_cost_bps: Basis points of underlying turnover for the index.
        contract_multiplier: Shares per option contract.
    """

    option_commission: float = 0.65
    option_half_spread: float = 0.01
    stock_hedge_cost_bps: float = 1.0
    index_hedge_cost_bps: float = 0.5
    contract_multiplier: float = 100.0

    def hedge_cost_bps(self, ticker: str) -> float:
        """Return the hedge cost in basis points for one security."""
        return self.index_hedge_cost_bps if ticker == "INDEX" else self.stock_hedge_cost_bps


@dataclass(frozen=True)
class StudyConfig:
    """Windows, horizons, and inference settings for one run of the study.

    Attributes:
        start_date: First date of the study period. Defaults to the first common date of the
            single-name volatility indexes, which is what bounds the sample.
        end_date: Last date of the study period. Frozen deliberately: both data providers extend
            their series every trading day, so an open-ended sample would give a different answer
            every time it was downloaded. Fixing the end date is what makes the published results
            reproducible from a download taken at any later time.
        risk_free_rate: Annual continuously compounded financing rate.
        core_horizon_days: Trading-day maturity of the core straddle.
        supplementary_horizons: Additional maturities run as robustness comparisons.
        moneyness_grid: Strike divided by the entry forward, for the moneyness comparison.
        core_moneyness: Moneyness of the core straddle.
        hedge_interval_days: Trading days between delta rebalances in the core protocol.
        supplementary_hedge_interval: Rebalance interval used for the robustness comparison.
        realized_window_days: Trailing window for the realised-volatility signal.
        train_end: Last date of the in-sample period used to choose signal thresholds.
        short_capital_spot_fraction: Research capital proxy as a fraction of spot notional.
        short_capital_premium_multiple: Alternative proxy as a multiple of entry premium.
        bootstrap_iterations: Resamples used for confidence intervals.
        random_seed: Seed for the bootstrap generator.
    """

    start_date: str = "2011-01-07"
    end_date: str = "2026-06-30"
    risk_free_rate: float = 0.0425
    core_horizon_days: int = 21
    supplementary_horizons: tuple[int, ...] = (42, 63)
    moneyness_grid: tuple[float, ...] = (0.95, 1.00, 1.05)
    core_moneyness: float = 1.00
    hedge_interval_days: int = 1
    supplementary_hedge_interval: int = 5
    realized_window_days: int = 21
    train_end: str = "2018-12-31"
    short_capital_spot_fraction: float = 0.20
    short_capital_premium_multiple: float = 1.5
    bootstrap_iterations: int = 10_000
    random_seed: int = 20_260_818

    def start(self) -> date:
        """Return :attr:`start_date` as a date."""
        return datetime.strptime(self.start_date, "%Y-%m-%d").date()

    def end(self) -> date:
        """Return :attr:`end_date` as a date."""
        return datetime.strptime(self.end_date, "%Y-%m-%d").date()

    def train_cutoff(self) -> date:
        """Return :attr:`train_end` as a date."""
        return datetime.strptime(self.train_end, "%Y-%m-%d").date()

    def all_horizons(self) -> tuple[int, ...]:
        """Return every maturity the study runs, core first."""
        return (self.core_horizon_days, *self.supplementary_horizons)

    def horizon_years(self, trading_days: int) -> float:
        """Convert a trading-day horizon to a year fraction."""
        return trading_days / TRADING_DAYS_PER_YEAR
