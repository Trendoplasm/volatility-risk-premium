"""Decomposing the volatility risk premium via delta-hedged option returns.

The measured study covers the S&P 500, Apple, and Amazon, using Cboe's public volatility indexes
for the implied side and observed closing prices for the realised side. See ``README.md`` for the
research design and for what the study deliberately does not claim.
"""

from __future__ import annotations

from vrp.config import CostModel, StudyConfig
from vrp.hedged import Trade
from vrp.models import Security
from vrp.pipeline import StudyResults, headline, run_study, write_outputs

__version__ = "1.0.0"

__all__ = [
    "CostModel",
    "Security",
    "StudyConfig",
    "StudyResults",
    "Trade",
    "__version__",
    "headline",
    "run_study",
    "write_outputs",
]
