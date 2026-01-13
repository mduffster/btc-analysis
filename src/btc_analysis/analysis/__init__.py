"""Analysis modules for BTC regulatory arbitrage research."""

from .cross_sectional import run_phase1_analysis
from .time_series import run_time_series_analysis
from .cash_substitution import run_cash_substitution_analysis

__all__ = [
    "run_phase1_analysis",
    "run_time_series_analysis",
    "run_cash_substitution_analysis",
]
