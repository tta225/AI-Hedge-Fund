"""Event-driven backtesting and performance measurement."""

from axiom.backtest.engine import Backtester, BacktestResult
from axiom.backtest.metrics import PerformanceReport, Trade, build_report, compute_metrics

__all__ = [
    "BacktestResult", "Backtester", "PerformanceReport", "Trade", "build_report",
    "compute_metrics",
]
