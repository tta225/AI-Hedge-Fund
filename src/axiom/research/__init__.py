"""Strategy search with overfitting controls."""

from axiom.research.lab import (
    Candidate,
    CandidateResult,
    SearchResult,
    StrategyLab,
    WalkForwardSplit,
    grid,
    walk_forward_splits,
)
from axiom.research.statistics import (
    PBOResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    parameter_stability,
    probability_of_backtest_overfitting,
)

__all__ = [
    "Candidate", "CandidateResult", "PBOResult", "SearchResult", "StrategyLab",
    "WalkForwardSplit", "deflated_sharpe_ratio", "expected_max_sharpe", "grid",
    "parameter_stability", "probability_of_backtest_overfitting",
    "walk_forward_splits",
]
