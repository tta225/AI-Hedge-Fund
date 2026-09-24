"""The agent desk: a governed, audited graph of specialist analyst roles.

Nine seats — regime, research, debate, red team, backtest, execution,
portfolio, risk, review — run as a dependency graph over facts the
deterministic engines measured. Every model call passes through a runtime that
enforces budgets, timeouts, retries, and caching; every generated narrative is
checked for figures the facts do not support; every run is recorded, content
addressed, and gated by a mandate before a human is ever asked to approve
anything.
"""

from axiom.agents.audit import AuditLog, RunRecord, code_version
from axiom.agents.base import Agent, AgentReport, AgentRole
from axiom.agents.governance import (
    ApprovalRequest,
    Mandate,
    MandateBreach,
    MandateDecision,
    MandateRule,
    RunContext,
    Verdict,
)
from axiom.agents.guardrails import NumericFidelityReport, check_numeric_fidelity
from axiom.agents.orchestrator import (
    AgentGraph,
    AgentPipeline,
    GraphCycleError,
    PipelineResult,
    Stage,
)
from axiom.agents.registry import AgentRegistry, default_registry
from axiom.agents.roles import (
    BacktestAgent,
    DebateAgent,
    ExecutionAgent,
    PortfolioAgent,
    QuantEnsembleAgent,
    RedTeamAgent,
    RegimeAgent,
    ResearchAgent,
    ReviewAgent,
    RiskAgent,
)
from axiom.agents.runtime import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitOpenError,
    LLMBudget,
    LLMResult,
    LLMRuntime,
    ResponseCache,
    Usage,
    estimate_cost,
)

__all__ = [
    "Agent",
    "AgentGraph",
    "AgentPipeline",
    "AgentRegistry",
    "AgentReport",
    "AgentRole",
    "ApprovalRequest",
    "AuditLog",
    "BacktestAgent",
    "BudgetExceededError",
    "CircuitBreaker",
    "CircuitOpenError",
    "DebateAgent",
    "ExecutionAgent",
    "GraphCycleError",
    "LLMBudget",
    "LLMResult",
    "LLMRuntime",
    "Mandate",
    "MandateBreach",
    "MandateDecision",
    "MandateRule",
    "NumericFidelityReport",
    "PipelineResult",
    "PortfolioAgent",
    "QuantEnsembleAgent",
    "RedTeamAgent",
    "RegimeAgent",
    "ResearchAgent",
    "ResponseCache",
    "ReviewAgent",
    "RiskAgent",
    "RunContext",
    "RunRecord",
    "Stage",
    "Usage",
    "Verdict",
    "check_numeric_fidelity",
    "code_version",
    "default_registry",
    "estimate_cost",
]
