"""Agent registry — how a new seat joins the desk without editing the pipeline.

Previously the five roles were constructed inline in ``AgentPipeline.__init__``
and sequenced by hand in ``run``. Adding a sixth meant editing both, and the
graph's shape lived in the order of a list literal.

Here, an agent declares its own class, its dependencies live on the
:class:`~axiom.agents.orchestrator.Stage` that wraps it, and the registry maps
role names to classes so a desk can be assembled from configuration — a subset
of roles for a fast pre-trade check, the full graph for research.
"""

from __future__ import annotations

from axiom.agents.base import Agent, AgentRole
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


class AgentRegistry:
    """Maps roles to agent classes. One registry per desk configuration."""

    def __init__(self, agents: dict[AgentRole, type[Agent]] | None = None) -> None:
        self._agents: dict[AgentRole, type[Agent]] = dict(agents or {})

    def register(self, agent_class: type[Agent], *, replace: bool = False) -> None:
        """Add a role. Refuses to shadow an existing one unless asked.

        Silent replacement is how a desk ends up running a research agent
        nobody remembers installing.
        """
        role = agent_class.role
        if role in self._agents and not replace:
            raise ValueError(
                f"{role.value} is already registered to "
                f"{self._agents[role].__name__}; pass replace=True to override"
            )
        self._agents[role] = agent_class

    def get(self, role: AgentRole) -> type[Agent]:
        try:
            return self._agents[role]
        except KeyError:
            raise KeyError(
                f"no agent registered for role {role.value}; "
                f"registered: {sorted(r.value for r in self._agents)}"
            ) from None

    def __contains__(self, role: object) -> bool:
        return role in self._agents

    @property
    def roles(self) -> tuple[AgentRole, ...]:
        return tuple(self._agents)


def default_registry() -> AgentRegistry:
    """Every built-in seat."""
    registry = AgentRegistry()
    for agent_class in (
        RegimeAgent,
        ResearchAgent,
        DebateAgent,
        RedTeamAgent,
        BacktestAgent,
        ExecutionAgent,
        PortfolioAgent,
        QuantEnsembleAgent,
        RiskAgent,
        ReviewAgent,
    ):
        registry.register(agent_class)
    return registry
