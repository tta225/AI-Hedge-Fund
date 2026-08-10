"""Backwards-compatible surface for the agent layer.

The desk used to live entirely in this module: five agent classes and a
sequential pipeline. It has since been split — roles into
:mod:`axiom.agents.roles`, execution into :mod:`axiom.agents.orchestrator` —
because the file was doing four jobs and none of them could be tested alone.

Every name that was importable from here still is, so existing call sites and
the CLI keep working unchanged. New code should import from the specific
module.
"""

from __future__ import annotations

from axiom.agents.orchestrator import AgentGraph, AgentPipeline, PipelineResult, Stage
from axiom.agents.roles import (
    BacktestAgent,
    DebateAgent,
    ExecutionAgent,
    PortfolioAgent,
    RedTeamAgent,
    RegimeAgent,
    ResearchAgent,
    ReviewAgent,
    RiskAgent,
    render_facts,
)

#: Retained under its original private name — the old module exposed it and
#: tests may reference it.
_render_facts = render_facts

__all__ = [
    "AgentGraph",
    "AgentPipeline",
    "BacktestAgent",
    "DebateAgent",
    "ExecutionAgent",
    "PipelineResult",
    "PortfolioAgent",
    "RedTeamAgent",
    "RegimeAgent",
    "ResearchAgent",
    "ReviewAgent",
    "RiskAgent",
    "Stage",
    "render_facts",
]
