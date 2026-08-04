"""The AI agent layer.

Implements the five-role pipeline: **Research → Debate → Backtest → Risk →
Review**, with a human approval gate before anything executes.

Two principles from the source workflow are enforced structurally rather than
requested politely in a prompt:

*"Don't invent data or numbers."*
    Agents never produce numbers. Every quantitative field on an
    :class:`AgentReport` is computed by the deterministic engines — the ICT
    engine, the backtester, the risk manager — and passed *to* the model as
    input. The model's job is to interpret and argue, and its narrative is kept
    in a separate field from the measured facts. An agent with no LLM
    configured still returns a complete, correct report; only the prose is
    missing.

*"Paper trade first."*
    :class:`AgentPipeline` has no path to a live venue. The execute stage
    produces an approval request, and a human acts on it.

The LLM is optional throughout. Without ``ANTHROPIC_API_KEY`` the pipeline runs
in deterministic mode and every stage still returns its measured findings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from axiom.core.config import AgentSettings
from axiom.core.provenance import Provenance


class AgentRole(str, Enum):
    RESEARCH = "research"
    DEBATE = "debate"
    BACKTEST = "backtest"
    RISK = "risk"
    REVIEW = "review"


@dataclass(slots=True)
class AgentReport:
    """One agent's output.

    The split between :attr:`facts` and :attr:`narrative` is the point of this
    class. ``facts`` are measured; ``narrative`` is generated. They are never
    merged, so a reader always knows which is which.
    """

    role: AgentRole
    #: Measured, engine-computed values. Never written by a model.
    facts: dict[str, Any] = field(default_factory=dict)
    #: Model-generated prose. Empty when running without an LLM.
    narrative: str = ""
    #: Concerns the agent is obliged to surface.
    warnings: list[str] = field(default_factory=list)
    #: Provenance of the data underlying ``facts``.
    provenance: Provenance | None = None
    produced_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    model: str = ""

    @property
    def used_llm(self) -> bool:
        return bool(self.narrative)

    def render(self) -> str:
        lines = [f"── {self.role.value.upper()} " + "─" * 40]
        if self.provenance is not None and not self.provenance.is_evidential:
            lines.append(
                f"⚠️  based on {self.provenance.kind.value} data — not evidence"
            )
        for key, value in self.facts.items():
            rendered = f"{value:,.4g}" if isinstance(value, float) else str(value)
            lines.append(f"  {key:<28} {rendered}")
        if self.warnings:
            lines.append("  concerns:")
            lines += [f"    ! {w}" for w in self.warnings]
        if self.narrative:
            lines += ["", f"  [{self.model}]", *(f"  {ln}" for ln in self.narrative.splitlines())]
        return "\n".join(lines)


class Agent(ABC):
    """Base class. Subclasses compute facts; the LLM only interprets them."""

    role: AgentRole

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self.settings = settings or AgentSettings()

    @property
    def llm_enabled(self) -> bool:
        return self.settings.enabled

    @abstractmethod
    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        """Compute measured facts and warnings. Must not call an LLM."""

    @abstractmethod
    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        """Build the interpretation prompt from already-measured facts."""

    def run(self, **inputs: Any) -> AgentReport:
        facts, warnings = self.gather_facts(**inputs)
        report = AgentReport(
            role=self.role,
            facts=facts,
            warnings=warnings,
            provenance=inputs.get("provenance"),
        )
        if self.llm_enabled:
            try:
                report.narrative = self._interpret(self.prompt_for(facts, **inputs))
                report.model = self.settings.agent_model
            except Exception as exc:
                # A model failure must never invalidate measured facts.
                report.warnings.append(f"LLM interpretation unavailable: {exc}")
        return report

    def _interpret(self, prompt: str) -> str:
        """Call the model. Isolated so it is trivial to stub in tests."""
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the agents extra is not installed (pip install 'axiom[agents]')"
            ) from exc

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        response = client.messages.create(
            model=self.settings.agent_model,
            max_tokens=1500,
            system=(
                "You are an analyst on a systematic trading desk. You will be "
                "given MEASURED values computed by the firm's engines. Interpret "
                "them. Never invent, extrapolate, or restate numbers that are "
                "not in the input. If the input is insufficient to reach a "
                "conclusion, say so plainly. Be skeptical rather than agreeable; "
                "your value is in identifying what could be wrong."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
