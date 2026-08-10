"""The AI agent layer.

Implements a **directed graph of specialist roles** — research, adversarial
review, execution, portfolio, risk, and post-hoc critique — with a human
approval gate before anything executes.

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

    This is now checked rather than trusted: every generated narrative is run
    through :func:`~axiom.agents.guardrails.check_numeric_fidelity`, and any
    figure that cannot be traced back to a measured fact is flagged on the
    report itself. See :mod:`axiom.agents.guardrails`.

*"Paper trade first."*
    :class:`~axiom.agents.orchestrator.AgentPipeline` has no path to a live
    venue. The graph terminates in an approval request, and a human acts on it.

The LLM is optional throughout. Without ``ANTHROPIC_API_KEY`` the graph runs in
deterministic mode and every stage still returns its measured findings.

Every model call goes through :class:`~axiom.agents.runtime.LLMRuntime`, which
owns budgets, timeouts, retries, caching, and cost accounting. No agent
constructs an SDK client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from axiom.agents.guardrails import NumericFidelityReport, check_numeric_fidelity
from axiom.agents.runtime import LLMRuntime, Usage
from axiom.core.config import AgentSettings
from axiom.core.provenance import Provenance

#: Shared across every agent, and byte-stable so it caches as a prompt prefix.
ANALYST_SYSTEM_PROMPT = (
    "You are an analyst on a systematic trading desk. You will be given "
    "MEASURED values computed by the firm's engines. Interpret them. Never "
    "invent, extrapolate, or restate numbers that are not in the input — every "
    "figure you write is checked against the input, and unverifiable figures "
    "are flagged to the reader. If the input is insufficient to reach a "
    "conclusion, say so plainly. Be skeptical rather than agreeable; your value "
    "is in identifying what could be wrong. Keep responses focused and brief."
)


class AgentRole(str, Enum):
    """Every seat on the desk. Order here is presentation order, not run order.

    Run order is a dependency graph — see :mod:`axiom.agents.orchestrator`.
    """

    RESEARCH = "research"
    REGIME = "regime"
    DEBATE = "debate"
    RED_TEAM = "red_team"
    BACKTEST = "backtest"
    EXECUTION = "execution"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    REVIEW = "review"


@dataclass(slots=True)
class AgentReport:
    """One agent's output.

    The split between :attr:`facts` and :attr:`narrative` is the point of this
    class. ``facts`` are measured; ``narrative`` is generated. They are never
    merged, so a reader always knows which is which — and :attr:`guardrail`
    records whether the generated half stayed inside the measured half.
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
    #: What this stage consumed. Zero in deterministic mode.
    usage: Usage = field(default_factory=Usage)
    #: Numeric-fidelity result for :attr:`narrative`.
    guardrail: NumericFidelityReport = field(default_factory=NumericFidelityReport)
    #: Content address of the prompt. Equal digests ⇒ identical question asked.
    prompt_digest: str = ""
    #: True when the narrative was served from cache rather than regenerated.
    cached: bool = False
    #: Wall-clock seconds for the whole stage, model call included.
    latency_s: float = 0.0
    #: Set when the stage failed outright. Facts may still be present.
    error: str = ""

    @property
    def used_llm(self) -> bool:
        return bool(self.narrative)

    @property
    def failed(self) -> bool:
        return bool(self.error)

    @property
    def trustworthy(self) -> bool:
        """Facts are measured and the narrative did not invent figures."""
        return not self.failed and self.guardrail.clean

    def as_dict(self, *, stable: bool = False) -> dict[str, Any]:
        """Serialisable form. The audit log's unit of record.

        Args:
            stable: omit everything that varies between two runs of the same
                analysis — wall-clock timestamps, latency, token counts, and
                whether the narrative came from cache. The audit record hashes
                the stable form, because a digest that changed with the clock
                could never establish that two runs produced the same result,
                which is the only thing a digest is for.
        """
        payload: dict[str, Any] = {
            "role": self.role.value,
            "facts": self.facts,
            "narrative": self.narrative,
            "warnings": list(self.warnings),
            "provenance": self.provenance.label if self.provenance else None,
            "model": self.model,
            "unsupported_figures": list(self.guardrail.unsupported),
            "prompt_digest": self.prompt_digest,
            "error": self.error,
        }
        if not stable:
            payload |= {
                "produced_at": self.produced_at.isoformat(),
                "usage": self.usage.as_dict(),
                "cached": self.cached,
                "latency_s": round(self.latency_s, 3),
            }
        return payload

    def render(self) -> str:
        lines = [f"── {self.role.value.upper()} " + "─" * 40]
        if self.error:
            lines.append(f"  ✗ stage failed: {self.error}")
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
            tag = f"{self.model}{' · cached' if self.cached else ''}"
            lines += ["", f"  [{tag}]", *(f"  {ln}" for ln in self.narrative.splitlines())]
        # Printed last and unmissably, because its whole purpose is to stop a
        # reader trusting the paragraph directly above it.
        if (guardrail := self.guardrail.warning()) is not None:
            lines += ["", f"  ⚠️  {guardrail}"]
        return "\n".join(lines)


class Agent(ABC):
    """Base class. Subclasses compute facts; the LLM only interprets them."""

    role: AgentRole
    #: Inputs this agent needs. The orchestrator checks these before running a
    #: stage, so a missing dependency is a named error rather than a KeyError
    #: thrown from somewhere inside `gather_facts`.
    requires: tuple[str, ...] = ()
    #: Output ceiling for this agent's model call. Generous by default: on
    #: current models this budget covers thinking *and* response text, and a
    #: tight cap truncates the answer rather than shortening it.
    max_tokens: int = 8_000

    def __init__(
        self,
        settings: AgentSettings | None = None,
        runtime: LLMRuntime | None = None,
    ) -> None:
        self.settings = settings or AgentSettings()
        self.runtime = runtime

    @property
    def llm_enabled(self) -> bool:
        return self.settings.enabled and self.runtime is not None

    @abstractmethod
    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        """Compute measured facts and warnings. Must not call an LLM."""

    @abstractmethod
    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        """Build the interpretation prompt from already-measured facts."""

    def run(self, **inputs: Any) -> AgentReport:
        """Measure, then interpret. Measurement failure is the only fatal one."""
        import time

        started = time.monotonic()
        missing = [name for name in self.requires if inputs.get(name) is None]
        if missing:
            return AgentReport(
                role=self.role,
                error=f"missing required input(s): {', '.join(missing)}",
                provenance=inputs.get("provenance"),
                latency_s=time.monotonic() - started,
            )

        try:
            facts, warnings = self.gather_facts(**inputs)
        except Exception as exc:
            # One stage's arithmetic blowing up must not cost the run every
            # other stage's findings. The orchestrator decides what to skip.
            return AgentReport(
                role=self.role,
                error=f"{type(exc).__name__}: {exc}",
                provenance=inputs.get("provenance"),
                latency_s=time.monotonic() - started,
            )

        report = AgentReport(
            role=self.role,
            facts=facts,
            warnings=warnings,
            provenance=inputs.get("provenance"),
        )

        if self.llm_enabled:
            assert self.runtime is not None
            prompt = self.prompt_for(facts, **inputs)
            try:
                result = self.runtime.complete(
                    system=ANALYST_SYSTEM_PROMPT,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                # A model failure must never invalidate measured facts.
                report.warnings.append(f"LLM interpretation unavailable: {exc}")
            else:
                report.narrative = result.text
                report.model = result.model
                report.usage = result.usage
                report.cached = result.cached
                report.prompt_digest = result.prompt_digest
                report.guardrail = check_numeric_fidelity(result.text, facts, prompt)
                if (warning := report.guardrail.warning()) is not None:
                    report.warnings.append(warning)

        report.latency_s = time.monotonic() - started
        return report
