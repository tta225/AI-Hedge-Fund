"""The mandate: what this desk is permitted to propose, checked mechanically.

:class:`~axiom.risk.manager.RiskManager` answers "is this position correctly
sized?". The mandate answers a different question — "is this the kind of trade
we are allowed to do at all?" — and it answers it *after* every agent has
spoken, because some of its checks are about the analysis rather than the trade.

The distinction matters because these two kinds of limit fail differently. A
sizing breach is arithmetic and the risk manager catches it deterministically.
A mandate breach is categorical: research ran on synthetic data, the red team's
objection went unanswered, half the desk failed to report, a narrative quoted
figures nobody measured. None of those make the position badly sized. All of
them make it unapprovable.

Every rule here is a hard gate, not a score. A weighted mandate score is a
number you can talk yourself past; a refusal naming the rule is not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from axiom.agents.base import AgentReport, AgentRole
from axiom.core.config import RiskSettings, Settings


class Verdict(str, Enum):
    """Outcome of the gate. There is no ``APPROVED`` — only a human approves."""

    #: Nothing blocks a human from considering this. Not an endorsement.
    PRESENTABLE = "presentable"
    #: Presentable, but a named concern must be acknowledged explicitly.
    REQUIRES_ACKNOWLEDGEMENT = "requires_acknowledgement"
    #: The desk may not put this in front of anyone as a trade candidate.
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class MandateBreach:
    """One rule, broken, named."""

    rule: str
    detail: str
    blocking: bool = True

    def render(self) -> str:
        marker = "✗" if self.blocking else "!"
        return f"  {marker} [{self.rule}] {self.detail}"


@dataclass(frozen=True, slots=True)
class MandateRule:
    """A named predicate over a finished run.

    Rules are data rather than method bodies so a desk can enumerate its own
    mandate — ``[r.name for r in mandate.rules]`` is the compliance answer to
    "what do you check?", and it cannot drift from what actually runs.
    """

    name: str
    description: str
    check: Callable[[RunContext], MandateBreach | None]
    blocking: bool = True


@dataclass(frozen=True, slots=True)
class RunContext:
    """Everything the mandate is allowed to look at."""

    reports: tuple[AgentReport, ...]
    settings: Settings
    risk_settings: RiskSettings
    has_signal: bool
    data_is_evidence: bool
    expected_roles: tuple[AgentRole, ...] = ()

    def report(self, role: AgentRole) -> AgentReport | None:
        return next((r for r in self.reports if r.role is role), None)


def _rule_kill_switch(ctx: RunContext) -> MandateBreach | None:
    if ctx.settings.kill_switch:
        return MandateBreach(
            "kill_switch",
            "The kill switch is engaged. No candidate may be presented for approval.",
        )
    return None


def _rule_real_data(ctx: RunContext) -> MandateBreach | None:
    if not ctx.data_is_evidence:
        return MandateBreach(
            "evidential_data",
            "Analysis ran on non-evidential data. It demonstrates the code runs "
            "and says nothing about this trade.",
        )
    return None


def _rule_numeric_fidelity(ctx: RunContext) -> MandateBreach | None:
    offenders = {
        r.role.value: r.guardrail.unsupported for r in ctx.reports if not r.guardrail.clean
    }
    if offenders:
        listed = "; ".join(f"{role}: {', '.join(figs)}" for role, figs in offenders.items())
        return MandateBreach(
            "numeric_fidelity",
            f"Generated narrative contains figures absent from the measured "
            f"facts ({listed}). The prose cannot be relied on.",
        )
    return None


def _rule_stage_completeness(ctx: RunContext) -> MandateBreach | None:
    failed = [r.role.value for r in ctx.reports if r.failed]
    produced = {r.role for r in ctx.reports}
    absent = [role.value for role in ctx.expected_roles if role not in produced]
    if failed or absent:
        parts = []
        if failed:
            parts.append(f"failed: {', '.join(failed)}")
        if absent:
            parts.append(f"never ran: {', '.join(absent)}")
        return MandateBreach(
            "stage_completeness",
            "The desk did not report in full — " + "; ".join(parts) + ".",
        )
    return None


def _rule_red_team_answered(ctx: RunContext) -> MandateBreach | None:
    red = ctx.report(AgentRole.RED_TEAM)
    if red is not None and red.warnings and ctx.has_signal:
        return MandateBreach(
            "red_team_objection",
            f"The red team raised {len(red.warnings)} unresolved objection(s). "
            "A human must address each before this is actionable.",
            blocking=False,
        )
    return None


def _rule_risk_approved(ctx: RunContext) -> MandateBreach | None:
    risk = ctx.report(AgentRole.RISK)
    if risk is None or not ctx.has_signal:
        return None
    if risk.facts.get("approved") is False:
        reasons = "; ".join(risk.warnings) or "no reason recorded"
        return MandateBreach(
            "risk_refusal",
            f"The risk manager declined this position: {reasons}",
        )
    return None


def _rule_sample_size(ctx: RunContext) -> MandateBreach | None:
    backtest = ctx.report(AgentRole.BACKTEST)
    if backtest is None:
        return None
    trades = float(backtest.facts.get("trades", 0) or 0)
    if trades < 30:
        return MandateBreach(
            "sample_size",
            f"Backtest sample is {trades:.0f} trades. Below 30 the performance "
            "statistics have no interpretation, favourable or otherwise.",
            blocking=False,
        )
    return None


def _rule_execution_viability(ctx: RunContext) -> MandateBreach | None:
    execution = ctx.report(AgentRole.EXECUTION)
    if execution is None:
        return None
    if execution.facts.get("survives_double_cost") is False:
        return MandateBreach(
            "cost_sensitivity",
            "The edge does not survive doubled execution costs — it is an "
            "artefact of the cost assumption, not of the strategy.",
            blocking=False,
        )
    return None


#: The standing mandate. Order is presentation order in a refusal.
DEFAULT_RULES: tuple[MandateRule, ...] = (
    MandateRule(
        "kill_switch",
        "Nothing is presentable while the kill switch is engaged.",
        _rule_kill_switch,
    ),
    MandateRule(
        "evidential_data",
        "A trade candidate must rest on real market data.",
        _rule_real_data,
    ),
    MandateRule(
        "numeric_fidelity",
        "No agent narrative may contain a figure absent from the measured facts.",
        _rule_numeric_fidelity,
    ),
    MandateRule(
        "stage_completeness",
        "Every seat on the desk must report before a candidate is presentable.",
        _rule_stage_completeness,
    ),
    MandateRule(
        "risk_refusal",
        "A position the risk manager declined is never presentable.",
        _rule_risk_approved,
    ),
    MandateRule(
        "red_team_objection",
        "Unresolved red-team objections require explicit human acknowledgement.",
        _rule_red_team_answered,
        blocking=False,
    ),
    MandateRule(
        "sample_size",
        "A sub-30-trade sample requires explicit acknowledgement.",
        _rule_sample_size,
        blocking=False,
    ),
    MandateRule(
        "cost_sensitivity",
        "An edge that dies at doubled costs requires explicit acknowledgement.",
        _rule_execution_viability,
        blocking=False,
    ),
)


@dataclass(frozen=True, slots=True)
class MandateDecision:
    """What the gate concluded, and exactly why."""

    verdict: Verdict
    breaches: tuple[MandateBreach, ...] = ()
    rules_checked: int = 0

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCKED

    @property
    def blocking_breaches(self) -> tuple[MandateBreach, ...]:
        return tuple(b for b in self.breaches if b.blocking)

    def render(self) -> str:
        header = {
            Verdict.PRESENTABLE: "MANDATE: clear — presentable for human decision",
            Verdict.REQUIRES_ACKNOWLEDGEMENT: (
                "MANDATE: presentable ONLY with explicit acknowledgement of:"
            ),
            Verdict.BLOCKED: "MANDATE: BLOCKED — this may not be presented as a candidate",
        }[self.verdict]
        lines = [header]
        lines += [b.render() for b in self.breaches]
        lines.append(f"  ({self.rules_checked} rules evaluated)")
        return "\n".join(lines)


class Mandate:
    """Evaluates the standing rules against a finished run."""

    def __init__(self, rules: Sequence[MandateRule] | None = None) -> None:
        self.rules: tuple[MandateRule, ...] = tuple(
            rules if rules is not None else DEFAULT_RULES
        )

    def evaluate(self, ctx: RunContext) -> MandateDecision:
        breaches: list[MandateBreach] = []
        for rule in self.rules:
            try:
                breach = rule.check(ctx)
            except Exception as exc:
                # A rule that cannot be evaluated is treated as breached. The
                # alternative — passing a check that did not run — is how a
                # compliance gate becomes decorative.
                breach = MandateBreach(
                    rule.name,
                    f"rule could not be evaluated ({type(exc).__name__}: {exc}); "
                    "treated as a breach",
                )
            if breach is not None:
                breaches.append(breach)

        if any(b.blocking for b in breaches):
            verdict = Verdict.BLOCKED
        elif breaches:
            verdict = Verdict.REQUIRES_ACKNOWLEDGEMENT
        else:
            verdict = Verdict.PRESENTABLE
        return MandateDecision(
            verdict=verdict, breaches=tuple(breaches), rules_checked=len(self.rules)
        )


@dataclass(slots=True)
class ApprovalRequest:
    """The terminal artefact. The only thing a human is ever asked to act on.

    There is no ``approve()`` method here and no execution path out of this
    class. Approval happens outside the process, by a person, against a
    :attr:`run_digest` they can look up.
    """

    run_digest: str
    summary: str
    decision: MandateDecision
    #: Concerns the human must acknowledge individually, not as a block.
    acknowledgements_required: tuple[str, ...] = ()
    #: Second-signature requirement. Recorded, never satisfied in software.
    four_eyes_required: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        lines = [
            "═" * 64,
            "HUMAN APPROVAL REQUIRED — this pipeline cannot place an order.",
            f"run: {self.run_digest[:12]}",
            "",
            self.summary,
            "",
            self.decision.render(),
        ]
        if self.acknowledgements_required:
            lines += [
                "",
                "Acknowledge each of the following explicitly before acting:",
                *(f"  [ ] {item}" for item in self.acknowledgements_required),
            ]
        if self.four_eyes_required:
            lines += [
                "",
                "Two-person rule: a second reviewer must sign off independently. "
                "This software records the requirement; it cannot satisfy it.",
            ]
        lines.append("Next step: paper trade. Live routing is not available here.")
        return "\n".join(lines)
