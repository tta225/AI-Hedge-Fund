"""Dependency-graph execution for the agent desk.

The previous pipeline ran five agents in a hardcoded sequence, and the shape of
the desk lived in the order of a list literal. That has three costs: stages
that share inputs and could run concurrently run one after another; a sixth
role means editing the pipeline class; and one stage raising takes down the
whole run, discarding the findings of every stage that already succeeded.

:class:`AgentGraph` fixes all three. Stages declare what they depend on, the
graph is levelised topologically, and each level runs concurrently. A stage
that fails is isolated: its dependents are skipped with a report saying *why*,
and every independent branch still completes.

**On concurrency and the deterministic engines.** Only the interpretation layer
runs in parallel. The ICT engine and the backtester run once, up front, in the
calling thread — they are the measurement, and measurement that varies by
thread scheduling is not measurement. What fans out is nine agents reading the
same immutable facts, each making at most one model call through a shared,
lock-guarded :class:`~axiom.agents.runtime.LLMRuntime`.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from axiom.agents.audit import AuditLog, RunRecord, code_version
from axiom.agents.base import Agent, AgentReport, AgentRole
from axiom.agents.governance import (
    ApprovalRequest,
    Mandate,
    MandateDecision,
    RunContext,
    Verdict,
)
from axiom.agents.registry import AgentRegistry, default_registry
from axiom.agents.runtime import LLMRuntime, Usage
from axiom.alpha.ensemble import AlphaEnsemble, BlendedView, default_ensemble
from axiom.alpha.panel import Panel
from axiom.backtest.engine import Backtester, BacktestResult
from axiom.core.config import AgentSettings, ExecutionSettings, RiskSettings, Settings
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.ict.engine import ICTEngine
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.strategy.base import Signal, Strategy


class GraphCycleError(ValueError):
    """The declared dependencies do not form a DAG."""


@dataclass(frozen=True, slots=True)
class Stage:
    """One seat, plus what it must wait for."""

    agent: Agent
    depends_on: tuple[AgentRole, ...] = ()
    #: Wall-clock ceiling for this stage. Exceeded, the stage is reported as
    #: timed out and its dependents are skipped.
    timeout_s: float = 180.0

    @property
    def role(self) -> AgentRole:
        return self.agent.role


class AgentGraph:
    """A DAG of agents, executed level by level with bounded concurrency."""

    def __init__(self, stages: list[Stage], *, max_workers: int = 6) -> None:
        self.stages = list(stages)
        self.max_workers = max_workers
        roles = [s.role for s in self.stages]
        if len(roles) != len(set(roles)):
            duplicates = sorted({r.value for r in roles if roles.count(r) > 1})
            raise ValueError(f"duplicate stages for role(s): {', '.join(duplicates)}")
        self._by_role = {s.role: s for s in self.stages}
        for stage in self.stages:
            for dependency in stage.depends_on:
                if dependency not in self._by_role:
                    raise ValueError(
                        f"{stage.role.value} depends on {dependency.value}, "
                        "which is not in the graph"
                    )

    @property
    def roles(self) -> tuple[AgentRole, ...]:
        return tuple(s.role for s in self.stages)

    def levels(self) -> tuple[tuple[Stage, ...], ...]:
        """Topological levels. Everything within a level may run concurrently.

        Levelising rather than emitting a flat topological order is what makes
        the fan-out available: a plain topological sort would serialise stages
        that have no relationship to each other.
        """
        remaining = {s.role: set(s.depends_on) for s in self.stages}
        levels: list[tuple[Stage, ...]] = []
        while remaining:
            ready = sorted(
                (role for role, deps in remaining.items() if not deps),
                key=lambda r: r.value,
            )
            if not ready:
                stuck = ", ".join(sorted(r.value for r in remaining))
                raise GraphCycleError(
                    f"dependency cycle among: {stuck}. The desk cannot be ordered."
                )
            levels.append(tuple(self._by_role[role] for role in ready))
            for role in ready:
                del remaining[role]
            for deps in remaining.values():
                deps.difference_update(ready)
        return tuple(levels)

    def run(self, inputs: dict[str, Any]) -> tuple[AgentReport, ...]:
        """Execute every stage whose dependencies succeeded.

        Returns reports in declaration order, so output is stable regardless of
        which stage happened to finish first.
        """
        reports: dict[AgentRole, AgentReport] = {}
        skipped: dict[AgentRole, str] = {}

        for level in self.levels():
            runnable: list[Stage] = []
            for stage in level:
                broken = [
                    dep.value
                    for dep in stage.depends_on
                    if dep in skipped or reports[dep].failed
                ]
                if broken:
                    # Running a stage whose inputs are missing produces a
                    # confident report built on nothing. Skipping and saying so
                    # is the only honest option.
                    skipped[stage.role] = (
                        f"skipped — depends on {', '.join(broken)}, which did not report"
                    )
                    continue
                runnable.append(stage)

            if not runnable:
                continue
            if len(runnable) == 1:
                stage = runnable[0]
                reports[stage.role] = self._run_stage(stage, inputs, reports)
                continue

            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(runnable)),
                thread_name_prefix="axiom-agent",
            ) as pool:
                futures = {
                    pool.submit(self._run_stage, stage, inputs, reports): stage
                    for stage in runnable
                }
                deadline = time.monotonic() + max(s.timeout_s for s in runnable)
                pending = set(futures)
                while pending:
                    remaining_time = max(deadline - time.monotonic(), 0.0)
                    done, pending = wait(
                        pending, timeout=remaining_time, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        stage = futures[future]
                        reports[stage.role] = future.result()
                    if not remaining_time and pending:
                        break
                for future in pending:
                    stage = futures[future]
                    # A Python thread cannot be killed. The stage is abandoned,
                    # not stopped — it may still finish and write nothing. The
                    # runtime's own timeout is the real bound; this is the
                    # backstop that keeps the run from hanging forever.
                    future.cancel()
                    reports[stage.role] = AgentReport(
                        role=stage.role,
                        error=f"timed out after {stage.timeout_s:.0f}s (abandoned)",
                        provenance=inputs.get("provenance"),
                    )

        for role, reason in skipped.items():
            reports[role] = AgentReport(
                role=role, error=reason, provenance=inputs.get("provenance")
            )
        return tuple(reports[s.role] for s in self.stages if s.role in reports)

    @staticmethod
    def _run_stage(
        stage: Stage, inputs: dict[str, Any], reports: dict[AgentRole, AgentReport]
    ) -> AgentReport:
        # Upstream reports are passed as inputs so a dependency can be more than
        # ordering — the red team reads research's actual thesis and attacks it.
        upstream = {
            dep: reports[dep] for dep in stage.depends_on if dep in reports
        }
        return stage.agent.run(**inputs, upstream=upstream)


@dataclass(slots=True)
class PipelineResult:
    """Every stage's report, plus the approval request the human acts on."""

    reports: list[AgentReport] = field(default_factory=list)
    approval: ApprovalRequest | None = None
    record: RunRecord | None = None
    usage: Usage = field(default_factory=Usage)

    #: Retained so existing callers keep working — the gate is unconditional.
    approval_required: bool = True

    def by_role(self, role: AgentRole) -> AgentReport | None:
        return next((r for r in self.reports if r.role is role), None)

    @property
    def all_warnings(self) -> list[str]:
        return [w for r in self.reports for w in r.warnings]

    @property
    def decision(self) -> MandateDecision | None:
        return self.approval.decision if self.approval else None

    @property
    def blocked(self) -> bool:
        return self.approval is not None and self.approval.decision.blocked

    @property
    def approval_summary(self) -> str:
        return self.approval.summary if self.approval else ""

    def render(self) -> str:
        body = "\n\n".join(r.render() for r in self.reports)
        footer = self.approval.render() if self.approval else ""
        cost = (
            f"\n\ncost: ${self.usage.cost_usd:,.4f} · {self.usage.calls} model call(s)"
            f"{f', {self.usage.cached_calls} cached' if self.usage.cached_calls else ''}"
            f" · {self.usage.latency_s:.1f}s"
        )
        return f"{body}\n\n{footer}{cost}"


class AgentPipeline:
    """Assembles and runs the desk.

    There is deliberately no ``execute`` stage. The graph terminates in an
    approval request that a human acts on, and the mandate can refuse to
    produce even that.
    """

    def __init__(
        self,
        settings: AgentSettings | None = None,
        *,
        runtime: LLMRuntime | None = None,
        registry: AgentRegistry | None = None,
        mandate: Mandate | None = None,
        audit_log: AuditLog | None = None,
        max_workers: int = 6,
    ) -> None:
        self.settings = settings or AgentSettings()
        self.registry = registry or default_registry()
        self.mandate = mandate or Mandate()
        self.audit_log = audit_log
        self.max_workers = max_workers
        self.runtime = runtime or (
            LLMRuntime(model=self.settings.agent_model, api_key=self.settings.anthropic_api_key)
            if self.settings.enabled
            else None
        )

    def _agent(self, role: AgentRole) -> Agent:
        return self.registry.get(role)(self.settings, self.runtime)

    @staticmethod
    def _blend_alpha(
        panel: Panel | None, ensemble: AlphaEnsemble | None
    ) -> list[BlendedView] | None:
        """Run the cross-sectional roster at the panel's last bar.

        Returns ``None`` — not an empty list — when no panel was supplied. The
        distinction is load-bearing: an empty list means the roster looked and
        found nothing, which is a finding; ``None`` means nobody looked, which
        is not, and the alpha seat is left unseated for it.
        """
        if panel is None:
            return None
        roster = ensemble or default_ensemble()
        return roster.run(panel, len(panel))

    @staticmethod
    def _fit_regimes(series: OHLCVSeries) -> object | None:
        """Causal regime series, or ``None`` when the model will not fit.

        Failure is returned rather than raised: not knowing the regime is a
        finding the regime seat reports, not a reason to lose the other eight.
        """
        from axiom.agents.roles import RegimeAgent
        from axiom.quant.regime import RegimeModel

        try:
            return RegimeModel(
                refit_every=RegimeAgent.refit_every,
                max_lookback=RegimeAgent.max_lookback,
            ).fit_causal(series)
        except Exception:
            return None

    def build_graph(self, *, with_alpha: bool = False) -> AgentGraph:
        """The standing desk.

        The dependency edges are the argument structure, not just ordering:
        debate needs research's read before it can dispute it, and the red team
        needs both a thesis and a measured track record before it has anything
        to attack.
        """
        role = AgentRole
        stages = [
            Stage(self._agent(role.REGIME)),
            Stage(self._agent(role.RESEARCH)),
            Stage(self._agent(role.BACKTEST)),
            Stage(self._agent(role.PORTFOLIO)),
            Stage(self._agent(role.DEBATE), depends_on=(role.RESEARCH,)),
            Stage(self._agent(role.EXECUTION), depends_on=(role.BACKTEST,)),
            Stage(self._agent(role.REVIEW), depends_on=(role.BACKTEST,)),
            Stage(
                self._agent(role.RED_TEAM),
                depends_on=(role.RESEARCH, role.BACKTEST),
            ),
            Stage(self._agent(role.RISK), depends_on=(role.PORTFOLIO,)),
        ]
        if with_alpha:
            # Seated only when there is a cross-section to read. A seat that
            # reports "no data" on every run trains a reader to skip it, and
            # the mandate's completeness rule would block every single-
            # instrument run for the absence of a view nobody could form.
            stages.append(Stage(self._agent(role.ALPHA)))
        return AgentGraph(stages, max_workers=self.max_workers)

    def run(
        self,
        series: OHLCVSeries,
        strategy: Strategy,
        *,
        risk_settings: RiskSettings | None = None,
        execution_settings: ExecutionSettings | None = None,
        settings: Settings | None = None,
        warmup: int = 100,
        panel: Panel | None = None,
        ensemble: AlphaEnsemble | None = None,
        context: dict[str, str] | None = None,
    ) -> PipelineResult:
        risk_config = risk_settings or RiskSettings()
        execution_config = execution_settings or ExecutionSettings()
        global_settings = settings or Settings()

        # Measurement first, single-threaded, before any agent sees anything.
        # Measurement that varies with thread scheduling is not measurement.
        state = ICTEngine().analyse(series)
        provenance: Provenance = series.provenance
        backtest_result = Backtester(strategy, risk_settings=risk_config).run(
            series, warmup=warmup
        )
        regimes = self._fit_regimes(series)
        views = self._blend_alpha(panel, ensemble)
        signal = backtest_result.signals[-1] if backtest_result.signals else None
        risk_manager = RiskManager(risk_config)
        portfolio = Portfolio(starting_cash=risk_config.account_equity)

        graph = self.build_graph(with_alpha=views is not None)
        reports = graph.run(
            {
                "series": series,
                "state": state,
                "provenance": provenance,
                "backtest": backtest_result,
                "regimes": regimes,
                "views": views,
                "signal": signal,
                "risk": risk_manager,
                "risk_settings": risk_config,
                "execution": execution_config,
                "portfolio": portfolio,
            }
        )

        usage = self.runtime.usage if self.runtime else Usage()
        decision = self.mandate.evaluate(
            RunContext(
                reports=reports,
                settings=global_settings,
                risk_settings=risk_config,
                has_signal=signal is not None,
                data_is_evidence=backtest_result.report.is_evidence,
                expected_roles=graph.roles,
            )
        )

        record = RunRecord(
            symbol=series.instrument.symbol,
            timeframe=str(series.timeframe),
            strategy=type(strategy).__name__,
            data_source=provenance.label,
            data_is_evidence=provenance.is_evidential,
            code_version=code_version(),
            model=self.settings.agent_model if self.settings.enabled else "",
            reports=reports,
            usage=usage,
            approval_summary=_summarise(signal, backtest_result, provenance),
            context=dict(context or {}),
        )
        if self.audit_log is not None:
            self.audit_log.append(record)

        approval = ApprovalRequest(
            run_digest=record.digest,
            summary=record.approval_summary,
            decision=decision,
            acknowledgements_required=tuple(
                b.detail for b in decision.breaches if not b.blocking
            ),
            warnings=tuple(w for r in reports for w in r.warnings),
        )
        return PipelineResult(
            reports=list(reports),
            approval=approval,
            record=record,
            usage=usage,
            approval_required=decision.verdict is not Verdict.BLOCKED,
        )


def _summarise(
    signal: Signal | None, result: BacktestResult, provenance: Provenance
) -> str:
    lines: list[str] = []
    if not provenance.is_evidential:
        lines.append(f"  Data is {provenance.kind.value.upper()}. Do not act on this.")
    if signal is None:
        lines.append("  No candidate signal. Nothing to approve.")
    else:
        if signal.reward_risk:
            lines.append(
                f"  Candidate: {signal.direction} @ {signal.entry} "
                f"stop {signal.stop} target {signal.primary_target} "
                f"(R:R {signal.reward_risk:.2f})"
            )
        else:
            lines.append(f"  Candidate: {signal.direction} @ {signal.entry}")
        lines.append(f"  Rationale: {signal.rationale}")
    lines.append(
        f"  Backtest sample: {int(result.report.metrics.get('trades', 0))} trades."
    )
    return "\n".join(lines)
