"""Graph execution, governance, and the audit record.

The properties under test: a failing stage costs the run only its own findings
and those that depend on it; the mandate refuses rather than scores; and a run
record is reproducible and tamper-evident.
"""

from __future__ import annotations

from typing import Any

import pytest

from axiom.agents.audit import AuditLog, RunRecord
from axiom.agents.base import Agent, AgentReport, AgentRole
from axiom.agents.governance import Mandate, MandateRule, RunContext, Verdict
from axiom.agents.guardrails import NumericFidelityReport
from axiom.agents.orchestrator import AgentGraph, GraphCycleError, Stage
from axiom.agents.registry import AgentRegistry, default_registry
from axiom.core.config import RiskSettings, Settings
from axiom.core.provenance import Provenance


class _Stub(Agent):
    """Records that it ran, and fails on demand."""

    def __init__(self, role: AgentRole, *, boom: bool = False) -> None:
        super().__init__()
        self.role = role  # type: ignore[misc]
        self.boom = boom
        self.ran = False
        self.saw_upstream: dict[AgentRole, AgentReport] = {}

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        self.ran = True
        self.saw_upstream = inputs.get("upstream") or {}
        if self.boom:
            raise ValueError("stage exploded")
        return {"role": self.role.value}, []

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return "prompt"


def _graph(stages: list[Stage]) -> AgentGraph:
    return AgentGraph(stages)


class TestLevelling:
    def test_independent_stages_share_a_level(self) -> None:
        graph = _graph(
            [Stage(_Stub(AgentRole.RESEARCH)), Stage(_Stub(AgentRole.BACKTEST))]
        )
        assert len(graph.levels()) == 1

    def test_a_dependency_creates_a_second_level(self) -> None:
        graph = _graph(
            [
                Stage(_Stub(AgentRole.RESEARCH)),
                Stage(_Stub(AgentRole.DEBATE), depends_on=(AgentRole.RESEARCH,)),
            ]
        )
        levels = graph.levels()
        assert [s.role for s in levels[0]] == [AgentRole.RESEARCH]
        assert [s.role for s in levels[1]] == [AgentRole.DEBATE]

    def test_a_cycle_is_refused_rather_than_hung(self) -> None:
        graph = _graph(
            [
                Stage(_Stub(AgentRole.RESEARCH), depends_on=(AgentRole.DEBATE,)),
                Stage(_Stub(AgentRole.DEBATE), depends_on=(AgentRole.RESEARCH,)),
            ]
        )
        with pytest.raises(GraphCycleError, match="cycle"):
            graph.levels()

    def test_an_unknown_dependency_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not in the graph"):
            _graph([Stage(_Stub(AgentRole.DEBATE), depends_on=(AgentRole.RESEARCH,))])

    def test_a_duplicate_role_is_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            _graph([Stage(_Stub(AgentRole.RESEARCH)), Stage(_Stub(AgentRole.RESEARCH))])


class TestFailureIsolation:
    def test_an_independent_branch_survives_a_failure(self) -> None:
        """One stage's arithmetic blowing up must not cost the run everything."""
        healthy = _Stub(AgentRole.BACKTEST)
        graph = _graph(
            [Stage(_Stub(AgentRole.RESEARCH, boom=True)), Stage(healthy)]
        )
        reports = graph.run({})
        assert healthy.ran
        by_role = {r.role: r for r in reports}
        assert by_role[AgentRole.RESEARCH].failed
        assert not by_role[AgentRole.BACKTEST].failed

    def test_dependents_of_a_failure_are_skipped_with_a_reason(self) -> None:
        dependent = _Stub(AgentRole.DEBATE)
        graph = _graph(
            [
                Stage(_Stub(AgentRole.RESEARCH, boom=True)),
                Stage(dependent, depends_on=(AgentRole.RESEARCH,)),
            ]
        )
        reports = {r.role: r for r in graph.run({})}
        assert not dependent.ran
        assert "research" in reports[AgentRole.DEBATE].error
        assert "skipped" in reports[AgentRole.DEBATE].error

    def test_a_missing_required_input_is_named_not_a_keyerror(self) -> None:
        class _Needy(_Stub):
            requires = ("series",)

        report = _Needy(AgentRole.RESEARCH).run(series=None)
        assert report.failed
        assert "series" in report.error

    def test_the_failure_message_carries_the_exception(self) -> None:
        report = _Stub(AgentRole.RESEARCH, boom=True).run()
        assert "ValueError" in report.error and "exploded" in report.error


class TestUpstreamInjection:
    def test_a_dependent_receives_its_dependency_report(self) -> None:
        """The edge is argument structure, not just ordering."""
        dependent = _Stub(AgentRole.DEBATE)
        graph = _graph(
            [
                Stage(_Stub(AgentRole.RESEARCH)),
                Stage(dependent, depends_on=(AgentRole.RESEARCH,)),
            ]
        )
        graph.run({})
        assert AgentRole.RESEARCH in dependent.saw_upstream
        assert dependent.saw_upstream[AgentRole.RESEARCH].facts["role"] == "research"


class TestRegistry:
    def test_the_default_registry_seats_every_role(self) -> None:
        registry = default_registry()
        assert set(registry.roles) == set(AgentRole)

    def test_shadowing_a_role_is_refused_by_default(self) -> None:
        registry = default_registry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(registry.get(AgentRole.RESEARCH))

    def test_an_explicit_replace_is_permitted(self) -> None:
        registry = default_registry()
        registry.register(registry.get(AgentRole.RESEARCH), replace=True)

    def test_an_unregistered_role_names_what_is_available(self) -> None:
        with pytest.raises(KeyError, match="research"):
            AgentRegistry().get(AgentRole.RESEARCH)


def _context(reports: tuple[AgentReport, ...], **kwargs: Any) -> RunContext:
    defaults: dict[str, Any] = {
        "settings": Settings(),
        "risk_settings": RiskSettings(),
        "has_signal": True,
        "data_is_evidence": True,
        "expected_roles": tuple(r.role for r in reports),
    }
    defaults.update(kwargs)
    return RunContext(reports=reports, **defaults)


class TestMandate:
    def test_a_clean_run_is_presentable(self) -> None:
        reports = (AgentReport(role=AgentRole.RESEARCH),)
        decision = Mandate().evaluate(_context(reports))
        assert decision.verdict is Verdict.PRESENTABLE

    def test_synthetic_data_blocks(self) -> None:
        reports = (AgentReport(role=AgentRole.RESEARCH),)
        decision = Mandate().evaluate(_context(reports, data_is_evidence=False))
        assert decision.blocked
        assert any(b.rule == "evidential_data" for b in decision.breaches)

    def test_the_kill_switch_blocks(self) -> None:
        decision = Mandate().evaluate(
            _context((AgentReport(role=AgentRole.RESEARCH),), settings=Settings(kill_switch=True))
        )
        assert decision.blocked
        assert any(b.rule == "kill_switch" for b in decision.breaches)

    def test_an_unsupported_figure_blocks(self) -> None:
        """A narrative that invented a number cannot back a trade decision."""
        report = AgentReport(
            role=AgentRole.RESEARCH,
            narrative="Profit factor 1.81.",
            guardrail=NumericFidelityReport(unsupported=("1.81",), checked=1),
        )
        decision = Mandate().evaluate(_context((report,)))
        assert decision.blocked
        assert any(b.rule == "numeric_fidelity" for b in decision.breaches)

    def test_a_risk_refusal_blocks(self) -> None:
        report = AgentReport(role=AgentRole.RISK, facts={"approved": False})
        decision = Mandate().evaluate(_context((report,)))
        assert decision.blocked

    def test_a_failed_stage_blocks(self) -> None:
        report = AgentReport(role=AgentRole.RESEARCH, error="boom")
        decision = Mandate().evaluate(_context((report,)))
        assert decision.blocked
        assert any(b.rule == "stage_completeness" for b in decision.breaches)

    def test_a_stage_that_never_ran_blocks(self) -> None:
        decision = Mandate().evaluate(
            _context(
                (AgentReport(role=AgentRole.RESEARCH),),
                expected_roles=(AgentRole.RESEARCH, AgentRole.RISK),
            )
        )
        assert decision.blocked

    def test_a_small_sample_requires_acknowledgement_not_a_block(self) -> None:
        report = AgentReport(role=AgentRole.BACKTEST, facts={"trades": 4})
        decision = Mandate().evaluate(_context((report,)))
        assert decision.verdict is Verdict.REQUIRES_ACKNOWLEDGEMENT
        assert not decision.blocking_breaches

    def test_a_red_team_objection_requires_acknowledgement(self) -> None:
        report = AgentReport(role=AgentRole.RED_TEAM, warnings=["the thesis is thin"])
        decision = Mandate().evaluate(_context((report,)))
        assert decision.verdict is Verdict.REQUIRES_ACKNOWLEDGEMENT

    def test_a_rule_that_raises_is_treated_as_breached(self) -> None:
        """A check that did not run must never read as a check that passed."""

        def _explode(ctx: RunContext) -> None:
            raise RuntimeError("rule is broken")

        mandate = Mandate([MandateRule("broken", "always raises", _explode)])
        decision = mandate.evaluate(_context((AgentReport(role=AgentRole.RESEARCH),)))
        assert decision.blocked
        assert "could not be evaluated" in decision.breaches[0].detail

    def test_the_mandate_can_enumerate_itself(self) -> None:
        """'What do you check?' must not be able to drift from what runs."""
        assert "numeric_fidelity" in {rule.name for rule in Mandate().rules}


def _record(**kwargs: Any) -> RunRecord:
    defaults: dict[str, Any] = {
        "symbol": "ES",
        "timeframe": "1h",
        "strategy": "SilverBullet",
        "data_source": "coinbase",
        "data_is_evidence": True,
        "code_version": "git:abc123",
        "model": "claude-opus-5",
        "reports": (AgentReport(role=AgentRole.RESEARCH, facts={"bars": 100}),),
    }
    defaults.update(kwargs)
    return RunRecord(**defaults)


class TestAuditRecord:
    def test_identical_runs_share_a_digest(self) -> None:
        """This is the reproducibility claim, made checkable."""
        assert _record().digest == _record().digest

    def test_the_clock_does_not_enter_the_digest(self) -> None:
        from datetime import UTC, datetime

        early = _record(started_at=datetime(2026, 1, 1, tzinfo=UTC))
        late = _record(started_at=datetime(2026, 6, 1, tzinfo=UTC))
        assert early.digest == late.digest

    def test_a_changed_fact_changes_the_digest(self) -> None:
        other = _record(
            reports=(AgentReport(role=AgentRole.RESEARCH, facts={"bars": 101}),)
        )
        assert _record().digest != other.digest

    def test_a_changed_code_version_changes_the_digest(self) -> None:
        assert _record().digest != _record(code_version="git:def456").digest

    def test_unsupported_figures_are_counted_across_stages(self) -> None:
        record = _record(
            reports=(
                AgentReport(
                    role=AgentRole.RESEARCH,
                    guardrail=NumericFidelityReport(unsupported=("1.8", "9.9")),
                ),
                AgentReport(
                    role=AgentRole.DEBATE,
                    guardrail=NumericFidelityReport(unsupported=("3.3",)),
                ),
            )
        )
        assert record.unsupported_figure_count == 3


class TestAuditLog:
    def test_a_written_record_reads_back_and_verifies(self, tmp_path: Any) -> None:
        log = AuditLog(tmp_path / "runs.jsonl")
        digest = log.append(_record())
        stored = log.find(digest[:8])
        assert stored is not None
        assert log.verify(stored)

    def test_tampering_is_detectable(self, tmp_path: Any) -> None:
        log = AuditLog(tmp_path / "runs.jsonl")
        log.append(_record())
        stored = next(iter(log))
        stored["data_is_evidence"] = not stored["data_is_evidence"]
        assert not log.verify(stored)

    def test_the_log_appends_rather_than_replaces(self, tmp_path: Any) -> None:
        log = AuditLog(tmp_path / "runs.jsonl")
        log.append(_record())
        log.append(_record(symbol="NQ"))
        assert len(list(log)) == 2

    def test_an_absent_log_reads_as_empty_not_an_error(self, tmp_path: Any) -> None:
        assert list(AuditLog(tmp_path / "missing.jsonl")) == []


class TestReportRendering:
    def test_an_unsupported_figure_is_rendered_after_the_prose(self) -> None:
        """Its whole purpose is to stop a reader trusting the paragraph above."""
        report = AgentReport(
            role=AgentRole.RESEARCH,
            narrative="Profit factor 1.81.",
            model="claude-opus-5",
            guardrail=NumericFidelityReport(unsupported=("1.81",), checked=1),
        )
        rendered = report.render()
        assert rendered.index("1.81") < rendered.index("UNVERIFIED")

    def test_a_report_serialises_for_the_audit_log(self) -> None:
        report = AgentReport(
            role=AgentRole.RESEARCH,
            facts={"bars": 10},
            provenance=Provenance.real("coinbase"),
        )
        payload = report.as_dict()
        assert payload["role"] == "research"
        assert payload["provenance"] == Provenance.real("coinbase").label

    def test_trustworthy_requires_both_success_and_clean_figures(self) -> None:
        assert AgentReport(role=AgentRole.RESEARCH).trustworthy
        assert not AgentReport(role=AgentRole.RESEARCH, error="boom").trustworthy
        assert not AgentReport(
            role=AgentRole.RESEARCH,
            guardrail=NumericFidelityReport(unsupported=("1.8",)),
        ).trustworthy
