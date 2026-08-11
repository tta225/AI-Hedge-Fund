"""Agent pipeline behaviour.

The property under test throughout: agents must produce complete, correct
*facts* with no LLM configured, and must never be the source of a number.
"""

from __future__ import annotations

from axiom.agents.base import AgentRole
from axiom.agents.pipeline import AgentPipeline, ResearchAgent, ReviewAgent
from axiom.core.config import AgentSettings, RiskSettings
from axiom.core.series import OHLCVSeries
from axiom.ict.engine import ICTEngine
from axiom.strategy.ict_strategies import SilverBulletStrategy


def _settings() -> AgentSettings:
    """Settings with no API key — deterministic mode."""
    return AgentSettings(ANTHROPIC_API_KEY="", AXIOM_AGENT_MODEL="claude-opus-5")


class TestDeterministicMode:
    def test_agents_are_disabled_without_a_key(self) -> None:
        assert not _settings().enabled

    def test_research_produces_facts_without_an_llm(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        report = ResearchAgent(_settings()).run(
            series=synthetic_series, state=state, provenance=synthetic_series.provenance
        )
        assert report.role is AgentRole.RESEARCH
        assert report.facts["symbol"] == "ES"
        assert "structural_bias" in report.facts
        assert "confluence_long" in report.facts
        assert not report.used_llm

    def test_report_renders_a_provenance_warning(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        report = ResearchAgent(_settings()).run(
            series=synthetic_series, state=state, provenance=synthetic_series.provenance
        )
        assert "not evidence" in report.render()


class TestPipeline:
    def _run(self, series: OHLCVSeries):  # type: ignore[no-untyped-def]
        return AgentPipeline(_settings()).run(
            series,
            SilverBulletStrategy(),
            risk_settings=RiskSettings(
                account_equity=250_000, max_gross_exposure_pct=2000.0
            ),
        )

    def test_every_seat_reports(self, synthetic_series: OHLCVSeries) -> None:
        """Reports come back in declaration order regardless of finish order.

        Stages run concurrently within a level, so a run that returned them in
        completion order would reorder itself between runs and break both the
        rendered report and the audit digest.
        """
        result = self._run(synthetic_series)
        # ALPHA is absent by design: no panel was supplied, so there is no
        # cross-section to read and the seat is not seated.
        assert {r.role for r in result.reports} == set(AgentRole) - {AgentRole.ALPHA}
        assert [r.role for r in result.reports] == [
            AgentRole.REGIME, AgentRole.RESEARCH, AgentRole.BACKTEST,
            AgentRole.PORTFOLIO, AgentRole.DEBATE, AgentRole.EXECUTION,
            AgentRole.REVIEW, AgentRole.RED_TEAM, AgentRole.RISK,
        ]

    def test_the_order_is_stable_across_runs(self, synthetic_series: OHLCVSeries) -> None:
        first = [r.role for r in self._run(synthetic_series).reports]
        second = [r.role for r in self._run(synthetic_series).reports]
        assert first == second

    def test_backtest_agent_reports_measured_metrics(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        report = self._run(synthetic_series).by_role(AgentRole.BACKTEST)
        assert report is not None
        assert report.facts["costs_included"] is True
        assert "trades" in report.facts

    def test_review_flags_generated_data_as_non_evidence(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        report = self._run(synthetic_series).by_role(AgentRole.REVIEW)
        assert report is not None
        assert report.facts["data_is_evidence"] is False
        assert any("generated data" in w for w in report.warnings)

    def test_pipeline_always_requires_human_approval(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        result = self._run(synthetic_series)
        assert "HUMAN APPROVAL REQUIRED" in result.render()
        assert "cannot place an order" in result.render()

    def test_synthetic_data_is_blocked_by_the_mandate(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Generated bars can never reach a human as a trade candidate."""
        result = self._run(synthetic_series)
        assert result.blocked
        assert any(b.rule == "evidential_data" for b in result.decision.breaches)

    def test_pipeline_has_no_execute_stage(self) -> None:
        """The safety property, asserted structurally.

        No role may name an execution seat, and the pipeline may not expose a
        method that routes anything. Both are checked because the graph is now
        assembled from a registry — a new seat is a config change, and this is
        what stops one from quietly becoming an order path.
        """
        assert not hasattr(AgentPipeline, "execute")
        assert not any(
            name in {"EXECUTE", "ORDER", "ROUTE", "TRADE"}
            for name in AgentRole.__members__
        )
        # EXECUTION analyses cost and feasibility; it must not route.
        from axiom.agents.roles import ExecutionAgent

        assert not hasattr(ExecutionAgent, "execute")
        assert not hasattr(ExecutionAgent, "submit")

    def test_narrative_is_empty_without_an_llm(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Facts are measured; only the prose depends on the model."""
        for report in self._run(synthetic_series).reports:
            assert report.narrative == ""
            assert report.facts, f"{report.role} produced no measured facts"

    def test_review_detects_profit_concentration(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        report = self._run(synthetic_series).by_role(AgentRole.REVIEW)
        assert report is not None
        # The field is computed whenever trades exist; its presence is what
        # matters, since a single dominant winner is the classic false positive.
        if report.facts.get("trades", 0) > 0:
            assert "best_trade_share_of_profit" in report.facts


class TestReviewAgentSkepticism:
    def test_small_sample_is_always_flagged(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from axiom.backtest.engine import Backtester

        result = Backtester(
            SilverBulletStrategy(),
            risk_settings=RiskSettings(
                account_equity=250_000, max_gross_exposure_pct=2000.0
            ),
        ).run(synthetic_series, warmup=100)

        report = ReviewAgent(_settings()).run(
            backtest=result,
            series=synthetic_series,
            state=ICTEngine().analyse(synthetic_series),
            provenance=synthetic_series.provenance,
        )
        if result.report.metrics.get("trades", 0) < 30:
            assert any("30 trades" in w for w in report.warnings)
