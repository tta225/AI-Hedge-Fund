"""The five agents and the pipeline that sequences them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axiom.agents.base import Agent, AgentReport, AgentRole
from axiom.backtest.engine import Backtester, BacktestResult
from axiom.core.config import AgentSettings, RiskSettings
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import ICTEngine, confluence_score
from axiom.ict.models import ICTState
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.strategy.base import Signal, Strategy


class ResearchAgent(Agent):
    """Summarises the current structural read. Facts come from the ICT engine."""

    role = AgentRole.RESEARCH

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        series: OHLCVSeries = inputs["series"]
        state: ICTState = inputs["state"]
        price = float(series.closes[state.index])

        facts: dict[str, Any] = {
            "symbol": series.instrument.symbol,
            "timeframe": str(series.timeframe),
            "bars": len(series),
            "as_of": state.as_of.isoformat(),
            "session": state.session,
            "last_price": price,
            "structural_bias": str(state.bias),
            "structure_events": len(state.structure_events),
            "active_fvgs": len(state.active_fvgs()),
            "active_order_blocks": len(state.active_order_blocks()),
            "unswept_pools": len(state.unswept_pools()),
            "recent_sweeps": len([s for s in state.sweeps if state.index - s.confirmed_index <= 20]),
            "confluence_long": round(confluence_score(state, price, Direction.BULLISH), 3),
            "confluence_short": round(confluence_score(state, price, Direction.BEARISH), 3),
        }

        if state.dealing_range is not None:
            dealing = state.dealing_range
            facts["range_high"] = dealing.high
            facts["range_low"] = dealing.low
            facts["range_position"] = round(dealing.position_of(price), 3)
            facts["premium_or_discount"] = "premium" if dealing.is_premium(price) else "discount"

        draw_up = state.nearest_draw(price, Direction.BULLISH)
        draw_down = state.nearest_draw(price, Direction.BEARISH)
        facts["draw_above"] = draw_up.price if draw_up else None
        facts["draw_below"] = draw_down.price if draw_down else None

        warnings: list[str] = []
        if state.bias is Direction.NEUTRAL:
            warnings.append("No structural bias established — no directional edge to trade.")
        if not state.active_fvgs() and not state.active_order_blocks():
            warnings.append("No live imbalances or order blocks; nothing to retrace into.")
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "Summarise the structural picture for a trader. Use ONLY these "
            "measured values; do not add prices or levels that are not listed.\n\n"
            + _render_facts(facts)
            + "\n\nCover: prevailing bias, where price sits relative to the range, "
            "and the most likely draw on liquidity. Be brief."
        )


class DebateAgent(Agent):
    """Argues both sides. The facts are the same; the point is disagreement."""

    role = AgentRole.DEBATE

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        state: ICTState = inputs["state"]
        series: OHLCVSeries = inputs["series"]
        price = float(series.closes[state.index])

        bull = confluence_score(state, price, Direction.BULLISH)
        bear = confluence_score(state, price, Direction.BEARISH)
        facts = {
            "confluence_long": round(bull, 3),
            "confluence_short": round(bear, 3),
            "edge_margin": round(abs(bull - bear), 3),
            "favoured_side": "long" if bull > bear else "short" if bear > bull else "neither",
            "buyside_pools_unswept": len(
                [p for p in state.unswept_pools() if p.price > price]
            ),
            "sellside_pools_unswept": len(
                [p for p in state.unswept_pools() if p.price < price]
            ),
        }
        warnings = []
        if abs(bull - bear) < 0.1:
            warnings.append(
                f"Confluence is near-symmetric ({bull:.2f} vs {bear:.2f}); the "
                "structure does not favour either side. This is a no-trade condition."
            )
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "Argue the strongest bull case AND the strongest bear case from these "
            "measured values. Then list explicit no-trade conditions and what "
            "would invalidate each case. Be skeptical, not agreeable.\n\n"
            + _render_facts(facts)
        )


class BacktestAgent(Agent):
    """Runs the real backtester. Every number here is measured, never estimated."""

    role = AgentRole.BACKTEST

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        result: BacktestResult = inputs["backtest"]
        report = result.report
        keys = (
            "trades", "win_rate_pct", "profit_factor", "expectancy", "avg_r",
            "avg_win", "avg_loss", "max_drawdown_pct", "sharpe", "sortino",
            "total_return_pct", "total_commission",
        )
        facts: dict[str, Any] = {k: report.metrics[k] for k in keys if k in report.metrics}
        facts["signals_generated"] = len(result.signals)
        facts["costs_included"] = True

        warnings = list(report.notes[-1:]) if report.notes else []
        if report.warning:
            warnings.insert(0, report.warning)
        if facts.get("trades", 0) < 30:
            warnings.append(
                "Sample too small for the win rate or profit factor to mean anything."
            )
        if result.risk_rejections:
            top = next(iter(result.risk_rejections.items()))
            warnings.append(f"Most common risk rejection ({top[1]}×): {top[0]}")
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "Interpret these backtest results. Costs are already included. State "
            "plainly whether the sample supports any conclusion, and identify the "
            "most likely ways these numbers mislead.\n\n" + _render_facts(facts)
        )


class RiskAgent(Agent):
    """Sizes the trade against hard limits. All arithmetic from the risk manager."""

    role = AgentRole.RISK

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        signal: Signal | None = inputs.get("signal")
        risk: RiskManager = inputs["risk"]
        portfolio: Portfolio = inputs["portfolio"]
        series: OHLCVSeries = inputs["series"]
        state: ICTState = inputs["state"]

        snapshot = risk.snapshot(portfolio, state.as_of)
        facts: dict[str, Any] = dict(snapshot)

        warnings: list[str] = []
        if signal is None:
            facts["decision"] = "no signal to size"
            return facts, warnings

        decision = risk.evaluate(
            instrument=series.instrument,
            direction=signal.direction,
            entry=signal.entry,
            stop=signal.stop,
            portfolio=portfolio,
            timestamp=state.as_of,
        )
        facts.update(
            {
                "direction": str(signal.direction),
                "entry": signal.entry,
                "stop": signal.stop,
                "target": signal.primary_target,
                "reward_risk": (
                    round(signal.reward_risk, 2) if signal.reward_risk else None
                ),
                "approved": decision.approved,
                "quantity": decision.quantity,
                "cash_at_risk": decision.sizing.risk_cash if decision.sizing else 0.0,
                "decision": decision.explain(),
            }
        )
        if not decision.approved:
            warnings.extend(decision.reasons)
        if signal.reward_risk is not None and signal.reward_risk < 2.0:
            warnings.append(f"Reward:risk of {signal.reward_risk:.2f} is below 2.0.")
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "Review this position-sizing decision. Confirm nothing exceeds the "
            "stated limits, and flag anything that concerns you. Do not "
            "recalculate the size — verify the reasoning.\n\n" + _render_facts(facts)
        )


class ReviewAgent(Agent):
    """Honest post-hoc report. Explicitly looks for overfitting."""

    role = AgentRole.REVIEW

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        result: BacktestResult = inputs["backtest"]
        report = result.report
        metrics = report.metrics

        facts: dict[str, Any] = {
            "data_is_evidence": report.is_evidence,
            "data_source": report.provenance.label,
            "trades": metrics.get("trades", 0),
            "total_return_pct": metrics.get("total_return_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "profit_factor": metrics.get("profit_factor"),
            "avg_r": metrics.get("avg_r"),
        }

        warnings: list[str] = []
        if not report.is_evidence:
            warnings.append(
                "These results come from generated data. They demonstrate that the "
                "code runs; they say nothing whatsoever about profitability."
            )
        if metrics.get("trades", 0) < 30:
            warnings.append("Fewer than 30 trades — no statistical conclusion is available.")

        # Concentration: one outsized winner carrying the whole result is the
        # most common way a backtest flatters a strategy that does not work.
        if result.trades:
            pnls = sorted((t.pnl for t in result.trades), reverse=True)
            total = sum(p for p in pnls if p > 0)
            if total > 0 and pnls[0] / total > 0.5:
                warnings.append(
                    f"The single best trade produced {pnls[0] / total:.0%} of all gross "
                    "profit — the result depends on one outcome, not an edge."
                )
            facts["best_trade_share_of_profit"] = (
                round(pnls[0] / total, 3) if total > 0 else None
            )

        drawdown = metrics.get("max_drawdown_pct") or 0.0
        returns = metrics.get("total_return_pct") or 0.0
        if drawdown > 0 and returns / drawdown < 1.0:
            warnings.append(
                f"Return ({returns:.2f}%) does not exceed max drawdown ({drawdown:.2f}%)."
            )
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "Write a plain-English, skeptical performance review. Address: what "
            "worked, what is concerning, drawdown risk, and whether this looks "
            "curve-fitted. No hype. If the data is not real, lead with that.\n\n"
            + _render_facts(facts)
        )


@dataclass(slots=True)
class PipelineResult:
    """Every stage's report, plus the approval request the human acts on."""

    reports: list[AgentReport] = field(default_factory=list)
    approval_required: bool = True
    approval_summary: str = ""

    def by_role(self, role: AgentRole) -> AgentReport | None:
        return next((r for r in self.reports if r.role is role), None)

    @property
    def all_warnings(self) -> list[str]:
        return [w for r in self.reports for w in r.warnings]

    def render(self) -> str:
        body = "\n\n".join(r.render() for r in self.reports)
        gate = (
            "\n\n"
            + "═" * 56
            + "\nHUMAN APPROVAL REQUIRED — this pipeline cannot place an order.\n"
            + self.approval_summary
        )
        return body + gate


class AgentPipeline:
    """Sequences Research → Debate → Backtest → Risk → Review.

    There is deliberately no ``execute`` stage. The pipeline terminates in an
    approval request that a human acts on.
    """

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self.settings = settings or AgentSettings()
        self.research = ResearchAgent(self.settings)
        self.debate = DebateAgent(self.settings)
        self.backtest = BacktestAgent(self.settings)
        self.risk = RiskAgent(self.settings)
        self.review = ReviewAgent(self.settings)

    def run(
        self,
        series: OHLCVSeries,
        strategy: Strategy,
        *,
        risk_settings: RiskSettings | None = None,
        warmup: int = 100,
    ) -> PipelineResult:
        settings = risk_settings or RiskSettings()
        state = ICTEngine().analyse(series)
        provenance: Provenance = series.provenance

        backtest_result = Backtester(strategy, risk_settings=settings).run(
            series, warmup=warmup
        )

        # The candidate signal is the strategy's most recent proposal, if any.
        signal = backtest_result.signals[-1] if backtest_result.signals else None
        risk_manager = RiskManager(settings)
        portfolio = Portfolio(starting_cash=settings.account_equity)

        common = {"series": series, "state": state, "provenance": provenance}
        reports = [
            self.research.run(**common),
            self.debate.run(**common),
            self.backtest.run(backtest=backtest_result, **common),
            self.risk.run(
                signal=signal, risk=risk_manager, portfolio=portfolio, **common
            ),
            self.review.run(backtest=backtest_result, **common),
        ]

        return PipelineResult(
            reports=reports,
            approval_required=True,
            approval_summary=self._approval_summary(
                signal, backtest_result, provenance
            ),
        )

    @staticmethod
    def _approval_summary(
        signal: Signal | None, result: BacktestResult, provenance: Provenance
    ) -> str:
        lines = []
        if not provenance.is_evidential:
            lines.append(
                f"  Data is {provenance.kind.value.upper()}. Do not act on this."
            )
        if signal is None:
            lines.append("  No candidate signal. Nothing to approve.")
        else:
            lines.append(
                f"  Candidate: {signal.direction} @ {signal.entry} "
                f"stop {signal.stop} target {signal.primary_target} "
                f"(R:R {signal.reward_risk:.2f})"
                if signal.reward_risk
                else f"  Candidate: {signal.direction} @ {signal.entry}"
            )
            lines.append(f"  Rationale: {signal.rationale}")
        lines.append(
            f"  Backtest sample: {int(result.report.metrics.get('trades', 0))} trades."
        )
        lines.append("  Next step: paper trade. Live routing is not available here.")
        return "\n".join(lines)


def _render_facts(facts: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in facts.items() if value is not None)
