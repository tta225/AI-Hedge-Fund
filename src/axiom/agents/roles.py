"""The seats on the desk.

Nine roles, each of which computes its own measured facts and then asks the
model to interpret only those. Four of them are adversarial or accounting roles
that exist because a desk that only asks "is this a good trade?" gets one
answer: research states the read, debate argues both sides, red team attacks the
thesis, execution prices the friction, portfolio checks concentration, risk
sizes it, and review looks for the ways the backtest flattered it.

Every agent here obeys the same contract: :meth:`gather_facts` does arithmetic
and touches no model; :meth:`prompt_for` renders those facts into a question.
"""

from __future__ import annotations

import math
from typing import Any

from axiom.agents.base import Agent, AgentRole
from axiom.backtest.engine import BacktestResult
from axiom.core.config import ExecutionSettings, RiskSettings
from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import confluence_score
from axiom.ict.models import ICTState
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.strategy.base import Signal


def render_facts(facts: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in facts.items() if value is not None)


def _finite(value: Any) -> float | None:
    """Drop NaN and infinity before a number reaches a prompt.

    ``profit_factor`` is NaN whenever there were no losing trades. Printed into
    a prompt it reads as a missing value at best and invites the model to
    reason about "nan" at worst; dropped, the absence is the message.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


class ResearchAgent(Agent):
    """Summarises the current structural read. Facts come from the ICT engine."""

    role = AgentRole.RESEARCH
    requires = ("series", "state")

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
            "recent_sweeps": len(
                [s for s in state.sweeps if state.index - s.confirmed_index <= 20]
            ),
            "confluence_long": round(confluence_score(state, price, Direction.BULLISH), 3),
            "confluence_short": round(confluence_score(state, price, Direction.BEARISH), 3),
        }

        if state.dealing_range is not None:
            dealing = state.dealing_range
            facts["range_high"] = dealing.high
            facts["range_low"] = dealing.low
            facts["range_position"] = round(dealing.position_of(price), 3)
            facts["premium_or_discount"] = (
                "premium" if dealing.is_premium(price) else "discount"
            )

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
            + render_facts(facts)
            + "\n\nCover: prevailing bias, where price sits relative to the range, "
            "and the most likely draw on liquidity. Be brief."
        )


class RegimeAgent(Agent):
    """States which market regime is in force, and whether it suits the strategy.

    This runs first and everything downstream is read in its light. A pattern
    edge measured across a whole sample is an average over regimes it works in
    and regimes it is shredded by; knowing which one is live is the difference
    between an edge and an average.
    """

    role = AgentRole.REGIME
    requires = ("series",)

    #: Coarser than the research default. This seat needs the *current* regime,
    #: not a publication-grade history, and an expanding-window refit every 500
    #: bars costs ~25s on a year of 15m data — paid on every pipeline run.
    #: The pipeline normally injects a precomputed series and this is unused.
    refit_every = 1_000
    max_lookback = 2_000

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        from axiom.quant.regime import RegimeLabel, RegimeModel

        series: OHLCVSeries = inputs["series"]
        facts: dict[str, Any] = {"symbol": series.instrument.symbol, "bars": len(series)}
        warnings: list[str] = []

        try:
            # Measurement is the pipeline's job and it hands the result down.
            # Fitting here is the fallback for an agent run standalone.
            regimes = inputs.get("regimes")
            if regimes is None:
                regimes = RegimeModel(
                    refit_every=self.refit_every, max_lookback=self.max_lookback
                ).fit_causal(series)
        except Exception as exc:
            # Not knowing the regime is a finding, not a failure. Reporting it
            # as "unknown" is honest; guessing one would be worse than silence.
            facts["regime"] = "unknown"
            facts["regime_model"] = f"unavailable: {type(exc).__name__}"
            warnings.append(
                "Regime model did not fit; treat every downstream read as "
                "regime-blind rather than regime-appropriate."
            )
            return facts, warnings

        index = min(len(series) - 1, len(regimes) - 1)
        label = regimes.label_at(index)
        confidence = regimes.confidence_at(index)
        facts["regime"] = label.value
        facts["regime_confidence"] = round(float(confidence), 3)
        facts["regime_is_tradable_trend"] = label.is_tradable_trend
        facts["warmup_bars"] = regimes.warmup_end
        facts["refits"] = len(regimes.refit_points)
        for occupied, share in regimes.occupancy().items():
            facts[f"occupancy_{occupied.value}"] = round(float(share), 3)

        if label is RegimeLabel.UNKNOWN:
            warnings.append(
                f"No regime established at the evaluation bar — the model needs "
                f"{regimes.warmup_end} bars and this read sits inside that window."
            )
        elif label is RegimeLabel.VOLATILE:
            warnings.append(
                "Volatile regime: directional structure rules degrade here, and "
                "stops sized for quiet conditions are the usual casualty."
            )
        if confidence < 0.6:
            warnings.append(
                f"Regime confidence is {confidence:.2f} — the classification is "
                "closer to a coin flip than a call."
            )
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "State which regime is in force and what it implies for a "
            "structure-following strategy. Say plainly if the regime is "
            "unknown or weakly identified.\n\n" + render_facts(facts)
        )


class DebateAgent(Agent):
    """Argues both sides. The facts are the same; the point is disagreement."""

    role = AgentRole.DEBATE
    requires = ("series", "state")

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
            + render_facts(facts)
        )


class RedTeamAgent(Agent):
    """Attacks the thesis. Its job is to be wrong about the trade being good.

    Debate argues both sides and can still land somewhere agreeable. This seat
    is asked only for the case against, because an adversary with a mandate
    finds objections a balanced reviewer talks itself out of. It has no vote —
    it produces objections that a human weighs at the approval gate.
    """

    role = AgentRole.RED_TEAM
    requires = ("series", "state", "backtest")

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        series: OHLCVSeries = inputs["series"]
        state: ICTState = inputs["state"]
        result: BacktestResult = inputs["backtest"]
        metrics = result.report.metrics
        price = float(series.closes[state.index])

        bull = confluence_score(state, price, Direction.BULLISH)
        bear = confluence_score(state, price, Direction.BEARISH)
        trades = int(metrics.get("trades", 0) or 0)

        facts: dict[str, Any] = {
            "data_is_evidence": result.report.is_evidence,
            "trades": trades,
            "edge_margin": round(abs(bull - bear), 3),
            "signals_generated": len(result.signals),
            "signal_to_trade_ratio": (
                round(len(result.signals) / trades, 2) if trades else None
            ),
            "profit_factor": _finite(metrics.get("profit_factor")),
            "sharpe": _finite(metrics.get("sharpe")),
            "max_drawdown_pct": _finite(metrics.get("max_drawdown_pct")),
            "total_commission": _finite(metrics.get("total_commission")),
            "net_pnl": _finite(metrics.get("net_pnl")),
        }

        # Costs exceeding profits is the single most common way a "profitable"
        # ICT backtest is actually a transfer to the broker.
        commission = facts.get("total_commission") or 0.0
        net = facts.get("net_pnl") or 0.0
        if commission > 0:
            facts["commission_vs_net_pnl_x"] = round(commission / abs(net), 2) if net else None

        warnings: list[str] = []
        if not result.report.is_evidence:
            warnings.append(
                "The entire case rests on generated data. There is nothing here "
                "to attack and nothing here to believe."
            )
        if commission > abs(net) and net > 0:
            warnings.append(
                f"Commission ({commission:,.0f}) exceeds net P&L ({net:,.0f}) — "
                "the edge is smaller than the cost of harvesting it."
            )
        if trades and len(result.signals) / max(trades, 1) > 5:
            warnings.append(
                "Most signals never became trades. The reported metrics describe "
                "the survivors, not the strategy as specified."
            )
        if facts["edge_margin"] is not None and facts["edge_margin"] < 0.15:
            warnings.append(
                "The directional edge is thin enough that ordinary noise flips it."
            )
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        prompt = (
            "You are the red team. Your job is to argue that this trade should "
            "NOT be taken. Build the strongest case against it from these "
            "measured values, name the assumption most likely to be false, and "
            "state what evidence would change your mind. Do not balance your "
            "answer — someone else already argued the other side.\n\n"
            + render_facts(facts)
        )
        # The graph hands this seat the research report so it can attack the
        # actual thesis rather than a generic one. Absent (deterministic mode,
        # or research failed), it falls back to attacking the facts alone.
        upstream: dict[AgentRole, Any] = inputs.get("upstream") or {}
        research = upstream.get(AgentRole.RESEARCH)
        if research is not None and research.narrative:
            prompt += (
                "\n\nThe research desk's thesis, which you are attacking — treat "
                "it as a claim to be tested, not as evidence, and do not adopt "
                "any figure that appears only here:\n"
                + research.narrative
            )
        return prompt


class BacktestAgent(Agent):
    """Runs the real backtester. Every number here is measured, never estimated."""

    role = AgentRole.BACKTEST
    requires = ("backtest",)

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
            "most likely ways these numbers mislead.\n\n" + render_facts(facts)
        )


class ExecutionAgent(Agent):
    """Prices the friction between the signal and the fill.

    A strategy is only as good as the price it actually gets. This seat measures
    what the modelled costs did to the result and how sensitive the edge is to
    those assumptions being optimistic — which, on a free IEX feed with a
    fixed-tick slippage model, they generally are.
    """

    role = AgentRole.EXECUTION
    requires = ("backtest",)

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        result: BacktestResult = inputs["backtest"]
        execution: ExecutionSettings = inputs.get("execution") or ExecutionSettings()
        metrics = result.report.metrics
        trades = result.trades

        gross_profit = _finite(metrics.get("gross_profit")) or 0.0
        gross_loss = _finite(metrics.get("gross_loss")) or 0.0
        commission = _finite(metrics.get("total_commission")) or 0.0
        net = _finite(metrics.get("net_pnl")) or 0.0

        facts: dict[str, Any] = {
            "commission_per_unit": execution.commission_per_unit,
            "slippage_ticks": execution.slippage_ticks,
            "trades": len(trades),
            "total_commission": commission,
            "gross_pnl": round(gross_profit - gross_loss, 2),
            "net_pnl": round(net, 2),
            "commission_per_trade": round(commission / len(trades), 2) if trades else None,
            "cost_share_of_gross_pct": (
                round(commission / gross_profit * 100.0, 2) if gross_profit > 0 else None
            ),
            "avg_duration_hours": _finite(metrics.get("avg_duration_hours")),
        }

        # The honest question is not "did it survive these costs" but "how much
        # worse would costs have to be to kill it". Doubling is the crude,
        # defensible answer for a free single-exchange feed.
        if trades:
            doubled = net - commission
            facts["net_pnl_at_double_cost"] = round(doubled, 2)
            facts["survives_double_cost"] = doubled > 0

        warnings: list[str] = []
        if trades and not facts.get("survives_double_cost", True):
            warnings.append(
                "The result does not survive doubling execution costs. On the "
                "free feed, doubled costs are a plausible reality, not a stress case."
            )
        if facts.get("cost_share_of_gross_pct") and facts["cost_share_of_gross_pct"] > 30:
            warnings.append(
                f"Costs consume {facts['cost_share_of_gross_pct']:.0f}% of gross "
                "profit — the strategy is mostly paying to trade."
            )
        if execution.slippage_ticks <= 0:
            warnings.append(
                "Slippage is modelled at zero. Every fill is assumed perfect, "
                "which no venue provides."
            )
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "Assess execution feasibility. These are modelled costs, not "
            "realised ones. State how much of the edge survives the friction "
            "and which cost assumption you would challenge first.\n\n"
            + render_facts(facts)
        )


class PortfolioAgent(Agent):
    """Looks at the book, not the trade. Concentration is the failure mode.

    Every other seat evaluates one candidate in isolation. This one asks what
    happens when it is added to what is already held — because a book of five
    uncorrelated-looking positions that are all long the same factor is one
    position with five commissions.
    """

    role = AgentRole.PORTFOLIO
    requires = ("portfolio", "risk_settings")

    def gather_facts(self, **inputs: Any) -> tuple[dict[str, Any], list[str]]:
        portfolio: Portfolio = inputs["portfolio"]
        settings: RiskSettings = inputs["risk_settings"]
        signal: Signal | None = inputs.get("signal")

        open_positions = list(portfolio.open_positions)
        equity = portfolio.equity if portfolio.equity > 0 else settings.account_equity
        gross = portfolio.gross_exposure

        facts: dict[str, Any] = {
            "equity": round(equity, 2),
            "open_positions": len(open_positions),
            "max_positions": settings.max_positions,
            "capacity_remaining": max(settings.max_positions - len(open_positions), 0),
            "gross_exposure": round(gross, 2),
            "gross_exposure_pct": round(gross / equity * 100.0, 2) if equity else None,
            "max_gross_exposure_pct": settings.max_gross_exposure_pct,
        }

        symbols = [p.instrument.symbol for p in open_positions]
        facts["held_symbols"] = ", ".join(sorted(symbols)) if symbols else "none"
        facts["distinct_symbols"] = len(set(symbols))

        # Same-name concentration is the one form of correlation measurable
        # without a covariance estimate, and the one most often overlooked.
        if signal is not None:
            candidate = inputs["series"].instrument.symbol if inputs.get("series") else ""
            facts["candidate_symbol"] = candidate
            facts["candidate_direction"] = str(signal.direction)
            facts["already_held"] = candidate in symbols

        directions = [str(p.side) for p in open_positions]
        if directions:
            dominant = max(set(directions), key=directions.count)
            share = directions.count(dominant) / len(directions)
            facts["dominant_side"] = dominant
            facts["dominant_side_share"] = round(share, 3)

        warnings: list[str] = []
        if facts.get("already_held"):
            warnings.append(
                "The book already holds this instrument. Adding to it concentrates "
                "rather than diversifies, whatever the per-trade risk says."
            )
        if facts.get("dominant_side_share", 0) >= 0.8 and len(open_positions) >= 3:
            warnings.append(
                f"{facts['dominant_side_share']:.0%} of open positions are "
                f"{facts['dominant_side']} — this is one directional bet, not a book."
            )
        if facts.get("gross_exposure_pct") and (
            facts["gross_exposure_pct"] > settings.max_gross_exposure_pct * 0.8
        ):
            warnings.append(
                "Gross exposure is within 20% of its cap; the next position may "
                "be refused regardless of its merit."
            )
        return facts, warnings

    def prompt_for(self, facts: dict[str, Any], **inputs: Any) -> str:
        return (
            "Assess this candidate as an addition to the existing book, not on "
            "its own merits. Concentration and correlation are your concern.\n\n"
            + render_facts(facts)
        )


class RiskAgent(Agent):
    """Sizes the trade against hard limits. All arithmetic from the risk manager."""

    role = AgentRole.RISK
    requires = ("series", "state", "risk", "portfolio")

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
            "recalculate the size — verify the reasoning.\n\n" + render_facts(facts)
        )


class ReviewAgent(Agent):
    """Honest post-hoc report. Explicitly looks for overfitting."""

    role = AgentRole.REVIEW
    requires = ("backtest",)

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
            + render_facts(facts)
        )
