"""Turn a pipeline run into a JSON-serialisable payload.

Everything here is extraction, never computation. If a value is not already on
the :class:`~axiom.agents.pipeline.PipelineResult`, the risk manager's snapshot,
or the performance report, it does not belong in the payload — a report that
quietly recalculates something is a report that can disagree with the run it
claims to describe.

NaN is preserved as ``None`` rather than zero. Metrics deliberately return NaN
when a sample is too short to support them (Sharpe under twenty daily
observations, profit factor with no losing trade), and rendering that as ``0.00``
would turn "we cannot say" into a confident, wrong claim.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from axiom.agents.pipeline import PipelineResult
from axiom.backtest.engine import BacktestResult
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import confluence_score
from axiom.ict.models import ICTState, LiquidityKind
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.strategy.base import Signal

#: Points kept in a rendered curve. Enough to show shape, small enough that the
#: HTML stays a file you can email rather than a payload you have to host.
CURVE_POINTS = 400

PAYLOAD_VERSION = 1


def _clean(value: Any) -> Any:
    """Make a value JSON-safe, mapping non-finite floats to ``None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


def _provenance_block(provenance: Provenance) -> dict[str, Any]:
    return {
        "source": provenance.source,
        "kind": provenance.kind.value,
        "detail": provenance.detail,
        "label": provenance.label,
        "describe": provenance.describe(),
        "is_evidential": provenance.is_evidential,
        "retrieved_at": provenance.retrieved_at.isoformat(timespec="seconds"),
    }


def _downsample(series: pd.Series, points: int = CURVE_POINTS) -> list[dict[str, Any]]:
    """Thin an equity curve for display, always keeping the final point.

    The last observation is the run's ending equity and appears elsewhere in the
    payload as a metric; dropping it to a stride boundary would put a visible
    gap between the curve's right edge and the number printed beside it.
    """
    if series.empty:
        return []
    step = max(len(series) // points, 1)
    thinned = series.iloc[::step]
    if thinned.index[-1] != series.index[-1]:
        thinned = pd.concat([thinned, series.iloc[[-1]]])
    return [
        {"t": ts.isoformat(), "equity": _clean(float(value))}
        for ts, value in thinned.items()
    ]


def _structure_block(series: OHLCVSeries, state: ICTState) -> dict[str, Any]:
    price = float(series.closes[state.index])
    block: dict[str, Any] = {
        "as_of": state.as_of.isoformat(),
        "session": state.session,
        "price": price,
        "bias": str(state.bias),
        "confluence_long": round(confluence_score(state, price, Direction.BULLISH), 3),
        "confluence_short": round(confluence_score(state, price, Direction.BEARISH), 3),
        "counts": {
            "swings": len(state.swings),
            "structure_events": len(state.structure_events),
            "fair_value_gaps": len(state.fair_value_gaps),
            "active_fvgs": len(state.active_fvgs()),
            "order_blocks": len(state.order_blocks),
            "active_order_blocks": len(state.active_order_blocks()),
            "liquidity_pools": len(state.liquidity_pools),
            "unswept_pools": len(state.unswept_pools()),
            "sweeps": len(state.sweeps),
        },
    }

    event = state.last_structure_event
    if event is not None:
        block["last_event"] = {
            "type": event.event_type.value,
            "direction": str(event.direction),
            "level": event.level,
            "bars_ago": state.index - event.confirmed_index,
        }

    dealing = state.dealing_range
    if dealing is not None:
        ote_low, ote_high = dealing.optimal_trade_entry(Direction.BULLISH)
        block["dealing_range"] = {
            "high": dealing.high,
            "low": dealing.low,
            "equilibrium": dealing.equilibrium,
            "position": round(dealing.position_of(price), 4),
            "zone": "premium" if dealing.is_premium(price) else "discount",
            "long_ote": [ote_low, ote_high],
        }

    block["pools"] = [
        {
            "side": "buyside" if pool.kind is LiquidityKind.BUYSIDE else "sellside",
            "price": pool.price,
            "distance": pool.price - price,
            "equal_cluster": bool(pool.is_equal_cluster),
        }
        for pool in sorted(state.unswept_pools(), key=lambda p: abs(p.price - price))[:12]
    ]

    zones: list[dict[str, Any]] = [
        {
            "kind": "OB",
            "direction": str(block_.effective_direction),
            "low": block_.bottom,
            "high": block_.top,
            "distance": abs(block_.midpoint - price),
        }
        for block_ in state.active_order_blocks()
    ]
    zones += [
        {
            "kind": "FVG",
            "direction": str(gap.direction),
            "low": gap.bottom,
            "high": gap.top,
            "distance": abs(gap.midpoint - price),
        }
        for gap in state.active_fvgs()
    ]
    block["zones"] = sorted(zones, key=lambda z: z["distance"])[:12]
    return block


def _signal_block(signal: Signal | None) -> dict[str, Any] | None:
    if signal is None:
        return None
    return {
        "direction": str(signal.direction),
        "entry": signal.entry,
        "stop": signal.stop,
        "targets": list(signal.targets),
        "primary_target": signal.primary_target,
        "reward_risk": signal.reward_risk,
        "risk_points": signal.risk_points,
        "confidence": signal.confidence,
        "rationale": signal.rationale,
        "tags": list(signal.tags),
    }


def _backtest_block(result: BacktestResult) -> dict[str, Any]:
    report = result.report
    return {
        "metrics": {k: _clean(v) for k, v in report.metrics.items()},
        "notes": list(report.notes),
        "warning": report.warning,
        "is_evidence": report.is_evidence,
        "starting_equity": report.starting_equity,
        "ending_equity": report.ending_equity,
        "equity_curve": _downsample(report.equity_curve),
        "funnel": {
            "signals": len(result.signals),
            "orders_routed": result.router.routed_count,
            "trades": len(result.trades),
            "risk_rejections": dict(result.risk_rejections),
            "router_rejections": dict(result.rejections),
        },
        "trades": [
            {
                "side": trade.side.value,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "duration_hours": trade.duration.total_seconds() / 3600.0,
                "pnl": trade.pnl,
                "commission": trade.commission,
                "r_multiple": _clean(trade.r_multiple),
                "exit_reason": trade.exit_reason,
                "tags": list(trade.tags),
                "is_win": trade.is_win,
            }
            for trade in result.trades
        ],
    }


def _price_block(series: OHLCVSeries, points: int = CURVE_POINTS) -> list[dict[str, Any]]:
    closes = series.closes
    step = max(len(closes) // points, 1)
    return [
        {"t": series.index[i].isoformat(), "close": float(closes[i])}
        for i in range(0, len(closes), step)
    ]


def build_payload(
    result: PipelineResult,
    series: OHLCVSeries,
    *,
    strategy_name: str,
    risk: RiskManager,
    portfolio: Portfolio | None = None,
) -> dict[str, Any]:
    """Assemble the full report payload from a completed pipeline run.

    Args:
        result: a run produced by :meth:`~axiom.agents.pipeline.AgentPipeline.run`.
            It must carry its backtest — a pipeline result built by hand without
            one cannot be rendered, and saying so here beats rendering a page
            with empty performance panels.
        risk: the manager whose limits the run was gated against. Its snapshot
            supplies the risk panel, so the page shows the configured limits
            rather than the defaults.
    """
    if result.backtest is None or result.ict_state is None:
        raise ValueError(
            "pipeline result carries no backtest — nothing measured to report"
        )

    state = result.ict_state
    book = portfolio if portfolio is not None else result.backtest.portfolio
    provenance = series.provenance

    return {
        "version": PAYLOAD_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "meta": {
            "symbol": series.instrument.symbol,
            "timeframe": str(series.timeframe),
            "bars": len(series),
            "start": series.start.isoformat(),
            "end": series.end.isoformat(),
            "strategy": strategy_name,
            "describe": series.describe(),
        },
        "provenance": _provenance_block(provenance),
        "structure": _structure_block(series, state),
        "price_curve": _price_block(series),
        "risk": {
            "snapshot": _clean(risk.snapshot(book, state.as_of)),
            "settings": {
                "account_equity": risk.settings.account_equity,
                "max_risk_per_trade_pct": risk.settings.max_risk_per_trade_pct,
                "max_risk_per_trade_cash": risk.settings.max_risk_per_trade_cash,
                "daily_loss_limit_pct": risk.settings.daily_loss_limit_pct,
                "daily_loss_limit_cash": risk.settings.daily_loss_limit_cash,
                "max_gross_exposure_pct": risk.settings.max_gross_exposure_pct,
                "max_positions": risk.settings.max_positions,
                "max_consecutive_losses": risk.settings.max_consecutive_losses,
            },
        },
        "stages": [
            {
                "role": report.role.value,
                "facts": {k: _clean(v) for k, v in report.facts.items()},
                "warnings": list(report.warnings),
                "narrative": report.narrative,
                "model": report.model,
                "used_llm": report.used_llm,
                "produced_at": report.produced_at.isoformat(timespec="seconds"),
            }
            for report in result.reports
        ],
        "backtest": _backtest_block(result.backtest),
        "approval": {
            "required": result.approval_required,
            "summary": result.approval_summary,
            "signal": _signal_block(result.signal),
            "warnings": list(result.all_warnings),
        },
    }


def write_report(
    payload: dict[str, Any],
    html_path: str | Path,
    json_path: str | Path | None = None,
) -> Path:
    """Render ``payload`` to a self-contained HTML file, optionally beside JSON."""
    from axiom.report.html import render_html

    target = Path(html_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(payload), encoding="utf-8")
    if json_path is not None:
        data = Path(json_path)
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
