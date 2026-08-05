"""Domain objects → JSON-safe dictionaries for the web API.

Why this is a separate module
-----------------------------
The engines speak in dataclasses, numpy arrays and pandas timestamps. None of
that survives a JSON encoder, and scattering conversions through the request
handlers is how a field quietly changes shape between two endpoints. Every
conversion lives here, so the wire format has one definition.

The provenance rule
-------------------
`series_payload` always emits the provenance block, and `is_evidential` travels
with it. The browser is a long way from the engine that computed a number, and
an operator must never be able to look at a chart and be unsure whether they are
seeing real market history or generated bars. Every payload carrying prices
carries its origin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from axiom.backtest.engine import BacktestResult
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import Instrument
from axiom.ict.models import (
    DealingRange,
    FairValueGap,
    ICTState,
    LiquidityPool,
    LiquiditySweep,
    OrderBlock,
    StructureEvent,
    SwingPoint,
)


def _num(value: Any) -> float | None:
    """Coerce to a JSON-safe float. NaN and inf become null, not `NaN`.

    `json.dumps` emits bare `NaN`, which is invalid JSON and which
    `JSON.parse` rejects — so a single non-finite metric would otherwise fail
    the whole response rather than one field.
    """
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _ms(stamp: pd.Timestamp) -> int:
    """Epoch milliseconds — what the browser's Date constructor expects."""
    return int(pd.Timestamp(stamp).timestamp() * 1000)


def provenance_payload(provenance: Provenance) -> dict[str, Any]:
    return {
        "source": provenance.source,
        "kind": provenance.kind.value,
        "is_evidential": provenance.is_evidential,
        "detail": provenance.detail,
        "retrieved_at": provenance.retrieved_at.isoformat(),
        "transforms": list(provenance.transforms),
        "describe": provenance.describe(),
    }


def instrument_payload(instrument: Instrument) -> dict[str, Any]:
    return {
        "symbol": instrument.symbol,
        "asset_class": instrument.asset_class.value,
        "exchange": instrument.exchange,
        "currency": instrument.currency,
        "tick_size": instrument.tick_size,
        "point_value": instrument.point_value,
        "description": instrument.description,
        "session_tz": instrument.session_tz,
    }


def series_payload(series: OHLCVSeries, *, max_bars: int = 1500) -> dict[str, Any]:
    """Bars as parallel arrays, newest `max_bars` only.

    Parallel arrays rather than a list of objects: a year of 15m bars is ~35k
    rows, and repeating five key names per row triples the payload for nothing.
    The cap exists because no chart can render more bars than a screen has
    pixels, and shipping them anyway is the difference between a snappy panel
    and a two-second parse.
    """
    view = series.tail(max_bars) if len(series) > max_bars else series
    offset = len(series) - len(view)
    return {
        "symbol": series.instrument.symbol,
        "timeframe": str(series.timeframe),
        "bars": len(view),
        "total_bars": len(series),
        "index_offset": offset,
        "start": _ms(view.start),
        "end": _ms(view.end),
        "time": [_ms(t) for t in view.index],
        "open": [_num(v) for v in view.opens],
        "high": [_num(v) for v in view.highs],
        "low": [_num(v) for v in view.lows],
        "close": [_num(v) for v in view.closes],
        "volume": [_num(v) for v in view.volumes],
        "provenance": provenance_payload(series.provenance),
        "describe": series.describe(),
    }


def _feature_base(feature: Any, offset: int) -> dict[str, Any]:
    """Common fields, with bar indices rebased onto the returned window.

    The chart addresses bars by their position in the payload, not their
    position in the full series. Rebasing here keeps that arithmetic out of the
    browser, where an off-by-one silently draws a zone against the wrong candle.
    """
    return {
        "origin_index": feature.origin_index - offset,
        "confirmed_index": feature.confirmed_index - offset,
        "timestamp": _ms(feature.timestamp),
    }


def _swing(swing: SwingPoint, offset: int) -> dict[str, Any]:
    return {
        **_feature_base(swing, offset),
        "price": _num(swing.price),
        "kind": swing.kind.value,
        "strength": swing.strength,
        "taken": swing.taken_index is not None,
    }


def _structure(event: StructureEvent, offset: int) -> dict[str, Any]:
    return {
        **_feature_base(event, offset),
        "event_type": event.event_type.value,
        "direction": str(event.direction),
        "level": _num(event.level),
        "displacement_atr": _num(event.displacement_atr),
        "is_reversal": event.is_reversal,
    }


def _fvg(gap: FairValueGap, offset: int) -> dict[str, Any]:
    return {
        **_feature_base(gap, offset),
        "top": _num(gap.top),
        "bottom": _num(gap.bottom),
        "midpoint": _num(gap.midpoint),
        "direction": str(gap.direction),
        "fill_ratio": _num(gap.fill_ratio),
        "displacement_atr": _num(gap.displacement_atr),
        "is_mitigated": gap.is_mitigated,
        "is_inverted": gap.is_inverted,
        "is_active": gap.is_active,
        "mitigated_index": (
            None if gap.mitigated_index is None else gap.mitigated_index - offset
        ),
    }


def _order_block(block: OrderBlock, offset: int) -> dict[str, Any]:
    return {
        **_feature_base(block, offset),
        "top": _num(block.top),
        "bottom": _num(block.bottom),
        "candle_high": _num(block.candle_high),
        "candle_low": _num(block.candle_low),
        "mean_threshold": _num(block.mean_threshold),
        "direction": str(block.direction),
        "effective_direction": str(block.effective_direction),
        "displacement_atr": _num(block.displacement_atr),
        "anchored_at_pivot": block.anchored_at_pivot,
        "has_imbalance": block.has_imbalance,
        "is_breaker": block.is_breaker,
        "is_active": block.is_active,
        "mitigated_index": (
            None if block.mitigated_index is None else block.mitigated_index - offset
        ),
    }


def _pool(pool: LiquidityPool, offset: int) -> dict[str, Any]:
    return {
        **_feature_base(pool, offset),
        "price": _num(pool.price),
        "kind": pool.kind.value,
        "size": pool.size,
        "is_equal_cluster": pool.is_equal_cluster,
        "tolerance_atr": _num(pool.tolerance_atr),
        "strength": _num(pool.strength),
        "is_swept": pool.is_swept,
        "swept_index": None if pool.swept_index is None else pool.swept_index - offset,
    }


def _sweep(sweep: LiquiditySweep, offset: int) -> dict[str, Any]:
    return {
        **_feature_base(sweep, offset),
        "pool_price": _num(sweep.pool_price),
        "kind": sweep.kind.value,
        "penetration_atr": _num(sweep.penetration_atr),
        "closed_back_inside": sweep.closed_back_inside,
        "reversal_direction": str(sweep.reversal_direction),
        "reversal_confirmed": sweep.reversal_confirmed_index is not None,
    }


def _dealing_range(dealing: DealingRange, price: float, offset: int) -> dict[str, Any]:
    from axiom.core.types import Direction

    long_ote = dealing.optimal_trade_entry(Direction.BULLISH)
    short_ote = dealing.optimal_trade_entry(Direction.BEARISH)
    return {
        **_feature_base(dealing, offset),
        "high": _num(dealing.high),
        "low": _num(dealing.low),
        "equilibrium": _num(dealing.equilibrium),
        "height": _num(dealing.height),
        "position": _num(dealing.position_of(price)),
        "is_premium": dealing.is_premium(price),
        "long_ote": [_num(long_ote[0]), _num(long_ote[1])],
        "short_ote": [_num(short_ote[0]), _num(short_ote[1])],
    }


def ict_payload(state: ICTState, series: OHLCVSeries, *, offset: int = 0) -> dict[str, Any]:
    """The full structural read, with every bar index rebased by `offset`."""
    price = float(series.closes[-1])
    return {
        "symbol": state.symbol,
        "timeframe": state.timeframe,
        "as_of": _ms(state.as_of),
        "bias": str(state.bias),
        "session": state.session,
        "last_price": _num(price),
        "swings": [_swing(s, offset) for s in state.swings],
        "structure_events": [_structure(e, offset) for e in state.structure_events],
        "fair_value_gaps": [_fvg(g, offset) for g in state.fair_value_gaps],
        "order_blocks": [_order_block(b, offset) for b in state.order_blocks],
        "liquidity_pools": [_pool(p, offset) for p in state.liquidity_pools],
        "sweeps": [_sweep(s, offset) for s in state.sweeps],
        "dealing_range": (
            None
            if state.dealing_range is None
            else _dealing_range(state.dealing_range, price, offset)
        ),
        "counts": {
            "swings": len(state.swings),
            "structure_events": len(state.structure_events),
            "fair_value_gaps": len(state.fair_value_gaps),
            "order_blocks": len(state.order_blocks),
            "liquidity_pools": len(state.liquidity_pools),
            "sweeps": len(state.sweeps),
        },
    }


def backtest_payload(result: BacktestResult, strategy: str) -> dict[str, Any]:
    """Metrics, equity curve, trades and the signal funnel.

    `notes` and `warning` are carried verbatim rather than summarised. They hold
    the caveats — too few trades, synthetic data, pessimistic fill assumptions —
    and a UI that drops them turns a hedged result into a claim.
    """
    report = result.report
    curve = report.equity_curve
    return {
        "strategy": strategy,
        "starting_equity": _num(report.starting_equity),
        "ending_equity": _num(report.ending_equity),
        "is_evidence": report.is_evidence,
        "warning": report.warning,
        "provenance": provenance_payload(report.provenance),
        "metrics": {k: _num(v) for k, v in report.metrics.items()},
        "notes": list(report.notes),
        "equity_curve": {
            "time": [_ms(t) for t in curve.index],
            "value": [_num(v) for v in curve.to_numpy()],
        },
        "trades": [
            {
                "symbol": t.instrument.symbol,
                "side": t.side.value,
                "quantity": _num(t.quantity),
                "entry_price": _num(t.entry_price),
                "exit_price": _num(t.exit_price),
                "entry_time": _ms(t.entry_time),
                "exit_time": _ms(t.exit_time),
                "pnl": _num(t.pnl),
                "commission": _num(t.commission),
                "exit_reason": t.exit_reason,
                "is_win": t.is_win,
                "r_multiple": _num(
                    t.pnl / (t.risk_points * t.quantity * t.instrument.point_value)
                    if t.risk_points and t.quantity
                    else float("nan")
                ),
                "tags": list(t.tags),
            }
            for t in report.trades
        ],
        "funnel": {
            "signals": len(result.signals),
            "orders": result.router.routed_count,
            "trades": len(result.trades),
            "risk_rejections": dict(result.risk_rejections),
            "rejections": dict(result.rejections),
        },
    }
