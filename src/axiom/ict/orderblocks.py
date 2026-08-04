"""Order blocks and breaker blocks.

An order block is the last opposing candle before a displacement leg that broke
structure. The reasoning: institutional buying that produced an up-leg had to be
absorbed somewhere, and the final down-close candle before the leg is where that
absorption is visible on the chart.

Quality gates applied here, in order of importance:

* the leg must **break structure** — a displacement that resolves nothing is
  not an order block;
* the leg should leave an **imbalance** (an FVG). Blocks whose leg left no gap
  are kept but flagged, because ICT rates them materially weaker;
* the block is invalidated once price **closes** through its far edge, at which
  point it becomes a *breaker* and trades with inverted polarity.

Confirmation is credited at the structure break, never at the candle itself —
the candle only becomes an order block in retrospect.
"""

from __future__ import annotations

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.models import FairValueGap, OrderBlock, StructureEvent

#: How far back from a structure break to search for the origin candle.
DEFAULT_LOOKBACK = 12


def find_order_blocks(
    series: OHLCVSeries,
    events: list[StructureEvent],
    gaps: list[FairValueGap] | None = None,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    atr_period: int = 14,
    require_imbalance: bool = False,
) -> list[OrderBlock]:
    """Derive order blocks from confirmed structure events."""
    if not events:
        return []

    opens, closes = series.opens, series.closes
    highs, lows = series.highs, series.lows
    atr = series.atr(atr_period)
    index = series.index
    n = len(series)
    gaps = gaps or []

    blocks: list[OrderBlock] = []
    seen: set[tuple[int, int]] = set()

    for event in events:
        break_index = event.confirmed_index
        origin = _find_origin_candle(
            opens, closes, break_index, event.direction, lookback
        )
        if origin is None:
            continue

        key = (origin, int(event.direction))
        if key in seen:
            continue
        seen.add(key)

        has_gap = _leg_left_imbalance(gaps, origin, break_index, event.direction)
        if require_imbalance and not has_gap:
            continue

        scale = max(atr[break_index], 1e-12)
        block = OrderBlock(
            origin_index=origin,
            confirmed_index=break_index,
            timestamp=index[origin],
            top=float(highs[origin]),
            bottom=float(lows[origin]),
            direction=event.direction,
            open_price=float(opens[origin]),
            close_price=float(closes[origin]),
            displacement_atr=float(abs(closes[break_index] - closes[origin]) / scale),
            has_imbalance=has_gap,
            structure_event_index=break_index,
        )
        blocks.append(block)

    _track_mitigation(blocks, highs, lows, closes, n)
    blocks.sort(key=lambda b: b.confirmed_index)
    return blocks


def _find_origin_candle(
    opens: np.ndarray,
    closes: np.ndarray,
    break_index: int,
    direction: Direction,
    lookback: int,
) -> int | None:
    """The last candle closing against ``direction`` before the break.

    Searching backwards from the break rather than forwards from a fixed offset
    means the block is anchored to the leg that actually did the work, however
    long that leg ran.
    """
    start = max(break_index - lookback, 0)
    for i in range(break_index - 1, start - 1, -1):
        opposing = closes[i] < opens[i] if direction is Direction.BULLISH else closes[i] > opens[i]
        if opposing:
            return i
    return None


def _leg_left_imbalance(
    gaps: list[FairValueGap], origin: int, break_index: int, direction: Direction
) -> bool:
    return any(
        g.direction is direction and origin <= g.origin_index <= break_index for g in gaps
    )


def _track_mitigation(
    blocks: list[OrderBlock],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    n: int,
) -> None:
    """Record first re-entry into each block, and the close that breaks it."""
    for block in blocks:
        bullish = block.direction is Direction.BULLISH
        for j in range(block.confirmed_index + 1, n):
            if block.mitigated_index is None:
                touched = lows[j] <= block.top if bullish else highs[j] >= block.bottom
                if touched:
                    block.mitigated_index = j
            broken = (
                closes[j] < block.bottom if bullish else closes[j] > block.top
            )
            if broken:
                block.broken_index = j
                break


def breaker_blocks(blocks: list[OrderBlock], index: int) -> list[OrderBlock]:
    """Blocks that have failed and now trade with inverted polarity."""
    return [
        b
        for b in blocks
        if b.is_known_at(index) and b.broken_index is not None and b.broken_index <= index
    ]


def unmitigated_blocks(
    blocks: list[OrderBlock], index: int, direction: Direction | None = None
) -> list[OrderBlock]:
    """Blocks known at ``index`` and still unmitigated entering that bar.

    ``mitigated_index >= index`` is deliberate, not an off-by-one. A block's
    mitigation index is the *first* bar that touches it — which is precisely the
    bar a retracement entry wants to trade. Requiring ``> index`` would exclude
    every block on the one bar it becomes actionable, making order-block entries
    unreachable. Blocks mitigated on an earlier bar are still excluded.
    """
    out = [
        b
        for b in blocks
        if b.is_known_at(index)
        and (b.mitigated_index is None or b.mitigated_index >= index)
        and (b.broken_index is None or b.broken_index > index)
    ]
    if direction is not None:
        out = [b for b in out if b.direction is direction]
    return out


def nearest_block(
    blocks: list[OrderBlock], price: float, direction: Direction | None = None
) -> OrderBlock | None:
    candidates = [b for b in blocks if b.is_active]
    if direction is not None:
        candidates = [b for b in candidates if b.effective_direction is direction]
    if not candidates:
        return None
    return min(candidates, key=lambda b: abs(b.entry_edge - price))
