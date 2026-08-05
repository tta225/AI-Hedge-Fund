"""Rejection blocks.

A rejection block is a candle whose **wick** documents a level being tested and
forcibly refused. Price went there, found no acceptance, and was pushed back.

The distinction from an order block is precise and worth holding onto: an order
block references the candle's **body** (where absorption happened), a rejection
block references the **wick** (where price was refused). They are different
objects built from different parts of the same candle, and conflating them
produces zones that are neither.

Canonical criteria (all required):

1. long wick — at least ``min_wick_ratio`` of the candle's total range;
2. the wick tip reaches a **known liquidity level** — this is what separates a
   rejection block from any random long-wicked candle, and it is the criterion
   most implementations quietly drop;
3. the close sits at the far end of the range, away from the wick tip;
4. the **next** candle confirms with displacement in the rejection direction.

Because criterion 4 looks forward one bar, a rejection block is confirmed at
``origin + 1``, never at the candle itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.models import LiquidityKind, LiquidityPool, PriceZone

#: Minimum wick share of the candle range.
DEFAULT_MIN_WICK_RATIO = 0.60
#: How close the close must sit to the far end of the range, as a share of it.
DEFAULT_CLOSE_POSITION = 0.60
#: Tolerance, in ATR units, for the wick tip "reaching" a liquidity level.
DEFAULT_LEVEL_TOLERANCE_ATR = 0.25


@dataclass(slots=True)
class RejectionBlock(PriceZone):
    """A candle whose wick was refused at a known level.

    The zone is the **wick**, from the body edge to the wick tip — the span of
    price that was rejected.
    """

    direction: Direction = Direction.NEUTRAL
    #: Wick length as a share of the candle's total range.
    wick_ratio: float = 0.0
    #: The liquidity level the wick reached.
    level: float = 0.0
    #: Displacement of the confirming candle, in ATR units.
    confirmation_atr: float = 0.0
    mitigated_index: int | None = None

    @property
    def wick_tip(self) -> float:
        """The extreme that was rejected. The natural stop anchor."""
        return self.bottom if self.direction is Direction.BULLISH else self.top

    @property
    def entry_edge(self) -> float:
        """Body edge — where the rejected span begins."""
        return self.top if self.direction is Direction.BULLISH else self.bottom

    @property
    def protective_edge(self) -> float:
        return self.wick_tip

    @property
    def is_active(self) -> bool:
        return self.mitigated_index is None


def find_rejection_blocks(
    series: OHLCVSeries,
    pools: list[LiquidityPool],
    *,
    atr_period: int = 14,
    min_wick_ratio: float = DEFAULT_MIN_WICK_RATIO,
    close_position: float = DEFAULT_CLOSE_POSITION,
    level_tolerance_atr: float = DEFAULT_LEVEL_TOLERANCE_ATR,
    require_confirmation: bool = True,
) -> list[RejectionBlock]:
    """Detect rejection blocks against known liquidity pools.

    Args:
        pools: liquidity levels the wick must reach. Without these the detector
            would return every long-wicked candle on the chart, which is noise —
            the level is what makes the rejection meaningful.
        require_confirmation: demand next-candle displacement. Disabling it
            finds candidate wicks one bar earlier at the cost of many that never
            resolve into anything.
    """
    n = len(series)
    if n < 3 or not pools:
        return []

    opens, highs, lows, closes = series.opens, series.highs, series.lows, series.closes
    atr = series.atr(atr_period)
    index = series.index

    # Levels usable at bar i are those confirmed at or before it.
    buyside = sorted(
        (p for p in pools if p.kind is LiquidityKind.BUYSIDE),
        key=lambda p: p.confirmed_index,
    )
    sellside = sorted(
        (p for p in pools if p.kind is LiquidityKind.SELLSIDE),
        key=lambda p: p.confirmed_index,
    )

    out: list[RejectionBlock] = []
    for i in range(1, n - 1):
        candle_range = highs[i] - lows[i]
        if candle_range <= 0:
            continue

        body_top = max(opens[i], closes[i])
        body_bottom = min(opens[i], closes[i])
        upper_wick = highs[i] - body_top
        lower_wick = body_bottom - lows[i]
        scale = max(atr[i], 1e-12)
        tolerance = level_tolerance_atr * scale

        # Bullish: long lower wick refused at sellside liquidity.
        if lower_wick / candle_range >= min_wick_ratio:
            level = _nearest_level(sellside, lows[i], i, tolerance)
            close_high_in_range = (closes[i] - lows[i]) / candle_range >= close_position
            if level is not None and close_high_in_range:
                block = _build(
                    series, i, Direction.BULLISH, body_bottom, lows[i],
                    lower_wick / candle_range, level, atr, index,
                    require_confirmation,
                )
                if block is not None:
                    out.append(block)

        # Bearish: long upper wick refused at buyside liquidity.
        if upper_wick / candle_range >= min_wick_ratio:
            level = _nearest_level(buyside, highs[i], i, tolerance)
            close_low_in_range = (highs[i] - closes[i]) / candle_range >= close_position
            if level is not None and close_low_in_range:
                block = _build(
                    series, i, Direction.BEARISH, highs[i], body_top,
                    upper_wick / candle_range, level, atr, index,
                    require_confirmation,
                )
                if block is not None:
                    out.append(block)

    _track_mitigation(out, highs, lows, n)
    out.sort(key=lambda b: b.confirmed_index)
    return out


def _nearest_level(
    pools: list[LiquidityPool], price: float, index: int, tolerance: float
) -> float | None:
    """A pool level within ``tolerance`` of ``price``, known at ``index``."""
    best: float | None = None
    best_distance = tolerance
    for pool in pools:
        if pool.confirmed_index > index:
            break  # sorted by confirmation; nothing later is knowable yet
        distance = abs(pool.price - price)
        if distance <= best_distance:
            best, best_distance = pool.price, distance
    return best


def _build(
    series: OHLCVSeries,
    i: int,
    direction: Direction,
    top: float,
    bottom: float,
    wick_ratio: float,
    level: float,
    atr: np.ndarray,
    index: pd.DatetimeIndex,
    require_confirmation: bool,
) -> RejectionBlock | None:
    """Validate the confirming candle and construct the block."""
    confirm_index = i + 1
    closes, opens = series.closes, series.opens
    scale = max(atr[confirm_index], 1e-12)
    move = (closes[confirm_index] - opens[confirm_index]) * int(direction)
    confirmation = move / scale

    if require_confirmation and confirmation <= 0:
        return None

    return RejectionBlock(
        origin_index=i,
        # Criterion 4 needs the next candle, so the block is only knowable then.
        confirmed_index=confirm_index,
        timestamp=index[i],
        top=float(top),
        bottom=float(bottom),
        direction=direction,
        wick_ratio=float(wick_ratio),
        level=float(level),
        confirmation_atr=float(confirmation),
    )


def _track_mitigation(
    blocks: list[RejectionBlock], highs: np.ndarray, lows: np.ndarray, n: int
) -> None:
    for block in blocks:
        bullish = block.direction is Direction.BULLISH
        for j in range(block.confirmed_index + 1, n):
            touched = lows[j] <= block.top if bullish else highs[j] >= block.bottom
            if touched:
                block.mitigated_index = j
                break


def active_rejection_blocks(
    blocks: list[RejectionBlock], index: int, direction: Direction | None = None
) -> list[RejectionBlock]:
    """Blocks known at ``index`` and not yet traded back into before it."""
    out = [
        b
        for b in blocks
        if b.is_known_at(index)
        and (b.mitigated_index is None or b.mitigated_index >= index)
    ]
    if direction is not None:
        out = [b for b in out if b.direction is direction]
    return out
