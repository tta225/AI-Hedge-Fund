"""Liquidity pools and stop raids.

ICT's central claim is that price is not seeking a fair value — it is seeking
*liquidity*. Resting stop orders cluster in obvious places: above swing highs,
below swing lows, and above all at **equal** highs and lows, where every chart
reader has drawn the same line.

This module locates those pools and then identifies **sweeps**: excursions past
a pool that reverse rather than continue. The distinction is the whole game. A
break that holds is expansion; a break that rejects was a raid to fill orders,
and the reversal is the trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom.core.series import OHLCVSeries
from axiom.ict.models import (
    LiquidityKind,
    LiquidityPool,
    LiquiditySweep,
    StructureEvent,
    SwingKind,
    SwingPoint,
)

#: Cluster half-width, in ATR units, for two extremes to count as "equal".
DEFAULT_EQUALITY_ATR = 0.12
#: Minimum excursion beyond a pool, in ATR units, to register as a sweep.
DEFAULT_MIN_PENETRATION_ATR = 0.05


def find_liquidity_pools(
    series: OHLCVSeries,
    swings: list[SwingPoint],
    *,
    atr_period: int = 14,
    equality_atr: float = DEFAULT_EQUALITY_ATR,
) -> list[LiquidityPool]:
    """Cluster swing extremes into liquidity pools.

    Adjacent pivots within ``equality_atr`` of one another collapse into a
    single pool flagged as an equal-highs/equal-lows cluster.
    """
    if not swings:
        return []

    atr = series.atr(atr_period)
    index = series.index

    pools: list[LiquidityPool] = []
    for kind, swing_kind in (
        (LiquidityKind.BUYSIDE, SwingKind.HIGH),
        (LiquidityKind.SELLSIDE, SwingKind.LOW),
    ):
        group = [p for p in swings if p.kind is swing_kind]
        pools.extend(_cluster(group, kind, atr, index, equality_atr))

    pools.sort(key=lambda p: p.confirmed_index)
    _mark_swept(pools, series.highs, series.lows)
    return pools


def _cluster(
    points: list[SwingPoint],
    kind: LiquidityKind,
    atr: np.ndarray,
    index: pd.DatetimeIndex,
    equality_atr: float,
) -> list[LiquidityPool]:
    pools: list[LiquidityPool] = []
    used: set[int] = set()

    for i, point in enumerate(points):
        if point.origin_index in used:
            continue
        tolerance = equality_atr * max(atr[point.origin_index], 1e-12)
        members = [point]
        for other in points[i + 1 :]:
            if other.origin_index in used:
                continue
            if abs(other.price - point.price) <= tolerance:
                members.append(other)

        used.update(m.origin_index for m in members)
        prices = [m.price for m in members]
        # A pool sits at the extreme of its members: buy stops rest above the
        # highest of a set of equal highs, not at their average.
        price = max(prices) if kind is LiquidityKind.BUYSIDE else min(prices)
        last = max(members, key=lambda m: m.confirmed_index)
        spread = (max(prices) - min(prices)) / max(atr[point.origin_index], 1e-12)

        pools.append(
            LiquidityPool(
                origin_index=point.origin_index,
                confirmed_index=last.confirmed_index,
                timestamp=index[point.origin_index],
                price=float(price),
                kind=kind,
                member_indices=tuple(m.origin_index for m in members),
                tolerance_atr=float(spread),
            )
        )
    return pools


def _mark_swept(pools: list[LiquidityPool], highs: np.ndarray, lows: np.ndarray) -> None:
    n = len(highs)
    for pool in pools:
        start = pool.confirmed_index + 1
        if start >= n:
            continue
        if pool.kind is LiquidityKind.BUYSIDE:
            hits = np.flatnonzero(highs[start:] > pool.price)
        else:
            hits = np.flatnonzero(lows[start:] < pool.price)
        if hits.size:
            pool.swept_index = int(start + hits[0])


def find_sweeps(
    series: OHLCVSeries,
    pools: list[LiquidityPool],
    *,
    atr_period: int = 14,
    min_penetration_atr: float = DEFAULT_MIN_PENETRATION_ATR,
    require_close_back: bool = True,
) -> list[LiquiditySweep]:
    """Identify raids on liquidity pools.

    Args:
        require_close_back: when True (the default) only excursions that close
            back on the origin side of the pool are returned. Setting it False
            also returns clean breakouts, which is useful for measuring how
            often a raid is *not* a reversal.
    """
    if not pools:
        return []

    highs, lows, closes = series.highs, series.lows, series.closes
    atr = series.atr(atr_period)
    index = series.index

    sweeps: list[LiquiditySweep] = []
    for pool in pools:
        j = pool.swept_index
        if j is None:
            continue

        scale = max(atr[j], 1e-12)
        if pool.kind is LiquidityKind.BUYSIDE:
            penetration = (highs[j] - pool.price) / scale
            closed_back = closes[j] < pool.price
        else:
            penetration = (pool.price - lows[j]) / scale
            closed_back = closes[j] > pool.price

        if penetration < min_penetration_atr:
            continue
        if require_close_back and not closed_back:
            continue

        sweeps.append(
            LiquiditySweep(
                origin_index=j,
                confirmed_index=j,
                timestamp=index[j],
                pool_price=float(pool.price),
                kind=pool.kind,
                penetration_atr=float(penetration),
                closed_back_inside=bool(closed_back),
            )
        )

    sweeps.sort(key=lambda s: s.confirmed_index)
    return sweeps


def confirm_sweep_reversals(
    sweeps: list[LiquiditySweep],
    events: list[StructureEvent],
    *,
    max_bars: int = 20,
) -> None:
    """Link each sweep to the structure break that validated it, if any.

    A raid alone is a hypothesis. The trade is only confirmed when structure
    breaks in the reversal direction within ``max_bars``; without that, price
    simply continued and the "sweep" was a breakout.
    """
    for sweep in sweeps:
        target = sweep.reversal_direction
        for event in events:
            if event.confirmed_index <= sweep.confirmed_index:
                continue
            if event.confirmed_index - sweep.confirmed_index > max_bars:
                break
            if event.direction is target and event.is_reversal:
                sweep.reversal_confirmed_index = event.confirmed_index
                break


def unswept_pools(
    pools: list[LiquidityPool], index: int, kind: LiquidityKind | None = None
) -> list[LiquidityPool]:
    """Pools known at ``index`` that price has not yet taken."""
    out = [
        p
        for p in pools
        if p.is_known_at(index) and (p.swept_index is None or p.swept_index > index)
    ]
    if kind is not None:
        out = [p for p in out if p.kind is kind]
    return out


def draw_on_liquidity(
    pools: list[LiquidityPool], price: float, index: int
) -> tuple[LiquidityPool | None, LiquidityPool | None]:
    """Nearest unswept pool above and below ``price``.

    These are the two magnets price is working between — the practical
    definition of a target in ICT terms.
    """
    live = unswept_pools(pools, index)
    above = [p for p in live if p.price > price]
    below = [p for p in live if p.price < price]
    return (
        min(above, key=lambda p: p.price - price) if above else None,
        min(below, key=lambda p: price - p.price) if below else None,
    )
