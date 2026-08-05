"""Market structure: BOS, CHoCH, and MSS.

The walk is strictly causal. At every bar we consider only pivots already
confirmed at that bar, so a break can never be scored against a swing the
market had not yet formed.

Classification:

* the break agrees with the prevailing bias  → **BOS** (continuation);
* the break opposes it, without displacement → **CHoCH**;
* the break opposes it, with displacement    → **MSS**.

Displacement is the close-through distance normalised by ATR. Requiring it to
promote CHoCH to MSS is what separates an energetic reversal from price merely
drifting one tick past a pivot.
"""

from __future__ import annotations

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.models import (
    StructureEvent,
    StructureEventType,
    SwingKind,
    SwingPoint,
)

#: Close-through distance, in ATR units, that qualifies a break as displacement.
DEFAULT_DISPLACEMENT_ATR = 0.25


def detect_structure(
    series: OHLCVSeries,
    swings: list[SwingPoint],
    *,
    atr_period: int = 14,
    displacement_atr: float = DEFAULT_DISPLACEMENT_ATR,
) -> list[StructureEvent]:
    """Walk the tape and emit structure events in chronological order."""
    if not swings:
        return []

    closes = series.closes
    atr = series.atr(atr_period)
    index = series.index
    n = len(series)

    highs_by_conf = _bucket(swings, SwingKind.HIGH, n)
    lows_by_conf = _bucket(swings, SwingKind.LOW, n)

    events: list[StructureEvent] = []
    bias = Direction.NEUTRAL
    # The pivots currently eligible to be broken.
    active_high: SwingPoint | None = None
    active_low: SwingPoint | None = None

    for i in range(n):
        for point in highs_by_conf[i]:
            if active_high is None or point.origin_index > active_high.origin_index:
                active_high = point
        for point in lows_by_conf[i]:
            if active_low is None or point.origin_index > active_low.origin_index:
                active_low = point

        close = closes[i]
        scale = max(atr[i], 1e-12)

        if active_high is not None and close > active_high.price:
            magnitude = (close - active_high.price) / scale
            event_type = _classify(bias, Direction.BULLISH, magnitude, displacement_atr)
            events.append(
                StructureEvent(
                    origin_index=active_high.origin_index,
                    confirmed_index=i,
                    timestamp=index[i],
                    event_type=event_type,
                    direction=Direction.BULLISH,
                    level=active_high.price,
                    displacement_atr=float(magnitude),
                    broken_swing_index=active_high.origin_index,
                )
            )
            bias = Direction.BULLISH
            active_high = None
            continue

        if active_low is not None and close < active_low.price:
            magnitude = (active_low.price - close) / scale
            event_type = _classify(bias, Direction.BEARISH, magnitude, displacement_atr)
            events.append(
                StructureEvent(
                    origin_index=active_low.origin_index,
                    confirmed_index=i,
                    timestamp=index[i],
                    event_type=event_type,
                    direction=Direction.BEARISH,
                    level=active_low.price,
                    displacement_atr=float(magnitude),
                    broken_swing_index=active_low.origin_index,
                )
            )
            bias = Direction.BEARISH
            active_low = None

    return events


def _classify(
    bias: Direction, break_dir: Direction, magnitude: float, threshold: float
) -> StructureEventType:
    if bias is break_dir:
        return StructureEventType.BOS
    if bias is Direction.NEUTRAL:
        # No established trend yet: the first break defines one rather than
        # reversing anything, so it is continuation by definition.
        return StructureEventType.BOS
    return (
        StructureEventType.MSS if magnitude >= threshold else StructureEventType.CHOCH
    )


def _bucket(swings: list[SwingPoint], kind: SwingKind, n: int) -> list[list[SwingPoint]]:
    """Group pivots by the bar at which they become known."""
    buckets: list[list[SwingPoint]] = [[] for _ in range(n)]
    for point in swings:
        if point.kind is kind and 0 <= point.confirmed_index < n:
            buckets[point.confirmed_index].append(point)
    return buckets


def current_bias(events: list[StructureEvent], index: int) -> Direction:
    """Structural bias implied by the last event known at ``index``."""
    known = [e for e in events if e.is_known_at(index)]
    return known[-1].direction if known else Direction.NEUTRAL


def displacement_legs(
    series: OHLCVSeries, *, atr_period: int = 14, threshold: float = 1.5
) -> np.ndarray:
    """Boolean mask of bars whose range exceeds ``threshold`` ATRs.

    A coarse volatility screen. For the canonical candle-level test — which
    looks at body dominance and wick asymmetry rather than raw range — use
    :func:`is_displacement_candle`.
    """
    atr = series.atr(atr_period)
    ranges = series.highs - series.lows
    return np.asarray(ranges >= threshold * np.maximum(atr, 1e-12))


#: Canonical displacement thresholds.
DISPLACEMENT_BODY_MULTIPLE = 1.5
DISPLACEMENT_BODY_RATIO = 0.70
DISPLACEMENT_MAX_OPPOSING_WICK = 0.20


def is_displacement_candle(
    series: OHLCVSeries,
    *,
    lookback: int = 20,
    body_multiple: float = DISPLACEMENT_BODY_MULTIPLE,
    body_ratio: float = DISPLACEMENT_BODY_RATIO,
    max_opposing_wick: float = DISPLACEMENT_MAX_OPPOSING_WICK,
) -> np.ndarray:
    """Canonical per-candle displacement test.

    A displacement candle satisfies all of:

    * **wide body** — at least ``body_multiple`` times the trailing average body;
    * **dominant body** — body covers at least ``body_ratio`` of the candle range;
    * **minimal opposing wick** — the wick against the direction of travel is at
      most ``max_opposing_wick`` of the range.

    This differs from the ATR-range screen above in a way that matters: a bar can
    have a huge range and still be indecision — a long wick both ways and a tiny
    body. Range alone cannot tell those apart; body dominance can.

    The trailing average excludes the candle itself, so the test is causal.
    """
    opens, highs, lows, closes = series.opens, series.highs, series.lows, series.closes
    n = len(series)
    ranges = np.maximum(highs - lows, 1e-12)
    bodies = np.abs(closes - opens)

    # Trailing mean of the previous `lookback` bodies, excluding the current bar.
    cumulative = np.concatenate(([0.0], np.cumsum(bodies)))
    avg_body = np.full(n, np.nan)
    for i in range(1, n):
        start = max(0, i - lookback)
        avg_body[i] = (cumulative[i] - cumulative[start]) / (i - start)

    bullish = closes > opens
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    opposing_wick = np.where(bullish, lower_wick, upper_wick)

    with np.errstate(invalid="ignore"):
        wide = bodies >= body_multiple * avg_body
    dominant = bodies / ranges >= body_ratio
    clean = opposing_wick / ranges <= max_opposing_wick
    return np.asarray(np.nan_to_num(wide, nan=0.0).astype(bool) & dominant & clean)
