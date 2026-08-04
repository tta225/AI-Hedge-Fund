"""Dealing ranges, premium/discount, and session reference ranges.

ICT's positional filter: within the range the algorithm is currently working,
only look for shorts in *premium* (above 50%) and longs in *discount* (below
50%). The Optimal Trade Entry band — the 62%-79% retracement — is where the
retracement is deep enough to be efficient but shallow enough to still be a
retracement rather than a reversal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom.core.series import OHLCVSeries
from axiom.core.sessions import SessionWindow, trading_day, window_mask
from axiom.core.types import Direction
from axiom.ict.models import DealingRange, SwingKind, SwingPoint


def build_dealing_range(
    series: OHLCVSeries, swings: list[SwingPoint], index: int, *, lookback: int = 100
) -> DealingRange | None:
    """The swing high/low pair defining the range price is working within.

    Anchored to the **extreme** confirmed pivots inside the lookback window —
    the highest swing high and the lowest swing low — rather than to the most
    recent pair of pivots.

    The recent-pair definition is tempting and wrong in a trend: consecutive
    pivots there are close together and both sit far behind price, producing a
    narrow stale range that reports price at 400% of its own dealing range. A
    premium/discount filter built on that is meaningless. The extremes give the
    range the algorithm is actually delivering within.

    Restricted to pivots confirmed at or before ``index``, so the range is one
    the market could have been referencing at that bar.
    """
    if index < 0 or index >= len(series):
        return None

    floor = max(index - lookback, 0)
    highs = [
        p for p in swings
        if p.kind is SwingKind.HIGH and floor <= p.origin_index and p.confirmed_index <= index
    ]
    lows = [
        p for p in swings
        if p.kind is SwingKind.LOW and floor <= p.origin_index and p.confirmed_index <= index
    ]
    if not highs or not lows:
        return None

    high = max(highs, key=lambda p: p.price)
    low = min(lows, key=lambda p: p.price)
    if high.price <= low.price:
        return None

    return DealingRange(
        origin_index=min(high.origin_index, low.origin_index),
        confirmed_index=max(high.confirmed_index, low.confirmed_index),
        timestamp=series.index[index],
        high=float(high.price),
        low=float(low.price),
        high_index=high.origin_index,
        low_index=low.origin_index,
    )


def premium_discount_bias(dealing_range: DealingRange | None, price: float) -> Direction:
    """Positional bias implied by where price sits in the range.

    Deliberately *contrarian to position*: price in premium argues for shorts.
    This is a filter on directional ideas, not a signal by itself.
    """
    if dealing_range is None or dealing_range.height <= 0:
        return Direction.NEUTRAL
    position = dealing_range.position_of(price)
    if position > 0.5:
        return Direction.BEARISH
    if position < 0.5:
        return Direction.BULLISH
    return Direction.NEUTRAL


def session_range(
    series: OHLCVSeries, window: SessionWindow, index: int
) -> tuple[float, float] | None:
    """High/low of ``window`` for the trading day containing bar ``index``.

    Only bars at or before ``index`` contribute, so a range still forming does
    not leak its eventual extremes backwards.
    """
    if index < 0 or index >= len(series):
        return None

    days = trading_day(series.index)
    mask = window_mask(series.index, window) & (days == days[index])
    positions = np.flatnonzero(mask)
    positions = positions[positions <= index]
    if positions.size == 0:
        return None
    return float(series.highs[positions].max()), float(series.lows[positions].min())


def standard_deviation_projections(
    low: float, high: float, direction: Direction, deviations: tuple[float, ...] = (1.0, 2.0, 2.5, 4.0)
) -> dict[float, float]:
    """Project multiples of a reference range beyond its extreme.

    ICT projects the CBDR / Asian range / Flout in standard deviations to
    estimate where the day's expansion is likely to terminate.
    """
    height = high - low
    if height <= 0:
        return {}
    if direction is Direction.BULLISH:
        return {d: high + d * height for d in deviations}
    return {d: low - d * height for d in deviations}


def true_day_open(series: OHLCVSeries, index: int) -> float | None:
    """The New York midnight opening price for bar ``index``'s trading day.

    ICT's reference for intraday premium/discount: above it is premium for the
    day, below it discount, independent of any swing-based range.
    """
    if index < 0 or index >= len(series):
        return None
    days = trading_day(series.index)
    positions = np.flatnonzero(days == days[index])
    positions = positions[positions <= index]
    if positions.size == 0:
        return None
    return float(series.opens[positions[0]])


def daily_bias_levels(series: OHLCVSeries, index: int) -> dict[str, float]:
    """Prior-day high/low/close plus the current true day open.

    The minimum set of reference levels a discretionary ICT read starts from.
    """
    if index < 0 or index >= len(series):
        return {}

    days = trading_day(series.index)
    today = days[index]
    prior_mask = days < today
    levels: dict[str, float] = {}

    if prior_mask.any():
        prior_day = days[prior_mask].max()
        positions = np.flatnonzero(days == prior_day)
        levels["prev_day_high"] = float(series.highs[positions].max())
        levels["prev_day_low"] = float(series.lows[positions].min())
        levels["prev_day_close"] = float(series.closes[positions[-1]])

    open_price = true_day_open(series, index)
    if open_price is not None:
        levels["true_day_open"] = open_price

    positions = np.flatnonzero(days == today)
    positions = positions[positions <= index]
    if positions.size:
        levels["day_high"] = float(series.highs[positions].max())
        levels["day_low"] = float(series.lows[positions].min())

    return levels


def judas_swing(
    series: OHLCVSeries, window: SessionWindow, index: int
) -> tuple[Direction, float] | None:
    """Detect a false move against the eventual direction inside ``window``.

    The judas swing is the opening manipulation: price runs one side of the
    prior range to collect stops, then reverses and delivers the real move.
    Returns the direction of the *false* move and its extent in points.
    """
    bounds = session_range(series, window, index)
    if bounds is None:
        return None
    high, low = bounds

    days = trading_day(series.index)
    mask = window_mask(series.index, window) & (days == days[index])
    positions = np.flatnonzero(mask)
    positions = positions[positions <= index]
    if positions.size < 2:
        return None

    open_price = float(series.opens[positions[0]])
    close_price = float(series.closes[positions[-1]])
    up_extension = high - max(open_price, close_price)
    down_extension = min(open_price, close_price) - low

    if close_price > open_price and down_extension > 0:
        return Direction.BEARISH, down_extension
    if close_price < open_price and up_extension > 0:
        return Direction.BULLISH, up_extension
    return None


def range_position_series(
    series: OHLCVSeries, swings: list[SwingPoint], *, lookback: int = 100
) -> pd.Series:
    """Per-bar position within the prevailing dealing range, in [0, 1]."""
    values = np.full(len(series), np.nan)
    for i in range(len(series)):
        dealing = build_dealing_range(series, swings, i, lookback=lookback)
        if dealing is not None:
            values[i] = dealing.position_of(float(series.closes[i]))
    return pd.Series(values, index=series.index, name="range_position")
