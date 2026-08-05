"""Fractal swing detection.

A swing high of strength *n* is a bar whose high exceeds the *n* bars before it
and is not exceeded by the *n* bars after it. The asymmetry (strict left,
non-strict right) means a flat double-top registers once, at its first bar,
rather than twice or not at all.

Confirmation lags detection by *n* bars, and :attr:`SwingPoint.confirmed_index`
records that honestly.
"""

from __future__ import annotations

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.ict.models import SwingKind, SwingPoint


def find_swings(series: OHLCVSeries, strength: int = 2) -> list[SwingPoint]:
    """Detect fractal swing highs and lows.

    Args:
        series: validated bars.
        strength: bars required on each side. Larger values yield fewer, more
            structurally significant pivots and a longer confirmation lag.
    """
    if strength < 1:
        raise ValueError("swing strength must be >= 1")
    n = len(series)
    if n < 2 * strength + 1:
        return []

    highs, lows = series.highs, series.lows
    index = series.index
    points: list[SwingPoint] = []

    for i in range(strength, n - strength):
        left = slice(i - strength, i)
        right = slice(i + 1, i + strength + 1)

        if highs[i] > highs[left].max() and highs[i] >= highs[right].max():
            points.append(
                SwingPoint(
                    origin_index=i,
                    confirmed_index=i + strength,
                    timestamp=index[i],
                    price=float(highs[i]),
                    kind=SwingKind.HIGH,
                    strength=strength,
                )
            )
        if lows[i] < lows[left].min() and lows[i] <= lows[right].min():
            points.append(
                SwingPoint(
                    origin_index=i,
                    confirmed_index=i + strength,
                    timestamp=index[i],
                    price=float(lows[i]),
                    kind=SwingKind.LOW,
                    strength=strength,
                )
            )

    points.sort(key=lambda p: (p.origin_index, p.kind.value))
    _mark_taken(points, highs, lows)
    return points


def _mark_taken(points: list[SwingPoint], highs: np.ndarray, lows: np.ndarray) -> None:
    """Record the first bar that traded through each pivot's level.

    Only bars at or after confirmation count: a pivot that is violated before
    anyone could have identified it was never tradable liquidity.
    """
    n = len(highs)
    for point in points:
        start = max(point.confirmed_index + 1, point.origin_index + 1)
        if start >= n:
            continue
        if point.is_high:
            hits = np.flatnonzero(highs[start:] > point.price)
        else:
            hits = np.flatnonzero(lows[start:] < point.price)
        if hits.size:
            point.taken_index = int(start + hits[0])


def last_swing(points: list[SwingPoint], kind: SwingKind, before: int) -> SwingPoint | None:
    """Most recent pivot of ``kind`` confirmed at or before bar ``before``."""
    candidates = [p for p in points if p.kind is kind and p.confirmed_index <= before]
    return candidates[-1] if candidates else None


def swings_confirmed_at(points: list[SwingPoint], index: int) -> list[SwingPoint]:
    return [p for p in points if p.confirmed_index <= index]
