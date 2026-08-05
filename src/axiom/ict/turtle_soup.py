"""Turtle Soup — the failed breakout.

Price takes out a known liquidity level, fails to hold beyond it, reclaims the
level, and then delivers in the opposite direction. The name is ICT's nod to the
original Turtle trend-following system: this is the trade that takes the other
side of a breakout entry.

Relationship to :mod:`axiom.ict.liquidity`
------------------------------------------
This overlaps with :class:`~axiom.ict.models.LiquiditySweep` and the two are
deliberately not merged, because their criteria differ in ways that matter:

* a **sweep** requires the *same bar* to close back inside the level. It is a
  single-bar rejection.
* a **turtle soup** allows the reclaim to take up to ``reclaim_bars`` bars
  (canonically 1-3), and additionally requires **post-event displacement** in
  the reversal direction.

So a turtle soup is the stricter, more complete pattern: it insists the market
actually did something after reclaiming, not merely that it poked and returned.
Every sweep that resolves is a turtle soup candidate; not every sweep resolves.

Confirmation is credited at the displacement bar, which is the first point at
which the whole pattern is knowable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.models import ICTFeature, LiquidityKind, LiquidityPool

#: Bars allowed between the violation and the reclaim close.
DEFAULT_RECLAIM_BARS = 3
#: Bars after the reclaim in which displacement must appear.
DEFAULT_DISPLACEMENT_WINDOW = 3
#: Minimum displacement, in ATR units, to confirm the reversal.
DEFAULT_MIN_DISPLACEMENT_ATR = 0.5


@dataclass(slots=True)
class TurtleSoup(ICTFeature):
    """A failed breakout: level violated, reclaimed, then reversed."""

    direction: Direction = Direction.NEUTRAL
    #: The liquidity level that was violated and reclaimed.
    level: float = 0.0
    kind: LiquidityKind = LiquidityKind.BUYSIDE
    #: Bar that traded beyond the level.
    violation_index: int = 0
    #: Bar that closed back inside.
    reclaim_index: int = 0
    #: How far beyond the level price reached, in ATR units.
    penetration_atr: float = 0.0
    #: Displacement of the confirming move, in ATR units.
    displacement_atr: float = 0.0
    #: The extreme reached beyond the level — the natural stop anchor.
    extreme: float = 0.0

    @property
    def reclaim_bars(self) -> int:
        """Bars taken to reclaim. Faster reclaims are cleaner failures."""
        return self.reclaim_index - self.violation_index

    @property
    def protective_level(self) -> float:
        """Beyond the swept extreme the failed-breakout read is wrong."""
        return self.extreme


def find_turtle_soups(
    series: OHLCVSeries,
    pools: list[LiquidityPool],
    *,
    atr_period: int = 14,
    reclaim_bars: int = DEFAULT_RECLAIM_BARS,
    displacement_window: int = DEFAULT_DISPLACEMENT_WINDOW,
    min_displacement_atr: float = DEFAULT_MIN_DISPLACEMENT_ATR,
) -> list[TurtleSoup]:
    """Detect failed breakouts at known liquidity levels.

    Args:
        reclaim_bars: bars allowed for the close to return inside the level.
            Canonically 1-3; beyond that the "reclaim" is a new move rather
            than a rejection of the old one.
        min_displacement_atr: without this the pattern is just a sweep. The
            displacement is what says the reversal was real.
    """
    n = len(series)
    if n < 4 or not pools:
        return []

    highs, lows, closes = series.highs, series.lows, series.closes
    atr = series.atr(atr_period)
    index = series.index

    out: list[TurtleSoup] = []
    for pool in pools:
        # The level must exist before it can be violated.
        start = pool.confirmed_index + 1
        if start >= n:
            continue

        buyside = pool.kind is LiquidityKind.BUYSIDE
        for i in range(start, n - 1):
            violated = highs[i] > pool.price if buyside else lows[i] < pool.price
            if not violated:
                continue

            reclaim = _find_reclaim(closes, pool.price, i, reclaim_bars, buyside, n)
            if reclaim is None:
                break  # level was taken and held; it is spent either way

            confirm = _find_displacement(
                closes, atr, reclaim, displacement_window,
                min_displacement_atr, buyside, n,
            )
            if confirm is None:
                break

            scale = max(atr[i], 1e-12)
            extreme = float(highs[i] if buyside else lows[i])
            penetration = abs(extreme - pool.price) / scale
            move = (closes[confirm] - closes[reclaim]) * (-1 if buyside else 1)

            out.append(
                TurtleSoup(
                    origin_index=i,
                    # Knowable only once the displacement has happened.
                    confirmed_index=confirm,
                    timestamp=index[i],
                    direction=Direction.BEARISH if buyside else Direction.BULLISH,
                    level=float(pool.price),
                    kind=pool.kind,
                    violation_index=i,
                    reclaim_index=reclaim,
                    penetration_atr=float(penetration),
                    displacement_atr=float(move / max(atr[confirm], 1e-12)),
                    extreme=extreme,
                )
            )
            break  # one turtle soup per pool: the level is now spent

    out.sort(key=lambda t: t.confirmed_index)
    return out


def _find_reclaim(
    closes: np.ndarray,
    level: float,
    violation: int,
    window: int,
    buyside: bool,
    n: int,
) -> int | None:
    """First bar within ``window`` that closes back inside the level."""
    for k in range(violation, min(violation + window + 1, n)):
        back_inside = closes[k] < level if buyside else closes[k] > level
        if back_inside:
            return k
    return None


def _find_displacement(
    closes: np.ndarray,
    atr: np.ndarray,
    reclaim: int,
    window: int,
    threshold: float,
    buyside: bool,
    n: int,
) -> int | None:
    """First bar after the reclaim showing displacement away from the level."""
    for j in range(reclaim + 1, min(reclaim + window + 1, n)):
        move = (closes[j] - closes[reclaim]) * (-1 if buyside else 1)
        if move / max(atr[j], 1e-12) >= threshold:
            return j
    return None


def recent_turtle_soups(
    soups: list[TurtleSoup], index: int, within: int = 20,
    direction: Direction | None = None,
) -> list[TurtleSoup]:
    """Turtle soups confirmed within ``within`` bars of ``index``."""
    out = [
        t
        for t in soups
        if t.is_known_at(index) and index - t.confirmed_index <= within
    ]
    if direction is not None:
        out = [t for t in out if t.direction is direction]
    return out
