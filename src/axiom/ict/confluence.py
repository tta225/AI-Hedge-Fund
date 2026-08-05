"""Higher-order ICT models built from overlapping primitives.

The primitives in the rest of ``axiom.ict`` are single features. The models here
are *confluences* — places where two independent features land on the same
price, which ICT treats as materially stronger than either alone.

:class:`UnicornSetup`
    The **Unicorn model**: a breaker block overlapping a fair value gap. The
    breaker says a prior order block failed and polarity flipped; the FVG says
    delivery was inefficient through that same band. Both point at the same
    prices for different reasons, and the overlap is the entry zone.

:func:`power_of_three`
    Accumulation → Manipulation → Distribution, ICT's model of how a session
    delivers. Classifies where in that cycle a session currently sits.

Both are derived from features that already carry ``confirmed_index``, and the
confluence inherits the *later* of its components' confirmations — you cannot
know two things have overlapped until you know both of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.core.sessions import SessionWindow, trading_day, window_mask
from axiom.core.types import Direction
from axiom.ict.models import FairValueGap, OrderBlock, PriceZone


@dataclass(slots=True)
class UnicornSetup(PriceZone):
    """A breaker block overlapping a fair value gap.

    The zone is the **intersection** of the two, not their union. Only the band
    where both agree carries the confluence; widening it to the union would
    dilute exactly the property that makes the setup worth taking.
    """

    direction: Direction = Direction.NEUTRAL
    breaker_index: int = 0
    gap_index: int = 0
    #: Overlap as a fraction of the smaller component. 1.0 = fully nested.
    overlap_ratio: float = 0.0
    #: Displacement of the leg that created the gap, in ATR units.
    displacement_atr: float = 0.0
    mitigated_index: int | None = None

    @property
    def is_active(self) -> bool:
        return self.mitigated_index is None

    @property
    def entry_edge(self) -> float:
        return self.top if self.direction is Direction.BULLISH else self.bottom

    @property
    def protective_edge(self) -> float:
        return self.bottom if self.direction is Direction.BULLISH else self.top


#: Minimum overlap, as a fraction of the smaller zone, to count as a unicorn.
DEFAULT_MIN_OVERLAP = 0.25


def find_unicorns(
    series: OHLCVSeries,
    order_blocks: list[OrderBlock],
    gaps: list[FairValueGap],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_age: int = 60,
) -> list[UnicornSetup]:
    """Find breaker/FVG overlaps.

    Args:
        min_overlap: fraction of the smaller zone that must be shared.
        max_age: maximum bars between the two components' confirmations. Two
            features formed hundreds of bars apart that happen to share a price
            are a coincidence, not a confluence.
    """
    breakers = [b for b in order_blocks if b.is_breaker]
    if not breakers or not gaps:
        return []

    setups: list[UnicornSetup] = []
    for breaker in breakers:
        # A breaker trades opposite to the block's original direction.
        direction = breaker.effective_direction
        assert breaker.broken_index is not None
        breaker_known = breaker.broken_index

        for gap in gaps:
            if gap.direction is not direction:
                continue
            if abs(gap.confirmed_index - breaker_known) > max_age:
                continue

            low = max(breaker.bottom, gap.bottom)
            high = min(breaker.top, gap.top)
            if high <= low:
                continue

            smaller = min(breaker.height, gap.height)
            if smaller <= 0:
                continue
            ratio = (high - low) / smaller
            if ratio < min_overlap:
                continue

            # Confirmed only once BOTH components are known.
            confirmed = max(breaker_known, gap.confirmed_index)
            if confirmed >= len(series):
                continue

            setups.append(
                UnicornSetup(
                    origin_index=min(breaker.origin_index, gap.origin_index),
                    confirmed_index=confirmed,
                    timestamp=series.index[confirmed],
                    top=high,
                    bottom=low,
                    direction=direction,
                    breaker_index=breaker.origin_index,
                    gap_index=gap.origin_index,
                    overlap_ratio=float(ratio),
                    displacement_atr=gap.displacement_atr,
                )
            )

    _track_mitigation(setups, series.highs, series.lows)
    setups.sort(key=lambda u: u.confirmed_index)
    return setups


def _track_mitigation(
    setups: list[UnicornSetup], highs: np.ndarray, lows: np.ndarray
) -> None:
    n = len(highs)
    for setup in setups:
        bullish = setup.direction is Direction.BULLISH
        for j in range(setup.confirmed_index + 1, n):
            touched = lows[j] <= setup.top if bullish else highs[j] >= setup.bottom
            if touched:
                setup.mitigated_index = j
                break


def active_unicorns(
    setups: list[UnicornSetup], index: int, direction: Direction | None = None
) -> list[UnicornSetup]:
    """Unicorns known at ``index`` and not yet traded into before it."""
    out = [
        u
        for u in setups
        if u.is_known_at(index)
        and (u.mitigated_index is None or u.mitigated_index >= index)
    ]
    if direction is not None:
        out = [u for u in out if u.direction is direction]
    return out


class PO3Phase(str):
    """Phase of the Power of Three cycle."""

    ACCUMULATION = "accumulation"
    MANIPULATION = "manipulation"
    DISTRIBUTION = "distribution"
    UNDEFINED = "undefined"


@dataclass(frozen=True, slots=True)
class PowerOfThree:
    """Where a session sits in accumulation → manipulation → distribution."""

    phase: str
    #: Direction of the manipulation leg — the false move, against the real one.
    manipulation_direction: Direction
    session_open: float
    session_high: float
    session_low: float
    #: Extent of the manipulation leg in ATR units.
    manipulation_atr: float

    @property
    def expected_direction(self) -> Direction:
        """Where distribution should run — opposite the manipulation."""
        return self.manipulation_direction.opposite


def power_of_three(
    series: OHLCVSeries,
    window: SessionWindow,
    index: int,
    *,
    atr_period: int = 14,
    manipulation_threshold: float = 0.5,
) -> PowerOfThree | None:
    """Classify the current session's Power of Three phase.

    The model: a session opens and consolidates (**accumulation**), runs one
    side of that range to collect stops (**manipulation**), then delivers in the
    opposite direction (**distribution**).

    Classification is causal — only bars up to ``index`` are considered, so a
    session still forming is judged on what has happened, not on how it ends.
    """
    if index < 0 or index >= len(series):
        return None

    days = trading_day(series.index)
    mask = window_mask(series.index, window) & (days == days[index])
    positions = np.flatnonzero(mask)
    positions = positions[positions <= index]
    if positions.size < 3:
        return None

    opens, highs, lows, closes = series.opens, series.highs, series.lows, series.closes
    session_open = float(opens[positions[0]])
    session_high = float(highs[positions].max())
    session_low = float(lows[positions].min())
    price = float(closes[index])
    atr = max(float(series.atr(atr_period)[index]), 1e-12)

    up_extent = (session_high - session_open) / atr
    down_extent = (session_open - session_low) / atr

    # Manipulation is the larger excursion from the open; distribution is
    # price having reversed back through the open and beyond.
    if max(up_extent, down_extent) < manipulation_threshold:
        return PowerOfThree(
            phase=PO3Phase.ACCUMULATION,
            manipulation_direction=Direction.NEUTRAL,
            session_open=session_open,
            session_high=session_high,
            session_low=session_low,
            manipulation_atr=float(max(up_extent, down_extent)),
        )

    if up_extent >= down_extent:
        manipulation = Direction.BULLISH
        reversed_through = price < session_open
        extent = up_extent
    else:
        manipulation = Direction.BEARISH
        reversed_through = price > session_open
        extent = down_extent

    return PowerOfThree(
        phase=PO3Phase.DISTRIBUTION if reversed_through else PO3Phase.MANIPULATION,
        manipulation_direction=manipulation,
        session_open=session_open,
        session_high=session_high,
        session_low=session_low,
        manipulation_atr=float(extent),
    )
