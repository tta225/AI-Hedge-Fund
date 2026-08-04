"""Fair value gaps and imbalance tracking.

A fair value gap is a three-bar pattern where the first and third bars fail to
overlap: delivery moved so fast that one side never traded. ICT treats the
unfilled band as an inefficiency the algorithm is obliged to revisit.

Life cycle tracked here:

1. **formed** — the third bar closes and the gap becomes visible;
2. **CE touched** — price reaches the 50% level (consequent encroachment),
   which ICT treats as the meaningful fill rather than the full close;
3. **mitigated** — price trades fully through the gap;
4. **inverted** — price *closes* decisively beyond the gap and later respects it
   from the far side. The gap flips polarity: a filled bullish FVG that now caps
   rallies is a bearish inverse FVG.

Steps 3 and 4 are distinguished because a mitigated gap is spent, whereas an
inverted one is a live level with the opposite bias.
"""

from __future__ import annotations

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.models import FairValueGap

#: Minimum gap height in ATR units. Filters the one-tick noise gaps that make
#: an ICT chart unreadable and a backtest meaningless.
DEFAULT_MIN_GAP_ATR = 0.08


def find_fair_value_gaps(
    series: OHLCVSeries,
    *,
    atr_period: int = 14,
    min_gap_atr: float = DEFAULT_MIN_GAP_ATR,
    track_inversions: bool = True,
) -> list[FairValueGap]:
    """Detect FVGs and track each one's fill state forward through the tape."""
    n = len(series)
    if n < 3:
        return []

    highs, lows, closes = series.highs, series.lows, series.closes
    atr = series.atr(atr_period)
    index = series.index

    gaps: list[FairValueGap] = []
    for i in range(1, n - 1):
        scale = max(atr[i], 1e-12)
        leg = abs(closes[i + 1] - closes[i - 1]) / scale

        # Bullish: the low of bar i+1 sits above the high of bar i-1.
        if lows[i + 1] > highs[i - 1]:
            height = lows[i + 1] - highs[i - 1]
            if height >= min_gap_atr * scale:
                gaps.append(
                    FairValueGap(
                        origin_index=i,
                        confirmed_index=i + 1,
                        timestamp=index[i + 1],
                        top=float(lows[i + 1]),
                        bottom=float(highs[i - 1]),
                        direction=Direction.BULLISH,
                        displacement_atr=float(leg),
                    )
                )

        # Bearish: the high of bar i+1 sits below the low of bar i-1.
        elif highs[i + 1] < lows[i - 1]:
            height = lows[i - 1] - highs[i + 1]
            if height >= min_gap_atr * scale:
                gaps.append(
                    FairValueGap(
                        origin_index=i,
                        confirmed_index=i + 1,
                        timestamp=index[i + 1],
                        top=float(lows[i - 1]),
                        bottom=float(highs[i + 1]),
                        direction=Direction.BEARISH,
                        displacement_atr=float(leg),
                    )
                )

    _track_fills(gaps, highs, lows, closes, track_inversions)
    return gaps


def _track_fills(
    gaps: list[FairValueGap],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    track_inversions: bool,
) -> None:
    """Walk each gap forward from confirmation, recording how it was consumed."""
    n = len(highs)
    for gap in gaps:
        start = gap.confirmed_index + 1
        if start >= n or gap.height <= 0:
            continue

        bullish = gap.direction is Direction.BULLISH
        ce = gap.midpoint

        for j in range(start, n):
            # Price approaches a bullish gap from above and a bearish gap from
            # below, so the relevant extreme differs.
            reach = lows[j] if bullish else highs[j]
            filled = (gap.top - reach) if bullish else (reach - gap.bottom)
            ratio = min(max(filled / gap.height, 0.0), 1.0)
            if ratio > gap.fill_ratio:
                gap.fill_ratio = float(ratio)

            touched_ce = lows[j] <= ce if bullish else highs[j] >= ce
            if gap.ce_touched_index is None and touched_ce:
                gap.ce_touched_index = j

            fully_filled = lows[j] <= gap.bottom if bullish else highs[j] >= gap.top
            if gap.mitigated_index is None and fully_filled:
                gap.mitigated_index = j
                if not track_inversions:
                    break

            # Inversion needs a decisive close beyond the gap, not just a wick.
            if track_inversions and gap.mitigated_index is not None:
                closed_through = (
                    closes[j] < gap.bottom if bullish else closes[j] > gap.top
                )
                if closed_through:
                    gap.inverted_index = j
                    break


def unmitigated_gaps(
    gaps: list[FairValueGap], index: int, direction: Direction | None = None
) -> list[FairValueGap]:
    """Gaps known at ``index`` and still unfilled entering that bar.

    ``mitigated_index >= index`` mirrors :func:`unmitigated_blocks`: the bar
    that completes the fill is the bar a rebalance entry trades, so it must
    remain visible. Gaps filled on an earlier bar are excluded.
    """
    out = [
        g
        for g in gaps
        if g.is_known_at(index)
        and (g.mitigated_index is None or g.mitigated_index >= index)
        and (g.inverted_index is None or g.inverted_index > index)
    ]
    if direction is not None:
        out = [g for g in out if g.direction is direction]
    return out


def nearest_gap(
    gaps: list[FairValueGap], price: float, direction: Direction | None = None
) -> FairValueGap | None:
    """The unfilled gap closest to ``price`` — the most likely near-term draw."""
    candidates = [g for g in gaps if g.is_active]
    if direction is not None:
        candidates = [g for g in candidates if g.direction is direction]
    if not candidates:
        return None
    return min(candidates, key=lambda g: abs(g.midpoint - price))


def volume_imbalance(series: OHLCVSeries, index: int) -> tuple[float, float] | None:
    """Body-to-body gap between consecutive bars, if one exists.

    Distinct from an FVG: this is the space between one bar's close and the
    next bar's open, where no *body* traded even though wicks may overlap.
    """
    if index < 1 or index >= len(series):
        return None
    prev_close = series.closes[index - 1]
    open_ = series.opens[index]
    if open_ > prev_close:
        return float(prev_close), float(open_)
    if open_ < prev_close:
        return float(open_), float(prev_close)
    return None
