"""Candle Range Theory (CRT).

Attribution — read this before using it
---------------------------------------
**CRT is not ICT-original.** It was popularised in 2024 by Romeo and TTrades,
and ICT's own public position is that it is *"based on my ideas but not my
concept"*. He acknowledges the lineage — the sweep-and-reverse mechanic comes
from his Power of Three and liquidity work — and declines to claim ownership.

It is implemented here because the trading community routinely discusses CRT
and ICT interchangeably, and having both in one engine with the attribution
stated is better than having the distinction quietly lost. Structurally, a CRT
setup is roughly *the manipulation phase of AMD applied to a single higher-
timeframe candle's range*.

CRT is also **less standardised than ICT**: the selection rules — which HTF
candle, which time filter, when to enter — vary materially between teachers.
The thresholds here are therefore even more provisional than the rest of the
engine, and the time-of-day filter is off by default because its canonical
values are genuinely disputed.

The mechanic
------------
1. take a completed higher-timeframe candle as the reference range;
2. a later bar sweeps one of its bounds and closes back inside;
3. enter on the reversal, targeting the **opposite bound**.

The lookahead trap
------------------
Resampling to H4 and reading "the H4 candle's high" while inside that candle is
a future leak — the high is not known until the candle closes. This module only
ever uses **completed** HTF candles, and :attr:`CRTSetup.reference_close_index`
records the bar at which the reference became knowable. Getting this wrong
produces a spectacular and entirely fictional backtest, because the reference
range then contains the very move being traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from axiom.core.series import OHLCVSeries
from axiom.core.sessions import to_ny
from axiom.core.timeframe import Timeframe
from axiom.core.types import Direction
from axiom.ict.models import ICTFeature

#: Most-cited CRT time-of-day anchors, New York time. Disputed between sources;
#: the filter is opt-in for that reason.
DEFAULT_TIME_ANCHORS: tuple[time, ...] = (
    time(2, 0), time(3, 0), time(5, 0), time(9, 0), time(13, 0),
)

#: Bars after the sweep in which the close must return inside the range.
DEFAULT_RECLAIM_BARS = 2
#: Minimum sweep beyond the bound, in ATR units.
DEFAULT_MIN_PENETRATION_ATR = 0.05


@dataclass(slots=True)
class CRTSetup(ICTFeature):
    """A sweep of one bound of a completed HTF candle, reversing to the other."""

    direction: Direction = Direction.NEUTRAL
    #: Bounds of the higher-timeframe reference candle.
    reference_high: float = 0.0
    reference_low: float = 0.0
    #: Base-series bar at which the reference candle closed — the first bar at
    #: which its range was knowable.
    reference_close_index: int = 0
    reference_timeframe: str = ""
    #: The bound that was swept.
    swept_bound: float = 0.0
    #: Sweep depth beyond the bound, in ATR units.
    penetration_atr: float = 0.0
    #: The extreme reached beyond the bound — the stop anchor.
    extreme: float = 0.0
    #: Opposite bound: the CRT target.
    target: float = 0.0

    @property
    def range_height(self) -> float:
        return self.reference_high - self.reference_low

    def reward_risk(self, stop_buffer: float = 0.0) -> float | None:
        """R:R targeting the opposite bound, with a stop beyond the extreme.

        ``stop_buffer`` is not optional in practice. A sweep that penetrated
        by a single tick gives a near-zero risk denominator and an R:R in the
        tens — arithmetically correct and completely unactionable, because no
        stop survives at the exact extreme of the bar that just tagged it.
        Pass a real buffer (an ATR fraction) to get a number worth acting on.
        """
        risk = abs(self.extreme - self.swept_bound) + stop_buffer
        if risk <= 0:
            return None
        return abs(self.target - self.swept_bound) / risk


def find_crt_setups(
    series: OHLCVSeries,
    reference_timeframe: str | Timeframe = "4h",
    *,
    atr_period: int = 14,
    reclaim_bars: int = DEFAULT_RECLAIM_BARS,
    min_penetration_atr: float = DEFAULT_MIN_PENETRATION_ATR,
    time_anchors: tuple[time, ...] | None = None,
    max_age_bars: int = 96,
) -> list[CRTSetup]:
    """Detect CRT setups against completed higher-timeframe candles.

    Args:
        reference_timeframe: the HTF whose candles form the reference ranges.
        time_anchors: restrict sweeps to these New York times. ``None`` (the
            default) disables the filter, because the canonical anchor set is
            disputed between CRT sources.
        max_age_bars: how long a reference candle stays eligible after closing.
    """
    target_tf = Timeframe.parse(reference_timeframe)
    if target_tf.seconds <= series.timeframe.seconds:
        raise ValueError(
            f"CRT reference timeframe {target_tf} must be higher than the base "
            f"series timeframe {series.timeframe}"
        )

    htf = series.resample(target_tf)
    if len(htf) < 2:
        return []

    n = len(series)
    highs, lows, closes = series.highs, series.lows, series.closes
    atr = series.atr(atr_period)
    base_index = series.index

    # For each HTF candle, the first base bar strictly after it closes. Until
    # that bar the candle is still forming and its range is not knowable.
    htf_starts = htf.index
    close_positions = np.searchsorted(
        base_index.values, (htf_starts + target_tf.timedelta).values, side="left"
    )

    ny_times = to_ny(base_index)
    anchor_minutes = (
        {t.hour * 60 + t.minute for t in time_anchors} if time_anchors else None
    )
    bar_minutes = ny_times.hour.to_numpy() * 60 + ny_times.minute.to_numpy()

    out: list[CRTSetup] = []
    for k in range(len(htf)):
        ref_high = float(htf.highs[k])
        ref_low = float(htf.lows[k])
        if ref_high <= ref_low:
            continue

        start = int(close_positions[k])
        if start >= n:
            continue
        stop = min(start + max_age_bars, n)

        setup = _scan_for_sweep(
            series, highs, lows, closes, atr, base_index,
            start, stop, ref_high, ref_low, k, str(target_tf),
            reclaim_bars, min_penetration_atr, anchor_minutes, bar_minutes,
        )
        if setup is not None:
            out.append(setup)

    out.sort(key=lambda s: s.confirmed_index)
    return out


def _scan_for_sweep(
    series: OHLCVSeries,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    base_index: pd.DatetimeIndex,
    start: int,
    stop: int,
    ref_high: float,
    ref_low: float,
    htf_position: int,
    tf_label: str,
    reclaim_bars: int,
    min_penetration_atr: float,
    anchor_minutes: set[int] | None,
    bar_minutes: np.ndarray,
) -> CRTSetup | None:
    """First qualifying sweep of either bound within the eligibility window."""
    for i in range(start, stop):
        if anchor_minutes is not None and int(bar_minutes[i]) not in anchor_minutes:
            continue

        scale = max(atr[i], 1e-12)

        for bound, is_high in ((ref_high, True), (ref_low, False)):
            swept = highs[i] > bound if is_high else lows[i] < bound
            if not swept:
                continue

            extreme = float(highs[i] if is_high else lows[i])
            penetration = abs(extreme - bound) / scale
            if penetration < min_penetration_atr:
                continue

            reclaim = _find_reclaim(closes, bound, i, reclaim_bars, is_high, len(closes))
            if reclaim is None:
                continue

            return CRTSetup(
                origin_index=i,
                # Knowable only once price has closed back inside the range.
                confirmed_index=reclaim,
                timestamp=base_index[i],
                direction=Direction.BEARISH if is_high else Direction.BULLISH,
                reference_high=ref_high,
                reference_low=ref_low,
                reference_close_index=start,
                reference_timeframe=tf_label,
                swept_bound=float(bound),
                penetration_atr=float(penetration),
                extreme=extreme,
                # Sweep the high -> target the low, and vice versa.
                target=ref_low if is_high else ref_high,
            )
    return None


def _find_reclaim(
    closes: np.ndarray, bound: float, sweep: int, window: int, is_high: bool, n: int
) -> int | None:
    for k in range(sweep, min(sweep + window + 1, n)):
        inside = closes[k] < bound if is_high else closes[k] > bound
        if inside:
            return k
    return None


def active_crt_setups(
    setups: list[CRTSetup], index: int, within: int = 24,
    direction: Direction | None = None,
) -> list[CRTSetup]:
    """Setups confirmed within ``within`` bars of ``index``."""
    out = [
        s
        for s in setups
        if s.is_known_at(index) and index - s.confirmed_index <= within
    ]
    if direction is not None:
        out = [s for s in out if s.direction is direction]
    return out
