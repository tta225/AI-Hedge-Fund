"""IPDA data ranges — the 20 / 40 / 60 trading-day lookback extremes.

ICT's Interbank Price Delivery Algorithm is an *explanatory model*, not a
documented protocol. What is operationally useful about it — and what this
module implements — is the reference set: the highest high and lowest low over
trailing 20, 40, and 60 **trading days**. These are the longest-horizon
liquidity pools on the chart, and they are where a daily or weekly draw
typically resolves.

Two details the canonical definition is specific about, both easy to get wrong:

* **trading days, not calendar days.** A 20-calendar-day window silently
  includes weekends and holidays and so covers roughly 14 trading days. On a
  60-day window the gap is nearly three weeks of data;
* **wick extremes, not closes.** The level is where price *traded*, because
  that is where the stops sit.

Everything here is computed causally: the levels at bar *i* use only bars up to
and including *i*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.core.sessions import trading_day
from axiom.core.types import Direction

#: The canonical lookback windows, in trading days.
IPDA_WINDOWS: tuple[int, ...] = (20, 40, 60)


@dataclass(frozen=True, slots=True)
class IPDALevel:
    """One extreme of one lookback window, as of some bar."""

    window_days: int
    is_high: bool
    price: float
    #: Bar at which the extreme was set.
    formed_index: int
    #: Trading days between formation and the bar this was computed at.
    age_days: int

    @property
    def direction(self) -> Direction:
        """Direction price must travel to reach this level."""
        return Direction.BULLISH if self.is_high else Direction.BEARISH

    @property
    def label(self) -> str:
        return f"ipda_{self.window_days}d_{'high' if self.is_high else 'low'}"


@dataclass(slots=True)
class IPDAState:
    """The full reference set at one bar."""

    index: int
    price: float
    levels: list[IPDALevel] = field(default_factory=list)
    #: Trading days actually available; fewer than 60 early in a series.
    days_available: int = 0

    def level(self, window_days: int, *, high: bool) -> IPDALevel | None:
        return next(
            (
                lv for lv in self.levels
                if lv.window_days == window_days and lv.is_high is high
            ),
            None,
        )

    def above(self) -> list[IPDALevel]:
        """Levels above current price, nearest first — the upside draw."""
        return sorted(
            (lv for lv in self.levels if lv.price > self.price),
            key=lambda lv: lv.price - self.price,
        )

    def below(self) -> list[IPDALevel]:
        return sorted(
            (lv for lv in self.levels if lv.price < self.price),
            key=lambda lv: self.price - lv.price,
        )

    def nearest_draw(self, direction: Direction) -> IPDALevel | None:
        """Closest IPDA level in ``direction`` — the longest-horizon target."""
        candidates = self.above() if direction is Direction.BULLISH else self.below()
        return candidates[0] if candidates else None

    def range_position(self, window_days: int = 20) -> float | None:
        """Where price sits inside a window's range, 0.0 at low, 1.0 at high."""
        high = self.level(window_days, high=True)
        low = self.level(window_days, high=False)
        if high is None or low is None or high.price <= low.price:
            return None
        return (self.price - low.price) / (high.price - low.price)

    def summary(self) -> str:
        parts = [
            f"{lv.label}={lv.price:,.2f}({lv.age_days}d)"
            for lv in sorted(self.levels, key=lambda lv: (-lv.window_days, lv.is_high))
        ]
        return " ".join(parts)


def compute_ipda_levels(
    series: OHLCVSeries,
    index: int,
    *,
    windows: tuple[int, ...] = IPDA_WINDOWS,
) -> IPDAState:
    """IPDA reference levels as of bar ``index``.

    Only bars at or before ``index`` contribute, so this is safe to call from a
    strategy or a backtest.
    """
    if not 0 <= index < len(series):
        raise IndexError(f"bar {index} out of range for {len(series)} bars")

    days = trading_day(series.index)
    highs, lows = series.highs, series.lows

    # Distinct trading days up to and including this bar, most recent last.
    seen = days[: index + 1]
    unique_days = np.unique(seen)
    available = len(unique_days)

    levels: list[IPDALevel] = []
    for window in windows:
        if available < 2:
            continue
        # Trading days, not calendar days: take the last `window` distinct
        # dates rather than slicing a fixed number of bars or timedelta.
        cutoff = unique_days[-min(window, available)]
        mask = seen >= cutoff
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            continue

        high_at = int(positions[np.argmax(highs[positions])])
        low_at = int(positions[np.argmin(lows[positions])])
        current_day = days[index]

        levels.append(
            IPDALevel(
                window_days=window,
                is_high=True,
                price=float(highs[high_at]),
                formed_index=high_at,
                age_days=int(np.count_nonzero(
                    (unique_days > days[high_at]) & (unique_days <= current_day)
                )),
            )
        )
        levels.append(
            IPDALevel(
                window_days=window,
                is_high=False,
                price=float(lows[low_at]),
                formed_index=low_at,
                age_days=int(np.count_nonzero(
                    (unique_days > days[low_at]) & (unique_days <= current_day)
                )),
            )
        )

    return IPDAState(
        index=index,
        price=float(series.closes[index]),
        levels=levels,
        days_available=available,
    )


def ipda_bias(state: IPDAState, window_days: int = 20) -> Direction:
    """Positional bias from where price sits in the IPDA range.

    Contrarian to position, like the dealing-range filter: near the 20-day high
    the draw is more plausibly the low. A filter on directional ideas, not a
    signal on its own.
    """
    position = state.range_position(window_days)
    if position is None:
        return Direction.NEUTRAL
    if position > 0.7:
        return Direction.BEARISH
    if position < 0.3:
        return Direction.BULLISH
    return Direction.NEUTRAL


def ipda_level_series(
    series: OHLCVSeries, window_days: int = 20, *, high: bool = True
) -> np.ndarray:
    """Per-bar IPDA level as an array, for plotting or feature construction.

    Computed with a rolling pass rather than by calling
    :func:`compute_ipda_levels` per bar, which would be quadratic.
    """
    days = trading_day(series.index)
    values = series.highs if high else series.lows
    n = len(series)
    out = np.full(n, np.nan)

    _, day_index = np.unique(days, return_inverse=True)
    for i in range(n):
        first_day = max(0, day_index[i] - window_days + 1)
        start = int(np.searchsorted(day_index, first_day, side="left"))
        segment = values[start : i + 1]
        if segment.size:
            out[i] = segment.max() if high else segment.min()
    return out
