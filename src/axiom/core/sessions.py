"""The ICT session clock.

ICT's model of the trading day is a schedule, not an indicator: algorithmic
price delivery is expected to seek liquidity and rebalance imbalances inside
specific windows, all defined in New York local time and all observing US
daylight-saving transitions. Everything here is therefore anchored to
``America/New_York`` and converted to UTC per-timestamp, so a backtest spanning
a DST change does not silently shift every killzone by an hour.

Windows are expressed as ``[start, end)`` half-open intervals so adjacent
windows never double-count a bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")
TOKYO_TZ = ZoneInfo("Asia/Tokyo")


class WindowKind(str, Enum):
    #: Broad regional session.
    SESSION = "session"
    #: A killzone — the window where a session's manipulation/distribution runs.
    KILLZONE = "killzone"
    #: Silver Bullet: the one-hour windows ICT singles out for FVG entries.
    SILVER_BULLET = "silver_bullet"
    #: Reference ranges used to project standard deviations.
    REFERENCE_RANGE = "reference_range"
    #: Macro windows — the :50-:10 algorithmic reprice periods.
    MACRO = "macro"


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """A recurring intraday window in New York local time."""

    name: str
    start: time
    end: time
    kind: WindowKind
    description: str = ""

    @property
    def crosses_midnight(self) -> bool:
        return self.end <= self.start

    def contains_local(self, t: time) -> bool:
        """Membership test for a New York wall-clock time."""
        if self.crosses_midnight:
            return t >= self.start or t < self.end
        return self.start <= t < self.end

    def __str__(self) -> str:
        return f"{self.name} ({self.start:%H:%M}-{self.end:%H:%M} ET)"


# --- Regional sessions -------------------------------------------------------

ASIA_SESSION = SessionWindow(
    "asia", time(20, 0), time(0, 0), WindowKind.SESSION,
    "Tokyo hours; typically consolidation that builds the Asian range.",
)
LONDON_SESSION = SessionWindow(
    "london", time(2, 0), time(5, 0), WindowKind.SESSION,
    "London hours; the day's first genuine expansion in FX and index futures.",
)
NEW_YORK_SESSION = SessionWindow(
    "new_york", time(7, 0), time(16, 0), WindowKind.SESSION,
    "New York hours through the equities close.",
)

# --- Killzones ---------------------------------------------------------------

ASIA_KILLZONE = SessionWindow(
    "asia_kz", time(20, 0), time(0, 0), WindowKind.KILLZONE,
    "Range-building. Its high/low become the first liquidity pools of the day.",
)
LONDON_KILLZONE = SessionWindow(
    "london_kz", time(2, 0), time(5, 0), WindowKind.KILLZONE,
    "London open killzone — judas swing against the Asian range, then reversal.",
)
NY_AM_KILLZONE = SessionWindow(
    "ny_am_kz", time(8, 0), time(11, 0), WindowKind.KILLZONE,
    "New York AM killzone — highest-probability window for the daily range extreme.",
)
LONDON_CLOSE_KILLZONE = SessionWindow(
    "london_close_kz", time(10, 0), time(12, 0), WindowKind.KILLZONE,
    "London close killzone — reversion/profit-taking as European desks flatten.",
)
NY_PM_KILLZONE = SessionWindow(
    "ny_pm_kz", time(13, 30), time(16, 0), WindowKind.KILLZONE,
    "New York PM killzone — afternoon expansion into the equities close.",
)
NY_LUNCH = SessionWindow(
    "ny_lunch", time(12, 0), time(13, 0), WindowKind.SESSION,
    "Lunch macro — low participation; ICT treats breakouts here as suspect.",
)

# --- Silver Bullet windows ---------------------------------------------------

SILVER_BULLET_LONDON = SessionWindow(
    "sb_london", time(3, 0), time(4, 0), WindowKind.SILVER_BULLET,
    "London Silver Bullet.",
)
SILVER_BULLET_AM = SessionWindow(
    "sb_am", time(10, 0), time(11, 0), WindowKind.SILVER_BULLET,
    "New York AM Silver Bullet.",
)
SILVER_BULLET_PM = SessionWindow(
    "sb_pm", time(14, 0), time(15, 0), WindowKind.SILVER_BULLET,
    "New York PM Silver Bullet.",
)

# --- Reference ranges --------------------------------------------------------

CBDR = SessionWindow(
    "cbdr", time(14, 0), time(20, 0), WindowKind.REFERENCE_RANGE,
    "Central Bank Dealers Range; standard deviations project next-day extremes.",
)
ASIAN_RANGE = SessionWindow(
    "asian_range", time(19, 0), time(0, 0), WindowKind.REFERENCE_RANGE,
    "Asian range used as the alternative std-dev projection basis.",
)
FLOUT = SessionWindow(
    "flout", time(15, 0), time(16, 0), WindowKind.REFERENCE_RANGE,
    "Flout — the final NY hour, a third projection basis.",
)

# --- Macros ------------------------------------------------------------------

# The five canonical macros. Earlier versions of this module generated a macro
# for every hour of the session, which is wrong: ICT names five specific
# :50-:10 windows, each anchored to a killzone. Generating eight diluted the
# concept and meant `active_windows` reported macros that do not exist.
MACRO_WINDOWS: tuple[SessionWindow, ...] = (
    SessionWindow(
        "macro_london_early", time(0, 50), time(1, 10), WindowKind.MACRO,
        "London early macro — end of Asia, ahead of the London open killzone.",
    ),
    SessionWindow(
        "macro_london_open", time(2, 50), time(3, 10), WindowKind.MACRO,
        "London open macro — inside the London open killzone.",
    ),
    SessionWindow(
        "macro_ny_pre_open", time(9, 50), time(10, 10), WindowKind.MACRO,
        "New York pre-open macro — inside the NY AM killzone.",
    ),
    SessionWindow(
        "macro_ny_first_pm", time(13, 50), time(14, 10), WindowKind.MACRO,
        "First New York PM macro — inside the NY PM killzone.",
    ),
    SessionWindow(
        "macro_ny_mid_pm", time(14, 50), time(15, 10), WindowKind.MACRO,
        "Mid New York PM macro — inside the NY PM killzone.",
    ),
)

KILLZONES: tuple[SessionWindow, ...] = (
    ASIA_KILLZONE, LONDON_KILLZONE, NY_AM_KILLZONE, LONDON_CLOSE_KILLZONE, NY_PM_KILLZONE,
)
SILVER_BULLETS: tuple[SessionWindow, ...] = (
    SILVER_BULLET_LONDON, SILVER_BULLET_AM, SILVER_BULLET_PM,
)
REFERENCE_RANGES: tuple[SessionWindow, ...] = (CBDR, ASIAN_RANGE, FLOUT)
SESSIONS: tuple[SessionWindow, ...] = (
    ASIA_SESSION, LONDON_SESSION, NEW_YORK_SESSION, NY_LUNCH,
)

ALL_WINDOWS: tuple[SessionWindow, ...] = (
    *SESSIONS, *KILLZONES, *SILVER_BULLETS, *REFERENCE_RANGES, *MACRO_WINDOWS,
)

WINDOWS_BY_NAME: dict[str, SessionWindow] = {w.name: w for w in ALL_WINDOWS}


def get_window(name: str) -> SessionWindow:
    try:
        return WINDOWS_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown session window {name!r}; known: {sorted(WINDOWS_BY_NAME)}"
        ) from None


# --- Vectorised helpers ------------------------------------------------------


def to_ny(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a UTC index to New York local time, honouring DST."""
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(NY_TZ)


def _minutes_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    ny = to_ny(index)
    return np.asarray(ny.hour.to_numpy() * 60 + ny.minute.to_numpy())


def _window_bounds(window: SessionWindow) -> tuple[int, int]:
    return window.start.hour * 60 + window.start.minute, window.end.hour * 60 + window.end.minute


def window_mask(index: pd.DatetimeIndex, window: SessionWindow) -> np.ndarray:
    """Boolean mask selecting bars whose open falls inside ``window``."""
    minutes = _minutes_of_day(index)
    start, end = _window_bounds(window)
    if end <= start:  # crosses midnight
        return (minutes >= start) | (minutes < end)
    return (minutes >= start) & (minutes < end)


def active_windows(ts: pd.Timestamp) -> tuple[SessionWindow, ...]:
    """Every window containing ``ts``. A bar is routinely in several at once."""
    local = ts.tz_convert(NY_TZ) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(NY_TZ)
    t = time(local.hour, local.minute)
    return tuple(w for w in ALL_WINDOWS if w.contains_local(t))


def active_killzone(ts: pd.Timestamp) -> SessionWindow | None:
    """The killzone containing ``ts``, if any."""
    for window in active_windows(ts):
        if window.kind is WindowKind.KILLZONE:
            return window
    return None


def in_silver_bullet(ts: pd.Timestamp) -> SessionWindow | None:
    for window in active_windows(ts):
        if window.kind is WindowKind.SILVER_BULLET:
            return window
    return None


def trading_day(index: pd.DatetimeIndex) -> np.ndarray:
    """The New York calendar date each bar belongs to.

    ICT anchors the day to NY midnight (the "true day open"), which is also the
    natural boundary for daily-bias and daily-loss-limit accounting.
    """
    return np.asarray(to_ny(index).normalize().tz_localize(None).to_numpy())


def session_label(ts: pd.Timestamp) -> str:
    """Short label for display: the killzone if in one, else the session."""
    kz = active_killzone(ts)
    if kz is not None:
        return kz.name
    for window in active_windows(ts):
        if window.kind is WindowKind.SESSION:
            return window.name
    return "off_session"
