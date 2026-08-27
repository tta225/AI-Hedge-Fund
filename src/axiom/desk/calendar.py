"""When a market is open, without pretending to know more than it does.

A desk that trades equities at 03:00 UTC on a Sunday sends orders into a
closed book. They rest until the open and fill on a gap, which is the worst
possible execution of a decision that was made on stale information anyway.

The honest difficulty here is that a real exchange calendar is a *data
problem*, not a logic problem: holidays move, half-days exist, and they change
by year and by venue. Hard-coding a list of dates produces something that is
correct until it silently is not — which for a calendar means the desk trades
on a holiday, once, and nobody notices until the fills do not arrive.

So this module does two things and refuses to do a third:

* :class:`AlwaysOpen` for continuously traded instruments. Correct, complete,
  and the right default for crypto.
* :class:`RegularHours` for a weekday session in a fixed timezone, with an
  **explicit** holiday set the caller supplies. It knows weekends, which never
  move, and it knows nothing else it was not told.
* It does **not** ship a holiday list. A stale built-in list is worse than no
  list, because it reads as authoritative. :meth:`RegularHours.us_equities`
  builds the schedule and takes the holidays as an argument, so a caller who
  passes none gets a calendar that is right about weekends and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from typing import Protocol, runtime_checkable

import pandas as pd

#: US equity regular session, in exchange-local time.
US_EQUITY_OPEN = time(9, 30)
US_EQUITY_CLOSE = time(16, 0)
US_EQUITY_TZ = "America/New_York"


@runtime_checkable
class TradingCalendar(Protocol):
    """Anything that can say whether a market is open at an instant."""

    def is_open(self, at: pd.Timestamp) -> bool: ...


@dataclass(frozen=True, slots=True)
class AlwaysOpen:
    """A market that never closes. Correct for crypto, and only for crypto."""

    name: str = "always_open"

    def is_open(self, at: pd.Timestamp) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class RegularHours:
    """A weekday session between two local times, minus known holidays.

    Args:
        open_time: session open, in ``timezone``.
        close_time: session close, in ``timezone``.
        timezone: the exchange's timezone, not the desk's. A server in UTC
            and an exchange in New York disagree by an amount that changes
            twice a year, and hard-coding the offset is how a desk trades an
            hour early every spring.
        holidays: dates the market is closed. Supplied by the caller because
            this module has no way to keep a list current, and a stale list
            that reads as authoritative is worse than an empty one.
        half_days: dates with an early close, mapped to that close time.
    """

    open_time: time = US_EQUITY_OPEN
    close_time: time = US_EQUITY_CLOSE
    timezone: str = US_EQUITY_TZ
    holidays: frozenset[date] = field(default_factory=frozenset)
    half_days: dict[date, time] = field(default_factory=dict)
    name: str = "regular_hours"

    def __post_init__(self) -> None:
        if self.open_time >= self.close_time:
            raise ValueError(
                f"open {self.open_time} is not before close {self.close_time}"
            )

    @classmethod
    def us_equities(
        cls,
        holidays: frozenset[date] | set[date] | None = None,
        half_days: dict[date, time] | None = None,
    ) -> RegularHours:
        """A US equity session. Holidays are the caller's to supply.

        Passing none is a supported and honest configuration: the result knows
        weekends, which is most of the closures, and
        :meth:`describe` will say plainly that it does not know holidays.
        """
        return cls(
            open_time=US_EQUITY_OPEN,
            close_time=US_EQUITY_CLOSE,
            timezone=US_EQUITY_TZ,
            holidays=frozenset(holidays or ()),
            half_days=dict(half_days or {}),
            name="us_equities",
        )

    def is_open(self, at: pd.Timestamp) -> bool:
        """Whether the session is running at ``at``.

        A naive timestamp is treated as UTC, this repository's convention
        everywhere. Reading the local hour off an aware timestamp without
        converting is how a calendar silently answers for the wrong timezone.
        """
        stamp = pd.Timestamp(at)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        local = stamp.tz_convert(self.timezone)

        if local.weekday() >= 5:
            return False
        day = local.date()
        if day in self.holidays:
            return False

        close = self.half_days.get(day, self.close_time)
        return bool(self.open_time <= local.time() < close)

    def describe(self) -> str:
        """What this calendar does and does not know."""
        lines = [
            f"{self.name}: {self.open_time}–{self.close_time} {self.timezone}, "
            "weekdays only",
        ]
        if self.holidays:
            lines.append(f"  {len(self.holidays)} holiday(s) configured")
        else:
            lines.append(
                "  ! no holidays configured — this calendar will report the "
                "market open on Thanksgiving. Supply them if that matters."
            )
        if self.half_days:
            lines.append(f"  {len(self.half_days)} half-day(s) configured")
        return "\n".join(lines)


def next_open(calendar: TradingCalendar, after: pd.Timestamp, *, limit_days: int = 10) -> pd.Timestamp | None:
    """The next minute the market is open, or None within ``limit_days``.

    Searched minute by minute rather than solved analytically, because a
    calendar is a Protocol and an arbitrary implementation has no closed form.
    Bounded so a misconfigured calendar that is never open returns rather than
    spinning.
    """
    if limit_days < 1:
        raise ValueError("limit_days must be at least 1")
    stamp = pd.Timestamp(after)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    stamp = stamp.ceil("min")

    for _ in range(limit_days * 24 * 60):
        if calendar.is_open(stamp):
            return stamp
        stamp += pd.Timedelta(minutes=1)
    return None
