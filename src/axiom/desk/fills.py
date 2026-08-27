"""Pull executions from the venue into the store, exactly once each.

The desk records what it *sent*. This records what actually *happened*, and the
gap between the two is where a live system goes wrong. An order submitted and
then filled at the venue changes the position whether or not this process
noticed, so without a poller the store's position book drifts from the broker's
the moment anything fills — and reconciliation, correctly, halts the desk.

Four properties, each closing a specific failure:

**Exactly-once, by the venue's fill id.** The store's ``venue_fill_id`` is
UNIQUE, so re-reading a window books nothing twice. That is what makes the
overlap below free rather than dangerous.

**A deliberate overlap on every poll.** The watermark is rewound by
:attr:`FillPoller.overlap` before each request. Venue clocks and this one
disagree, and activity endpoints order by *transaction* time rather than by
insertion, so a fill can appear with a timestamp fractionally before the last
watermark. Polling from exactly where the last one ended loses those silently.

**Unattributed fills are recorded, not dropped.** A bracket's protective legs
fill under order ids the desk never submitted, and a human can trade the
account directly. Those are real position changes; discarding them because they
do not match a known order would recreate the exact drift this module exists to
prevent. They are stored with a null order link and reported.

**The watermark advances only after the write commits.** Advancing first means
a crash between the two loses the window permanently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from axiom.execution.alpaca import VenueFill
from axiom.store.db import Store

logger = logging.getLogger(__name__)

#: Store key holding the poll watermark.
WATERMARK_KEY = "fills_watermark_us"
#: How far to rewind the watermark before each poll. Generous, because the
#: deduplication makes an over-large overlap cost only bandwidth while an
#: under-large one costs a lost fill.
DEFAULT_OVERLAP = pd.Timedelta(minutes=5)
#: How far back a first-ever poll reaches. A desk with no watermark has no idea
#: what it missed, so it looks back far enough to find a recent session.
DEFAULT_COLD_START = pd.Timedelta(days=1)


@dataclass(slots=True)
class PollOutcome:
    """What one poll found."""

    at: pd.Timestamp
    fetched: int = 0
    recorded: int = 0
    duplicates: int = 0
    unattributed: list[str] = field(default_factory=list)
    error: str = ""
    watermark: pd.Timestamp | None = None

    @property
    def changed_positions(self) -> bool:
        return self.recorded > 0

    def render(self) -> str:
        if self.error:
            return f"[{self.at}] fill poll FAILED — {self.error}"
        parts = [
            f"[{self.at}] {self.fetched} activities, {self.recorded} new fill(s)"
        ]
        if self.duplicates:
            parts.append(f"{self.duplicates} already booked")
        if self.unattributed:
            parts.append(
                f"{len(self.unattributed)} unattributed "
                f"({', '.join(self.unattributed[:3])})"
            )
        return " | ".join(parts)


@runtime_checkable
class FillsSource(Protocol):
    """Anything that can report executions.

    A Protocol rather than a base class so a venue satisfies it by having the
    method, without inheriting from this module. That keeps the dependency
    pointing one way — the poller knows what it needs, and a test, a replay
    harness or a second broker can stand in without dragging an HTTP client
    into the type.
    """

    def fills_since(
        self, since: pd.Timestamp, limit: int = 500
    ) -> list[VenueFill]: ...


class FillPoller:
    """Reads executions from a venue and books them, once each.

    Owns no timer, like :class:`~axiom.desk.runner.DeskRunner`. The caller
    decides when to :meth:`poll` — after a tick, on a schedule, from a test.
    A scheduler in here would make the interesting logic untestable without
    waiting in real time.

    Args:
        venue: anything exposing ``fills_since``.
        store: where fills are booked and the watermark lives.
        overlap: how far to rewind the watermark each poll.
        cold_start: lookback when no watermark exists yet.
        limit: maximum activities per request.
    """

    def __init__(
        self,
        venue: FillsSource,
        store: Store,
        *,
        overlap: pd.Timedelta = DEFAULT_OVERLAP,
        cold_start: pd.Timedelta = DEFAULT_COLD_START,
        limit: int = 500,
    ) -> None:
        if overlap < pd.Timedelta(0):
            raise ValueError("overlap must not be negative")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.venue = venue
        self.store = store
        self.overlap = overlap
        self.cold_start = cold_start
        self.limit = limit

    def watermark(self, now: pd.Timestamp | None = None) -> pd.Timestamp:
        """The point this poller last read up to.

        Falls back to ``now - cold_start`` when nothing is recorded, rather
        than to the epoch: a first poll that requests every activity since 1970
        is a slow request that will be rate-limited, and a desk that cannot
        start is a desk that cannot trade.
        """
        raw = self.store.get_meta(WATERMARK_KEY)
        if raw:
            return pd.Timestamp(int(raw) * 1_000, unit="ns", tz="UTC")
        moment = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        return moment - self.cold_start

    def poll(self, now: pd.Timestamp | None = None) -> PollOutcome:
        """Fetch, attribute and book every execution since the watermark."""
        moment = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        outcome = PollOutcome(at=moment)

        since = self.watermark(moment) - self.overlap
        try:
            venue_fills = self.venue.fills_since(since, limit=self.limit)
        except Exception as exc:
            # Deliberately broad, and deliberately not advancing the watermark:
            # a transient venue error should be retried on the next poll, and
            # the window it covered must remain unread.
            outcome.error = f"{type(exc).__name__}: {exc}"
            logger.error("fill poll failed: %s", outcome.error)
            return outcome

        outcome.fetched = len(venue_fills)
        latest = since
        for venue_fill in venue_fills:
            order_id = self._resolve(venue_fill, outcome)
            booked = self.store.record_fill(
                order_id, venue_fill.fill, venue_fill.venue_fill_id or None
            )
            if booked:
                outcome.recorded += 1
            else:
                outcome.duplicates += 1
            latest = max(latest, pd.Timestamp(venue_fill.fill.timestamp))

        # Advanced last, and only to what was actually read. A crash before
        # this point re-reads the window; a crash after it would have lost it.
        outcome.watermark = latest
        self.store.set_meta(WATERMARK_KEY, str(_to_us(latest)))
        return outcome

    def _resolve(self, venue_fill: VenueFill, outcome: PollOutcome) -> int | None:
        """Map a venue order id onto a local order, or report that it cannot be.

        Returning None is a normal outcome, not a failure. Bracket legs and
        manual trades genuinely have no local order, and the position change is
        just as real.
        """
        if not venue_fill.venue_order_id:
            outcome.unattributed.append(
                f"{venue_fill.fill.instrument.symbol} (no venue order id)"
            )
            return None
        stored = self.store.order_by_venue_id(venue_fill.venue_order_id)
        if stored is None:
            outcome.unattributed.append(
                f"{venue_fill.fill.instrument.symbol} "
                f"({venue_fill.venue_order_id[:8]})"
            )
            return None
        return stored.id


def _to_us(timestamp: pd.Timestamp) -> int:
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value // 1_000)
