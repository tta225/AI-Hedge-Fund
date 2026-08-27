"""Preconditions that must hold before the desk is allowed to send anything.

A backtest can assume its data is present, fresh, and correct. A live desk
cannot assume any of those, and the failure mode when it does is specific and
expensive: **stale data looks exactly like a quiet market.**

A feed that froze twenty minutes ago serves the same last price on every poll.
Every indicator computed from it is stable, no error is raised, and the
strategy happily trades a market that has moved somewhere else entirely. By the
time the feed reconnects the position is against a price nobody saw coming.

So the guards here are all of the form "prove it is safe to trade", not "detect
that something went wrong". They run before every decision, and any failure
stops the desk rather than degrading it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class GuardStatus(str, Enum):
    """Outcome of a single guard."""

    PASS = "pass"
    #: Trading may continue but something needs attention.
    WARN = "warn"
    #: Trading stops. No new orders; existing protective orders stay at the
    #: venue, which is why they belong at the venue rather than in this process.
    HALT = "halt"

    @property
    def blocks_trading(self) -> bool:
        return self is GuardStatus.HALT


@dataclass(frozen=True, slots=True)
class GuardReport:
    """What every guard said, and whether the desk may act."""

    at: pd.Timestamp
    results: dict[str, tuple[GuardStatus, str]] = field(default_factory=dict)

    @property
    def status(self) -> GuardStatus:
        """The worst outcome across all guards."""
        if any(s is GuardStatus.HALT for s, _ in self.results.values()):
            return GuardStatus.HALT
        if any(s is GuardStatus.WARN for s, _ in self.results.values()):
            return GuardStatus.WARN
        return GuardStatus.PASS

    @property
    def may_trade(self) -> bool:
        return not self.status.blocks_trading

    @property
    def blocking(self) -> list[str]:
        return [
            f"{name}: {detail}"
            for name, (status, detail) in sorted(self.results.items())
            if status.blocks_trading
        ]

    def render(self) -> str:
        lines = [f"Guards at {self.at}: {self.status.value.upper()}"]
        for name, (status, detail) in sorted(self.results.items()):
            marker = {"pass": " ", "warn": "!", "halt": "X"}[status.value]
            lines.append(f"  [{marker}] {name:<22} {detail}")
        return "\n".join(lines)


def check_data_freshness(
    last_bar_at: pd.Timestamp | None,
    now: pd.Timestamp,
    *,
    expected_interval: pd.Timedelta,
    stale_multiple: float = 2.0,
    halt_multiple: float = 4.0,
) -> tuple[GuardStatus, str]:
    """Whether the most recent bar is recent enough to trade on.

    Thresholds are multiples of the bar interval rather than fixed durations,
    because "two minutes old" is fine on a daily bar and catastrophic on a
    one-minute one.

    A missing timestamp is a HALT, not a WARN. No data at all is strictly worse
    than old data, and defaulting the unknown case to "proceed" is how a guard
    becomes decorative.
    """
    if last_bar_at is None:
        return GuardStatus.HALT, "no bars received"
    if expected_interval <= pd.Timedelta(0):
        raise ValueError("expected_interval must be positive")
    if halt_multiple < stale_multiple:
        raise ValueError("halt_multiple must be at least stale_multiple")

    age = pd.Timestamp(now) - pd.Timestamp(last_bar_at)
    if age < pd.Timedelta(0):
        # A bar stamped in the future means a clock problem somewhere, and a
        # desk that cannot trust its clock cannot trust its bar ordering.
        return GuardStatus.HALT, f"last bar is {-age} in the future — check NTP"

    if age > expected_interval * halt_multiple:
        return GuardStatus.HALT, f"last bar {age} old (>{halt_multiple}x interval)"
    if age > expected_interval * stale_multiple:
        return GuardStatus.WARN, f"last bar {age} old (>{stale_multiple}x interval)"
    return GuardStatus.PASS, f"last bar {age} old"


def check_drawdown(
    equity: float,
    peak_equity: float,
    *,
    derisk_pct: float = 5.0,
    halt_pct: float = 10.0,
) -> tuple[GuardStatus, str]:
    """Drawdown from the high-water mark.

    Two thresholds rather than one: the first says "something is wrong, take
    less risk", the second says "stop". A single halt level makes the desk fine
    right up until it is finished, with no intermediate state in which a human
    might look at it.
    """
    if halt_pct < derisk_pct:
        raise ValueError("halt_pct must be at least derisk_pct")
    if peak_equity <= 0:
        return GuardStatus.HALT, "no positive peak equity recorded"

    drawdown = max(0.0, (peak_equity - equity) / peak_equity * 100.0)
    if drawdown >= halt_pct:
        return GuardStatus.HALT, f"drawdown {drawdown:.2f}% >= {halt_pct:.1f}%"
    if drawdown >= derisk_pct:
        return GuardStatus.WARN, f"drawdown {drawdown:.2f}% >= {derisk_pct:.1f}%"
    return GuardStatus.PASS, f"drawdown {drawdown:.2f}%"


def check_guards(
    *,
    now: pd.Timestamp,
    last_bar_at: pd.Timestamp | None,
    expected_interval: pd.Timedelta,
    equity: float,
    peak_equity: float,
    store_halted: bool,
    kill_switch: bool,
    reconciled: bool,
    extra: dict[str, Callable[[], tuple[GuardStatus, str]]] | None = None,
) -> GuardReport:
    """Run every precondition and collect the answers.

    The order is deliberate but not short-circuiting: every guard runs even
    when an earlier one has already halted, because an incident review wants
    the whole picture rather than the first thing that tripped.
    """
    results: dict[str, tuple[GuardStatus, str]] = {
        "kill_switch": (
            (GuardStatus.HALT, "kill switch engaged")
            if kill_switch
            else (GuardStatus.PASS, "released")
        ),
        "persisted_halt": (
            (GuardStatus.HALT, "an uncleared halt is recorded in the store")
            if store_halted
            else (GuardStatus.PASS, "none active")
        ),
        "reconciliation": (
            (GuardStatus.PASS, "positions agree with the broker")
            if reconciled
            else (GuardStatus.HALT, "position book disagrees with the broker")
        ),
        "data_freshness": check_data_freshness(
            last_bar_at, now, expected_interval=expected_interval
        ),
        "drawdown": check_drawdown(equity, peak_equity),
    }

    for name, guard in (extra or {}).items():
        try:
            results[name] = guard()
        except Exception as exc:
            # A guard that raises has told us nothing about whether it is safe
            # to trade, and "unknown" resolves to "no".
            results[name] = (GuardStatus.HALT, f"guard raised {type(exc).__name__}: {exc}")

    return GuardReport(at=pd.Timestamp(now), results=results)
