"""Is the desk alive, and is what it believes still current?

Deliberately readable from the **store alone**. A health check that has to ask
the running process is useless in the one case that matters — the process being
wedged — because a hung desk answers "fine" or does not answer at all, and both
are indistinguishable from a network problem.

Reading the database instead means a watchdog, a cron job, or a human can
answer the question from outside, and the answer is derived from evidence the
desk had to write down in order to act.

The distinction that makes this worth having:

*Liveness* is "did the desk do anything recently". A desk that has not marked
equity in an hour is either stopped or wedged, and neither is fine.

*Correctness* is "is what it recorded still true" — an open halt, a position
with no equity mark behind it, a fill watermark that stopped advancing while
orders were still working.

A desk can be perfectly live and completely wrong, so both are reported
separately rather than collapsed into one green light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from axiom.store.db import Store

#: Beyond this without an equity mark, the desk is presumed not running.
DEFAULT_HEARTBEAT_WARN = pd.Timedelta(minutes=15)
DEFAULT_HEARTBEAT_FAIL = pd.Timedelta(hours=1)
#: Beyond this without a fill poll, positions may have drifted unnoticed.
DEFAULT_FILL_POLL_WARN = pd.Timedelta(minutes=30)


class HealthStatus(str, Enum):
    OK = "ok"
    #: Working, but something needs looking at.
    DEGRADED = "degraded"
    #: Not working, or working on state known to be wrong.
    DOWN = "down"

    @property
    def is_healthy(self) -> bool:
        return self is HealthStatus.OK


@dataclass(frozen=True, slots=True)
class HealthReport:
    """One health assessment, and the evidence behind it."""

    at: pd.Timestamp
    status: HealthStatus
    checks: dict[str, tuple[HealthStatus, str]] = field(default_factory=dict)
    #: Facts a dashboard wants without re-querying.
    facts: dict[str, object] = field(default_factory=dict)

    @property
    def failing(self) -> list[str]:
        return [
            f"{name}: {detail}"
            for name, (status, detail) in sorted(self.checks.items())
            if status is HealthStatus.DOWN
        ]

    @property
    def exit_code(self) -> int:
        """0 healthy, 1 degraded, 2 down — for a shell watchdog."""
        return {HealthStatus.OK: 0, HealthStatus.DEGRADED: 1, HealthStatus.DOWN: 2}[
            self.status
        ]

    def render(self) -> str:
        lines = [f"Desk health at {self.at}: {self.status.value.upper()}"]
        for name, (status, detail) in sorted(self.checks.items()):
            marker = {"ok": " ", "degraded": "!", "down": "X"}[status.value]
            lines.append(f"  [{marker}] {name:<20} {detail}")
        if self.facts:
            lines.append("")
            for key, value in sorted(self.facts.items()):
                lines.append(f"  {key:<20} {value}")
        return "\n".join(lines)


def check_health(
    store: Store,
    *,
    now: pd.Timestamp | None = None,
    heartbeat_warn: pd.Timedelta = DEFAULT_HEARTBEAT_WARN,
    heartbeat_fail: pd.Timedelta = DEFAULT_HEARTBEAT_FAIL,
    fill_poll_warn: pd.Timedelta = DEFAULT_FILL_POLL_WARN,
) -> HealthReport:
    """Assess a desk from its database.

    Args:
        store: the desk's store, opened read-only or otherwise.
        now: assessment time. Defaults to now, UTC.
        heartbeat_warn: equity-mark age at which liveness degrades.
        heartbeat_fail: age at which the desk is presumed down.
        fill_poll_warn: fill-watermark age at which drift becomes plausible.
    """
    if heartbeat_fail < heartbeat_warn:
        raise ValueError("heartbeat_fail must be at least heartbeat_warn")

    moment = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    checks: dict[str, tuple[HealthStatus, str]] = {}
    facts: dict[str, object] = {}

    curve = store.equity_curve()
    if curve.empty:
        # Never traded is not the same as stopped, and calling a fresh desk
        # "down" would train whoever reads this to ignore it.
        checks["heartbeat"] = (
            HealthStatus.DEGRADED,
            "no equity marks recorded — the desk has never run",
        )
    else:
        last_mark = pd.Timestamp(curve.index[-1])
        age = moment - last_mark
        facts["last_heartbeat"] = str(last_mark)
        facts["equity"] = f"{float(curve.iloc[-1]):,.2f}"
        peak = float(curve.max())
        facts["peak_equity"] = f"{peak:,.2f}"
        if peak > 0:
            facts["drawdown_pct"] = f"{(peak - float(curve.iloc[-1])) / peak * 100:.2f}"

        if age > heartbeat_fail:
            checks["heartbeat"] = (
                HealthStatus.DOWN,
                f"last equity mark {age} ago — the desk is not running",
            )
        elif age > heartbeat_warn:
            checks["heartbeat"] = (HealthStatus.DEGRADED, f"last equity mark {age} ago")
        else:
            checks["heartbeat"] = (HealthStatus.OK, f"last equity mark {age} ago")

    halts = store.active_halts()
    facts["active_halts"] = len(halts)
    if halts:
        # DOWN rather than DEGRADED: a halted desk is not trading, which is
        # exactly the condition someone needs to be told about.
        checks["halts"] = (
            HealthStatus.DOWN,
            f"{len(halts)} uncleared: " + "; ".join(str(h["reason"]) for h in halts),
        )
    else:
        checks["halts"] = (HealthStatus.OK, "none active")

    # Liveness of the poller is the *poll* timestamp, not the watermark. The
    # watermark is the newest fill's transaction time and legitimately sits
    # days in the past on a quiet desk that is polling perfectly.
    last_poll = store.get_meta("fills_last_poll_us")
    watermark = store.get_meta("fills_watermark_us")
    open_orders = store.open_orders()
    facts["open_orders"] = len(open_orders)
    if watermark:
        facts["newest_fill"] = str(
            pd.Timestamp(int(watermark) * 1_000, unit="ns", tz="UTC")
        )
    if not last_poll:
        checks["fill_polling"] = (
            HealthStatus.DEGRADED if open_orders else HealthStatus.OK,
            "no fill poll recorded"
            + (" while orders are working" if open_orders else " (nothing to poll)"),
        )
    else:
        polled_at = pd.Timestamp(int(last_poll) * 1_000, unit="ns", tz="UTC")
        age = moment - polled_at
        facts["last_fill_poll"] = str(polled_at)
        if age > fill_poll_warn:
            # Degraded rather than down: the fills are not lost, they are
            # unbooked, and reconciliation will catch the drift they cause.
            checks["fill_polling"] = (
                HealthStatus.DEGRADED,
                f"fills last polled {age} ago — positions may have drifted",
            )
        else:
            checks["fill_polling"] = (HealthStatus.OK, f"fills polled {age} ago")

    positions = store.positions()
    facts["positions"] = len(positions)
    if positions and curve.empty:
        checks["consistency"] = (
            HealthStatus.DOWN,
            f"{len(positions)} position(s) held with no equity ever marked — "
            "the store is inconsistent",
        )
    else:
        checks["consistency"] = (HealthStatus.OK, "positions and equity agree")

    return HealthReport(
        at=moment, status=_worst(checks), checks=checks, facts=facts
    )


def _worst(checks: dict[str, tuple[HealthStatus, str]]) -> HealthStatus:
    if any(status is HealthStatus.DOWN for status, _ in checks.values()):
        return HealthStatus.DOWN
    if any(status is HealthStatus.DEGRADED for status, _ in checks.values()):
        return HealthStatus.DEGRADED
    return HealthStatus.OK
