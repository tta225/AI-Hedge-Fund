"""What is worth waking someone for, said once.

The failure mode of alerting is not missing an alert. It is sending so many
that people stop reading them, and then missing one inside the noise. A desk
that is halted emits the same condition on every poll; a naive router sends it
every thirty seconds until someone silences the channel, and the next real
alert arrives into a muted channel.

So this router does two things a plain callback does not:

**Deduplicates by key, with a cooldown.** The same condition re-alerts only
after :attr:`AlertRouter.cooldown`. A persistent problem produces a heartbeat
of reminders rather than a flood.

**Sends a resolution.** A condition that clears sends exactly one "recovered"
message. Without it, the only way to know a problem ended is that the alerts
stopped — which is indistinguishable from the alerter itself dying.

Sinks are injected callables. Nothing here knows about email, Slack or PagerDuty
on purpose: a delivery integration would drag credentials and a network
dependency into the one component that has to keep working when the network is
the problem. A sink that raises is caught and reported, because an alerter that
dies while alerting is worse than one that says nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)

#: Minimum gap before the same condition alerts again.
DEFAULT_COOLDOWN = pd.Timedelta(minutes=30)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    #: Trading has stopped, or is running on state known to be wrong.
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass(frozen=True, slots=True)
class Alert:
    """One thing worth telling a human."""

    key: str
    severity: Severity
    summary: str
    at: pd.Timestamp
    detail: str = ""
    #: True when this reports a condition clearing rather than occurring.
    resolved: bool = False
    context: dict[str, object] = field(default_factory=dict)

    def render(self) -> str:
        prefix = "RESOLVED" if self.resolved else self.severity.value.upper()
        line = f"[{prefix}] {self.summary}"
        if self.detail:
            line += f" — {self.detail}"
        return line


Sink = Callable[[Alert], None]


def console_sink(alert: Alert) -> None:
    """Log an alert as a structured event. The default, and dependency-free."""
    from axiom.ops.logs import log_event

    log_event(
        logger,
        "alert",
        level=logging.ERROR if alert.severity is Severity.CRITICAL else logging.WARNING,
        alert_key=alert.key,
        severity=alert.severity.value,
        resolved=alert.resolved,
        summary=alert.summary,
        detail=alert.detail,
        **alert.context,
    )


class AlertRouter:
    """Sends alerts, once, to every configured sink.

    Args:
        sinks: delivery callables. Defaults to :func:`console_sink`.
        cooldown: minimum gap before an unresolved condition re-alerts.
        min_severity: alerts below this are dropped.
    """

    def __init__(
        self,
        sinks: list[Sink] | None = None,
        *,
        cooldown: pd.Timedelta = DEFAULT_COOLDOWN,
        min_severity: Severity = Severity.WARNING,
    ) -> None:
        if cooldown < pd.Timedelta(0):
            raise ValueError("cooldown must not be negative")
        self.sinks = list(sinks) if sinks is not None else [console_sink]
        self.cooldown = cooldown
        self.min_severity = min_severity
        #: Last time each key was sent, for deduplication.
        self._sent: dict[str, pd.Timestamp] = {}
        #: Keys currently in an alerting state, so a resolution can be paired.
        self._active: set[str] = set()

    def send(self, alert: Alert) -> bool:
        """Deliver ``alert`` unless it is a duplicate. Returns whether it went.

        A resolution always goes if the condition was active, regardless of
        cooldown — the whole point of a resolution is that it arrives promptly,
        and suppressing it would leave the last word being the problem.
        """
        if alert.severity.rank < self.min_severity.rank and not alert.resolved:
            return False

        if alert.resolved:
            if alert.key not in self._active:
                # Never alerted, so there is nothing to resolve. Sending anyway
                # would tell someone a problem ended that they never heard of.
                return False
            self._active.discard(alert.key)
            self._sent.pop(alert.key, None)
            self._deliver(alert)
            return True

        last = self._sent.get(alert.key)
        if last is not None and alert.at - last < self.cooldown:
            return False

        self._sent[alert.key] = alert.at
        self._active.add(alert.key)
        self._deliver(alert)
        return True

    def _deliver(self, alert: Alert) -> None:
        for sink in self.sinks:
            try:
                sink(alert)
            except Exception:
                # An alerter that dies while alerting is worse than one that
                # says nothing: the condition it was reporting is still there,
                # and now nobody will hear about the next one either.
                logger.exception("alert sink %r failed", getattr(sink, "__name__", sink))

    def resolve(self, key: str, at: pd.Timestamp, summary: str = "") -> bool:
        """Convenience for clearing a condition by key."""
        return self.send(
            Alert(
                key=key,
                severity=Severity.INFO,
                summary=summary or f"{key} recovered",
                at=at,
                resolved=True,
            )
        )

    @property
    def active(self) -> set[str]:
        """Conditions currently alerting and not yet resolved."""
        return set(self._active)


def alerts_for_health(report: object) -> list[Alert]:
    """Translate a :class:`~axiom.ops.health.HealthReport` into alerts.

    One alert per failing check rather than one for the whole report, so a
    router can deduplicate and resolve each condition independently — a desk
    whose halt clears while its heartbeat is still stale has genuinely improved,
    and a single combined alert cannot express that.
    """
    from axiom.ops.health import HealthReport, HealthStatus

    if not isinstance(report, HealthReport):
        raise TypeError(f"expected a HealthReport, got {type(report).__name__}")

    severity = {
        HealthStatus.OK: Severity.INFO,
        HealthStatus.DEGRADED: Severity.WARNING,
        HealthStatus.DOWN: Severity.CRITICAL,
    }
    return [
        Alert(
            key=f"health.{name}",
            severity=severity[status],
            summary=f"desk {name} is {status.value}",
            detail=detail,
            at=report.at,
            context=dict(report.facts),
        )
        for name, (status, detail) in sorted(report.checks.items())
        if status is not HealthStatus.OK
    ]
