"""Getting an alert to a human who is not looking at the terminal.

:mod:`axiom.ops.alerts` decides *what* is worth saying and refuses to say it
twice. It ships only ``console_sink``, which writes a structured log line — and
an alert nobody receives is a log. This closes that gap.

Every sink here is written against the same three constraints, which is why
none of them takes an SDK dependency:

**The network is often the thing that broke.** A sink that hangs for sixty
seconds inside a desk tick has converted a notification problem into an
execution problem. Every sink has a short timeout and swallows its own failures
— :meth:`~axiom.ops.alerts.AlertRouter._deliver` already catches, but a sink
that raises on a routine 500 produces an exception traceback on every poll.

**A webhook URL is a credential.** Anyone holding a Slack or PagerDuty URL can
post as the desk. They resolve through :mod:`axiom.ops.secrets`, so they are
fingerprinted rather than printed and are scoped to an environment — a paper
desk should not be able to page the person carrying the live one.

**Delivery must be attempted for the resolution too.** A router that sends the
problem and not the recovery leaves the last word being the failure, and the
only way to learn it ended is that alerts stopped — indistinguishable from the
alerter dying.

The deliberate omission: no retry, no queue, no store-and-forward. A missed
alert is not recovered by trying harder inside a trading loop; it is recovered
by the health check, which is readable from the store by anything, at any time,
and does not depend on this module having worked.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from axiom.ops.alerts import Alert, Severity

logger = logging.getLogger(__name__)

#: Seconds a sink may spend delivering before it gives up. Deliberately short:
#: this runs on the desk's own thread, and a slow notifier must never be able
#: to delay a trading decision.
DEFAULT_TIMEOUT = 5.0

#: Slack's colour convention, so severity is visible before the text is read.
_SLACK_COLOURS = {
    Severity.INFO: "#36a64f",
    Severity.WARNING: "#daa038",
    Severity.CRITICAL: "#d00000",
}
#: PagerDuty's own severity vocabulary.
_PAGERDUTY_SEVERITY = {
    Severity.INFO: "info",
    Severity.WARNING: "warning",
    Severity.CRITICAL: "critical",
}


def _post(url: str, payload: dict[str, object], timeout: float) -> bool:
    """POST JSON, swallowing every failure. Returns whether it landed.

    Swallowing is correct here and nowhere else in the codebase: the caller is
    an alert router reporting that something is already wrong, and an exception
    escaping this would replace a useful warning with a traceback about the
    warning.
    """
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return bool(200 <= response.status < 300)
    except urllib.error.HTTPError as exc:
        # The URL is a credential; it must not reach the log.
        logger.warning("alert delivery rejected with HTTP %s", exc.code)
    except Exception as exc:
        logger.warning("alert delivery failed: %s", type(exc).__name__)
    return False


@dataclass(frozen=True, slots=True)
class SlackSink:
    """Post to a Slack incoming webhook.

    ``webhook_url`` is a credential — resolve it through
    :func:`~axiom.ops.secrets.CredentialSet.get` and pass ``.reveal()`` here,
    rather than reading an environment variable directly, so it is scoped to an
    environment and never rendered by accident.
    """

    webhook_url: str
    timeout: float = DEFAULT_TIMEOUT
    #: Prefixed to every message, so two desks posting to one channel are
    #: distinguishable. A paper alert that looks like a live alert is worse
    #: than no alert.
    environment: str = "paper"

    def __call__(self, alert: Alert) -> None:
        state = "RESOLVED" if alert.resolved else alert.severity.value.upper()
        fields = [
            {"title": key, "value": str(value), "short": True}
            for key, value in sorted(alert.context.items())
        ][:8]
        _post(
            self.webhook_url,
            {
                "attachments": [
                    {
                        "color": (
                            _SLACK_COLOURS[Severity.INFO]
                            if alert.resolved
                            else _SLACK_COLOURS[alert.severity]
                        ),
                        "title": f"[{self.environment}] {state}: {alert.summary}",
                        "text": alert.detail or "",
                        "fields": fields,
                        "footer": f"axiom · {alert.key}",
                        "ts": int(alert.at.timestamp()),
                    }
                ]
            },
            self.timeout,
        )


@dataclass(frozen=True, slots=True)
class PagerDutySink:
    """Trigger and resolve PagerDuty incidents through the Events API v2.

    The one sink that can wake someone at night, so it defaults to critical
    only. Routing warnings to a pager is how a pager gets ignored.

    ``dedup_key`` is the alert's own key, which means PagerDuty performs the
    same deduplication the router already does — belt and braces, and it makes
    the resolution automatic: a resolved alert closes the incident it opened
    rather than creating a second one.
    """

    routing_key: str
    timeout: float = DEFAULT_TIMEOUT
    min_severity: Severity = Severity.CRITICAL
    environment: str = "paper"
    url: str = "https://events.pagerduty.com/v2/enqueue"

    def __call__(self, alert: Alert) -> None:
        # A resolution must go through even when its severity is below the
        # threshold, or an incident opened by a critical alert never closes.
        if not alert.resolved and alert.severity.rank < self.min_severity.rank:
            return
        payload: dict[str, object] = {
            "routing_key": self.routing_key,
            "event_action": "resolve" if alert.resolved else "trigger",
            "dedup_key": f"axiom.{self.environment}.{alert.key}",
        }
        if not alert.resolved:
            payload["payload"] = {
                "summary": f"[{self.environment}] {alert.summary}",
                "severity": _PAGERDUTY_SEVERITY[alert.severity],
                "source": f"axiom-{self.environment}",
                "component": alert.key,
                "custom_details": {k: str(v) for k, v in alert.context.items()},
            }
        _post(self.url, payload, self.timeout)


@dataclass(frozen=True, slots=True)
class WebhookSink:
    """POST the alert as plain JSON to an arbitrary endpoint.

    For anything this module does not speak natively — an internal incident
    system, a Discord relay, a Lambda. The payload is the alert's own fields
    rather than a vendor's schema, so the receiving end decides what to do.
    """

    url: str
    timeout: float = DEFAULT_TIMEOUT
    environment: str = "paper"

    def __call__(self, alert: Alert) -> None:
        _post(
            self.url,
            {
                "environment": self.environment,
                "key": alert.key,
                "severity": alert.severity.value,
                "resolved": alert.resolved,
                "summary": alert.summary,
                "detail": alert.detail,
                "at": alert.at.isoformat(),
                "context": {k: str(v) for k, v in alert.context.items()},
            },
            self.timeout,
        )


@dataclass(frozen=True, slots=True)
class FileSink:
    """Append alerts as JSON lines to a file.

    The sink with no network dependency, which is exactly why it is worth
    having: when the outage *is* the network, this is the one that still works,
    and it leaves a local record an operator can read after the fact without
    reconstructing anything from a third party's retention policy.
    """

    path: Path
    environment: str = "paper"

    def __call__(self, alert: Alert) -> None:
        line = json.dumps(
            {
                "at": alert.at.isoformat(),
                "environment": self.environment,
                "key": alert.key,
                "severity": alert.severity.value,
                "resolved": alert.resolved,
                "summary": alert.summary,
                "detail": alert.detail,
                "context": {k: str(v) for k, v in alert.context.items()},
            }
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning("could not append alert to %s: %s", self.path, exc)


def sinks_from_environment(
    environment: str = "paper", *, log_path: str | Path | None = None
) -> list[Callable[[Alert], None]]:
    """Assemble sinks from whatever credentials are configured.

    Reads through :mod:`axiom.ops.secrets`, so a live desk cannot page using a
    paper webhook and vice versa. Anything not configured is simply absent —
    there is no partial or degraded mode, because a sink that silently drops
    messages is worse than one that was never wired up.

    ``console_sink`` is always included. It is the only one guaranteed to work
    when the network is the thing that failed.
    """
    from axiom.ops.alerts import console_sink
    from axiom.ops.secrets import Environment, default_credentials

    credentials = default_credentials(Environment(environment))
    out: list[Callable[[Alert], None]] = [console_sink]

    if log_path is not None:
        out.append(FileSink(Path(log_path), environment=environment))

    slack = credentials.get_optional("SLACK_WEBHOOK_URL")
    if slack is not None:
        out.append(SlackSink(slack.reveal(), environment=environment))

    pagerduty = credentials.get_optional("PAGERDUTY_ROUTING_KEY")
    if pagerduty is not None:
        out.append(PagerDutySink(pagerduty.reveal(), environment=environment))

    webhook = credentials.get_optional("ALERT_WEBHOOK_URL")
    if webhook is not None:
        out.append(WebhookSink(webhook.reveal(), environment=environment))

    return out
