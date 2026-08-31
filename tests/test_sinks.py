"""Alert delivery.

The properties worth pinning are the ones whose violation is silent or
actively harmful:

* a sink must never raise into the desk loop — it is reporting that something
  is already wrong, and a traceback about the warning replaces the warning,
* a resolution must go out even when its severity is below the sink's
  threshold, or an incident opened by a critical alert never closes,
* a webhook URL must not reach the log, because it is a credential.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from axiom.ops.alerts import Alert, AlertRouter, Severity
from axiom.ops.sinks import (
    FileSink,
    PagerDutySink,
    SlackSink,
    WebhookSink,
    sinks_from_environment,
)

AT = pd.Timestamp("2026-08-31T12:00:00")


def _alert(
    severity: Severity = Severity.CRITICAL, *, resolved: bool = False, key: str = "desk.halted"
) -> Alert:
    return Alert(
        key=key,
        severity=severity,
        summary="desk is halted",
        at=AT,
        detail="reconciliation failed",
        resolved=resolved,
        context={"symbol": "AAPL", "equity": 100_000.0},
    )


@pytest.fixture
def posted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Capture what each sink would have sent, without a network."""
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, payload: dict, timeout: float) -> bool:
        calls.append((url, payload))
        return True

    monkeypatch.setattr("axiom.ops.sinks._post", fake_post)
    return calls


class TestSlackSink:
    def test_sends_severity_as_colour(self, posted) -> None:
        SlackSink("https://hooks.slack.test/x")(_alert())
        _, payload = posted[0]
        attachment = payload["attachments"][0]
        assert attachment["color"] == "#d00000"
        assert "CRITICAL" in attachment["title"]

    def test_environment_is_visible_in_the_title(self, posted) -> None:
        """A paper alert that looks like a live alert is worse than none."""
        SlackSink("https://hooks.slack.test/x", environment="live")(_alert())
        assert "[live]" in posted[0][1]["attachments"][0]["title"]

    def test_a_resolution_is_marked_and_recoloured(self, posted) -> None:
        SlackSink("https://hooks.slack.test/x")(_alert(resolved=True))
        attachment = posted[0][1]["attachments"][0]
        assert "RESOLVED" in attachment["title"]
        assert attachment["color"] == "#36a64f"

    def test_context_becomes_fields_and_is_capped(self, posted) -> None:
        alert = Alert(
            key="k", severity=Severity.WARNING, summary="s", at=AT,
            context={f"f{i}": i for i in range(20)},
        )
        SlackSink("https://hooks.slack.test/x")(alert)
        assert len(posted[0][1]["attachments"][0]["fields"]) == 8


class TestPagerDutySink:
    def test_critical_triggers(self, posted) -> None:
        PagerDutySink("routing-key")(_alert())
        _, payload = posted[0]
        assert payload["event_action"] == "trigger"
        assert payload["payload"]["severity"] == "critical"

    def test_warnings_do_not_page_by_default(self, posted) -> None:
        """Routing warnings to a pager is how a pager gets ignored."""
        PagerDutySink("routing-key")(_alert(Severity.WARNING))
        assert posted == []

    def test_a_resolution_goes_through_below_the_threshold(self, posted) -> None:
        """The failure this guards: an incident opened by a critical alert
        must close, and its resolution carries INFO severity."""
        PagerDutySink("routing-key")(_alert(Severity.INFO, resolved=True))
        assert posted, "a resolution below min_severity must still be delivered"
        assert posted[0][1]["event_action"] == "resolve"

    def test_dedup_key_pairs_trigger_with_resolve(self, posted) -> None:
        sink = PagerDutySink("routing-key", environment="live")
        sink(_alert())
        sink(_alert(Severity.INFO, resolved=True))
        assert posted[0][1]["dedup_key"] == posted[1][1]["dedup_key"]
        assert posted[0][1]["dedup_key"] == "axiom.live.desk.halted"

    def test_a_resolution_carries_no_payload(self, posted) -> None:
        PagerDutySink("routing-key")(_alert(resolved=True))
        assert "payload" not in posted[0][1]


class TestWebhookSink:
    def test_posts_the_alert_fields(self, posted) -> None:
        WebhookSink("https://example.test/hook", environment="live")(_alert())
        _, payload = posted[0]
        assert payload["key"] == "desk.halted"
        assert payload["severity"] == "critical"
        assert payload["environment"] == "live"
        assert payload["context"]["symbol"] == "AAPL"


class TestFileSink:
    def test_appends_json_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "alerts" / "alerts.jsonl"
        sink = FileSink(path)
        sink(_alert())
        sink(_alert(resolved=True))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["summary"] == "desk is halted"
        assert json.loads(lines[1])["resolved"] is True

    def test_an_unwritable_path_does_not_raise(self, tmp_path: Path) -> None:
        """When the outage is the disk, the alerter still must not die."""
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        FileSink(blocker / "alerts.jsonl")(_alert())


class TestFailureIsolation:
    def test_a_network_failure_does_not_raise(self, monkeypatch) -> None:
        """The desk loop calls this. It must never see an exception."""

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        SlackSink("https://hooks.slack.test/x")(_alert())
        WebhookSink("https://example.test/hook")(_alert())
        PagerDutySink("routing-key")(_alert())

    def test_the_url_never_reaches_the_log(self, monkeypatch, caplog) -> None:
        """A webhook URL is a credential: anyone holding it can post as the desk."""
        secret = "https://hooks.slack.test/T00/B00/SUPERSECRETTOKEN"

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with caplog.at_level("DEBUG"):
            SlackSink(secret)(_alert())
        assert "SUPERSECRETTOKEN" not in caplog.text

    def test_router_survives_a_sink_that_raises(self) -> None:
        delivered: list[Alert] = []

        def broken(alert: Alert) -> None:
            raise RuntimeError("sink is down")

        router = AlertRouter([broken, delivered.append])
        assert router.send(_alert())
        assert len(delivered) == 1, "a failing sink must not block the others"


class TestSinksFromEnvironment:
    def test_console_is_always_present(self, monkeypatch) -> None:
        for name in ("SLACK_WEBHOOK_URL", "PAGERDUTY_ROUTING_KEY", "ALERT_WEBHOOK_URL"):
            monkeypatch.delenv(name, raising=False)
            monkeypatch.delenv(f"AXIOM_PAPER_{name}", raising=False)
        monkeypatch.delenv("AXIOM_SECRETS_DIR", raising=False)
        monkeypatch.delenv("AXIOM_SECRETS_COMMAND", raising=False)
        assert len(sinks_from_environment()) == 1

    def test_configured_sinks_are_added(self, monkeypatch) -> None:
        monkeypatch.delenv("AXIOM_SECRETS_DIR", raising=False)
        monkeypatch.delenv("AXIOM_SECRETS_COMMAND", raising=False)
        monkeypatch.setenv("AXIOM_PAPER_SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
        monkeypatch.setenv("AXIOM_PAPER_PAGERDUTY_ROUTING_KEY", "rk")
        sinks = sinks_from_environment("paper")
        kinds = {type(s).__name__ for s in sinks}
        assert "SlackSink" in kinds
        assert "PagerDutySink" in kinds

    def test_live_does_not_inherit_paper_webhooks(self, monkeypatch) -> None:
        """A paper desk must not be able to page whoever carries the live one."""
        monkeypatch.delenv("AXIOM_SECRETS_DIR", raising=False)
        monkeypatch.delenv("AXIOM_SECRETS_COMMAND", raising=False)
        monkeypatch.delenv("AXIOM_LIVE_PAGERDUTY_ROUTING_KEY", raising=False)
        monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "unscoped-key")
        kinds = {type(s).__name__ for s in sinks_from_environment("live")}
        assert "PagerDutySink" not in kinds

    def test_file_sink_is_opt_in(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("AXIOM_SECRETS_DIR", raising=False)
        monkeypatch.delenv("AXIOM_SECRETS_COMMAND", raising=False)
        sinks = sinks_from_environment(log_path=tmp_path / "alerts.jsonl")
        assert any(isinstance(s, FileSink) for s in sinks)
