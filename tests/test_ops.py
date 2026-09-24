"""Tests for the ops layer: structured logs, health checks, alerting."""

from __future__ import annotations

import io
import json
import logging

import pandas as pd
import pytest

from axiom.core.types import Side, get_instrument
from axiom.execution.base import Fill, Order
from axiom.ops.alerts import (
    Alert,
    AlertRouter,
    Severity,
    alerts_for_health,
    console_sink,
)
from axiom.ops.health import HealthStatus, check_health
from axiom.ops.logs import (
    JsonFormatter,
    correlation_id,
    current_correlation_id,
    log_event,
    setup_logging,
)
from axiom.store import Store

AAPL = get_instrument("AAPL")
T0 = pd.Timestamp("2026-01-01T12:00:00Z")


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def captured() -> io.StringIO:
    """A stream capturing JSON log lines, restored afterwards."""
    stream = io.StringIO()
    setup_logging(logging.DEBUG, stream=stream)
    yield stream
    logging.getLogger().handlers.clear()


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


class TestStructuredLogs:
    def test_every_line_is_one_json_object(self, captured: io.StringIO) -> None:
        logging.getLogger("t").info("hello")
        assert len(_lines(captured)) == 1

    def test_carries_the_standard_fields(self, captured: io.StringIO) -> None:
        logging.getLogger("t").warning("something")
        line = _lines(captured)[0]
        assert line["level"] == "WARNING"
        assert line["logger"] == "t"
        assert line["message"] == "something"
        assert "ts" in line

    def test_timestamps_are_utc_with_an_offset(self, captured: io.StringIO) -> None:
        """Local-time logs from three sessions cannot be ordered against each other."""
        logging.getLogger("t").info("x")
        assert str(_lines(captured)[0]["ts"]).endswith("+00:00")

    def test_extra_fields_become_top_level(self, captured: io.StringIO) -> None:
        """A queryable log, rather than merely a readable one."""
        log_event(logging.getLogger("t"), "order_submitted", symbol="AAPL", qty=5)
        line = _lines(captured)[0]
        assert line["event"] == "order_submitted"
        assert line["symbol"] == "AAPL"
        assert line["qty"] == 5

    def test_an_exception_stays_on_one_line(self, captured: io.StringIO) -> None:
        """A multi-line record breaks every shipper that splits on newline."""
        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("t").exception("failed")
        lines = _lines(captured)
        assert len(lines) == 1
        assert "ValueError" in str(lines[0]["error"])
        assert "\n" not in str(lines[0]["error"])

    def test_unserialisable_values_do_not_break_the_record(
        self, captured: io.StringIO
    ) -> None:
        """A formatter that raises destroys the record that mattered most."""
        log_event(logging.getLogger("t"), "e", obj=object(), when=T0)
        line = _lines(captured)[0]
        assert isinstance(line["obj"], str)

    def test_nested_structures_survive(self, captured: io.StringIO) -> None:
        log_event(logging.getLogger("t"), "e", detail={"a": [1, 2], "b": {"c": 3}})
        assert _lines(captured)[0]["detail"] == {"a": [1, 2], "b": {"c": 3}}

    def test_setup_replaces_rather_than_stacks_handlers(self) -> None:
        """Calling twice otherwise doubles every line and inflates volume."""
        first, second = io.StringIO(), io.StringIO()
        setup_logging(logging.INFO, stream=first)
        setup_logging(logging.INFO, stream=second)
        logging.getLogger("t").info("once")
        assert first.getvalue() == ""
        assert len(_lines(second)) == 1
        logging.getLogger().handlers.clear()

    def test_human_format_is_available(self) -> None:
        stream = io.StringIO()
        setup_logging(logging.INFO, stream=stream, json_format=False)
        logging.getLogger("t").info("readable")
        assert "readable" in stream.getvalue()
        with pytest.raises(json.JSONDecodeError):
            json.loads(stream.getvalue())
        logging.getLogger().handlers.clear()


class TestCorrelationId:
    def test_absent_outside_a_context(self) -> None:
        assert current_correlation_id() is None

    def test_bound_inside(self) -> None:
        with correlation_id("abc") as value:
            assert value == "abc"
            assert current_correlation_id() == "abc"

    def test_generated_when_not_supplied(self) -> None:
        with correlation_id() as value:
            assert value and len(value) == 12

    def test_reset_on_exit(self) -> None:
        """Leaking an id attributes an unrelated failure to this decision."""
        with correlation_id("abc"):
            pass
        assert current_correlation_id() is None

    def test_reset_even_when_the_block_raises(self) -> None:
        with pytest.raises(RuntimeError), correlation_id("abc"):
            raise RuntimeError("boom")
        assert current_correlation_id() is None

    def test_nested_blocks_inherit(self) -> None:
        """A helper opening its own context must not sever the chain."""
        with correlation_id("outer"), correlation_id():
            assert current_correlation_id() == "outer"

    def test_a_nested_block_can_override(self) -> None:
        with correlation_id("outer"):
            with correlation_id("inner"):
                assert current_correlation_id() == "inner"
            assert current_correlation_id() == "outer"

    def test_it_reaches_the_log_line(self, captured: io.StringIO) -> None:
        with correlation_id("trace-me"):
            logging.getLogger("t").info("inside")
        logging.getLogger("t").info("outside")
        lines = _lines(captured)
        assert lines[0]["correlation_id"] == "trace-me"
        assert "correlation_id" not in lines[1]

    def test_a_whole_tick_shares_one_id(self, captured: io.StringIO) -> None:
        """The point: one decision reads as one causal chain."""
        with correlation_id():
            log_event(logging.getLogger("t"), "guards_passed")
            log_event(logging.getLogger("t"), "order_submitted")
        ids = {line["correlation_id"] for line in _lines(captured)}
        assert len(ids) == 1


class TestFormatterDirectly:
    def test_handles_a_record_with_no_extras(self) -> None:
        record = logging.LogRecord("n", logging.INFO, "p", 1, "m", None, None)
        assert json.loads(JsonFormatter().format(record))["message"] == "m"


class TestHealth:
    def test_a_fresh_store_is_degraded_not_down(self, store: Store) -> None:
        """Calling a never-run desk 'down' trains people to ignore the check."""
        report = check_health(store, now=T0)
        assert report.status is HealthStatus.DEGRADED
        assert "never run" in report.checks["heartbeat"][1]

    def test_a_recent_heartbeat_is_healthy(self, store: Store) -> None:
        store.record_equity(T0, 100_000.0, 50_000.0)
        assert check_health(store, now=T0 + pd.Timedelta(minutes=1)).status is (
            HealthStatus.OK
        )

    def test_a_stale_heartbeat_degrades(self, store: Store) -> None:
        store.record_equity(T0, 100_000.0, 50_000.0)
        report = check_health(store, now=T0 + pd.Timedelta(minutes=20))
        assert report.checks["heartbeat"][0] is HealthStatus.DEGRADED

    def test_a_very_stale_heartbeat_is_down(self, store: Store) -> None:
        """A wedged desk answers 'fine' or not at all; the store cannot lie."""
        store.record_equity(T0, 100_000.0, 50_000.0)
        report = check_health(store, now=T0 + pd.Timedelta(hours=3))
        assert report.checks["heartbeat"][0] is HealthStatus.DOWN
        assert report.status is HealthStatus.DOWN

    def test_a_halt_is_down_not_degraded(self, store: Store) -> None:
        store.record_equity(T0, 100_000.0, 50_000.0)
        store.record_halt("drawdown", T0)
        report = check_health(store, now=T0)
        assert report.checks["halts"][0] is HealthStatus.DOWN
        assert "drawdown" in report.checks["halts"][1]

    def test_a_cleared_halt_is_healthy(self, store: Store) -> None:
        store.record_equity(T0, 100_000.0, 50_000.0)
        halt_id = store.record_halt("drawdown", T0)
        store.clear_halt(halt_id, T0)
        assert check_health(store, now=T0).checks["halts"][0] is HealthStatus.OK

    def test_no_fill_poll_with_open_orders_degrades(self, store: Store) -> None:
        store.record_equity(T0, 100_000.0, 50_000.0)
        order = Order(instrument=AAPL, side=Side.BUY, quantity=1.0, stop_loss=90.0)
        order_id, _ = store.record_order(order, "k", T0)
        from axiom.core.types import OrderStatus

        store.record_status(order_id, OrderStatus.SUBMITTED, T0)
        report = check_health(store, now=T0)
        assert report.checks["fill_polling"][0] is HealthStatus.DEGRADED

    def test_no_fill_poll_without_orders_is_fine(self, store: Store) -> None:
        store.record_equity(T0, 100_000.0, 50_000.0)
        assert check_health(store, now=T0).checks["fill_polling"][0] is HealthStatus.OK

    def test_a_stale_fill_poll_degrades(self, store: Store) -> None:
        store.record_equity(T0, 100_000.0, 50_000.0)
        store.set_meta("fills_last_poll_us", str(int(T0.value // 1_000)))
        report = check_health(store, now=T0 + pd.Timedelta(hours=2))
        assert report.checks["fill_polling"][0] is HealthStatus.DEGRADED
        assert "drifted" in report.checks["fill_polling"][1]

    def test_a_quiet_desk_polling_fine_is_healthy(self, store: Store) -> None:
        """The watermark is the newest fill, not the last poll.

        A desk polling every minute with no new fills has a watermark days old
        and is working perfectly. Reading the watermark for liveness would
        report it permanently degraded.
        """
        store.record_equity(T0, 100_000.0, 50_000.0)
        old_fill = T0 - pd.Timedelta(days=17)
        store.set_meta("fills_watermark_us", str(int(old_fill.value // 1_000)))
        store.set_meta("fills_last_poll_us", str(int(T0.value // 1_000)))
        report = check_health(store, now=T0 + pd.Timedelta(minutes=1))
        assert report.checks["fill_polling"][0] is HealthStatus.OK
        assert report.facts["newest_fill"].startswith("2025-12-15")

    def test_positions_without_equity_are_inconsistent(self, store: Store) -> None:
        order = Order(instrument=AAPL, side=Side.BUY, quantity=1.0, stop_loss=90.0)
        order_id, _ = store.record_order(order, "k", T0)
        store.record_fill(
            order_id,
            Fill(order_id=order_id, instrument=AAPL, side=Side.BUY, quantity=5.0,
                 price=100.0, commission=0.0, timestamp=T0),
            "f1",
        )
        report = check_health(store, now=T0)
        assert report.checks["consistency"][0] is HealthStatus.DOWN

    def test_facts_carry_drawdown(self, store: Store) -> None:
        store.record_equity(T0, 100.0, 100.0)
        store.record_equity(T0 + pd.Timedelta(minutes=1), 90.0, 90.0)
        report = check_health(store, now=T0 + pd.Timedelta(minutes=2))
        assert report.facts["drawdown_pct"] == "10.00"

    def test_exit_codes_are_distinct(self, store: Store) -> None:
        """A shell watchdog needs to tell degraded from down."""
        assert check_health(store, now=T0).exit_code == 1
        store.record_equity(T0, 100.0, 100.0)
        assert check_health(store, now=T0).exit_code == 0
        store.record_halt("x", T0)
        assert check_health(store, now=T0).exit_code == 2

    def test_render_is_readable(self, store: Store) -> None:
        store.record_equity(T0, 100.0, 100.0)
        assert "Desk health" in check_health(store, now=T0).render()

    def test_rejects_inverted_thresholds(self, store: Store) -> None:
        with pytest.raises(ValueError, match="heartbeat_fail"):
            check_health(
                store,
                heartbeat_warn=pd.Timedelta(hours=1),
                heartbeat_fail=pd.Timedelta(minutes=1),
            )


class TestAlertRouter:
    @staticmethod
    def _alert(key: str = "k", at: pd.Timestamp = T0, **kwargs: object) -> Alert:
        defaults: dict[str, object] = {
            "key": key,
            "severity": Severity.CRITICAL,
            "summary": "something broke",
            "at": at,
        }
        defaults.update(kwargs)
        return Alert(**defaults)  # type: ignore[arg-type]

    def test_sends_the_first_occurrence(self) -> None:
        sent: list[Alert] = []
        router = AlertRouter([sent.append])
        assert router.send(self._alert()) is True
        assert len(sent) == 1

    def test_suppresses_a_repeat_inside_the_cooldown(self) -> None:
        """A halted desk emits this every poll; a flood mutes the channel."""
        sent: list[Alert] = []
        router = AlertRouter([sent.append], cooldown=pd.Timedelta(minutes=30))
        router.send(self._alert(at=T0))
        assert router.send(self._alert(at=T0 + pd.Timedelta(minutes=5))) is False
        assert len(sent) == 1

    def test_re_alerts_after_the_cooldown(self) -> None:
        sent: list[Alert] = []
        router = AlertRouter([sent.append], cooldown=pd.Timedelta(minutes=30))
        router.send(self._alert(at=T0))
        router.send(self._alert(at=T0 + pd.Timedelta(hours=1)))
        assert len(sent) == 2

    def test_different_keys_do_not_suppress_each_other(self) -> None:
        sent: list[Alert] = []
        router = AlertRouter([sent.append])
        router.send(self._alert(key="a"))
        router.send(self._alert(key="b"))
        assert len(sent) == 2

    def test_a_resolution_is_sent_once(self) -> None:
        """Otherwise the only sign a problem ended is that alerts stopped."""
        sent: list[Alert] = []
        router = AlertRouter([sent.append])
        router.send(self._alert())
        assert router.resolve("k", T0) is True
        assert sent[-1].resolved
        assert router.resolve("k", T0) is False

    def test_resolving_something_never_alerted_is_silent(self) -> None:
        sent: list[Alert] = []
        assert AlertRouter([sent.append]).resolve("never", T0) is False
        assert sent == []

    def test_a_resolution_ignores_the_cooldown(self) -> None:
        sent: list[Alert] = []
        router = AlertRouter([sent.append], cooldown=pd.Timedelta(hours=6))
        router.send(self._alert(at=T0))
        assert router.resolve("k", T0 + pd.Timedelta(seconds=1)) is True

    def test_re_alerting_after_a_resolution_works(self) -> None:
        sent: list[Alert] = []
        router = AlertRouter([sent.append], cooldown=pd.Timedelta(hours=6))
        router.send(self._alert(at=T0))
        router.resolve("k", T0)
        assert router.send(self._alert(at=T0 + pd.Timedelta(minutes=1))) is True

    def test_severity_floor_drops_noise(self) -> None:
        sent: list[Alert] = []
        router = AlertRouter([sent.append], min_severity=Severity.CRITICAL)
        assert router.send(self._alert(severity=Severity.WARNING)) is False

    def test_a_failing_sink_does_not_kill_the_router(self) -> None:
        """An alerter that dies while alerting is worse than one that is silent."""
        def broken(alert: Alert) -> None:
            raise RuntimeError("pager is down")

        sent: list[Alert] = []
        router = AlertRouter([broken, sent.append])
        assert router.send(self._alert()) is True
        assert len(sent) == 1

    def test_active_conditions_are_tracked(self) -> None:
        router = AlertRouter([lambda a: None])
        router.send(self._alert(key="a"))
        router.send(self._alert(key="b"))
        router.resolve("a", T0)
        assert router.active == {"b"}

    def test_rejects_a_negative_cooldown(self) -> None:
        with pytest.raises(ValueError, match="cooldown"):
            AlertRouter(cooldown=pd.Timedelta(minutes=-1))

    def test_the_default_sink_logs(self, captured: io.StringIO) -> None:
        console_sink(self._alert())
        line = _lines(captured)[0]
        assert line["event"] == "alert"
        assert line["alert_key"] == "k"


class TestHealthToAlerts:
    def test_one_alert_per_failing_check(self, store: Store) -> None:
        """A combined alert cannot express one condition clearing on its own."""
        store.record_equity(T0, 100.0, 100.0)
        store.record_halt("drawdown", T0)
        alerts = alerts_for_health(
            check_health(store, now=T0 + pd.Timedelta(hours=3))
        )
        keys = {a.key for a in alerts}
        assert "health.halts" in keys
        assert "health.heartbeat" in keys

    def test_healthy_checks_produce_nothing(self, store: Store) -> None:
        store.record_equity(T0, 100.0, 100.0)
        assert alerts_for_health(check_health(store, now=T0)) == []

    def test_severity_maps_from_status(self, store: Store) -> None:
        store.record_equity(T0, 100.0, 100.0)
        store.record_halt("x", T0)
        alerts = {a.key: a for a in alerts_for_health(check_health(store, now=T0))}
        assert alerts["health.halts"].severity is Severity.CRITICAL

    def test_rejects_the_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="HealthReport"):
            alerts_for_health("not a report")

    def test_the_pair_deduplicates_across_polls(self, store: Store) -> None:
        """End to end: a persistent halt alerts once, not every poll."""
        store.record_equity(T0, 100.0, 100.0)
        store.record_halt("drawdown", T0)
        sent: list[Alert] = []
        router = AlertRouter([sent.append], cooldown=pd.Timedelta(hours=1))
        for minute in range(5):
            moment = T0 + pd.Timedelta(minutes=minute)
            for alert in alerts_for_health(check_health(store, now=moment)):
                router.send(alert)
        assert len([a for a in sent if a.key == "health.halts"]) == 1
