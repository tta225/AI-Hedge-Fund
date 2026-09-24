"""Tests for the supervisor loop and the trading calendar."""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest
from tests.conftest import make_series

from axiom.core.types import Direction, OrderStatus, get_instrument
from axiom.desk.calendar import AlwaysOpen, RegularHours, next_open
from axiom.desk.fills import FillPoller
from axiom.desk.runner import DeskRunner
from axiom.desk.supervisor import MAX_INTERVAL, CycleOutcome, Supervisor
from axiom.execution.base import ExecutionError, ExecutionVenue, Order
from axiom.ict.models import ICTState
from axiom.ops.alerts import Alert, AlertRouter
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.store import Store
from axiom.strategy.base import Signal, Strategy, StrategyContext

BTC = get_instrument("BTC-USD")
T0 = pd.Timestamp("2026-01-05T15:00:00Z")  # a Monday, inside US equity hours


class TestAlwaysOpen:
    def test_open_at_every_hour(self) -> None:
        calendar = AlwaysOpen()
        assert all(
            calendar.is_open(T0 + pd.Timedelta(hours=h)) for h in range(0, 72, 3)
        )

    def test_open_at_the_weekend(self) -> None:
        """Correct for crypto, which is the only thing it claims."""
        assert AlwaysOpen().is_open(pd.Timestamp("2026-01-04T03:00:00Z"))


class TestRegularHours:
    calendar = RegularHours.us_equities()

    def test_open_during_the_session(self) -> None:
        assert self.calendar.is_open(pd.Timestamp("2026-01-05T15:00:00Z"))

    def test_closed_before_the_open(self) -> None:
        assert not self.calendar.is_open(pd.Timestamp("2026-01-05T13:00:00Z"))

    def test_closed_after_the_close(self) -> None:
        assert not self.calendar.is_open(pd.Timestamp("2026-01-05T21:30:00Z"))

    def test_closed_at_the_weekend(self) -> None:
        assert not self.calendar.is_open(pd.Timestamp("2026-01-04T15:00:00Z"))

    def test_the_close_is_exclusive(self) -> None:
        assert not self.calendar.is_open(pd.Timestamp("2026-01-05T21:00:00Z"))

    def test_the_open_is_inclusive(self) -> None:
        assert self.calendar.is_open(pd.Timestamp("2026-01-05T14:30:00Z"))

    def test_a_naive_timestamp_is_utc(self) -> None:
        assert self.calendar.is_open(pd.Timestamp("2026-01-05T15:00:00"))

    def test_it_converts_rather_than_reading_the_local_hour(self) -> None:
        """Reading the hour off an aware timestamp answers the wrong timezone."""
        tokyo = pd.Timestamp("2026-01-06T00:00:00+09:00")  # 15:00 UTC Monday
        assert self.calendar.is_open(tokyo)

    def test_dst_is_handled_by_the_timezone_not_an_offset(self) -> None:
        """A hard-coded offset trades an hour early every spring."""
        # 2026-07-06 is a Monday in US daylight time: 14:30 UTC is 10:30 ET.
        summer = RegularHours.us_equities()
        assert summer.is_open(pd.Timestamp("2026-07-06T14:30:00Z"))
        # In January the same UTC time is 09:30 ET — also open, but only
        # because the timezone was applied rather than a fixed offset.
        assert summer.is_open(pd.Timestamp("2026-01-05T14:30:00Z"))
        # 13:30 UTC is 08:30 ET in January, before the open.
        assert not summer.is_open(pd.Timestamp("2026-01-05T13:30:00Z"))

    def test_a_configured_holiday_closes_the_market(self) -> None:
        calendar = RegularHours.us_equities(holidays={date(2026, 1, 5)})
        assert not calendar.is_open(T0)

    def test_a_half_day_closes_early(self) -> None:
        calendar = RegularHours.us_equities(
            half_days={date(2026, 1, 5): time(13, 0)}
        )
        assert calendar.is_open(pd.Timestamp("2026-01-05T17:00:00Z"))  # 12:00 ET
        assert not calendar.is_open(pd.Timestamp("2026-01-05T19:00:00Z"))  # 14:00 ET

    def test_it_says_when_it_does_not_know_holidays(self) -> None:
        """A stale built-in list would read as authoritative."""
        assert "no holidays configured" in RegularHours.us_equities().describe()

    def test_configured_holidays_are_reported(self) -> None:
        described = RegularHours.us_equities(holidays={date(2026, 1, 1)}).describe()
        assert "1 holiday(s) configured" in described

    def test_rejects_an_inverted_session(self) -> None:
        with pytest.raises(ValueError, match="not before close"):
            RegularHours(open_time=time(16, 0), close_time=time(9, 30))


class TestNextOpen:
    def test_finds_the_monday_open_from_a_weekend(self) -> None:
        calendar = RegularHours.us_equities()
        found = next_open(calendar, pd.Timestamp("2026-01-03T12:00:00Z"))
        assert found is not None
        assert calendar.is_open(found)
        assert found.date() == date(2026, 1, 5)

    def test_returns_the_current_minute_when_already_open(self) -> None:
        found = next_open(RegularHours.us_equities(), T0)
        assert found == T0

    def test_a_never_open_calendar_returns_none_rather_than_spinning(self) -> None:
        class _NeverOpen:
            def is_open(self, at: pd.Timestamp) -> bool:
                return False

        assert next_open(_NeverOpen(), T0, limit_days=1) is None

    def test_rejects_a_zero_limit(self) -> None:
        with pytest.raises(ValueError, match="limit_days"):
            next_open(AlwaysOpen(), T0, limit_days=0)


class _Venue(ExecutionVenue):
    name = "stub"
    is_live = False

    def __init__(self) -> None:
        self.submitted: list[Order] = []

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        self.submitted.append(order)
        order.status = OrderStatus.SUBMITTED
        return order

    def cancel(self, order: Order) -> Order:
        order.status = OrderStatus.CANCELLED
        return order

    def working_orders(self) -> list[Order]:
        return []


class _AlwaysLong(Strategy):
    name = "always_long"
    requires_ict = False

    def evaluate(self, context: StrategyContext) -> Signal | None:
        price = context.price
        return Signal(
            direction=Direction.BULLISH,
            entry=price,
            stop=price * 0.95,
            targets=(price * 1.10,),
        )


class _Source:
    """A fill source that can be made to fail."""

    def __init__(self) -> None:
        self.calls = 0
        self.raise_with: Exception | None = None

    def fills_since(self, since: pd.Timestamp, limit: int = 500) -> list:
        self.calls += 1
        if self.raise_with is not None:
            raise self.raise_with
        return []


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def _context(at: pd.Timestamp = T0) -> StrategyContext:
    """A context whose newest bar is `at`.

    Freshness matters: the runner's data-freshness guard halts on a stale bar,
    so a fixture whose bars predate the cycle time silently tests the halt path
    rather than the trading path.
    """
    rows = [(100.0, 101.0, 99.0, 100.0) for _ in range(60)]
    series = make_series(
        rows,
        instrument=BTC,
        timeframe="1h",
        start=(at - pd.Timedelta(hours=59)).to_pydatetime(),
    )
    last = len(series) - 1
    return StrategyContext(
        series=series,
        index=last,
        ict=ICTState(
            symbol="BTC-USD", timeframe="1h", as_of=series.index[last], index=last
        ),
        portfolio=Portfolio(starting_cash=1_000_000),
        timestamp=series.index[last],
    )


def _supervisor(
    store: Store,
    *,
    venue: _Venue | None = None,
    source: _Source | None = None,
    is_open: object = None,
    context: StrategyContext | None = None,
    alerts: AlertRouter | None = None,
) -> tuple[Supervisor, _Venue]:
    venue = venue or _Venue()
    ctx = context if context is not None else _context()
    runner = DeskRunner(
        strategy=_AlwaysLong(),
        venue=venue,
        store=store,
        risk=RiskManager(),
        portfolio=ctx.portfolio,
    )
    supervisor = Supervisor(
        runner=runner,
        store=store,
        context_factory=lambda: context if context is not None else _context(),
        poller=FillPoller(source, store) if source is not None else None,  # type: ignore[arg-type]
        is_open=is_open,  # type: ignore[arg-type]
        alerts=alerts,
        interval=0.001,
    )
    return supervisor, venue


class TestCycleOrdering:
    def test_fills_are_polled_before_trading(self, store: Store) -> None:
        """Deciding first sizes against a book that already changed."""
        source = _Source()
        supervisor, venue = _supervisor(store, source=source)
        supervisor.cycle(T0)
        assert source.calls == 1
        assert len(venue.submitted) == 1

    def test_a_clean_cycle_reports_everything(self, store: Store) -> None:
        supervisor, _ = _supervisor(store, source=_Source())
        outcome = supervisor.cycle(T0)
        assert outcome.ok
        assert outcome.poll is not None
        assert outcome.tick is not None
        assert outcome.health is not None

    def test_health_is_checked_last(self, store: Store) -> None:
        """So the report describes the cycle that just ran."""
        supervisor, _ = _supervisor(store, source=_Source())
        outcome = supervisor.cycle(T0)
        assert outcome.health is not None
        # The tick marked equity, so health can see a heartbeat.
        assert "heartbeat" in outcome.health.checks


class TestGating:
    def test_a_closed_market_skips_trading(self, store: Store) -> None:
        supervisor, venue = _supervisor(store, is_open=lambda _: False)
        outcome = supervisor.cycle(T0)
        assert outcome.skipped_trading == "market closed"
        assert not venue.submitted

    def test_a_halted_desk_skips_trading(self, store: Store) -> None:
        store.record_halt("manual", T0)
        supervisor, venue = _supervisor(store)
        outcome = supervisor.cycle(T0)
        assert outcome.skipped_trading == "desk is halted"
        assert not venue.submitted

    def test_a_halted_desk_still_polls_and_reports(self, store: Store) -> None:
        """A halt is exactly when someone is looking at the desk."""
        store.record_halt("manual", T0)
        source = _Source()
        supervisor, _ = _supervisor(store, source=source)
        outcome = supervisor.cycle(T0)
        assert source.calls == 1
        assert outcome.health is not None

    def test_no_fresh_bar_skips_trading(self, store: Store) -> None:
        supervisor, venue = _supervisor(store)
        supervisor.context_factory = lambda: None
        outcome = supervisor.cycle(T0)
        assert outcome.skipped_trading == "no fresh bar"
        assert not venue.submitted


class TestResilience:
    def test_a_failing_poll_does_not_stop_the_cycle(self, store: Store) -> None:
        """A supervisor that dies leaves open positions with nothing watching."""
        source = _Source()
        source.raise_with = ExecutionError("venue down")
        supervisor, venue = _supervisor(store, source=source)
        outcome = supervisor.cycle(T0)
        assert outcome.errors
        assert len(venue.submitted) == 1  # trading still happened

    def test_a_raising_context_factory_is_caught(self, store: Store) -> None:
        supervisor, _ = _supervisor(store)

        def boom() -> StrategyContext | None:
            raise RuntimeError("data feed exploded")

        supervisor.context_factory = boom
        outcome = supervisor.cycle(T0)
        assert any("tick raised" in e for e in outcome.errors)
        assert outcome.health is not None  # reporting still ran

    def test_a_raising_health_check_is_caught(self, store: Store) -> None:
        supervisor, _ = _supervisor(store)
        store.close()  # every subsequent read raises
        outcome = supervisor.cycle(T0)
        assert outcome.errors

    def test_the_loop_survives_a_failing_cycle(self, store: Store) -> None:
        source = _Source()
        source.raise_with = ExecutionError("down")
        supervisor, _ = _supervisor(store, source=source)
        outcomes = supervisor.run(max_cycles=3)
        assert len(outcomes) == 3


class TestBackoff:
    def test_a_healthy_cycle_sleeps_the_base_interval(self, store: Store) -> None:
        supervisor, _ = _supervisor(store, source=_Source())
        supervisor.interval = 60.0
        supervisor.cycle(T0)
        assert supervisor.sleep_seconds() == 60.0

    def test_failures_widen_the_interval(self, store: Store) -> None:
        source = _Source()
        source.raise_with = ExecutionError("down")
        supervisor, _ = _supervisor(store, source=source)
        supervisor.interval = 10.0
        sleeps = []
        for _ in range(3):
            supervisor.cycle(T0)
            sleeps.append(supervisor.sleep_seconds())
        assert sleeps == sorted(sleeps)
        assert sleeps[-1] > sleeps[0]

    def test_backoff_is_capped(self, store: Store) -> None:
        source = _Source()
        source.raise_with = ExecutionError("down")
        supervisor, _ = _supervisor(store, source=source)
        supervisor.interval = 60.0
        for _ in range(20):
            supervisor.cycle(T0)
        assert supervisor.sleep_seconds() == MAX_INTERVAL

    def test_one_success_clears_the_backoff(self, store: Store) -> None:
        source = _Source()
        source.raise_with = ExecutionError("down")
        supervisor, _ = _supervisor(store, source=source)
        supervisor.interval = 10.0
        supervisor.cycle(T0)
        supervisor.cycle(T0)
        assert supervisor.sleep_seconds() > 10.0
        source.raise_with = None
        supervisor.cycle(T0)
        assert supervisor.sleep_seconds() == 10.0

    def test_rejects_an_impossible_interval(self, store: Store) -> None:
        runner = DeskRunner(
            strategy=_AlwaysLong(), venue=_Venue(), store=store,
            risk=RiskManager(), portfolio=Portfolio(starting_cash=1.0),
        )
        with pytest.raises(ValueError, match="interval must be positive"):
            Supervisor(
                runner=runner, store=store, context_factory=lambda: None, interval=0
            )

    def test_rejects_a_max_below_the_interval(self, store: Store) -> None:
        runner = DeskRunner(
            strategy=_AlwaysLong(), venue=_Venue(), store=store,
            risk=RiskManager(), portfolio=Portfolio(starting_cash=1.0),
        )
        with pytest.raises(ValueError, match="max_interval"):
            Supervisor(
                runner=runner, store=store, context_factory=lambda: None,
                interval=60.0, max_interval=10.0,
            )


class TestStopping:
    def test_a_stop_request_ends_the_loop(self, store: Store) -> None:
        supervisor, _ = _supervisor(store)
        supervisor.request_stop()
        assert supervisor.run(max_cycles=10) == []

    def test_max_cycles_bounds_the_run(self, store: Store) -> None:
        supervisor, _ = _supervisor(store)
        assert len(supervisor.run(max_cycles=2)) == 2

    def test_resume_runs_before_the_first_cycle(self, store: Store) -> None:
        """A restart must reconstruct state before deciding anything."""
        store.record_halt("left over from last run", T0)
        supervisor, venue = _supervisor(store)
        supervisor.run(max_cycles=1)
        assert not venue.submitted


class TestAlerting:
    def test_a_failing_health_check_alerts(self, store: Store) -> None:
        sent: list[Alert] = []
        store.record_halt("drawdown", T0)
        supervisor, _ = _supervisor(store, alerts=AlertRouter([sent.append]))
        supervisor.cycle(T0)
        assert any(a.key == "health.halts" for a in sent)

    def test_a_persistent_condition_alerts_once(self, store: Store) -> None:
        sent: list[Alert] = []
        store.record_halt("drawdown", T0)
        router = AlertRouter([sent.append], cooldown=pd.Timedelta(hours=1))
        supervisor, _ = _supervisor(store, alerts=router)
        for minute in range(4):
            supervisor.cycle(T0 + pd.Timedelta(minutes=minute))
        assert len([a for a in sent if a.key == "health.halts"]) == 1

    def test_a_cleared_condition_sends_a_resolution(self, store: Store) -> None:
        """Otherwise the only sign a problem ended is that alerts stopped."""
        sent: list[Alert] = []
        halt_id = store.record_halt("drawdown", T0)
        supervisor, _ = _supervisor(store, alerts=AlertRouter([sent.append]))
        supervisor.cycle(T0)
        store.clear_halt(halt_id, T0)
        supervisor.cycle(T0 + pd.Timedelta(minutes=1))
        assert any(a.resolved and a.key == "health.halts" for a in sent)


class TestCycleOutcomeRendering:
    def test_renders_a_clean_cycle(self, store: Store) -> None:
        supervisor, _ = _supervisor(store, source=_Source())
        assert "health" in supervisor.cycle(T0).render()

    def test_renders_a_skip_reason(self, store: Store) -> None:
        supervisor, _ = _supervisor(store, is_open=lambda _: False)
        assert "market closed" in supervisor.cycle(T0).render()

    def test_renders_errors(self) -> None:
        outcome = CycleOutcome(at=T0, errors=["boom"])
        assert "ERRORS" in outcome.render()
        assert not outcome.ok
