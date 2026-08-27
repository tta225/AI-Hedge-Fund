"""Tests for the live desk: persistence, reconciliation, guards, and the loop."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.conftest import make_series

from axiom.core.types import OrderStatus, Side, get_instrument
from axiom.desk.guards import (
    GuardStatus,
    check_data_freshness,
    check_drawdown,
    check_guards,
)
from axiom.desk.runner import DeskConfig, DeskRunner, flatten_all
from axiom.execution.base import ExecutionError, ExecutionVenue, Fill, Order
from axiom.ict.models import ICTState
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.store import Store, reconcile_positions
from axiom.store.db import StoreError
from axiom.strategy.base import Signal, Strategy, StrategyContext

BTC = get_instrument("BTC-USD")
T0 = pd.Timestamp("2026-01-01T00:00:00Z")


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def _order(quantity: float = 1.0, side: Side = Side.BUY) -> Order:
    return Order(
        instrument=BTC, side=side, quantity=quantity, stop_loss=90.0, strategy="test"
    )


def _fill(order_id: int, quantity: float = 1.0, price: float = 100.0,
          side: Side = Side.BUY, at: pd.Timestamp = T0) -> Fill:
    return Fill(
        order_id=order_id, instrument=BTC, side=side, quantity=quantity,
        price=price, commission=0.1, timestamp=at,
    )


class TestIdempotency:
    def test_the_same_key_never_creates_a_second_order(self, store: Store) -> None:
        """The dangerous failure is sending twice, not losing one."""
        first, was_new_first = store.record_order(_order(), "key-1", T0)
        second, was_new_second = store.record_order(_order(), "key-1", T0)
        assert first == second
        assert was_new_first and not was_new_second

    def test_different_keys_create_different_orders(self, store: Store) -> None:
        a, _ = store.record_order(_order(), "key-a", T0)
        b, _ = store.record_order(_order(), "key-b", T0)
        assert a != b

    def test_a_key_is_required(self, store: Store) -> None:
        with pytest.raises(ValueError, match="idempotency key"):
            store.record_order(_order(), "", T0)

    def test_runner_keys_are_stable_for_the_same_decision(self) -> None:
        order = _order()
        assert DeskRunner._idempotency_key(order, T0) == DeskRunner._idempotency_key(
            order, T0
        )

    def test_runner_keys_differ_across_bars(self) -> None:
        """Same setup on a later bar is a new decision, not a retry."""
        order = _order()
        later = T0 + pd.Timedelta(hours=1)
        assert DeskRunner._idempotency_key(order, T0) != DeskRunner._idempotency_key(
            order, later
        )

    def test_runner_keys_differ_by_size(self) -> None:
        assert DeskRunner._idempotency_key(_order(1.0), T0) != (
            DeskRunner._idempotency_key(_order(2.0), T0)
        )


class TestOrderHistory:
    def test_status_changes_append_rather_than_overwrite(self, store: Store) -> None:
        """An incident review needs the sequence, and UPDATE destroys it."""
        order_id, _ = store.record_order(_order(), "k", T0)
        store.record_status(order_id, OrderStatus.SUBMITTED, T0)
        store.record_status(order_id, OrderStatus.PARTIALLY_FILLED, T0)
        store.record_status(order_id, OrderStatus.FILLED, T0)
        statuses = [s for s, _, _ in store.order_history(order_id)]
        assert statuses == [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
        ]

    def test_current_status_is_the_latest_event(self, store: Store) -> None:
        order_id, _ = store.record_order(_order(), "k", T0)
        store.record_status(order_id, OrderStatus.FILLED, T0)
        stored = store.get_order(order_id)
        assert stored is not None and stored.status is OrderStatus.FILLED

    def test_open_orders_excludes_terminal_ones(self, store: Store) -> None:
        working, _ = store.record_order(_order(), "open", T0)
        done, _ = store.record_order(_order(), "done", T0)
        store.record_status(working, OrderStatus.SUBMITTED, T0)
        store.record_status(done, OrderStatus.FILLED, T0)
        assert [o.id for o in store.open_orders()] == [working]

    def test_missing_order_is_none(self, store: Store) -> None:
        assert store.get_order(999) is None


class TestFills:
    def test_a_venue_fill_id_is_booked_once(self, store: Store) -> None:
        """A websocket replay must not double the position."""
        order_id, _ = store.record_order(_order(), "k", T0)
        assert store.record_fill(order_id, _fill(order_id), "venue-1") is True
        assert store.record_fill(order_id, _fill(order_id), "venue-1") is False
        assert store.positions()["BTC-USD"] == pytest.approx(1.0)

    def test_positions_are_derived_from_fills(self, store: Store) -> None:
        order_id, _ = store.record_order(_order(), "k", T0)
        store.record_fill(order_id, _fill(order_id, 2.0), "f1")
        store.record_fill(order_id, _fill(order_id, 0.5, side=Side.SELL), "f2")
        assert store.positions()["BTC-USD"] == pytest.approx(1.5)

    def test_a_flat_symbol_disappears(self, store: Store) -> None:
        order_id, _ = store.record_order(_order(), "k", T0)
        store.record_fill(order_id, _fill(order_id, 1.0), "f1")
        store.record_fill(order_id, _fill(order_id, 1.0, side=Side.SELL), "f2")
        assert "BTC-USD" not in store.positions()

    def test_fills_roll_up_into_the_order(self, store: Store) -> None:
        order_id, _ = store.record_order(_order(quantity=2.0), "k", T0)
        store.record_fill(order_id, _fill(order_id, 1.0, price=100.0), "f1")
        store.record_fill(order_id, _fill(order_id, 1.0, price=102.0), "f2")
        stored = store.get_order(order_id)
        assert stored is not None
        assert stored.filled_quantity == pytest.approx(2.0)
        assert stored.average_fill_price == pytest.approx(101.0)


class TestHaltsSurviveRestart:
    def test_a_halt_stays_active_until_cleared(self, store: Store) -> None:
        """A halt that does not survive a restart is only a delay."""
        store.record_halt("data_stale", T0)
        assert store.is_halted()

    def test_clearing_releases_it(self, store: Store) -> None:
        halt_id = store.record_halt("data_stale", T0)
        store.clear_halt(halt_id, T0)
        assert not store.is_halted()

    def test_survives_reopening_the_database(self, tmp_path: object) -> None:
        path = f"{tmp_path}/desk.db"
        with Store(path) as first:
            first.record_halt("drawdown", T0)
        with Store(path) as second:
            assert second.is_halted()


class TestEquity:
    def test_curve_is_time_ordered(self, store: Store) -> None:
        for i in range(3):
            store.record_equity(T0 + pd.Timedelta(hours=i), 100.0 + i, 50.0)
        curve = store.equity_curve()
        assert list(curve.values) == [100.0, 101.0, 102.0]

    def test_empty_curve_is_empty(self, store: Store) -> None:
        assert store.equity_curve().empty


class TestSchemaGuard:
    def test_a_mismatched_schema_refuses_to_open(self, tmp_path: object) -> None:
        """Reading misparsed state is how a desk trades on fiction."""
        path = f"{tmp_path}/desk.db"
        with Store(path) as store:
            store._connection.execute(
                "UPDATE meta SET value = '99' WHERE key = 'schema_version'"
            )
            store._connection.commit()
        with pytest.raises(StoreError, match="schema version"):
            Store(path)


class TestReconciliation:
    def test_agreement_is_clean(self) -> None:
        result = reconcile_positions({"BTC-USD": 1.0}, {"BTC-USD": 1.0})
        assert result.is_clean and not result.should_halt

    def test_a_phantom_position_halts(self) -> None:
        """The desk manages a stop for something it cannot close."""
        result = reconcile_positions({"BTC-USD": 1.0}, {})
        assert result.should_halt
        assert result.discrepancies[0].kind == "phantom_position"

    def test_an_unknown_position_halts(self) -> None:
        """Exposure with no risk limit attached to it."""
        result = reconcile_positions({}, {"BTC-USD": 1.0})
        assert result.should_halt
        assert result.discrepancies[0].kind == "unknown_position"

    def test_a_wrong_side_is_named(self) -> None:
        result = reconcile_positions({"BTC-USD": 1.0}, {"BTC-USD": -1.0})
        assert result.discrepancies[0].kind == "wrong_side"

    def test_quantity_mismatch_is_named(self) -> None:
        result = reconcile_positions({"BTC-USD": 2.0}, {"BTC-USD": 1.0})
        assert result.discrepancies[0].kind == "quantity_mismatch"

    def test_tolerance_absorbs_rounding_only(self) -> None:
        assert reconcile_positions(
            {"BTC-USD": 1.0}, {"BTC-USD": 1.0 + 1e-12}
        ).is_clean

    def test_both_books_are_scanned(self) -> None:
        """Iterating one side would miss half the failures."""
        result = reconcile_positions({"A": 1.0}, {"B": 1.0})
        assert {d.symbol for d in result.discrepancies} == {"A", "B"}

    def test_render_says_the_broker_wins(self) -> None:
        assert "broker is authoritative" in reconcile_positions(
            {"BTC-USD": 1.0}, {}
        ).render()


class TestFreshnessGuard:
    interval = pd.Timedelta(hours=1)

    def test_fresh_data_passes(self) -> None:
        status, _ = check_data_freshness(
            T0, T0 + pd.Timedelta(minutes=30), expected_interval=self.interval
        )
        assert status is GuardStatus.PASS

    def test_slightly_stale_warns(self) -> None:
        status, _ = check_data_freshness(
            T0, T0 + pd.Timedelta(hours=3), expected_interval=self.interval
        )
        assert status is GuardStatus.WARN

    def test_very_stale_halts(self) -> None:
        """Stale data looks exactly like a quiet market."""
        status, _ = check_data_freshness(
            T0, T0 + pd.Timedelta(hours=9), expected_interval=self.interval
        )
        assert status is GuardStatus.HALT

    def test_no_data_at_all_halts(self) -> None:
        status, detail = check_data_freshness(
            None, T0, expected_interval=self.interval
        )
        assert status is GuardStatus.HALT
        assert "no bars" in detail

    def test_a_future_bar_halts(self) -> None:
        status, detail = check_data_freshness(
            T0 + pd.Timedelta(hours=1), T0, expected_interval=self.interval
        )
        assert status is GuardStatus.HALT
        assert "NTP" in detail

    def test_thresholds_scale_with_the_interval(self) -> None:
        """Two minutes is fine on daily bars and fatal on one-minute bars."""
        age = pd.Timedelta(minutes=10)
        daily, _ = check_data_freshness(
            T0, T0 + age, expected_interval=pd.Timedelta(days=1)
        )
        minutely, _ = check_data_freshness(
            T0, T0 + age, expected_interval=pd.Timedelta(minutes=1)
        )
        assert daily is GuardStatus.PASS
        assert minutely is GuardStatus.HALT


class TestDrawdownGuard:
    def test_shallow_drawdown_passes(self) -> None:
        assert check_drawdown(98.0, 100.0)[0] is GuardStatus.PASS

    def test_derisk_band_warns(self) -> None:
        assert check_drawdown(94.0, 100.0)[0] is GuardStatus.WARN

    def test_deep_drawdown_halts(self) -> None:
        assert check_drawdown(89.0, 100.0)[0] is GuardStatus.HALT

    def test_no_peak_halts(self) -> None:
        assert check_drawdown(100.0, 0.0)[0] is GuardStatus.HALT

    def test_rejects_inverted_thresholds(self) -> None:
        with pytest.raises(ValueError, match="halt_pct"):
            check_drawdown(100.0, 100.0, derisk_pct=10.0, halt_pct=5.0)


class TestGuardAggregation:
    @staticmethod
    def _kwargs(**overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "now": T0,
            "last_bar_at": T0,
            "expected_interval": pd.Timedelta(hours=1),
            "equity": 100.0,
            "peak_equity": 100.0,
            "store_halted": False,
            "kill_switch": False,
            "reconciled": True,
        }
        base.update(overrides)
        return base

    def test_all_clear_permits_trading(self) -> None:
        assert check_guards(**self._kwargs()).may_trade  # type: ignore[arg-type]

    def test_kill_switch_blocks(self) -> None:
        assert not check_guards(**self._kwargs(kill_switch=True)).may_trade  # type: ignore[arg-type]

    def test_persisted_halt_blocks(self) -> None:
        assert not check_guards(**self._kwargs(store_halted=True)).may_trade  # type: ignore[arg-type]

    def test_failed_reconciliation_blocks(self) -> None:
        assert not check_guards(**self._kwargs(reconciled=False)).may_trade  # type: ignore[arg-type]

    def test_every_guard_runs_even_after_one_halts(self) -> None:
        """An incident review wants the whole picture, not the first trip."""
        report = check_guards(**self._kwargs(kill_switch=True, store_halted=True))  # type: ignore[arg-type]
        assert len(report.blocking) == 2

    def test_a_raising_guard_halts_rather_than_passes(self) -> None:
        """A guard that told us nothing resolves to 'no'."""
        def broken() -> tuple[GuardStatus, str]:
            raise RuntimeError("feed handler exploded")

        report = check_guards(**self._kwargs(extra={"custom": broken}))  # type: ignore[arg-type]
        assert not report.may_trade
        assert "feed handler exploded" in report.results["custom"][1]

    def test_warnings_do_not_block(self) -> None:
        report = check_guards(**self._kwargs(equity=94.0, peak_equity=100.0))  # type: ignore[arg-type]
        assert report.status is GuardStatus.WARN
        assert report.may_trade


class _RecordingVenue(ExecutionVenue):
    """Accepts everything and remembers it."""

    name = "recording"
    is_live = False

    def __init__(self, *, fail: bool = False) -> None:
        self.submitted: list[Order] = []
        self.fail = fail

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        if self.fail:
            raise ExecutionError("venue is down")
        self.submitted.append(order)
        order.status = OrderStatus.SUBMITTED
        return order

    def cancel(self, order: Order) -> Order:
        order.status = OrderStatus.CANCELLED
        return order

    def working_orders(self) -> list[Order]:
        return [o for o in self.submitted if o.is_open]


class _AlwaysLong(Strategy):
    name = "always_long"
    requires_ict = False

    def evaluate(self, context: StrategyContext) -> Signal | None:
        from axiom.core.types import Direction

        price = context.price
        return Signal(
            direction=Direction.BULLISH,
            entry=price,
            stop=price * 0.95,
            targets=(price * 1.10,),
        )


class _Silent(Strategy):
    name = "silent"
    requires_ict = False

    def evaluate(self, context: StrategyContext) -> Signal | None:
        return None


def _runner(
    store: Store,
    *,
    strategy: Strategy | None = None,
    venue: _RecordingVenue | None = None,
    broker: dict[str, float] | None = None,
    config: DeskConfig | None = None,
) -> tuple[DeskRunner, StrategyContext]:
    rows = [(100.0, 101.0, 99.0, 100.0) for _ in range(60)]
    series = make_series(rows, instrument=BTC, timeframe="1h")
    portfolio = Portfolio(starting_cash=1_000_000)
    context = StrategyContext(
        series=series,
        index=50,
        ict=ICTState(symbol="BTC-USD", timeframe="1h", as_of=series.index[50], index=50),
        portfolio=portfolio,
        timestamp=series.index[50],
    )
    runner = DeskRunner(
        strategy=strategy or _AlwaysLong(),
        venue=venue or _RecordingVenue(),
        store=store,
        risk=RiskManager(),
        portfolio=portfolio,
        config=config,
        broker_positions=(lambda: broker) if broker is not None else None,
    )
    return runner, context


class TestDeskLoop:
    def test_a_clean_tick_sends_one_order(self, store: Store) -> None:
        venue = _RecordingVenue()
        runner, context = _runner(store, venue=venue)
        outcome = runner.tick(context)
        assert outcome.orders_sent == 1
        assert len(venue.submitted) == 1

    def test_no_signal_sends_nothing(self, store: Store) -> None:
        venue = _RecordingVenue()
        runner, context = _runner(store, strategy=_Silent(), venue=venue)
        outcome = runner.tick(context)
        assert outcome.signals == 0 and not venue.submitted

    def test_the_order_is_persisted_before_it_is_sent(self, store: Store) -> None:
        """A send that dies mid-flight must leave evidence, not a ghost."""
        venue = _RecordingVenue(fail=True)
        runner, context = _runner(store, venue=venue)
        outcome = runner.tick(context)
        assert outcome.orders_sent == 0
        assert "venue rejected" in outcome.rejected[0]
        # The row exists despite the failure, so reconciliation can find it.
        assert store.get_order(1) is not None
        assert store.get_order(1).status is OrderStatus.REJECTED  # type: ignore[union-attr]

    def test_a_kill_switch_stops_the_desk(self, store: Store) -> None:
        venue = _RecordingVenue()
        runner, context = _runner(store, venue=venue)
        runner.risk.engage_kill_switch("manual")
        outcome = runner.tick(context)
        assert outcome.halted_for
        assert not venue.submitted

    def test_a_broker_disagreement_stops_the_desk(self, store: Store) -> None:
        venue = _RecordingVenue()
        runner, context = _runner(store, venue=venue, broker={"BTC-USD": 5.0})
        outcome = runner.tick(context)
        assert outcome.reconciliation is not None
        assert outcome.reconciliation.should_halt
        assert not venue.submitted

    def test_the_halt_is_persisted(self, store: Store) -> None:
        runner, context = _runner(store, broker={"BTC-USD": 5.0})
        runner.tick(context)
        assert store.is_halted()

    def test_a_halt_is_recorded_once_not_every_tick(self, store: Store) -> None:
        runner, context = _runner(store, broker={"BTC-USD": 5.0})
        for _ in range(4):
            runner.tick(context)
        assert len(store.active_halts()) == 1

    def test_dry_run_records_but_never_sends(self, store: Store) -> None:
        venue = _RecordingVenue()
        runner, context = _runner(store, venue=venue, config=DeskConfig(dry_run=True))
        runner.tick(context)
        assert not venue.submitted
        assert store.get_order(1) is not None

    def test_equity_is_marked_every_tick(self, store: Store) -> None:
        runner, context = _runner(store)
        runner.tick(context)
        assert not store.equity_curve().empty

    def test_a_venue_without_resting_stops_is_refused(self, store: Store) -> None:
        """An in-process stop is a stop that vanishes on restart."""
        class _NoBrackets(_RecordingVenue):
            supports_brackets = False

        venue = _NoBrackets()
        runner, context = _runner(store, venue=venue)
        outcome = runner.tick(context)
        assert not venue.submitted
        assert "cannot hold a resting stop" in outcome.rejected[0]

    def test_that_venue_is_allowed_when_brackets_are_not_required(
        self, store: Store
    ) -> None:
        class _NoBrackets(_RecordingVenue):
            supports_brackets = False

        venue = _NoBrackets()
        runner, context = _runner(
            store, venue=venue, config=DeskConfig(require_brackets=False)
        )
        assert runner.tick(context).orders_sent == 1

    def test_repeating_the_same_bar_does_not_resend(self, store: Store) -> None:
        """The idempotency key is what makes a retry safe."""
        venue = _RecordingVenue()
        runner, context = _runner(store, venue=venue)
        runner.tick(context)
        second = runner.tick(context)
        assert len(venue.submitted) == 1
        assert second.orders_sent == 0


class TestResume:
    def test_a_clean_store_resumes_silently(self, store: Store) -> None:
        runner, _ = _runner(store)
        assert runner.resume(T0) == []

    def test_open_orders_are_reported_not_cancelled(self, store: Store) -> None:
        """Cancelling on restart would strip the stop off a live position."""
        order_id, _ = store.record_order(_order(), "k", T0)
        store.record_status(order_id, OrderStatus.SUBMITTED, T0)
        runner, _ = _runner(store)
        findings = runner.resume(T0)
        assert any("still open at restart" in f for f in findings)
        assert store.get_order(order_id).is_open  # type: ignore[union-attr]

    def test_positions_are_rebuilt_from_fills(self, store: Store) -> None:
        order_id, _ = store.record_order(_order(), "k", T0)
        store.record_fill(order_id, _fill(order_id, 3.0), "f1")
        runner, _ = _runner(store)
        assert any("positions restored" in f for f in runner.resume(T0))

    def test_an_uncleared_halt_is_surfaced(self, store: Store) -> None:
        store.record_halt("drawdown", T0)
        runner, _ = _runner(store)
        assert any("uncleared halt" in f for f in runner.resume(T0))

    def test_peak_equity_is_restored(self, store: Store) -> None:
        """Otherwise the drawdown guard resets to zero on every restart."""
        store.record_equity(T0, 200.0, 100.0)
        runner, _ = _runner(store)
        runner.resume(T0)
        assert runner._peak_equity == pytest.approx(200.0)

    def test_a_disagreement_on_resume_halts(self, store: Store) -> None:
        order_id, _ = store.record_order(_order(), "k", T0)
        store.record_fill(order_id, _fill(order_id, 1.0), "f1")
        runner, _ = _runner(store, broker={})
        findings = runner.resume(T0)
        assert any("RECONCILIATION FAILED" in f for f in findings)
        assert store.is_halted()


class TestFlattenAll:
    def test_flattens_both_directions(self) -> None:
        venue = _RecordingVenue()
        sent = flatten_all(
            venue, {"BTC-USD": 2.0, "ETH-USD": -3.0},
            {"BTC-USD": BTC, "ETH-USD": get_instrument("ETH-USD")}, T0,
        )
        assert len(sent) == 2
        sides = {o.instrument.symbol: o.side for o in sent}
        assert sides["BTC-USD"] is Side.SELL
        assert sides["ETH-USD"] is Side.BUY

    def test_skips_flat_symbols(self) -> None:
        venue = _RecordingVenue()
        assert flatten_all(venue, {"BTC-USD": 0.0}, {"BTC-USD": BTC}, T0) == []

    def test_one_failure_does_not_strand_the_rest(self) -> None:
        """A partial flatten must still flatten what it can."""
        class _FlakyVenue(_RecordingVenue):
            def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
                if order.instrument.symbol == "BTC-USD":
                    raise ExecutionError("rejected")
                return super().submit(order, timestamp)

        sent = flatten_all(
            _FlakyVenue(), {"BTC-USD": 1.0, "ETH-USD": 1.0},
            {"BTC-USD": BTC, "ETH-USD": get_instrument("ETH-USD")}, T0,
        )
        assert [o.instrument.symbol for o in sent] == ["ETH-USD"]

    def test_an_unknown_instrument_is_skipped_not_fatal(self) -> None:
        assert flatten_all(_RecordingVenue(), {"WAT": 1.0}, {}, T0) == []
