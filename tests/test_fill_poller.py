"""Tests for the fill poller: exactly-once booking, overlap, attribution."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from axiom.core.types import Side, get_instrument
from axiom.desk.fills import WATERMARK_KEY, FillPoller, FillsSource
from axiom.execution.alpaca import VenueFill
from axiom.execution.base import ExecutionError, Fill, Order
from axiom.store import Store
from axiom.store.db import SCHEMA_VERSION, StoreError

AAPL = get_instrument("AAPL")
T0 = pd.Timestamp("2026-01-01T12:00:00Z")


def _venue_fill(
    fill_id: str = "f1",
    order_id: str = "o1",
    *,
    quantity: float = 5.0,
    price: float = 100.0,
    side: Side = Side.BUY,
    at: pd.Timestamp = T0,
) -> VenueFill:
    return VenueFill(
        venue_fill_id=fill_id,
        venue_order_id=order_id,
        fill=Fill(
            order_id=0,
            instrument=AAPL,
            side=side,
            quantity=quantity,
            price=price,
            commission=0.0,
            timestamp=at,
        ),
    )


class _StubSource(FillsSource):
    """Replays canned fill batches and records the windows requested."""

    def __init__(self, batches: list[list[VenueFill]] | None = None) -> None:
        self.batches = list(batches or [])
        self.requested: list[pd.Timestamp] = []
        self.raise_with: Exception | None = None

    def fills_since(self, since: pd.Timestamp, limit: int = 500) -> list[VenueFill]:
        self.requested.append(since)
        if self.raise_with is not None:
            raise self.raise_with
        return self.batches.pop(0) if self.batches else []


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def _submitted_order(store: Store, venue_id: str = "o1") -> int:
    """Record an order and attach a venue id, as the desk does on submit."""
    order = Order(instrument=AAPL, side=Side.BUY, quantity=5.0, stop_loss=90.0)
    order_id, _ = store.record_order(order, f"key-{venue_id}", T0)
    store.attach_venue_id(order_id, venue_id)
    return order_id


class TestExactlyOnce:
    def test_a_fill_is_booked_once(self, store: Store) -> None:
        _submitted_order(store)
        source = _StubSource([[_venue_fill()]])
        outcome = FillPoller(source, store).poll(T0)
        assert outcome.recorded == 1
        assert store.positions()["AAPL"] == pytest.approx(5.0)

    def test_re_reading_the_window_books_nothing_twice(self, store: Store) -> None:
        """This is what makes the overlap free rather than dangerous."""
        _submitted_order(store)
        fill = _venue_fill()
        source = _StubSource([[fill], [fill], [fill]])
        poller = FillPoller(source, store)
        for _ in range(3):
            poller.poll(T0)
        assert store.positions()["AAPL"] == pytest.approx(5.0)

    def test_duplicates_are_counted_not_hidden(self, store: Store) -> None:
        _submitted_order(store)
        fill = _venue_fill()
        source = _StubSource([[fill], [fill]])
        poller = FillPoller(source, store)
        poller.poll(T0)
        second = poller.poll(T0)
        assert second.recorded == 0
        assert second.duplicates == 1

    def test_distinct_fills_all_book(self, store: Store) -> None:
        _submitted_order(store)
        source = _StubSource([[_venue_fill("f1"), _venue_fill("f2")]])
        FillPoller(source, store).poll(T0)
        assert store.positions()["AAPL"] == pytest.approx(10.0)

    def test_sells_reduce_the_position(self, store: Store) -> None:
        _submitted_order(store)
        source = _StubSource([[
            _venue_fill("f1", quantity=5.0),
            _venue_fill("f2", quantity=2.0, side=Side.SELL),
        ]])
        FillPoller(source, store).poll(T0)
        assert store.positions()["AAPL"] == pytest.approx(3.0)


class TestWatermark:
    def test_cold_start_looks_back_a_bounded_distance(self, store: Store) -> None:
        """Requesting everything since 1970 is a slow, rate-limited request."""
        poller = FillPoller(_StubSource(), store, cold_start=pd.Timedelta(days=1))
        assert poller.watermark(T0) == T0 - pd.Timedelta(days=1)

    def test_advances_to_the_latest_fill_read(self, store: Store) -> None:
        _submitted_order(store)
        later = T0 + pd.Timedelta(minutes=30)
        source = _StubSource([[_venue_fill(at=T0), _venue_fill("f2", at=later)]])
        poller = FillPoller(source, store)
        outcome = poller.poll(T0)
        assert outcome.watermark == later
        assert poller.watermark() == later

    def test_the_watermark_lives_under_a_stable_key(self, store: Store) -> None:
        """Renaming it would silently cold-start every desk on upgrade."""
        _submitted_order(store)
        FillPoller(_StubSource([[_venue_fill()]]), store).poll(T0)
        assert store.get_meta(WATERMARK_KEY)

    def test_survives_a_restart(self, tmp_path: object) -> None:
        """A watermark that does not persist re-reads or loses a window."""
        path = f"{tmp_path}/desk.db"
        with Store(path) as store:
            _submitted_order(store)
            FillPoller(_StubSource([[_venue_fill()]]), store).poll(T0)
        with Store(path) as store:
            assert FillPoller(_StubSource(), store).watermark() == T0

    def test_each_poll_rewinds_by_the_overlap(self, store: Store) -> None:
        """Venue clocks disagree; polling from exactly the last point loses fills."""
        _submitted_order(store)
        overlap = pd.Timedelta(minutes=5)
        source = _StubSource([[_venue_fill(at=T0)], []])
        poller = FillPoller(source, store, overlap=overlap)
        poller.poll(T0)
        poller.poll(T0 + pd.Timedelta(minutes=10))
        assert source.requested[1] == T0 - overlap

    def test_an_empty_poll_does_not_rewind_the_watermark(self, store: Store) -> None:
        _submitted_order(store)
        poller = FillPoller(
            _StubSource([[_venue_fill(at=T0)], []]), store, overlap=pd.Timedelta(0)
        )
        poller.poll(T0)
        poller.poll(T0 + pd.Timedelta(hours=1))
        assert poller.watermark() == T0

    def test_a_negative_overlap_is_refused(self, store: Store) -> None:
        with pytest.raises(ValueError, match="overlap"):
            FillPoller(_StubSource(), store, overlap=pd.Timedelta(minutes=-1))


class TestAttribution:
    def test_a_known_order_is_linked(self, store: Store) -> None:
        order_id = _submitted_order(store, "o1")
        FillPoller(_StubSource([[_venue_fill(order_id="o1")]]), store).poll(T0)
        assert store.fills_for("AAPL")[0]["order_id"] == order_id

    def test_an_unknown_order_is_still_booked(self, store: Store) -> None:
        """A bracket leg fills under an id the desk never submitted."""
        outcome = FillPoller(
            _StubSource([[_venue_fill(order_id="never-seen")]]), store
        ).poll(T0)
        assert outcome.recorded == 1
        assert store.positions()["AAPL"] == pytest.approx(5.0)

    def test_an_unattributed_fill_is_reported(self, store: Store) -> None:
        """Dropping it silently would recreate the drift this prevents."""
        outcome = FillPoller(
            _StubSource([[_venue_fill(order_id="never-seen")]]), store
        ).poll(T0)
        assert len(outcome.unattributed) == 1
        assert "unattributed" in outcome.render()

    def test_a_missing_venue_order_id_is_handled(self, store: Store) -> None:
        outcome = FillPoller(_StubSource([[_venue_fill(order_id="")]]), store).poll(T0)
        assert outcome.recorded == 1
        assert "no venue order id" in outcome.unattributed[0]

    def test_an_unattributed_fill_stores_a_null_link(self, store: Store) -> None:
        FillPoller(_StubSource([[_venue_fill(order_id="unknown")]]), store).poll(T0)
        assert store.fills_for("AAPL")[0]["order_id"] is None


class TestFailureHandling:
    def test_a_venue_error_does_not_raise(self, store: Store) -> None:
        """A transient failure must not kill the loop that calls this."""
        source = _StubSource()
        source.raise_with = ExecutionError("alpaca: HTTP 429")
        outcome = FillPoller(source, store).poll(T0)
        assert outcome.error
        assert "429" in outcome.error
        assert "FAILED" in outcome.render()

    def test_a_failed_poll_leaves_the_window_unread(self, store: Store) -> None:
        """Advancing past an unread window loses those fills permanently."""
        _submitted_order(store)
        poller = FillPoller(_StubSource([[_venue_fill(at=T0)]]), store)
        poller.poll(T0)
        before = poller.watermark()

        failing = _StubSource()
        failing.raise_with = ExecutionError("boom")
        FillPoller(failing, store).poll(T0 + pd.Timedelta(hours=1))
        assert poller.watermark() == before

    def test_a_fill_missed_during_an_outage_is_caught_on_recovery(
        self, store: Store
    ) -> None:
        """The end-to-end property: an outage delays booking, never loses it."""
        _submitted_order(store)
        failing = _StubSource()
        failing.raise_with = ExecutionError("venue down")
        FillPoller(failing, store).poll(T0)
        assert store.positions() == {}

        recovered = _StubSource([[_venue_fill(at=T0)]])
        FillPoller(recovered, store).poll(T0 + pd.Timedelta(minutes=1))
        assert store.positions()["AAPL"] == pytest.approx(5.0)


class TestStoreAdditions:
    def test_venue_id_lookup(self, store: Store) -> None:
        order_id = _submitted_order(store, "venue-xyz")
        found = store.order_by_venue_id("venue-xyz")
        assert found is not None and found.id == order_id

    def test_unknown_venue_id_is_none(self, store: Store) -> None:
        assert store.order_by_venue_id("nope") is None

    def test_meta_round_trips(self, store: Store) -> None:
        store.set_meta("k", "v")
        assert store.get_meta("k") == "v"

    def test_meta_default(self, store: Store) -> None:
        assert store.get_meta("absent", "fallback") == "fallback"

    def test_the_schema_key_is_not_a_scratchpad(self, store: Store) -> None:
        with pytest.raises(ValueError, match="managed by migrations"):
            store.set_meta("schema_version", "99")


class TestMigration:
    @staticmethod
    def _build_v1(path: str) -> None:
        """A database exactly as schema version 1 left it."""
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                venue_order_id TEXT, symbol TEXT NOT NULL, side TEXT NOT NULL,
                quantity REAL NOT NULL, order_type TEXT NOT NULL,
                limit_price REAL, stop_price REAL, stop_loss REAL,
                take_profit REAL, strategy TEXT NOT NULL DEFAULT '',
                tag TEXT NOT NULL DEFAULT '', created_us INTEGER NOT NULL
            );
            CREATE TABLE order_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL REFERENCES orders (id),
                status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '',
                at_us INTEGER NOT NULL
            );
            CREATE TABLE fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL REFERENCES orders (id),
                venue_fill_id TEXT UNIQUE, symbol TEXT NOT NULL,
                side TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL,
                commission REAL NOT NULL DEFAULT 0.0,
                slippage REAL NOT NULL DEFAULT 0.0,
                strategy TEXT NOT NULL DEFAULT '', at_us INTEGER NOT NULL
            );
            CREATE TABLE equity (
                at_us INTEGER PRIMARY KEY, equity REAL NOT NULL,
                cash REAL NOT NULL, exposure REAL NOT NULL DEFAULT 0.0
            );
            CREATE TABLE halts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, reason TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '', at_us INTEGER NOT NULL,
                cleared_us INTEGER
            );
            INSERT INTO meta VALUES ('schema_version', '1');
            INSERT INTO orders (idempotency_key, symbol, side, quantity,
                                order_type, created_us)
                VALUES ('k1', 'AAPL', 'buy', 5.0, 'market', 1);
            INSERT INTO fills (order_id, venue_fill_id, symbol, side, quantity,
                               price, at_us)
                VALUES (1, 'vf1', 'AAPL', 'buy', 5.0, 100.0, 1);
            """
        )
        connection.commit()
        connection.close()

    def test_a_v1_database_opens(self, tmp_path: object) -> None:
        path = f"{tmp_path}/v1.db"
        self._build_v1(path)
        with Store(path) as store:
            assert int(store.get_meta("schema_version")) == SCHEMA_VERSION

    def test_existing_fills_are_preserved(self, tmp_path: object) -> None:
        """A migration that loses position history is worse than no migration."""
        path = f"{tmp_path}/v1.db"
        self._build_v1(path)
        with Store(path) as store:
            assert store.positions()["AAPL"] == pytest.approx(5.0)

    def test_unattributed_fills_become_possible(self, tmp_path: object) -> None:
        """The point of the migration: bracket legs can now be booked."""
        path = f"{tmp_path}/v1.db"
        self._build_v1(path)
        with Store(path) as store:
            assert store.record_fill(None, _venue_fill("vf2").fill, "vf2") is True

    def test_a_newer_database_is_refused_not_guessed(self, tmp_path: object) -> None:
        path = f"{tmp_path}/future.db"
        with Store(path) as store:
            store._connection.execute(
                "UPDATE meta SET value = '99' WHERE key = 'schema_version'"
            )
            store._connection.commit()
        with pytest.raises(StoreError, match="schema version"):
            Store(path)
