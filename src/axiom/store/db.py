"""Append-mostly SQLite persistence for live trading state.

Design constraints, and why each is what it is:

**WAL mode.** Readers never block the writer. A dashboard querying equity while
the desk is submitting an order is the normal case, not an edge case, and the
default rollback journal makes it a lock contention bug.

**Idempotency keys on orders.** The dangerous failure in live trading is not
losing an order, it is sending it twice — a retry after a timeout that already
succeeded doubles the position. Every order carries a caller-supplied
``idempotency_key`` under a UNIQUE constraint, so a duplicate submit is a
no-op that returns the original order id rather than a second order.

**Orders are append-only in spirit.** Status changes are recorded as new rows
in ``order_events`` rather than by mutating the order. What actually happened
and in what sequence is the thing an incident review needs, and an UPDATE
destroys it. The current status is a view over the event log.

**Timestamps are stored as integer microseconds UTC.** SQLite has no date type;
storing ISO strings invites timezone bugs, and floats lose precision. Integer
microseconds sort correctly as integers and round-trip exactly.

Not designed for: multi-process writers (SQLite allows it, but the reconcile
loop assumes one desk owns its database), or a tick store. This holds decisions
and outcomes, not market data.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from axiom.core.types import Instrument, OrderStatus, Side, get_instrument
from axiom.execution.base import Fill, Order

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key    TEXT    NOT NULL UNIQUE,
    venue_order_id     TEXT,
    symbol             TEXT    NOT NULL,
    side               TEXT    NOT NULL,
    quantity           REAL    NOT NULL,
    order_type         TEXT    NOT NULL,
    limit_price        REAL,
    stop_price         REAL,
    stop_loss          REAL,
    take_profit        REAL,
    strategy           TEXT    NOT NULL DEFAULT '',
    tag                TEXT    NOT NULL DEFAULT '',
    created_us         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_created ON orders (created_us);
CREATE INDEX IF NOT EXISTS orders_symbol  ON orders (symbol);

CREATE TABLE IF NOT EXISTS order_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER NOT NULL REFERENCES orders (id),
    status      TEXT    NOT NULL,
    detail      TEXT    NOT NULL DEFAULT '',
    at_us       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS order_events_order ON order_events (order_id, id);

CREATE TABLE IF NOT EXISTS fills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders (id),
    venue_fill_id TEXT    UNIQUE,
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    quantity      REAL    NOT NULL,
    price         REAL    NOT NULL,
    commission    REAL    NOT NULL DEFAULT 0.0,
    slippage      REAL    NOT NULL DEFAULT 0.0,
    strategy      TEXT    NOT NULL DEFAULT '',
    at_us         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS fills_order  ON fills (order_id);
CREATE INDEX IF NOT EXISTS fills_symbol ON fills (symbol, at_us);

CREATE TABLE IF NOT EXISTS equity (
    at_us     INTEGER PRIMARY KEY,
    equity    REAL NOT NULL,
    cash      REAL NOT NULL,
    exposure  REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS halts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    reason   TEXT    NOT NULL,
    detail   TEXT    NOT NULL DEFAULT '',
    at_us    INTEGER NOT NULL,
    cleared_us INTEGER
);
"""


class StoreError(RuntimeError):
    """The store could not satisfy a request."""


def _to_us(timestamp: pd.Timestamp) -> int:
    """UTC microseconds since epoch."""
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value // 1_000)


def _from_us(value: int) -> pd.Timestamp:
    return pd.Timestamp(value * 1_000, unit="ns", tz="UTC")


@dataclass(frozen=True, slots=True)
class StoredOrder:
    """An order as persisted, with its current status resolved."""

    id: int
    idempotency_key: str
    venue_order_id: str | None
    instrument: Instrument
    side: Side
    quantity: float
    status: OrderStatus
    strategy: str
    created_at: pd.Timestamp
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status.is_open


class Store:
    """SQLite-backed record of what the desk did.

    Args:
        path: database file. ``":memory:"`` is supported for tests, but note
            that an in-memory store defeats the entire purpose in production —
            it is exactly the state that must survive a restart.
    """

    def __init__(self, path: str | Path = "data/desk.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False plus an explicit lock: the desk loop and a
        # dashboard thread both touch this, and SQLite's own thread check is
        # blunter than serialising the writes ourselves.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._configure()

    def _configure(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            # NORMAL rather than FULL: with WAL this still survives process
            # death (the case we care about) and only risks the last
            # transaction on OS/power loss, at a large write-throughput gain.
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._connection.commit()
        self._check_schema_version()

    def _check_schema_version(self) -> None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        found = int(row["value"]) if row else 0
        if found != SCHEMA_VERSION:
            raise StoreError(
                f"database at {self.path} is schema version {found}, this build "
                f"expects {SCHEMA_VERSION}. Migrate it or point at a new file — "
                "silently reading a mismatched schema is how a desk trades on "
                "misparsed state."
            )

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One transaction, committed on success and rolled back on failure."""
        with self._lock:
            try:
                yield self._connection
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- orders -----------------------------------------------------------

    def record_order(
        self, order: Order, idempotency_key: str, timestamp: pd.Timestamp
    ) -> tuple[int, bool]:
        """Persist an order before it is sent. Returns ``(id, was_new)``.

        Call this **before** submitting to the venue, never after. An order
        recorded after a successful send is an order that is invisible if the
        process dies between the two, and an invisible order is an unmanaged
        position.

        A repeated ``idempotency_key`` returns the existing id and
        ``was_new=False``, which is what makes a retry after an ambiguous
        timeout safe.
        """
        if not idempotency_key:
            raise ValueError("an idempotency key is required")

        with self._write() as connection:
            existing = connection.execute(
                "SELECT id FROM orders WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                return int(existing["id"]), False

            cursor = connection.execute(
                """INSERT INTO orders (
                       idempotency_key, venue_order_id, symbol, side, quantity,
                       order_type, limit_price, stop_price, stop_loss,
                       take_profit, strategy, tag, created_us
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    idempotency_key,
                    None,
                    order.instrument.symbol,
                    order.side.value,
                    float(order.quantity),
                    order.order_type.value,
                    order.limit_price,
                    order.stop_price,
                    order.stop_loss,
                    order.take_profit,
                    order.strategy,
                    order.tag,
                    _to_us(timestamp),
                ),
            )
            order_id = int(cursor.lastrowid or 0)
            connection.execute(
                "INSERT INTO order_events (order_id, status, detail, at_us) "
                "VALUES (?,?,?,?)",
                (order_id, OrderStatus.PENDING.value, "recorded", _to_us(timestamp)),
            )
            return order_id, True

    def attach_venue_id(self, order_id: int, venue_order_id: str) -> None:
        """Record the venue's identifier once the submit returns."""
        with self._write() as connection:
            connection.execute(
                "UPDATE orders SET venue_order_id = ? WHERE id = ?",
                (venue_order_id, order_id),
            )

    def record_status(
        self,
        order_id: int,
        status: OrderStatus,
        timestamp: pd.Timestamp,
        detail: str = "",
    ) -> None:
        """Append a status transition. History is never overwritten."""
        with self._write() as connection:
            connection.execute(
                "INSERT INTO order_events (order_id, status, detail, at_us) "
                "VALUES (?,?,?,?)",
                (order_id, status.value, detail, _to_us(timestamp)),
            )

    def order_history(self, order_id: int) -> list[tuple[OrderStatus, str, pd.Timestamp]]:
        """Every status this order passed through, oldest first."""
        rows = self._connection.execute(
            "SELECT status, detail, at_us FROM order_events "
            "WHERE order_id = ? ORDER BY id",
            (order_id,),
        ).fetchall()
        return [
            (OrderStatus(row["status"]), row["detail"], _from_us(row["at_us"]))
            for row in rows
        ]

    def get_order(self, order_id: int) -> StoredOrder | None:
        row = self._connection.execute(
            "SELECT * FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        return None if row is None else self._hydrate(row)

    def open_orders(self) -> list[StoredOrder]:
        """Orders whose latest status is still working.

        This is the first thing a restarting desk asks, and the answer is the
        difference between resuming and double-trading.
        """
        rows = self._connection.execute("SELECT * FROM orders").fetchall()
        return [order for order in map(self._hydrate, rows) if order.is_open]

    def _hydrate(self, row: sqlite3.Row) -> StoredOrder:
        latest = self._connection.execute(
            "SELECT status FROM order_events WHERE order_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        fills = self._connection.execute(
            "SELECT COALESCE(SUM(quantity), 0) AS q, "
            "       COALESCE(SUM(quantity * price), 0) AS notional "
            "FROM fills WHERE order_id = ?",
            (row["id"],),
        ).fetchone()
        filled = float(fills["q"])
        return StoredOrder(
            id=int(row["id"]),
            idempotency_key=row["idempotency_key"],
            venue_order_id=row["venue_order_id"],
            instrument=get_instrument(row["symbol"]),
            side=Side(row["side"]),
            quantity=float(row["quantity"]),
            status=OrderStatus(latest["status"]) if latest else OrderStatus.PENDING,
            strategy=row["strategy"],
            created_at=_from_us(row["created_us"]),
            filled_quantity=filled,
            average_fill_price=float(fills["notional"]) / filled if filled else 0.0,
        )

    # --- fills ------------------------------------------------------------

    def record_fill(
        self, order_id: int, fill: Fill, venue_fill_id: str | None = None
    ) -> bool:
        """Persist an execution report. Returns False if already recorded.

        ``venue_fill_id`` is UNIQUE, so a websocket replay or a poll that
        overlaps a push cannot book the same fill twice. Passing None disables
        that protection, which is only correct for a simulated venue.
        """
        with self._write() as connection:
            if venue_fill_id is not None:
                seen = connection.execute(
                    "SELECT 1 FROM fills WHERE venue_fill_id = ?", (venue_fill_id,)
                ).fetchone()
                if seen is not None:
                    return False
            connection.execute(
                """INSERT INTO fills (
                       order_id, venue_fill_id, symbol, side, quantity, price,
                       commission, slippage, strategy, at_us
                   ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    order_id,
                    venue_fill_id,
                    fill.instrument.symbol,
                    fill.side.value,
                    float(fill.quantity),
                    float(fill.price),
                    float(fill.commission),
                    float(fill.slippage),
                    fill.strategy,
                    _to_us(fill.timestamp),
                ),
            )
            return True

    def positions(self) -> dict[str, float]:
        """Net signed quantity per symbol, derived from fills.

        Derived rather than stored. A separately maintained position table is a
        second source of truth that can drift from the fills that produced it,
        and when they disagree there is no way to tell which is wrong.
        """
        rows = self._connection.execute(
            "SELECT symbol, side, SUM(quantity) AS q FROM fills GROUP BY symbol, side"
        ).fetchall()
        out: dict[str, float] = {}
        for row in rows:
            sign = 1.0 if Side(row["side"]) is Side.BUY else -1.0
            out[row["symbol"]] = out.get(row["symbol"], 0.0) + sign * float(row["q"])
        return {symbol: qty for symbol, qty in out.items() if qty != 0.0}

    def fills_for(self, symbol: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        query = "SELECT * FROM fills"
        params: tuple[Any, ...] = ()
        if symbol is not None:
            query += " WHERE symbol = ?"
            params = (symbol,)
        query += " ORDER BY at_us DESC LIMIT ?"
        params += (limit,)
        return [dict(row) for row in self._connection.execute(query, params).fetchall()]

    # --- equity and halts -------------------------------------------------

    def record_equity(
        self, timestamp: pd.Timestamp, equity: float, cash: float, exposure: float = 0.0
    ) -> None:
        """Mark the account. Same-microsecond marks replace rather than stack."""
        with self._write() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO equity (at_us, equity, cash, exposure) "
                "VALUES (?,?,?,?)",
                (_to_us(timestamp), float(equity), float(cash), float(exposure)),
            )

    def equity_curve(self) -> pd.Series:
        rows = self._connection.execute(
            "SELECT at_us, equity FROM equity ORDER BY at_us"
        ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        return pd.Series(
            [float(row["equity"]) for row in rows],
            index=pd.DatetimeIndex([_from_us(row["at_us"]) for row in rows]),
        )

    def record_halt(self, reason: str, timestamp: pd.Timestamp, detail: str = "") -> int:
        """Record that trading stopped, and why."""
        with self._write() as connection:
            cursor = connection.execute(
                "INSERT INTO halts (reason, detail, at_us) VALUES (?,?,?)",
                (reason, detail, _to_us(timestamp)),
            )
            return int(cursor.lastrowid or 0)

    def clear_halt(self, halt_id: int, timestamp: pd.Timestamp) -> None:
        with self._write() as connection:
            connection.execute(
                "UPDATE halts SET cleared_us = ? WHERE id = ? AND cleared_us IS NULL",
                (_to_us(timestamp), halt_id),
            )

    def active_halts(self) -> list[dict[str, Any]]:
        """Halts nobody has cleared.

        A restarting desk must check this before trading. A halt that does not
        survive a restart is not a halt — it is a delay, and the failure it was
        protecting against is still there.
        """
        rows = self._connection.execute(
            "SELECT * FROM halts WHERE cleared_us IS NULL ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def is_halted(self) -> bool:
        return bool(self.active_halts())
