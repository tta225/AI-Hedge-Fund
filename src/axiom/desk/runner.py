"""The desk loop: guards, signal, risk, record, send — in that order.

This is the trading bot. It is deliberately small, because the interesting
parts live elsewhere and the loop's only job is to sequence them correctly.

The ordering is the design, and each step is before the next for a reason:

1. **Guards first.** Nothing else runs if it is not safe to trade. Checking
   afterwards means the decision has already been made and someone will be
   tempted to act on it.
2. **Reconcile before sizing.** Position size is computed from current
   exposure; sizing against a stale position book produces a correct-looking
   number for the wrong account.
3. **Record before sending.** An order persisted after a successful send is
   invisible if the process dies in between, and an invisible order is an
   unmanaged position. Recording first can leave a phantom row if the send
   fails — which reconciliation catches and a human resolves. Those two failure
   modes are not symmetric.
4. **Brackets go to the venue, not into this process.** A stop held in memory
   is a stop that does not exist during a restart, a network partition, or a
   crash — exactly the moments it is needed. See :attr:`DeskConfig.require_brackets`.

Crash-only: there is no shutdown handler, because a shutdown handler is code
that does not run when the machine is killed. State is reconstructed from
:mod:`axiom.store` on startup, so a hard kill and a graceful stop are the same
thing.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from axiom.core.types import Instrument, OrderStatus, Side
from axiom.desk.guards import GuardReport, check_guards
from axiom.execution.base import ExecutionError, ExecutionVenue, Order
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.store.db import Store
from axiom.store.reconcile import Reconciliation, reconcile_positions
from axiom.strategy.base import Signal, Strategy, StrategyContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeskConfig:
    """Operating limits for a live desk.

    Args:
        expected_interval: cadence of the bar feed, used by the freshness
            guard. Set it to the actual timeframe being traded.
        require_brackets: refuse to send an entry whose protective stop cannot
            be attached at the venue. Defaults True and should stay True — an
            unbracketed entry is an unbounded loss waiting for a network
            partition. Only turn it off for a venue that genuinely cannot hold
            resting stops, and then only with an external liquidator.
        max_orders_per_tick: circuit breaker on a runaway loop. A strategy bug
            that emits a signal every evaluation should cost a few orders, not
            the account.
        dry_run: run every step including persistence, but never call the
            venue. The honest way to exercise the whole path before arming it.
    """

    expected_interval: pd.Timedelta = field(
        default_factory=lambda: pd.Timedelta(hours=1)
    )
    require_brackets: bool = True
    max_orders_per_tick: int = 3
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.expected_interval <= pd.Timedelta(0):
            raise ValueError("expected_interval must be positive")
        if self.max_orders_per_tick < 1:
            raise ValueError("max_orders_per_tick must be at least 1")


@dataclass(slots=True)
class TickOutcome:
    """What one pass of the loop did, and why."""

    at: pd.Timestamp
    guards: GuardReport
    reconciliation: Reconciliation | None = None
    signals: int = 0
    orders_sent: int = 0
    rejected: list[str] = field(default_factory=list)
    halted_for: str = ""

    @property
    def traded(self) -> bool:
        return self.orders_sent > 0

    def render(self) -> str:
        if self.halted_for:
            return f"[{self.at}] HALTED — {self.halted_for}"
        parts = [f"[{self.at}] {self.signals} signal(s), {self.orders_sent} sent"]
        if self.rejected:
            parts.append(f"rejected: {'; '.join(self.rejected)}")
        return " | ".join(parts)


class DeskRunner:
    """Sequences one trading decision, safely, and records all of it.

    The runner owns no timer. It exposes :meth:`tick`, and the caller decides
    when to call it — on a bar close, on a schedule, from a test. Embedding a
    scheduler here would make the interesting logic untestable without waiting
    in real time, and every bug in it would only reproduce live.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        venue: ExecutionVenue,
        store: Store,
        risk: RiskManager,
        portfolio: Portfolio,
        config: DeskConfig | None = None,
        broker_positions: Callable[[], Mapping[str, float]] | None = None,
    ) -> None:
        self.strategy = strategy
        self.venue = venue
        self.store = store
        self.risk = risk
        self.portfolio = portfolio
        self.config = config or DeskConfig()
        # Injected rather than pulled off the venue so a venue that cannot
        # report positions is a wiring decision rather than a silent skip of
        # the most important guard.
        self.broker_positions = broker_positions
        self._peak_equity = 0.0

    # --- startup ----------------------------------------------------------

    def resume(self, timestamp: pd.Timestamp) -> list[str]:
        """Rebuild what the process forgot. Call once, before the first tick.

        Returns a list of findings a human should read. An empty list means the
        desk came back to exactly the state it left.
        """
        findings: list[str] = []

        halts = self.store.active_halts()
        if halts:
            findings.append(
                f"{len(halts)} uncleared halt(s): "
                + "; ".join(str(h["reason"]) for h in halts)
            )

        open_orders = self.store.open_orders()
        if open_orders:
            # Not cancelled automatically: an open order may be a working
            # protective stop, and cancelling it on restart would strip the
            # protection from a live position.
            findings.append(
                f"{len(open_orders)} order(s) still open at restart: "
                + ", ".join(f"#{o.id} {o.instrument.symbol} {o.side.value}" for o in open_orders)
            )

        positions = self.store.positions()
        if positions:
            findings.append(
                "positions restored from fills: "
                + ", ".join(f"{s}={q:+g}" for s, q in sorted(positions.items()))
            )

        curve = self.store.equity_curve()
        if not curve.empty:
            self._peak_equity = float(curve.max())
            findings.append(f"peak equity restored as {self._peak_equity:,.2f}")

        reconciliation = self.reconcile(timestamp)
        if reconciliation is not None and reconciliation.should_halt:
            self.store.record_halt(
                "reconciliation_failed_on_resume",
                timestamp,
                reconciliation.render(),
            )
            findings.append("RECONCILIATION FAILED ON RESUME — trading is halted")

        return findings

    def reconcile(self, timestamp: pd.Timestamp) -> Reconciliation | None:
        """Compare the store's position book against the broker's."""
        if self.broker_positions is None:
            return None
        return reconcile_positions(
            self.store.positions(), dict(self.broker_positions()), at=timestamp
        )

    # --- the loop ---------------------------------------------------------

    def tick(self, context: StrategyContext, now: pd.Timestamp | None = None) -> TickOutcome:
        """One decision: check, ask, size, record, send."""
        moment = pd.Timestamp(now) if now is not None else context.timestamp
        equity = self.portfolio.equity
        self._peak_equity = max(self._peak_equity, equity)
        self.store.record_equity(moment, equity, self.portfolio.cash)

        reconciliation = self.reconcile(moment)
        guards = check_guards(
            now=moment,
            last_bar_at=context.timestamp,
            expected_interval=self.config.expected_interval,
            equity=equity,
            peak_equity=self._peak_equity,
            store_halted=self.store.is_halted(),
            kill_switch=self.risk.kill_switch,
            reconciled=reconciliation is None or reconciliation.is_clean,
        )
        outcome = TickOutcome(at=moment, guards=guards, reconciliation=reconciliation)

        if not guards.may_trade:
            outcome.halted_for = "; ".join(guards.blocking)
            self._halt_once(moment, outcome.halted_for)
            return outcome

        signal = self.strategy.evaluate(context)
        if signal is None:
            return outcome
        outcome.signals = 1

        order = self._size(signal, context, outcome)
        if order is None:
            return outcome

        if self._send(order, moment, outcome):
            outcome.orders_sent += 1
        return outcome

    def _halt_once(self, timestamp: pd.Timestamp, reason: str) -> None:
        """Record a halt unless one is already open, so the log stays readable."""
        if not self.store.is_halted():
            self.store.record_halt("guard_failed", timestamp, reason)
            logger.error("desk halted: %s", reason)

    def _size(
        self, signal: Signal, context: StrategyContext, outcome: TickOutcome
    ) -> Order | None:
        """Turn a signal into a sized order, or explain why not."""
        decision = self.risk.evaluate(
            instrument=context.instrument,
            direction=signal.direction,
            entry=signal.entry,
            stop=signal.stop,
            portfolio=self.portfolio,
            timestamp=context.timestamp,
        )
        if not decision.approved or decision.sizing is None:
            outcome.rejected.append(decision.explain())
            return None

        quantity = float(decision.sizing.quantity)
        if quantity <= 0:
            outcome.rejected.append("risk sized the position to zero")
            return None

        # `Signal` guarantees a stop exists — it validates one against entry in
        # __post_init__ — so the open question is never whether the strategy
        # supplied one, it is whether the venue will *hold* it. A stop this
        # process has to watch is a stop that vanishes on restart.
        if self.config.require_brackets and not self.venue.supports_brackets:
            outcome.rejected.append(
                f"venue {self.venue.name!r} cannot hold a resting stop, and "
                "require_brackets is on — an in-process stop is an unbounded "
                "loss waiting for a restart"
            )
            return None

        return Order(
            instrument=context.instrument,
            side=Side.from_direction(signal.direction),
            quantity=quantity,
            strategy=self.strategy.name,
            created_at=context.timestamp,
            stop_loss=signal.stop,
            take_profit=signal.primary_target,
            reference_price=signal.entry,
        )

    def _send(self, order: Order, timestamp: pd.Timestamp, outcome: TickOutcome) -> bool:
        """Persist, then submit. Never the other way round."""
        if outcome.orders_sent >= self.config.max_orders_per_tick:
            outcome.rejected.append(
                f"per-tick order cap of {self.config.max_orders_per_tick} reached"
            )
            return False

        key = self._idempotency_key(order, timestamp)
        # Carried on the order so a venue that supports client-side order ids
        # can enforce the same guarantee. Alpaca rejects a duplicate outright,
        # which is what turns "the submit timed out, did it land?" from a
        # position-doubling hazard into a safe retry.
        order.tag = key
        order_id, was_new = self.store.record_order(order, key, timestamp)
        if not was_new:
            # The same decision, already sent. This is the retry path working.
            logger.info("order %s already recorded as #%d; not resending", key, order_id)
            return False

        if self.config.dry_run:
            self.store.record_status(order_id, OrderStatus.CANCELLED, timestamp, "dry_run")
            outcome.rejected.append("dry_run: order recorded but not sent")
            return False

        try:
            submitted = self.venue.submit(order, timestamp)
        except ExecutionError as exc:
            # The order is already persisted, so this leaves a row with no
            # venue id. That is the intended residue: reconciliation will
            # surface it, rather than the order vanishing.
            self.store.record_status(order_id, OrderStatus.REJECTED, timestamp, str(exc))
            outcome.rejected.append(f"venue rejected: {exc}")
            return False

        self.store.attach_venue_id(order_id, str(submitted.order_id))
        self.store.record_status(order_id, submitted.status, timestamp, "submitted")
        return True

    @staticmethod
    def _idempotency_key(order: Order, timestamp: pd.Timestamp) -> str:
        """A key that is stable across a retry of the *same* decision.

        Derived from the decision's content and the bar it was made on, not
        from a random UUID or the wall clock. A random key would make every
        retry a new order, which is precisely the duplicate-submission bug the
        key exists to prevent.

        A UUID namespace rather than a raw hash so the value is fixed-length
        and obviously opaque to anyone reading the database.
        """
        material = "|".join(
            [
                order.instrument.symbol,
                order.side.value,
                f"{order.quantity:.10g}",
                f"{order.stop_loss if order.stop_loss is not None else 'na'}",
                order.strategy,
                str(pd.Timestamp(timestamp).value),
            ]
        )
        return str(uuid.uuid5(uuid.NAMESPACE_OID, material))


def flatten_all(
    venue: ExecutionVenue,
    positions: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    timestamp: pd.Timestamp,
) -> list[Order]:
    """Emergency exit: market out of everything.

    Deliberately not a method on :class:`DeskRunner`. Flattening is the thing a
    human reaches for when the desk itself is the problem, and it must not
    depend on the runner's state being sane — the guards, the risk manager, and
    the strategy are all bypassed on purpose.

    Protective stops are *not* cancelled first. Leaving them in place means a
    partial failure here still leaves the position protected.
    """
    sent: list[Order] = []
    for symbol, quantity in positions.items():
        if quantity == 0.0:
            continue
        instrument = instruments.get(symbol)
        if instrument is None:
            logger.error("cannot flatten %s: no instrument definition", symbol)
            continue
        order = Order(
            instrument=instrument,
            side=Side.SELL if quantity > 0 else Side.BUY,
            quantity=abs(quantity),
            strategy="flatten_all",
            created_at=timestamp,
            tag="emergency_flatten",
        )
        try:
            sent.append(venue.submit(order, timestamp))
        except ExecutionError:
            # Keep going: one symbol failing to flatten must not strand the
            # rest, and the caller gets back only what actually went out.
            logger.exception("failed to flatten %s", symbol)
    return sent
