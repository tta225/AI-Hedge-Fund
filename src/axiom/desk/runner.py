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

from axiom.alpha.panel import Panel
from axiom.core.types import Instrument, OrderStatus, Side
from axiom.desk.compliance import ComplianceEngine
from axiom.desk.compliance import Context as ComplianceContext
from axiom.desk.guards import GuardReport, check_guards
from axiom.execution.base import ExecutionError, ExecutionVenue, Order
from axiom.ops.logs import correlation_id, log_event
from axiom.ops.metrics import (
    EQUITY,
    GROSS_EXPOSURE,
    HALTS,
    IS_HALTED,
    OPEN_POSITIONS,
    ORDERS_REJECTED,
    ORDERS_SENT,
    SIGNALS_BLOCKED,
    SIGNALS_GENERATED,
    TICK_SECONDS,
)
from axiom.portfolio.positions import Portfolio
from axiom.portfolio.risk import PortfolioRiskManager
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
        portfolio_risk: PortfolioRiskManager | None = None,
        risk_panel: Panel | None = None,
        compliance: ComplianceEngine | None = None,
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
        # Both or neither: a portfolio risk manager with no panel has no
        # covariance to reason about, and a panel with no manager is unused.
        if (portfolio_risk is None) != (risk_panel is None):
            raise ValueError(
                "portfolio_risk and risk_panel must be supplied together — a "
                "risk manager with no universe cannot estimate correlation, "
                "and would silently pass every size through unchanged"
            )
        self.portfolio_risk = portfolio_risk
        self.risk_panel = risk_panel
        # Defaults to an empty engine rather than to a set of limits. Silently
        # imposing position caps a caller did not ask for would block orders
        # for reasons that appear nowhere in their configuration; an empty
        # engine says so in `describe()`.
        self.compliance = compliance or ComplianceEngine()
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
        """One decision: check, ask, size, record, send.

        The whole tick runs under one correlation id, so the guard that
        passed, the size that was chosen and the order that carried it can be
        pulled out of a log as a single causal chain rather than grepped for by
        symbol and hoped over.
        """
        with correlation_id(), TICK_SECONDS.time(strategy=self.strategy.name):
            return self._tick(context, now)

    def _compliance_context(self) -> ComplianceContext:
        """The account snapshot every rule sees, taken once.

        Positions come from the portfolio rather than the store because the
        portfolio is what sizing just used; a compliance check against a
        different position book than the sizer used would block or permit for
        reasons the sizer cannot see.
        """
        positions = {
            symbol: position.quantity
            for symbol, position in self.portfolio.positions.items()
        }
        prices = {
            symbol: position.last_price
            for symbol, position in self.portfolio.positions.items()
            if position.last_price
        }
        return ComplianceContext(
            equity=self.portfolio.equity, positions=positions, prices=prices
        )

    def _tick(self, context: StrategyContext, now: pd.Timestamp | None) -> TickOutcome:
        moment = pd.Timestamp(now) if now is not None else context.timestamp
        equity = self.portfolio.equity
        self._peak_equity = max(self._peak_equity, equity)
        self.store.record_equity(moment, equity, self.portfolio.cash)

        open_positions = [p for p in self.portfolio.positions.values() if not p.is_flat]
        EQUITY.set(equity)
        OPEN_POSITIONS.set(len(open_positions))
        GROSS_EXPOSURE.set(sum(abs(p.exposure) for p in open_positions))

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

        IS_HALTED.set(0.0 if guards.may_trade else 1.0)
        if not guards.may_trade:
            outcome.halted_for = "; ".join(guards.blocking)
            HALTS.inc()
            log_event(
                logger, "guards_blocked", level=logging.WARNING,
                symbol=context.instrument.symbol, reason=outcome.halted_for,
            )
            self._halt_once(moment, outcome.halted_for)
            return outcome

        signal = self.strategy.evaluate(context)
        if signal is None:
            return outcome
        outcome.signals = 1
        SIGNALS_GENERATED.inc(strategy=self.strategy.name)

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

        # The per-trade budget above is blind to what is already on the book:
        # three 0.5% bets in correlated names are one 1.5% bet, and the count
        # says "diversified" while the covariance says otherwise. This scales
        # down for correlation, realised volatility and drawdown — it can only
        # ever reduce, never raise, so the per-trade guarantee still holds.
        quantity, portfolio_reason = self._apply_portfolio_risk(
            quantity, context, outcome
        )
        if quantity <= 0:
            outcome.rejected.append(f"portfolio risk sized to zero: {portfolio_reason}")
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

    def _apply_portfolio_risk(
        self, quantity: float, context: StrategyContext, outcome: TickOutcome
    ) -> tuple[float, str]:
        """Scale a per-trade size down for what the book already carries.

        A no-op when no :class:`~axiom.portfolio.risk.PortfolioRiskManager` or
        risk panel was supplied. That is a deliberate default rather than a
        silent one: the alternative is refusing to trade without a covariance
        model, and a single-instrument desk legitimately has no correlation to
        model. The absence is reported on the tick either way.
        """
        if self.portfolio_risk is None or self.risk_panel is None:
            return quantity, "no portfolio risk model configured"

        equity = self.portfolio.equity
        if equity <= 0:
            return quantity, "no equity to weight against"

        symbol = context.instrument.symbol
        notional = quantity * context.price * context.instrument.point_value
        proposed_weight = notional / equity
        current = {
            position.instrument.symbol: (
                position.quantity
                * position.last_price
                * position.instrument.point_value
                / equity
            )
            for position in self.portfolio.open_positions
        }

        # Score against the most recent bar of the panel that this decision's
        # timestamp permits, so the covariance never sees past the decision.
        index = int(self.risk_panel.index.searchsorted(context.timestamp, side="left"))
        scalar, reason = self.portfolio_risk.scale_for(
            symbol,
            proposed_weight,
            current,
            self.risk_panel,
            index,
            equity=equity,
            peak_equity=max(self._peak_equity, equity),
        )
        if scalar < 1.0:
            outcome.rejected.append(f"portfolio risk x{scalar:.2f}: {reason}")
        return quantity * scalar, reason

    def _send(self, order: Order, timestamp: pd.Timestamp, outcome: TickOutcome) -> bool:
        """Persist, then submit. Never the other way round."""
        if outcome.orders_sent >= self.config.max_orders_per_tick:
            outcome.rejected.append(
                f"per-tick order cap of {self.config.max_orders_per_tick} reached"
            )
            return False

        # Compliance runs before the order is recorded, not after. A blocked
        # order is one that was never permitted to exist, and writing it into
        # the order log first would leave a row that reconciliation has to
        # explain away on every subsequent startup.
        decision = self.compliance.check(order, self._compliance_context())
        if not decision.allowed:
            SIGNALS_BLOCKED.inc(len(decision.breaches), reason="compliance")
            outcome.rejected.append(f"compliance: {decision.reason}")
            log_event(
                logger, "order_blocked",
                symbol=order.instrument.symbol, side=order.side.value,
                quantity=order.quantity, strategy=order.strategy,
                rules=[breach.rule for breach in decision.breaches],
                detail=decision.reason,
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
            ORDERS_REJECTED.inc(venue=self.venue.name)
            outcome.rejected.append(f"venue rejected: {exc}")
            return False

        self.store.attach_venue_id(order_id, str(submitted.order_id))
        self.store.record_status(order_id, submitted.status, timestamp, "submitted")
        ORDERS_SENT.inc(venue=self.venue.name, strategy=order.strategy or "unattributed")
        log_event(
            logger, "order_submitted",
            order_id=order_id, venue_order_id=str(submitted.order_id),
            symbol=order.instrument.symbol, side=order.side.value,
            quantity=order.quantity, stop_loss=order.stop_loss,
            take_profit=order.take_profit, strategy=order.strategy,
        )
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
