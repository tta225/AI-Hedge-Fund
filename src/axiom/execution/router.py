"""Order router — the single choke point between strategy and venue.

Nothing in AXIOM submits to a venue directly. Every order passes through here,
which enforces, in order:

1. the **kill switch**;
2. **mode compatibility** — a live venue is unreachable unless the platform is
   explicitly in live mode with the confirmation phrase set;
3. **risk approval** — the router will not size an order itself; it only routes
   what :class:`~axiom.risk.RiskManager` has already approved.

Rejections are recorded rather than raised, so a strategy loop cannot crash the
platform by proposing something inadmissible, and the audit trail keeps every
refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from axiom.core.types import OrderStatus, TradingMode
from axiom.execution.base import ExecutionVenue, Order
from axiom.risk.manager import RiskManager


@dataclass(slots=True)
class RoutingRecord:
    """One routing attempt, approved or not."""

    timestamp: pd.Timestamp
    order: Order
    routed: bool
    reason: str = ""


@dataclass(slots=True)
class OrderRouter:
    """Routes approved orders to a venue and keeps the audit trail."""

    venue: ExecutionVenue
    risk: RiskManager
    mode: TradingMode = TradingMode.BACKTEST
    live_confirmed: bool = False
    audit: list[RoutingRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.venue.is_live and not (
            self.mode is TradingMode.LIVE and self.live_confirmed
        ):
            raise PermissionError(
                f"venue {self.venue.name!r} is a live venue but the router is in "
                f"{self.mode.value} mode with live_confirmed={self.live_confirmed}. "
                "Refusing to construct a router that could reach real money."
            )

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        """Route ``order`` if permitted; otherwise reject and record why."""
        blocked = self._blocking_reason()
        if blocked:
            order.reject(blocked)
            self.audit.append(RoutingRecord(timestamp, order, routed=False, reason=blocked))
            return order

        self.venue.submit(order, timestamp)
        self.audit.append(RoutingRecord(timestamp, order, routed=True))
        return order

    def _blocking_reason(self) -> str:
        if self.risk.kill_switch:
            return "kill switch engaged — order not routed"
        if self.venue.is_live and self.mode is not TradingMode.LIVE:
            return (
                f"live venue unreachable in {self.mode.value} mode"
            )
        if self.mode is TradingMode.LIVE and not self.live_confirmed:
            return "live mode without confirmation — order not routed"
        return ""

    def cancel_all(self) -> list[Order]:
        return [self.venue.cancel(o) for o in self.venue.working_orders()]

    @property
    def rejected(self) -> list[RoutingRecord]:
        return [r for r in self.audit if not r.routed]

    @property
    def routed_count(self) -> int:
        return sum(1 for r in self.audit if r.routed)

    def flatten_reason_counts(self) -> dict[str, int]:
        """Rejection reasons by frequency — the first thing to read when a
        strategy mysteriously takes no trades."""
        counts: dict[str, int] = {}
        for record in self.audit:
            if not record.routed:
                counts[record.reason] = counts.get(record.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def emergency_flatten(router: OrderRouter, reason: str) -> list[Order]:
    """Engage the kill switch and cancel every working order.

    Does not liquidate open positions — closing a position is itself an order,
    and the kill switch has just forbidden orders. Flattening inventory is a
    deliberate human action, not an automatic consequence of a halt.
    """
    router.risk.engage_kill_switch(reason)
    return router.cancel_all()


def order_is_actionable(order: Order) -> bool:
    return order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
