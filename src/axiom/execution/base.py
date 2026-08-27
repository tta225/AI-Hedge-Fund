"""Order model and the execution venue interface."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from axiom.core.types import Instrument, OrderStatus, OrderType, Side, TimeInForce

_ORDER_IDS = itertools.count(1)


class ExecutionError(RuntimeError):
    """An order could not be routed or filled."""


@dataclass(slots=True)
class Order:
    """A single order. Mutable: status and fill state evolve in place."""

    instrument: Instrument
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    #: Strategy that produced this order, for attribution.
    strategy: str = ""
    tag: str = ""
    order_id: int = field(default_factory=lambda: next(_ORDER_IDS))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: pd.Timestamp | None = None
    reject_reason: str = ""
    #: Bracket legs, priced when the order is created.
    stop_loss: float | None = None
    take_profit: float | None = None
    #: The price this order was sized against — the signal's intended entry.
    reference_price: float | None = None
    #: Maximum adverse distance, in price points, between ``reference_price``
    #: and the fill. A fill worse than this expires the order instead.
    #:
    #: This is the venue half of the risk-budget guarantee: sizing assumes the
    #: worst case within this bound, and the venue refuses anything outside it,
    #: so realised risk cannot exceed the budget.
    #:
    #: Deliberately ``None`` on protective exits. A stop-loss must always fill,
    #: however bad the price — refusing to exit because the fill is unfavourable
    #: is how a bounded loss becomes an unbounded one.
    max_entry_deviation: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} order requires a limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires a stop_price")

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    def reject(self, reason: str) -> Order:
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason
        return self

    def risk_per_unit(self) -> float | None:
        """Distance from entry to stop, in points, if both are known."""
        entry = self.limit_price if self.limit_price is not None else self.average_fill_price
        if not entry or self.stop_loss is None:
            return None
        return abs(entry - self.stop_loss)


@dataclass(frozen=True, slots=True)
class Fill:
    """An execution report."""

    order_id: int
    instrument: Instrument
    side: Side
    quantity: float
    price: float
    commission: float
    timestamp: pd.Timestamp
    #: Signed distance between the reference price and the fill, in points.
    slippage: float = 0.0
    strategy: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price * self.instrument.point_value


class ExecutionVenue(ABC):
    """Where orders go. Simulated, paper, and live venues share this contract."""

    name: str = "venue"
    #: True when this venue can move real money. Guarded by the router.
    is_live: bool = False
    #: True when the venue itself will hold a protective stop after the entry
    #: fills, rather than relying on this process to watch the price.
    #:
    #: The distinction is the difference between a bounded and an unbounded
    #: loss. A stop that lives in the trading process disappears during a
    #: restart, a network partition, or a crash — precisely the moments a stop
    #: is needed. A desk configured with ``require_brackets`` refuses to send
    #: entries to a venue that declares False here.
    supports_brackets: bool = True

    @abstractmethod
    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        """Accept an order. Returns the same object with updated status."""

    @abstractmethod
    def cancel(self, order: Order) -> Order:
        """Cancel a working order."""

    @abstractmethod
    def working_orders(self) -> list[Order]:
        """Orders still live at the venue."""
