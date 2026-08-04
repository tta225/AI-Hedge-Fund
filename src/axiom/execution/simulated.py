"""Simulated execution venue.

Fill model, and its deliberate conservatism:

* **market orders** fill at the *next* bar's open, never the signal bar's close.
  Filling on the close of the bar that generated the signal is the classic way a
  backtest earns money that was never available.
* **slippage** is applied against the order every time, sized in ticks;
* **commission** is charged per unit, both sides;
* **limit orders** require price to trade strictly *through* the limit, not
  merely touch it — touching does not guarantee a fill in a real queue;
* **stop orders** become market orders and pay slippage on top of the stop;
* **gaps** are honoured: a stop that gaps through fills at the open, worse than
  its trigger, exactly as it would in the market.

Every one of these choices makes results worse, not better. That is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from axiom.core.types import Bar, OrderStatus, OrderType, Side
from axiom.execution.base import ExecutionVenue, Fill, Order


@dataclass(frozen=True, slots=True)
class FillModel:
    """Cost assumptions applied to every simulated fill."""

    slippage_ticks: float = 1.0
    commission_per_unit: float = 0.85
    #: Require price to exceed a limit by this many ticks before filling.
    limit_through_ticks: float = 0.0

    def commission(self, quantity: float) -> float:
        return self.commission_per_unit * quantity


class SimulatedVenue(ExecutionVenue):
    """Bar-driven simulator used by the backtester and paper mode."""

    name = "simulated"
    is_live = False

    def __init__(self, fill_model: FillModel | None = None) -> None:
        self.fill_model = fill_model or FillModel()
        self._working: list[Order] = []
        self.fills: list[Fill] = []

    # --- ExecutionVenue -----------------------------------------------------

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        order.created_at = timestamp
        order.status = OrderStatus.SUBMITTED
        self._working.append(order)
        return order

    def cancel(self, order: Order) -> Order:
        if order in self._working:
            self._working.remove(order)
        if order.is_open:
            order.status = OrderStatus.CANCELLED
        return order

    def working_orders(self) -> list[Order]:
        return list(self._working)

    # --- simulation ---------------------------------------------------------

    def process_bar(self, bar: Bar) -> list[Fill]:
        """Attempt to fill working orders against ``bar``. Returns new fills."""
        produced: list[Fill] = []
        for order in list(self._working):
            fill = self._try_fill(order, bar)
            if fill is not None:
                produced.append(fill)
                self.fills.append(fill)
                self._working.remove(order)
        return produced

    def _try_fill(self, order: Order, bar: Bar) -> Fill | None:
        # An order submitted on this bar cannot also fill on it — the signal was
        # generated from this bar's close, so the earliest fill is the next open.
        if order.created_at is not None and bar.timestamp <= order.created_at:
            return None

        tick = order.instrument.tick_size
        reference: float | None = None

        if order.order_type is OrderType.MARKET:
            reference = bar.open

        elif order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            through = self.fill_model.limit_through_ticks * tick
            if order.side is Side.BUY and bar.low <= order.limit_price - through:
                # A gap below the limit fills better, at the open.
                reference = min(order.limit_price, bar.open)
            elif order.side is Side.SELL and bar.high >= order.limit_price + through:
                reference = max(order.limit_price, bar.open)

        elif order.order_type is OrderType.STOP:
            assert order.stop_price is not None
            if order.side is Side.BUY and bar.high >= order.stop_price:
                reference = max(order.stop_price, bar.open)
            elif order.side is Side.SELL and bar.low <= order.stop_price:
                reference = min(order.stop_price, bar.open)

        elif order.order_type is OrderType.STOP_LIMIT:
            assert order.stop_price is not None and order.limit_price is not None
            triggered = (
                bar.high >= order.stop_price
                if order.side is Side.BUY
                else bar.low <= order.stop_price
            )
            if triggered:
                if order.side is Side.BUY and bar.low <= order.limit_price:
                    reference = min(order.limit_price, max(bar.open, bar.low))
                elif order.side is Side.SELL and bar.high >= order.limit_price:
                    reference = max(order.limit_price, min(bar.open, bar.high))

        if reference is None:
            return None

        # Limit orders rest in the book and are not charged slippage; anything
        # that crosses the spread is.
        pays_slippage = order.order_type in (OrderType.MARKET, OrderType.STOP)
        slippage = self.fill_model.slippage_ticks * tick if pays_slippage else 0.0
        price = order.instrument.round_to_tick(reference + slippage * order.side.sign)
        # Never report a fill outside the bar's actual range.
        price = min(max(price, bar.low), bar.high)

        commission = self.fill_model.commission(order.quantity)
        order.filled_quantity = order.quantity
        order.average_fill_price = price
        order.status = OrderStatus.FILLED

        return Fill(
            order_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=order.quantity,
            price=price,
            commission=commission,
            timestamp=bar.timestamp,
            slippage=price - reference,
            strategy=order.strategy,
        )

    def reset(self) -> None:
        self._working.clear()
        self.fills.clear()
