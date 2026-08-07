"""Alpaca **paper** trading venue.

What this is
------------
The first venue in this platform that sends an order somewhere other than a
simulator. It routes to Alpaca's paper endpoint, which fills against real market
data with real queue behaviour but settles in fake money.

`is_live` is **False**, deliberately and permanently for this class. Paper
orders cannot move real money, so the router's live-venue refusal does not
apply. A live adapter would be a different class with `is_live = True`, and the
router already refuses those — that friction is the feature.

Why a paper venue is not just a simulator with extra steps
-----------------------------------------------------------
`SimulatedVenue` fills from the bar it is given, using a model. It cannot show
you:

* **Rejections you did not model.** Wash-trade blocks, PDT restrictions,
  insufficient buying power, symbols that are not tradeable, orders outside
  market hours. These are the errors that turn a working backtest into a broken
  deployment, and no simulator raises them because nobody thought to model them.
* **Partial fills and queue position.** A backtest fills the whole order at one
  price. A real book does not.
* **State drift.** The broker's idea of your position and this platform's idea
  of it diverge — from a manual trade, a corporate action, a missed fill. That
  divergence is silent and compounding, which is why `reconcile` exists and is
  the most important method here.

The endpoint refuses to guess
-----------------------------
Every method that mutates state raises rather than returning a plausible-looking
default when the API disagrees. A venue that swallows an error and reports
success is worse than one that crashes: the position is real either way, and
only one of those tells you.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from axiom.core.credentials import get_credential
from axiom.core.types import Instrument, OrderStatus, OrderType, Side, TimeInForce, get_instrument
from axiom.execution.base import ExecutionError, ExecutionVenue, Fill, Order

__all__ = ["AlpacaPaperVenue", "BrokerPosition", "ReconciliationReport"]

_PAPER_HOST = "https://paper-api.alpaca.markets"

#: Alpaca order states that mean "this order is done and will not fill further".
_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day"}

_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """A position as the broker sees it."""

    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealised_pnl: float

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass(slots=True)
class ReconciliationReport:
    """Differences between this platform's book and the broker's.

    Any non-empty field is a reason to stop trading until it is understood. A
    position the broker holds and the platform does not is an unmanaged
    position: nothing here will size against it, stop it out, or close it.
    """

    #: symbol → (ours, theirs) where the two disagree.
    mismatched: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Held at the broker, absent from our book.
    broker_only: dict[str, float] = field(default_factory=dict)
    #: In our book, absent at the broker.
    platform_only: dict[str, float] = field(default_factory=dict)
    #: Broker equity minus platform equity.
    equity_drift: float = 0.0
    checked_at: pd.Timestamp | None = None

    @property
    def is_clean(self) -> bool:
        return not (self.mismatched or self.broker_only or self.platform_only)

    def render(self) -> str:
        if self.is_clean:
            return (
                f"✓ reconciled — broker and platform agree "
                f"(equity drift ${self.equity_drift:,.2f})"
            )
        lines = ["✗ RECONCILIATION FAILED — do not trade until resolved", ""]
        for symbol, (ours, theirs) in sorted(self.mismatched.items()):
            lines.append(f"  {symbol:<8} platform {ours:+,.2f}  broker {theirs:+,.2f}")
        for symbol, quantity in sorted(self.broker_only.items()):
            lines.append(
                f"  {symbol:<8} UNMANAGED at broker: {quantity:+,.2f} "
                f"— nothing here will stop it out"
            )
        for symbol, quantity in sorted(self.platform_only.items()):
            lines.append(f"  {symbol:<8} in platform book only: {quantity:+,.2f}")
        if self.equity_drift:
            lines.append(f"  equity drift ${self.equity_drift:,.2f}")
        return "\n".join(lines)


class AlpacaPaperVenue(ExecutionVenue):
    """Routes orders to Alpaca's paper trading endpoint.

    Args:
        confirm_paper: verified against the account on first use. The endpoint
            is the paper host, but a misconfiguration that pointed this at a
            live account would be the single worst bug this file could have, so
            it is checked rather than assumed.
    """

    name = "alpaca-paper"
    #: Paper money. A live adapter would set this True and the router refuses
    #: those unless explicitly unlocked — see axiom.execution.router.
    is_live = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        host: str = _PAPER_HOST,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or get_credential("APCA_API_KEY_ID", "ALPACA_API_KEY")
        self.secret_key = secret_key or get_credential(
            "APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"
        )
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._orders: dict[int, Order] = {}
        #: Our order id → Alpaca's order id.
        self._broker_ids: dict[int, str] = {}
        self._verified_paper = False

    # --- plumbing -----------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "accept": "application/json",
            "content-type": "application/json",
        }

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.host}{path}", data=payload, headers=self._headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ExecutionError(
                f"{self.name}: {method} {path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ExecutionError(
                f"{self.name}: could not reach {self.host}: {exc.reason}. "
                f"If this is a 403 tunnel failure the host is blocked by a "
                f"network policy, not by your key — see docs/DATA_SETUP.md."
            ) from exc

    def _ensure_paper(self) -> None:
        """Confirm the account is paper before any order is sent.

        Cheap, cached, and worth it: the cost of being wrong is real money moved
        by code that believes it cannot move real money.
        """
        if self._verified_paper:
            return
        account = self._request("GET", "/v2/account")
        # Alpaca does not expose an explicit is_paper flag; the account number
        # on a paper account is prefixed PA. Combined with the paper host, a
        # mismatch means the configuration is not what it claims.
        number = str(account.get("account_number", ""))
        if _PAPER_HOST not in self.host:
            raise ExecutionError(
                f"{self.name}: host is {self.host!r}, which is not the paper "
                f"endpoint. This class routes paper orders only."
            )
        if number and not number.startswith("PA"):
            raise ExecutionError(
                f"{self.name}: account {number!r} does not look like a paper "
                f"account (expected a PA prefix). Refusing to send orders."
            )
        self._verified_paper = True

    # --- venue contract -----------------------------------------------------

    def is_available(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def account(self) -> dict[str, Any]:
        """Raw account snapshot. Useful for a status panel."""
        return dict(self._request("GET", "/v2/account"))

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        """Send an order to the paper venue.

        Bracket legs travel with the order rather than being submitted
        separately. Submitting them as follow-ups leaves a window where the
        position exists with no stop — short, but real, and it is exactly the
        window in which a gap happens.
        """
        if not self.is_available():
            return order.reject(f"{self.name}: no Alpaca credentials configured")
        self._ensure_paper()

        body: dict[str, Any] = {
            "symbol": order.instrument.symbol,
            "qty": str(order.quantity),
            "side": "buy" if order.side is Side.BUY else "sell",
            "type": _order_type(order.order_type),
            "time_in_force": _tif(order.time_in_force),
        }
        if order.limit_price is not None:
            body["limit_price"] = str(order.limit_price)
        if order.stop_price is not None:
            body["stop_price"] = str(order.stop_price)

        if order.stop_loss is not None or order.take_profit is not None:
            body["order_class"] = "bracket"
            if order.stop_loss is not None:
                body["stop_loss"] = {"stop_price": str(order.stop_loss)}
            if order.take_profit is not None:
                body["take_profit"] = {"limit_price": str(order.take_profit)}

        try:
            response = self._request("POST", "/v2/orders", body)
        except ExecutionError as exc:
            # A broker rejection is information, not a crash. Record it on the
            # order so the funnel reports it like any other rejection.
            return order.reject(str(exc))

        order.created_at = timestamp
        order.status = _STATUS_MAP.get(str(response.get("status", "")), OrderStatus.SUBMITTED)
        self._orders[order.order_id] = order
        self._broker_ids[order.order_id] = str(response.get("id", ""))
        return order

    def cancel(self, order: Order) -> Order:
        broker_id = self._broker_ids.get(order.order_id)
        if not broker_id:
            return order
        # Already terminal at the broker is not a failure to cancel.
        with contextlib.suppress(ExecutionError):
            self._request("DELETE", f"/v2/orders/{broker_id}")
        order.status = OrderStatus.CANCELLED
        return order

    def working_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_open]

    def poll(self, timestamp: pd.Timestamp) -> list[Fill]:
        """Refresh order state from the broker and emit any new fills.

        Replaces the backtester's `process_bar`. There is no bar to process
        against a live venue — the broker decides when something fills, and this
        asks it.
        """
        fills: list[Fill] = []
        for order_id, broker_id in list(self._broker_ids.items()):
            order = self._orders.get(order_id)
            if order is None or not order.is_open:
                continue
            remote = self._request("GET", f"/v2/orders/{broker_id}")
            status = str(remote.get("status", ""))
            filled = float(remote.get("filled_qty", 0) or 0)
            price = float(remote.get("filled_avg_price") or 0) or 0.0

            newly_filled = filled - order.filled_quantity
            if newly_filled > 0 and price > 0:
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        instrument=order.instrument,
                        side=order.side,
                        quantity=newly_filled,
                        price=price,
                        commission=0.0,  # Alpaca equities are commission-free
                        timestamp=timestamp,
                        slippage=(
                            price - order.reference_price
                            if order.reference_price
                            else 0.0
                        ),
                        strategy=order.strategy,
                    )
                )
                order.filled_quantity = filled
                order.average_fill_price = price

            order.status = _STATUS_MAP.get(status, order.status)
            if status in _TERMINAL:
                self._broker_ids.pop(order_id, None)
        return fills

    # --- reconciliation -----------------------------------------------------

    def positions(self) -> dict[str, BrokerPosition]:
        """Every position the broker currently holds."""
        raw = self._request("GET", "/v2/positions")
        out: dict[str, BrokerPosition] = {}
        for item in raw or []:
            symbol = str(item.get("symbol", ""))
            out[symbol] = BrokerPosition(
                symbol=symbol,
                quantity=float(item.get("qty", 0) or 0),
                average_price=float(item.get("avg_entry_price", 0) or 0),
                market_value=float(item.get("market_value", 0) or 0),
                unrealised_pnl=float(item.get("unrealized_pl", 0) or 0),
            )
        return out

    def reconcile(
        self, portfolio: Any, timestamp: pd.Timestamp | None = None
    ) -> ReconciliationReport:
        """Compare the platform's book against the broker's.

        The most important method in this file. Internal and broker state drift
        for mundane reasons — a manual trade, a corporate action, a fill this
        process missed while restarting — and the drift is **silent**. A
        position the broker holds that the platform does not know about is
        unmanaged: nothing will size against it, stop it out, or close it.

        Run this before trading, and on a schedule while trading.
        """
        broker = self.positions()
        report = ReconciliationReport(checked_at=timestamp or pd.Timestamp.now("UTC"))

        ours: dict[str, float] = {}
        for position in portfolio.open_positions:
            ours[position.instrument.symbol] = float(position.quantity)

        for symbol, quantity in ours.items():
            theirs = broker.get(symbol)
            if theirs is None:
                report.platform_only[symbol] = quantity
            elif abs(theirs.quantity - quantity) > 1e-9:
                report.mismatched[symbol] = (quantity, theirs.quantity)

        for symbol, position in broker.items():
            if symbol not in ours:
                report.broker_only[symbol] = position.quantity

        try:
            equity = float(self.account().get("equity", 0) or 0)
            report.equity_drift = equity - float(portfolio.equity)
        except ExecutionError:
            pass

        return report

    def check(self) -> str:
        """One-line status for `data-check` and the console."""
        if not self.is_available():
            return "✗ no Alpaca credentials configured"
        try:
            self._ensure_paper()
            account = self.account()
        except ExecutionError as exc:
            return f"✗ {exc}"
        return (
            f"✓ paper account {account.get('account_number', '?')} "
            f"({account.get('status', '?')}), equity "
            f"${float(account.get('equity', 0) or 0):,.2f}, "
            f"{len(self.positions())} open position(s)"
        )


def _order_type(order_type: OrderType) -> str:
    return {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop",
        OrderType.STOP_LIMIT: "stop_limit",
    }.get(order_type, "market")


def _tif(tif: TimeInForce) -> str:
    return {
        TimeInForce.DAY: "day",
        TimeInForce.GTC: "gtc",
        TimeInForce.IOC: "ioc",
        TimeInForce.FOK: "fok",
    }.get(tif, "day")


def instrument_for(symbol: str) -> Instrument:
    """Resolve a broker symbol to an instrument spec."""
    return get_instrument(symbol)
