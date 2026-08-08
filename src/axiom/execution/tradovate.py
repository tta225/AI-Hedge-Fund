"""Tradovate futures venue — demo and live.

Why Tradovate and not NinjaTrader for the primary path
------------------------------------------------------
Same parent company, very different integration surface. Tradovate is a real
REST + WebSocket API reachable from anywhere: no desktop application, no
Windows requirement, no DLL. It runs from a Mac, a Linux container, or a server.
`NinjaTraderVenue` exists alongside it for people already running the desktop,
but this is the one that works unattended.

The `is_live` flag is computed, not declared
--------------------------------------------
Every other venue in this platform hardcodes `is_live`. This one cannot: the
same class serves both `demo.tradovateapi.com` and `live.tradovateapi.com`, and
the difference is one string. So `is_live` is derived from the host, and
`ExecutionVenue.is_live` — which the router reads to decide whether to refuse —
becomes a property rather than a class attribute.

That is the most safety-critical line in the file. Get it wrong and the router
constructs happily against a live account it believes is a demo.

Live also requires an explicit opt-in argument. Pointing the host at production
is not enough; you must also pass `allow_live=True`, so a copied config string
cannot arm real trading on its own.

CME requires automated orders to say so
---------------------------------------
Tradovate requires `isAutomated: true` on any order not placed by a human
touching a button. This is a CME Group regulatory requirement, not a Tradovate
preference. It is set unconditionally here, because every order this platform
produces is by definition automated, and a venue that let a caller turn it off
would be offering a compliance violation as a configuration option.

Session tokens expire
---------------------
Access tokens are short-lived. The venue tracks `expirationTime` and
re-authenticates before it lapses rather than after a 401 — a token that expires
between placing an entry and placing its protective stop is a position with no
stop, which is the specific failure this platform spends the most effort
preventing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from axiom.core.credentials import get_credential
from axiom.core.types import OrderStatus, OrderType, Side, TimeInForce
from axiom.execution.base import ExecutionError, ExecutionVenue, Fill, Order

__all__ = ["TradovatePosition", "TradovateVenue"]

DEMO_HOST = "https://demo.tradovateapi.com/v1"
LIVE_HOST = "https://live.tradovateapi.com/v1"

#: Re-authenticate this long before the token actually expires. A token that
#: lapses between an entry and its stop leaves an unprotected position.
_TOKEN_MARGIN = timedelta(minutes=5)

_STATUS_MAP: dict[str, OrderStatus] = {
    "Pending": OrderStatus.PENDING,
    "PendingNew": OrderStatus.PENDING,
    "Working": OrderStatus.SUBMITTED,
    "Accepted": OrderStatus.SUBMITTED,
    "Filled": OrderStatus.FILLED,
    "Canceled": OrderStatus.CANCELLED,
    "Cancelled": OrderStatus.CANCELLED,
    "Expired": OrderStatus.EXPIRED,
    "Rejected": OrderStatus.REJECTED,
    "Suspended": OrderStatus.SUBMITTED,
}

_TERMINAL = {"Filled", "Canceled", "Cancelled", "Expired", "Rejected"}

_ORDER_TYPES: dict[OrderType, str] = {
    OrderType.MARKET: "Market",
    OrderType.LIMIT: "Limit",
    OrderType.STOP: "Stop",
    OrderType.STOP_LIMIT: "StopLimit",
}

_TIF: dict[TimeInForce, str] = {
    TimeInForce.DAY: "Day",
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
}


@dataclass(frozen=True, slots=True)
class TradovatePosition:
    """A futures position as Tradovate reports it."""

    symbol: str
    quantity: float
    average_price: float
    contract_id: int = 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass(slots=True)
class _Session:
    """An authenticated session, with the expiry that governs renewal."""

    token: str = ""
    expires_at: datetime | None = None
    user_id: int = 0
    account_id: int = 0
    account_spec: str = ""

    @property
    def is_valid(self) -> bool:
        if not self.token or self.expires_at is None:
            return False
        return datetime.now(UTC) + _TOKEN_MARGIN < self.expires_at


class TradovateVenue(ExecutionVenue):
    """Routes futures orders to Tradovate.

    Args:
        allow_live: required to point at the live host. The host string alone
            does not arm real trading — a config copied from somewhere else
            should not be able to do that by itself.
        account_spec: account name. Discovered from `/account/list` when unset.
    """

    name = "tradovate"

    def __init__(
        self,
        *,
        username: str | None = None,
        password: str | None = None,
        app_id: str = "AXIOM",
        app_version: str = "1.0",
        cid: str | None = None,
        secret: str | None = None,
        host: str = DEMO_HOST,
        account_spec: str = "",
        allow_live: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.username = username or get_credential("TRADOVATE_USERNAME")
        self.password = password or get_credential("TRADOVATE_PASSWORD")
        self.app_id = app_id
        self.app_version = app_version
        self.cid = cid or get_credential("TRADOVATE_CID")
        self.secret = secret or get_credential("TRADOVATE_SECRET")
        self.host = host.rstrip("/")
        self.account_spec = account_spec
        self.timeout = timeout
        self.allow_live = allow_live

        if self.host.startswith(LIVE_HOST) and not allow_live:
            raise PermissionError(
                f"{self.name}: host is the LIVE endpoint but allow_live=False. "
                f"Refusing to construct a venue that could reach real money. "
                f"Pass allow_live=True deliberately, and note the router still "
                f"requires live mode plus a confirmation phrase on top of this."
            )

        self._session = _Session()
        self._orders: dict[int, Order] = {}
        #: Our order id → Tradovate's order id.
        self._broker_ids: dict[int, int] = {}

    @property
    def is_live(self) -> bool:  # type: ignore[override]
        """Derived from the host, never declared.

        One class serves demo and live, so a hardcoded flag would be a lie half
        the time — and the half it lies about is the one that moves money. The
        router reads this to decide whether to refuse construction.
        """
        return self.host.startswith(LIVE_HOST)

    def is_available(self) -> bool:
        return bool(self.username and self.password and self.cid and self.secret)

    # --- plumbing -----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        authenticate: bool = True,
    ) -> Any:
        headers = {"accept": "application/json", "content-type": "application/json"}
        if authenticate:
            headers["Authorization"] = f"Bearer {self._token()}"

        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.host}{path}", data=payload, headers=headers, method=method
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
                f"A 403 tunnel failure means the host is blocked by a network "
                f"policy, not by your credentials — allow "
                f"demo.tradovateapi.com and live.tradovateapi.com."
            ) from exc

    def _token(self) -> str:
        """Return a valid access token, renewing before expiry rather than after.

        Renewing on a 401 means discovering the problem by failing a request —
        and the request that fails might be the one placing a protective stop.
        """
        if self._session.is_valid:
            return self._session.token
        self.authenticate()
        return self._session.token

    def authenticate(self) -> _Session:
        """Exchange credentials for an access token."""
        if not self.is_available():
            raise ExecutionError(
                f"{self.name}: missing credentials. Set TRADOVATE_USERNAME, "
                f"TRADOVATE_PASSWORD, TRADOVATE_CID and TRADOVATE_SECRET."
            )

        response = self._request(
            "POST",
            "/auth/accesstokenrequest",
            {
                "name": self.username,
                "password": self.password,
                "appId": self.app_id,
                "appVersion": self.app_version,
                "cid": self.cid,
                "sec": self.secret,
            },
            authenticate=False,
        )

        token = str(response.get("accessToken", ""))
        if not token:
            raise ExecutionError(
                f"{self.name}: authentication returned no access token: "
                f"{str(response)[:300]}"
            )

        expires = response.get("expirationTime")
        expires_at = (
            pd.Timestamp(expires).tz_convert(UTC).to_pydatetime()
            if expires
            else datetime.now(UTC) + timedelta(hours=1)
        )
        self._session = _Session(
            token=token,
            expires_at=expires_at,
            user_id=int(response.get("userId", 0) or 0),
        )
        self._resolve_account()
        return self._session

    def _resolve_account(self) -> None:
        """Find the account to trade. Refuses to guess between several."""
        accounts = self._request("GET", "/account/list")
        if not accounts:
            raise ExecutionError(f"{self.name}: no accounts on this login")

        if self.account_spec:
            match = next(
                (a for a in accounts if str(a.get("name")) == self.account_spec), None
            )
            if match is None:
                names = [str(a.get("name")) for a in accounts]
                raise ExecutionError(
                    f"{self.name}: account {self.account_spec!r} not found. "
                    f"Available: {names}"
                )
        elif len(accounts) == 1:
            match = accounts[0]
        else:
            names = [str(a.get("name")) for a in accounts]
            raise ExecutionError(
                f"{self.name}: {len(accounts)} accounts available and no "
                f"account_spec given: {names}. Refusing to pick one — trading "
                f"the wrong account is not a recoverable mistake."
            )

        self._session.account_id = int(match.get("id", 0) or 0)
        self._session.account_spec = str(match.get("name", ""))

    # --- venue contract -----------------------------------------------------

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        """Place an order. Always flagged as automated.

        Bracket legs are sent as a single OSO (One-Sends-Other) request rather
        than as follow-ups, so the stop exists the moment the entry does.
        """
        if not self.is_available():
            return order.reject(f"{self.name}: no Tradovate credentials configured")

        try:
            session = self._session if self._session.is_valid else self.authenticate()
            body: dict[str, Any] = {
                "accountSpec": session.account_spec,
                "accountId": session.account_id,
                "action": "Buy" if order.side is Side.BUY else "Sell",
                "symbol": order.instrument.symbol,
                "orderQty": int(order.quantity),
                "orderType": _ORDER_TYPES.get(order.order_type, "Market"),
                "timeInForce": _TIF.get(order.time_in_force, "Day"),
                # CME Group requires automated orders to be flagged. Every order
                # this platform produces is automated, so this is unconditional
                # rather than configurable.
                "isAutomated": True,
            }
            if order.limit_price is not None:
                body["price"] = order.limit_price
            if order.stop_price is not None:
                body["stopPrice"] = order.stop_price

            brackets = self._brackets(order)
            if brackets:
                response = self._request(
                    "POST",
                    "/order/placeOSO",
                    {**body, **brackets},
                )
            else:
                response = self._request("POST", "/order/placeOrder", body)
        except (ExecutionError, PermissionError) as exc:
            return order.reject(str(exc))

        failure = response.get("failureReason") or response.get("failureText")
        if failure:
            return order.reject(f"{self.name}: {failure}")

        broker_id = response.get("orderId") or response.get("id")
        if not broker_id:
            return order.reject(
                f"{self.name}: order accepted but no id returned: {str(response)[:200]}"
            )

        order.created_at = timestamp
        order.status = OrderStatus.SUBMITTED
        self._orders[order.order_id] = order
        self._broker_ids[order.order_id] = int(broker_id)
        return order

    def _brackets(self, order: Order) -> dict[str, Any]:
        """OSO legs for the stop and target, if the order carries them."""
        legs: list[dict[str, Any]] = []
        opposite = "Sell" if order.side is Side.BUY else "Buy"
        if order.stop_loss is not None:
            legs.append({
                "action": opposite,
                "orderType": "Stop",
                "stopPrice": order.stop_loss,
                "orderQty": int(order.quantity),
                "timeInForce": "GTC",
                "isAutomated": True,
            })
        if order.take_profit is not None:
            legs.append({
                "action": opposite,
                "orderType": "Limit",
                "price": order.take_profit,
                "orderQty": int(order.quantity),
                "timeInForce": "GTC",
                "isAutomated": True,
            })
        if not legs:
            return {}
        out: dict[str, Any] = {"bracket1": legs[0]}
        if len(legs) > 1:
            out["bracket2"] = legs[1]
        return out

    def cancel(self, order: Order) -> Order:
        """Cancel at the broker, and only claim it once the broker agrees.

        The tempting shape here is to fire the cancel and set the status to
        CANCELLED regardless, on the reasoning that a rejected cancel means the
        order was already terminal. That reasoning holds for one of the two ways
        a cancel fails and not the other. A 404 does mean "already filled or
        already gone". A timeout, a 500, or a dropped connection means *unknown*
        — and an order that is still working at the broker while this side has
        written it off is precisely a live position nothing is managing. So on
        failure we ask the broker what the order actually is, and adopt that.
        If we cannot find out, the error propagates rather than being papered
        over with an optimistic status.
        """
        broker_id = self._broker_ids.get(order.order_id)
        if broker_id is None:
            return order
        try:
            self._request("POST", "/order/cancelOrder", {"orderId": broker_id})
        except ExecutionError:
            status = self._request("GET", f"/order/item?id={broker_id}").get("ordStatus")
            if status not in _TERMINAL:
                raise
            order.status = _STATUS_MAP.get(str(status), OrderStatus.CANCELLED)
            return order
        order.status = OrderStatus.CANCELLED
        return order

    def working_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_open]

    def poll(self, timestamp: pd.Timestamp) -> list[Fill]:
        """Refresh order state and emit any fills seen since the last call."""
        fills: list[Fill] = []
        for order_id, broker_id in list(self._broker_ids.items()):
            order = self._orders.get(order_id)
            if order is None or not order.is_open:
                continue

            remote = self._request("GET", f"/order/item?id={broker_id}")
            status = str(remote.get("ordStatus", ""))
            filled = float(remote.get("cumQty", 0) or 0)
            price = float(remote.get("avgPx", 0) or 0)

            new_quantity = filled - order.filled_quantity
            if new_quantity > 0 and price > 0:
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        instrument=order.instrument,
                        side=order.side,
                        quantity=new_quantity,
                        price=price,
                        commission=0.0,  # settled separately by Tradovate
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

    # --- account state ------------------------------------------------------

    def account(self) -> dict[str, Any]:
        session = self._session if self._session.is_valid else self.authenticate()
        snapshot = self._request(
            "GET", f"/cashBalance/getcashbalancesnapshot?accountId={session.account_id}"
        )
        return {
            "account_id": session.account_id,
            "account_spec": session.account_spec,
            "is_live": self.is_live,
            **(snapshot if isinstance(snapshot, dict) else {}),
        }

    def positions(self) -> dict[str, TradovatePosition]:
        raw = self._request("GET", "/position/list")
        out: dict[str, TradovatePosition] = {}
        for item in raw or []:
            quantity = float(item.get("netPos", 0) or 0)
            if quantity == 0:
                continue  # a flat row is not a position
            symbol = str(item.get("symbol") or item.get("contractId") or "")
            out[symbol] = TradovatePosition(
                symbol=symbol,
                quantity=quantity,
                average_price=float(item.get("netPrice", 0) or 0),
                contract_id=int(item.get("contractId", 0) or 0),
            )
        return out

    def reconcile(
        self, portfolio: Any, timestamp: pd.Timestamp | None = None
    ) -> Any:
        """Compare the platform's book against Tradovate's.

        Shares `ReconciliationReport` with the Alpaca venue deliberately: an
        operator reading two brokers' reconciliations should not have to learn
        two report formats to spot the same class of problem.
        """
        from axiom.execution.alpaca_paper import ReconciliationReport

        broker = self.positions()
        report = ReconciliationReport(checked_at=timestamp or pd.Timestamp.now("UTC"))

        ours = {p.instrument.symbol: float(p.quantity) for p in portfolio.open_positions}

        for symbol, quantity in ours.items():
            theirs = broker.get(symbol)
            if theirs is None:
                report.platform_only[symbol] = quantity
            elif abs(theirs.quantity - quantity) > 1e-9:
                report.mismatched[symbol] = (quantity, theirs.quantity)

        for symbol, position in broker.items():
            if symbol not in ours:
                report.broker_only[symbol] = position.quantity

        return report

    def check(self) -> str:
        """One-line status for the CLI and console."""
        if not self.is_available():
            return (
                "✗ no Tradovate credentials — set TRADOVATE_USERNAME, "
                "TRADOVATE_PASSWORD, TRADOVATE_CID, TRADOVATE_SECRET"
            )
        try:
            session = self.authenticate()
            positions = self.positions()
        except ExecutionError as exc:
            return f"✗ {exc}"
        environment = "LIVE — REAL MONEY" if self.is_live else "demo"
        return (
            f"✓ {environment}: account {session.account_spec!r} "
            f"(id {session.account_id}), {len(positions)} open position(s)"
        )
