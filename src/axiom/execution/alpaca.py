"""Order routing to Alpaca, paper-first.

This is the first venue in AXIOM that can reach an account holding money, and
the module is written around that fact rather than around convenience.

Three things make it safe by construction:

**Paper is the default and live is opt-in.** :class:`AlpacaVenue` points at
``paper-api.alpaca.markets`` unless someone passes ``paper=False``, and doing so
sets :attr:`~axiom.execution.base.ExecutionVenue.is_live`, which
:class:`~axiom.execution.router.OrderRouter` refuses to construct against
outside live mode with an explicit confirmation. Reaching real money therefore
requires three independent deliberate acts, not one flag.

**Brackets are submitted with the entry, in one request.** Alpaca's
``order_class=bracket`` attaches the stop and target at the venue atomically, so
the protection exists from the instant the entry fills — including during a
restart, a network partition, or a crash of this process. A stop this process
has to watch is a stop that is absent exactly when it is needed, so
:attr:`supports_brackets` is True only because the venue really does hold them.

**Every write carries a client order id.** Alpaca rejects a duplicate
``client_order_id`` outright, which turns the dangerous ambiguity — a submit
that timed out but may have succeeded — into a safe retry. The desk's
idempotency key flows straight through to it, so the same decision cannot
become two positions.

Credentials are the same pair the data adapter uses::

    APCA_API_KEY_ID
    APCA_API_SECRET_KEY

Not implemented here, deliberately: a websocket fill stream. Polling
:meth:`AlpacaVenue.fills_since` is slower but has no reconnect-gap failure mode,
and correctness before latency is the right order for a desk that is not yet
proven to have an edge.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd

from axiom.core.credentials import get_credential
from axiom.core.types import AssetClass, Instrument, OrderStatus, OrderType, Side, TimeInForce
from axiom.execution.base import ExecutionError, ExecutionVenue, Fill, Order

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

#: Our order types mapped onto Alpaca's spelling.
_ORDER_TYPES: dict[OrderType, str] = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP: "stop",
    OrderType.STOP_LIMIT: "stop_limit",
}

_TIME_IN_FORCE: dict[TimeInForce, str] = {
    TimeInForce.DAY: "day",
    TimeInForce.GTC: "gtc",
    TimeInForce.IOC: "ioc",
    TimeInForce.FOK: "fok",
}

#: Alpaca's order states mapped onto ours. Anything unlisted is treated as open,
#: because assuming a terminal state we do not recognise is how a live position
#: stops being managed.
_STATUS: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "held": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "done_for_day": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "stopped": OrderStatus.REJECTED,
}

#: Alpaca's hard cap on activities per page. Discovered the way these things
#: usually are: a 422 from the live API rejecting page_size=500.
MAX_FILL_PAGE = 100

#: Quote currencies recognised when a crypto pair arrives without a separator.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD", "EUR", "GBP", "BTC")


@dataclass(frozen=True, slots=True)
class VenueFill:
    """One execution as the venue reports it, before local attribution.

    Kept distinct from :class:`~axiom.execution.base.Fill` because the two ids
    live in different namespaces: the venue's are opaque strings it assigns,
    ours are integers this process assigns. Conflating them is how a fill gets
    attributed to whichever local order happens to share a number.
    """

    venue_fill_id: str
    venue_order_id: str
    fill: Fill


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """What Alpaca says the account is, at a moment."""

    equity: float
    cash: float
    buying_power: float
    #: True when Alpaca has restricted the account from opening new positions.
    trading_blocked: bool
    #: True when the account is flagged as a pattern day trader.
    pattern_day_trader: bool
    currency: str = "USD"

    @property
    def is_tradable(self) -> bool:
        return not self.trading_blocked


class AlpacaVenue(ExecutionVenue):
    """Submit, cancel and read back orders at Alpaca.

    Args:
        paper: point at the paper endpoint. Defaults True. Setting it False
            marks the venue live, which the router will refuse to touch outside
            confirmed live mode.
        timeout: per-request timeout in seconds.
    """

    name = "alpaca"
    #: Alpaca holds bracket legs itself, so a stop survives this process dying.
    supports_brackets = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or get_credential("APCA_API_KEY_ID", "ALPACA_API_KEY")
        self.secret_key = secret_key or get_credential(
            "APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"
        )
        self.paper = paper
        # Set on the instance rather than the class: two venues in one process
        # (paper and live) must not share a liveness flag.
        self.is_live = not paper
        self.base_url = PAPER_URL if paper else LIVE_URL
        self.timeout = timeout
        #: Maps our order ids to Alpaca's, so cancel and status lookups work
        #: without a round trip through the store.
        self._venue_ids: dict[int, str] = {}

    def is_available(self) -> bool:
        return bool(self.api_key and self.secret_key)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "accept": "application/json",
            "content-type": "application/json",
        }

    # --- symbol translation ------------------------------------------------

    @staticmethod
    def venue_symbol(instrument: Instrument) -> str:
        """Translate an instrument symbol into Alpaca's trading spelling.

        The same ``BASE/QUOTE`` requirement as the data endpoint: ``BTC-USD``
        is rejected with a 400, and because a rejected order looks like a
        venue outage rather than a naming bug, getting this wrong is expensive
        and silent.
        """
        symbol = instrument.symbol.upper()
        if instrument.asset_class is not AssetClass.CRYPTO:
            return symbol
        if "/" in symbol:
            return symbol
        base, separator, quote = symbol.partition("-")
        if separator and base and quote:
            return f"{base}/{quote}"
        for candidate in _CRYPTO_QUOTES:
            if symbol.endswith(candidate) and len(symbol) > len(candidate):
                return f"{symbol[: -len(candidate)]}/{candidate}"
        return f"{symbol}/USD"

    @staticmethod
    def local_symbol(venue_symbol: str) -> str:
        """Invert :meth:`venue_symbol` for reading positions back.

        Alpaca reports crypto positions as ``BTC/USD`` while the platform keys
        everything by ``BTC-USD``. Without this the reconciler compares two
        spellings of the same holding, finds neither in the other's book, and
        reports both a phantom and an unknown position.
        """
        return venue_symbol.replace("/", "-").upper()

    # --- HTTP --------------------------------------------------------------

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        if not self.is_available():
            raise ExecutionError(
                "Alpaca credentials not found. Set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY (see docs/DATA_SETUP.md)."
            )

        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            url, data=data, headers=self._headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ExecutionError(
                f"{self.name}: HTTP {exc.code} on {method} {path}"
                f"{self._hint(exc.code)}. {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            # Deliberately not retried here. A retry on an ambiguous network
            # failure is safe only because of the client order id, and that
            # decision belongs to the caller who owns the idempotency key.
            raise ExecutionError(
                f"{self.name}: network error on {method} {path}: {exc.reason}"
            ) from exc

    @staticmethod
    def _hint(code: int) -> str:
        return {
            401: " — check APCA_API_KEY_ID / APCA_API_SECRET_KEY",
            403: " — the account may be blocked from trading, or the key is "
                 "for the other environment (paper keys do not work live)",
            422: " — Alpaca rejected the request parameters. On an order "
                 "POST this is usually a duplicate client_order_id, meaning "
                 "the original order already exists; on a GET it is usually a "
                 "page size or date range outside the endpoint's limits",
            429: " — rate limited",
        }.get(code, "")

    # --- account and positions --------------------------------------------

    def account(self) -> AccountSnapshot:
        """Read the account. Cheap, and the right first call before trading."""
        payload = self._request("GET", "/v2/account")
        return AccountSnapshot(
            equity=float(payload.get("equity", 0.0)),
            cash=float(payload.get("cash", 0.0)),
            buying_power=float(payload.get("buying_power", 0.0)),
            trading_blocked=bool(payload.get("trading_blocked", False)),
            pattern_day_trader=bool(payload.get("pattern_day_trader", False)),
            currency=str(payload.get("currency", "USD")),
        )

    def positions(self) -> dict[str, float]:
        """Signed quantity per symbol, keyed the way the platform keys things.

        This is what :func:`~axiom.store.reconcile.reconcile_positions` compares
        against, so the symbols must come back in local spelling.
        """
        payload = self._request("GET", "/v2/positions")
        out: dict[str, float] = {}
        for row in payload or []:
            quantity = float(row.get("qty", 0.0))
            # Alpaca reports shorts with a negative qty and side="short"; trust
            # the sign, but fall back to the side if the sign is unavailable.
            if quantity > 0 and str(row.get("side", "long")).lower() == "short":
                quantity = -quantity
            if quantity != 0.0:
                out[self.local_symbol(str(row.get("symbol", "")))] = quantity
        return out

    # --- orders ------------------------------------------------------------

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        """Send an order, with its bracket legs attached if it has any."""
        body = self._order_body(order)
        payload = self._request("POST", "/v2/orders", body)

        venue_id = str(payload.get("id", ""))
        if venue_id:
            self._venue_ids[order.order_id] = venue_id
        order.status = _STATUS.get(str(payload.get("status", "")), OrderStatus.SUBMITTED)
        order.created_at = order.created_at or timestamp

        filled = float(payload.get("filled_qty") or 0.0)
        if filled:
            order.filled_quantity = filled
            order.average_fill_price = float(payload.get("filled_avg_price") or 0.0)
        return order

    def _order_body(self, order: Order) -> dict[str, Any]:
        """Build the request payload, including bracket legs.

        Alpaca requires a bracket parent to be ``time_in_force=gtc`` (or ``day``
        for equities) and refuses ``ioc``/``fok``, because a bracket has to
        outlive the entry to be worth anything. Rather than silently rewriting
        the caller's intent, a bracketed order with an incompatible TIF is a
        rejection with the reason spelled out.
        """
        order_type = _ORDER_TYPES.get(order.order_type)
        if order_type is None:
            raise ExecutionError(f"{self.name}: unsupported order type {order.order_type}")

        has_bracket = order.stop_loss is not None or order.take_profit is not None
        time_in_force = _TIME_IN_FORCE[order.time_in_force]
        if has_bracket and time_in_force in {"ioc", "fok"}:
            raise ExecutionError(
                f"{self.name}: a bracketed order cannot use "
                f"time_in_force={time_in_force!r} — the protective legs must "
                "outlive the entry. Use DAY or GTC."
            )

        body: dict[str, Any] = {
            "symbol": self.venue_symbol(order.instrument),
            "qty": _format_quantity(order.quantity),
            "side": "buy" if order.side is Side.BUY else "sell",
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if order.limit_price is not None:
            body["limit_price"] = _format_price(order.limit_price)
        if order.stop_price is not None:
            body["stop_price"] = _format_price(order.stop_price)

        # The desk's idempotency key, passed straight through. Alpaca rejects a
        # duplicate with 422, which is what makes retrying an ambiguous submit
        # safe rather than position-doubling.
        if order.tag:
            body["client_order_id"] = order.tag[:128]

        if has_bracket:
            # Both legs make it a bracket; one leg is what Alpaca calls OTO,
            # and a single venue-held stop is still the property that matters.
            body["order_class"] = (
                "bracket"
                if order.stop_loss is not None and order.take_profit is not None
                else "oto"
            )
            if order.stop_loss is not None:
                body["stop_loss"] = {"stop_price": _format_price(order.stop_loss)}
            if order.take_profit is not None:
                body["take_profit"] = {"limit_price": _format_price(order.take_profit)}
        return body

    def cancel(self, order: Order) -> Order:
        """Cancel a working order.

        A 404 is treated as success: the order is already gone, which is the
        state the caller wanted. Raising there would make a retry of a
        successful cancel look like a failure.
        """
        venue_id = self._venue_ids.get(order.order_id)
        if venue_id is None:
            raise ExecutionError(
                f"{self.name}: no venue id known for order {order.order_id}; "
                "it was never submitted through this venue instance"
            )
        try:
            self._request("DELETE", f"/v2/orders/{venue_id}")
        except ExecutionError as exc:
            if "HTTP 404" not in str(exc):
                raise
        order.status = OrderStatus.CANCELLED
        return order

    def working_orders(self) -> list[Order]:
        """Orders still live at the venue.

        Read from Alpaca rather than from local memory, because the venue is
        the source of truth and a restarted process has no local memory at all.
        """
        payload = self._request("GET", "/v2/orders?status=open&limit=500")
        return [self._to_order(row) for row in payload or []]

    def _to_order(self, row: dict[str, Any]) -> Order:
        """Rebuild a local Order from Alpaca's representation."""
        from axiom.core.types import get_instrument

        instrument = get_instrument(self.local_symbol(str(row.get("symbol", ""))))
        order = Order(
            instrument=instrument,
            side=Side.BUY if str(row.get("side")) == "buy" else Side.SELL,
            quantity=float(row.get("qty") or 0.0) or 1.0,
            order_type=OrderType(str(row.get("type", "market")).replace("-", "_")),
            limit_price=_optional_float(row.get("limit_price")),
            stop_price=_optional_float(row.get("stop_price")),
            tag=str(row.get("client_order_id", "")),
        )
        order.status = _STATUS.get(str(row.get("status", "")), OrderStatus.SUBMITTED)
        order.filled_quantity = float(row.get("filled_qty") or 0.0)
        order.average_fill_price = float(row.get("filled_avg_price") or 0.0)
        venue_id = str(row.get("id", ""))
        if venue_id:
            self._venue_ids[order.order_id] = venue_id
        return order

    def fills_since(self, since: pd.Timestamp, limit: int = MAX_FILL_PAGE) -> list[VenueFill]:
        """Every execution after ``since``, oldest first, following pagination.

        Each carries the venue's fill id, so
        :meth:`~axiom.store.db.Store.record_fill` can deduplicate, and the
        venue's *order* id, which is the only way to attribute an execution
        back to the decision that caused it.

        Polling rather than streaming: a poll that overlaps a previous window
        books nothing twice, whereas a websocket that reconnects past a message
        loses a fill silently, and a lost fill is an unmanaged position.

        **Pages until exhausted, deliberately.** Alpaca caps a page at
        :data:`MAX_FILL_PAGE` activities and returns the first page only. A
        single request would therefore truncate a busy session silently, and
        the poller — having no way to know it was truncated — would advance its
        watermark past the fills it never saw. That loses them permanently,
        which is the exact failure this whole module exists to prevent.

        Args:
            since: exclusive lower bound on transaction time.
            limit: page size, clamped to Alpaca's maximum. Total results are
                not limited; every page is followed.
        """
        after = pd.Timestamp(since)
        if after.tzinfo is None:
            after = after.tz_localize("UTC")

        from axiom.core.types import get_instrument

        rows: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            query: dict[str, str] = {
                "activity_types": "FILL",
                "after": after.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
                "page_size": str(min(max(limit, 1), MAX_FILL_PAGE)),
                "direction": "asc",
            }
            if page_token:
                query["page_token"] = page_token
            page = self._request(
                "GET", f"/v2/account/activities?{urllib.parse.urlencode(query)}"
            )
            batch = list(page or [])
            rows.extend(batch)
            # Alpaca pages activities by the id of the last row returned, and
            # signals the end by returning fewer rows than asked for.
            if len(batch) < min(max(limit, 1), MAX_FILL_PAGE):
                break
            page_token = str(batch[-1].get("id", ""))
            if not page_token:
                break

        out: list[VenueFill] = []
        for row in rows:
            quantity = abs(float(row.get("qty") or 0.0))
            if quantity <= 0:
                continue
            symbol = self.local_symbol(str(row.get("symbol", "")))
            out.append(
                VenueFill(
                    venue_fill_id=str(row.get("id", "")),
                    venue_order_id=str(row.get("order_id", "")),
                    fill=Fill(
                        # Alpaca's order id is a UUID and our Order.order_id is
                        # an int, so the linkage is resolved by the caller
                        # against its own record rather than invented here.
                        order_id=0,
                        instrument=get_instrument(symbol),
                        side=Side.BUY if str(row.get("side")) == "buy" else Side.SELL,
                        quantity=quantity,
                        price=float(row.get("price") or 0.0),
                        commission=0.0,
                        timestamp=pd.Timestamp(str(row.get("transaction_time"))),
                    ),
                )
            )
        return out

    def check_credentials(self) -> str:
        """Verify the keys reach a tradable account, and say which one."""
        if not self.is_available():
            return (
                "✗ No Alpaca credentials found.\n"
                "  Create a .env file at the project root with\n"
                "    APCA_API_KEY_ID=your_key_id\n"
                "    APCA_API_SECRET_KEY=your_secret_key"
            )
        try:
            snapshot = self.account()
        except ExecutionError as exc:
            return f"✗ {exc}"

        environment = "PAPER" if self.paper else "[LIVE — REAL MONEY]"
        status = "tradable" if snapshot.is_tradable else "TRADING BLOCKED"
        return (
            f"✓ Alpaca {environment} account reachable — {status}\n"
            f"  equity {snapshot.equity:,.2f} {snapshot.currency} | "
            f"cash {snapshot.cash:,.2f} | buying power {snapshot.buying_power:,.2f}"
            + ("\n  ! flagged as a pattern day trader" if snapshot.pattern_day_trader else "")
        )


def _format_quantity(quantity: float) -> str:
    """Alpaca wants a string, and fractional quantities are legal for crypto.

    Formatted with ``repr``-like precision rather than a fixed number of
    decimals so a small crypto size is not rounded to zero.
    """
    return f"{quantity:.9f}".rstrip("0").rstrip(".") or "0"


def _format_price(price: float) -> str:
    return f"{price:.4f}".rstrip("0").rstrip(".")


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
