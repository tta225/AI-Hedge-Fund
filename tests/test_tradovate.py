"""Tradovate futures venue. Offline — the HTTP layer is stubbed.

This venue differs from the Alpaca one in the way that matters most: a single
class serves both the demo host and the live host. Alpaca's paper venue can
assert `is_live is False` and be done, because it is structurally incapable of
reaching real money. Here, whether real money is reachable is a *runtime*
property of the host string, so the tests that carry the weight are the ones
that pin how `is_live` is derived and what refuses to be constructed.

Three failure modes are worth more than the rest:

* **Trading the live account while believing it is demo.** `is_live` is what
  the router keys off; if it ever reports False against the live host, every
  guard downstream is disarmed at once.
* **An entry with no stop.** Brackets go with the order, and the token is
  renewed before it expires, both for the same reason — a gap between placing
  an entry and placing its protection is an unprotected position.
* **Believing an order is cancelled when it is still working.** A failed cancel
  that gets recorded as success leaves a live order nothing is managing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from axiom.core.types import OrderStatus, OrderType, Side, get_instrument
from axiom.execution.base import ExecutionError, Order
from axiom.execution.tradovate import (
    DEMO_HOST,
    LIVE_HOST,
    TradovateVenue,
    _Session,
)

NOW = pd.Timestamp("2026-03-02 15:00", tz="UTC")
ES = get_instrument("ES")


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this module off any real Tradovate account.

    Every credential argument falls back to the environment when passed as
    None, which is right for the application and wrong here: on a machine with
    real keys set, a test asserting "no credentials" would instead authenticate
    against a live login. The Alpaca suite learned this the expensive way.
    """
    for name in (
        "TRADOVATE_USERNAME",
        "TRADOVATE_PASSWORD",
        "TRADOVATE_CID",
        "TRADOVATE_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)


class _StubVenue(TradovateVenue):
    """Serves canned API responses instead of reaching the network."""

    def __init__(
        self, responses: dict[tuple[str, str], Any] | None = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("username", "user")
        kwargs.setdefault("password", "pass")
        kwargs.setdefault("cid", "cid")
        kwargs.setdefault("secret", "sec")
        super().__init__(**kwargs)
        self.responses = responses or {}
        self.calls: list[tuple[str, str, Any]] = []

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        authenticate: bool = True,
    ) -> Any:
        self.calls.append((method, path, body))
        for (verb, prefix), value in self.responses.items():
            if verb == method and path.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return value
        return {}

    def body_for(self, path_prefix: str) -> dict[str, Any]:
        return next(b for _, p, b in self.calls if p.startswith(path_prefix) and b)


_AUTH = {
    "accessToken": "tok",
    "userId": 7,
    "expirationTime": "2026-03-02T18:00:00Z",
}


def _authed(extra: dict[tuple[str, str], Any] | None = None, **kwargs: Any) -> _StubVenue:
    responses: dict[tuple[str, str], Any] = {
        ("POST", "/auth/accesstokenrequest"): _AUTH,
        ("GET", "/account/list"): [{"id": 42, "name": "DEMO123"}],
    }
    responses.update(extra or {})
    return _StubVenue(responses, **kwargs)


def _order(**kwargs: Any) -> Order:
    defaults: dict[str, Any] = {
        "instrument": ES,
        "side": Side.BUY,
        "quantity": 2.0,
        "order_type": OrderType.MARKET,
        "strategy": "test",
    }
    return Order(**{**defaults, **kwargs})


class TestLiveGuard:
    def test_the_demo_host_is_not_live(self) -> None:
        assert _authed().is_live is False

    def test_the_live_host_is_live(self) -> None:
        """The router reads this. Reporting False here disarms every guard."""
        assert _authed(host=LIVE_HOST, allow_live=True).is_live is True

    def test_is_live_is_derived_not_declared(self) -> None:
        """One class, two hosts — a fixed class attribute would be a lie half
        the time, and the half it lies about is the one that moves money."""
        assert isinstance(TradovateVenue.__dict__["is_live"], property)

    def test_the_live_host_refuses_to_construct_without_allow_live(self) -> None:
        with pytest.raises(PermissionError, match="allow_live=False"):
            _StubVenue(host=LIVE_HOST)

    def test_allow_live_alone_does_not_arm_the_router(self) -> None:
        """Two independent gates. This one only clears construction."""
        from axiom.core.types import TradingMode
        from axiom.execution.router import OrderRouter
        from axiom.risk.manager import RiskManager

        venue = _authed(host=LIVE_HOST, allow_live=True)
        with pytest.raises(PermissionError, match="live venue"):
            OrderRouter(venue=venue, risk=RiskManager(), mode=TradingMode.PAPER)

    def test_check_names_the_environment_in_words(self) -> None:
        """'demo' vs 'live' as hostnames is a substring away from a bad read."""
        assert "LIVE — REAL MONEY" in _authed(
            host=LIVE_HOST, allow_live=True
        ).check()
        assert "demo" in _authed().check()


class TestAuthentication:
    def test_a_token_is_renewed_before_it_expires_not_after(self) -> None:
        """Renewing on a 401 means learning about it by failing a request, and
        the request that fails may be the one placing a protective stop."""
        session = _Session(
            token="tok", expires_at=datetime.now(UTC) + timedelta(minutes=2)
        )
        assert not session.is_valid  # inside the 5-minute margin

    def test_a_comfortably_fresh_token_is_reused(self) -> None:
        session = _Session(
            token="tok", expires_at=datetime.now(UTC) + timedelta(minutes=30)
        )
        assert session.is_valid

    def test_an_empty_token_is_never_valid(self) -> None:
        assert not _Session(expires_at=datetime.now(UTC) + timedelta(days=1)).is_valid

    def test_authentication_without_a_token_in_the_reply_is_an_error(self) -> None:
        venue = _StubVenue({("POST", "/auth"): {"errorText": "bad credentials"}})
        with pytest.raises(ExecutionError, match="no access token"):
            venue.authenticate()

    def test_missing_credentials_are_named_not_guessed_at(self) -> None:
        venue = TradovateVenue(username=None, password=None, cid=None, secret=None)
        assert not venue.is_available()
        assert "TRADOVATE_USERNAME" in venue.check()


class TestAccountResolution:
    def test_a_single_account_is_adopted(self) -> None:
        assert _authed().authenticate().account_spec == "DEMO123"

    def test_several_accounts_and_no_choice_is_refused(self) -> None:
        """Picking one is not a recoverable mistake, so it is not guessed."""
        venue = _authed({
            ("GET", "/account/list"): [
                {"id": 1, "name": "DEMO123"},
                {"id": 2, "name": "LIVE456"},
            ]
        })
        with pytest.raises(ExecutionError, match="Refusing to pick one"):
            venue.authenticate()

    def test_a_named_account_is_selected_from_several(self) -> None:
        venue = _authed(
            {
                ("GET", "/account/list"): [
                    {"id": 1, "name": "DEMO123"},
                    {"id": 2, "name": "LIVE456"},
                ]
            },
            account_spec="LIVE456",
        )
        assert venue.authenticate().account_id == 2

    def test_a_name_that_does_not_exist_lists_the_ones_that_do(self) -> None:
        venue = _authed(account_spec="TYPO")
        with pytest.raises(ExecutionError, match="DEMO123"):
            venue.authenticate()

    def test_a_login_with_no_accounts_is_an_error(self) -> None:
        venue = _authed({("GET", "/account/list"): []})
        with pytest.raises(ExecutionError, match="no accounts"):
            venue.authenticate()


class TestSubmit:
    _UNSET: Any = object()

    def _venue(self, order_reply: Any = _UNSET) -> _StubVenue:
        # Sentinel rather than `or`: one test needs an *empty* reply, which a
        # truthiness default would silently replace with a successful one.
        if order_reply is self._UNSET:
            order_reply = {"orderId": 9001}
        return _authed({
            ("POST", "/order/placeOrder"): order_reply,
            ("POST", "/order/placeOSO"): order_reply,
        })

    def test_the_order_is_sent_and_the_broker_id_recorded(self) -> None:
        venue = self._venue()
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.SUBMITTED
        assert venue._broker_ids[order.order_id] == 9001
        assert order.created_at == NOW

    def test_every_order_is_flagged_automated(self) -> None:
        """A CME Group requirement, so it is unconditional rather than a flag —
        nothing this platform emits is hand-entered."""
        venue = self._venue()
        venue.submit(_order(), NOW)
        assert venue.body_for("/order/placeOrder")["isAutomated"] is True

    def test_bracket_legs_are_also_flagged_automated(self) -> None:
        venue = self._venue()
        venue.submit(_order(stop_loss=4990.0, take_profit=5050.0), NOW)
        body = venue.body_for("/order/placeOSO")
        assert body["bracket1"]["isAutomated"] is True
        assert body["bracket2"]["isAutomated"] is True

    def test_brackets_travel_with_the_entry_as_one_request(self) -> None:
        """Sending legs afterwards leaves a window with an unprotected position,
        and that window is exactly when a gap happens."""
        venue = self._venue()
        venue.submit(_order(stop_loss=4990.0, take_profit=5050.0), NOW)
        assert not any(p.startswith("/order/placeOrder") for _, p, _ in venue.calls)
        body = venue.body_for("/order/placeOSO")
        assert body["bracket1"]["stopPrice"] == 4990.0
        assert body["bracket2"]["price"] == 5050.0

    def test_bracket_legs_face_the_opposite_way_to_the_entry(self) -> None:
        venue = self._venue()
        venue.submit(_order(side=Side.SELL, stop_loss=5050.0), NOW)
        assert venue.body_for("/order/placeOSO")["bracket1"]["action"] == "Buy"

    def test_a_stop_only_order_still_gets_its_stop(self) -> None:
        venue = self._venue()
        venue.submit(_order(stop_loss=4990.0), NOW)
        body = venue.body_for("/order/placeOSO")
        assert body["bracket1"]["orderType"] == "Stop"
        assert "bracket2" not in body

    def test_a_broker_failure_reason_becomes_a_rejection_not_a_crash(self) -> None:
        venue = self._venue({"failureReason": "InsufficientFunds"})
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED
        assert "InsufficientFunds" in order.reject_reason

    def test_an_accepted_order_with_no_id_is_rejected_not_tracked(self) -> None:
        """An untracked order cannot be polled or cancelled. Better to know now."""
        venue = self._venue({})
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED
        assert order.order_id not in venue._broker_ids

    def test_a_transport_failure_becomes_a_rejection(self) -> None:
        venue = _authed({
            ("POST", "/order/placeOrder"): ExecutionError("host unreachable")
        })
        assert venue.submit(_order(), NOW).status is OrderStatus.REJECTED

    def test_missing_credentials_reject_before_any_request(self) -> None:
        venue = TradovateVenue(username=None, password=None, cid=None, secret=None)
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED
        assert "credentials" in order.reject_reason

    def test_limit_and_stop_prices_are_forwarded(self) -> None:
        venue = self._venue()
        venue.submit(
            _order(order_type=OrderType.STOP_LIMIT, limit_price=5001.0, stop_price=5000.0),
            NOW,
        )
        body = venue.body_for("/order/placeOrder")
        assert body["orderType"] == "StopLimit"
        assert body["price"] == 5001.0 and body["stopPrice"] == 5000.0


class TestCancel:
    def _submitted(self, extra: dict[tuple[str, str], Any]) -> tuple[_StubVenue, Order]:
        venue = _authed({("POST", "/order/placeOrder"): {"orderId": 9001}, **extra})
        return venue, venue.submit(_order(), NOW)

    def test_a_successful_cancel_is_recorded(self) -> None:
        venue, order = self._submitted({("POST", "/order/cancelOrder"): {"ok": True}})
        assert venue.cancel(order).status is OrderStatus.CANCELLED

    def test_a_cancel_rejected_because_the_order_already_filled_adopts_filled(
        self,
    ) -> None:
        """Not CANCELLED. The position is real and something has to manage it."""
        venue, order = self._submitted({
            ("POST", "/order/cancelOrder"): ExecutionError("HTTP 404"),
            ("GET", "/order/item"): {"ordStatus": "Filled"},
        })
        assert venue.cancel(order).status is OrderStatus.FILLED

    def test_a_cancel_that_fails_on_a_still_working_order_raises(self) -> None:
        """The failure this test exists for: a 500 or a timeout means *unknown*,
        and writing the order off as cancelled leaves it live at the broker with
        nothing on this side tracking it."""
        venue, order = self._submitted({
            ("POST", "/order/cancelOrder"): ExecutionError("HTTP 503"),
            ("GET", "/order/item"): {"ordStatus": "Working"},
        })
        with pytest.raises(ExecutionError, match="503"):
            venue.cancel(order)
        assert order.status is not OrderStatus.CANCELLED

    def test_an_order_that_was_never_submitted_is_left_alone(self) -> None:
        venue = _authed()
        order = _order()
        assert venue.cancel(order).status is order.status


class TestPolling:
    def _venue(self, remote: dict[str, Any]) -> tuple[_StubVenue, Order]:
        venue = _authed({
            ("POST", "/order/placeOrder"): {"orderId": 9001},
            ("GET", "/order/item"): remote,
        })
        return venue, venue.submit(_order(), NOW)

    def test_a_fill_is_emitted_once(self) -> None:
        venue, order = self._venue(
            {"ordStatus": "Filled", "cumQty": 2, "avgPx": 5000.25}
        )
        fills = venue.poll(NOW)
        assert len(fills) == 1
        assert fills[0].quantity == 2.0
        assert fills[0].price == 5000.25
        assert order.status is OrderStatus.FILLED
        # Terminal orders stop being polled, so the fill cannot be double counted.
        assert venue.poll(NOW) == []

    def test_a_partial_fill_emits_only_the_new_quantity(self) -> None:
        venue, order = self._venue(
            {"ordStatus": "Working", "cumQty": 1, "avgPx": 5000.0}
        )
        assert venue.poll(NOW)[0].quantity == 1.0
        assert order.filled_quantity == 1.0
        assert venue.poll(NOW) == []  # nothing new since last time

    def test_a_rejected_order_emits_no_fill(self) -> None:
        venue, order = self._venue({"ordStatus": "Rejected", "cumQty": 0})
        assert venue.poll(NOW) == []
        assert order.status is OrderStatus.REJECTED

    def test_slippage_is_measured_against_the_reference_price(self) -> None:
        venue = _authed({
            ("POST", "/order/placeOrder"): {"orderId": 9001},
            ("GET", "/order/item"): {
                "ordStatus": "Filled", "cumQty": 2, "avgPx": 5001.0
            },
        })
        venue.submit(_order(reference_price=5000.0), NOW)
        assert venue.poll(NOW)[0].slippage == pytest.approx(1.0)

    def test_working_orders_excludes_terminal_ones(self) -> None:
        venue, _ = self._venue({"ordStatus": "Filled", "cumQty": 2, "avgPx": 5000.0})
        assert len(venue.working_orders()) == 1
        venue.poll(NOW)
        assert venue.working_orders() == []


@dataclass
class _Position:
    instrument: Any
    quantity: float


@dataclass
class _Portfolio:
    open_positions: list[_Position]
    equity: float = 100_000.0


class TestPositionsAndReconciliation:
    def _venue(self, rows: list[dict[str, Any]]) -> _StubVenue:
        return _authed({("GET", "/position/list"): rows})

    def test_a_flat_row_is_not_a_position(self) -> None:
        """Tradovate keeps rows for closed contracts. Counting them as open
        positions would make every reconciliation dirty forever."""
        assert self._venue([{"symbol": "ESM6", "netPos": 0, "netPrice": 5000}]).positions() == {}

    def test_agreement_is_clean(self) -> None:
        venue = self._venue([{"symbol": "ES", "netPos": 2, "netPrice": 5000}])
        assert venue.reconcile(_Portfolio([_Position(ES, 2.0)])).is_clean

    def test_a_broker_only_position_is_flagged(self) -> None:
        """Nothing on this side will size against it or stop it out."""
        report = self._venue(
            [{"symbol": "NQ", "netPos": 1, "netPrice": 18000}]
        ).reconcile(_Portfolio([]))
        assert report.broker_only == {"NQ": 1.0}
        assert not report.is_clean
        assert "UNMANAGED" in report.render()

    def test_a_quantity_mismatch_reports_both_sides(self) -> None:
        report = self._venue(
            [{"symbol": "ES", "netPos": 1, "netPrice": 5000}]
        ).reconcile(_Portfolio([_Position(ES, 2.0)]))
        assert report.mismatched == {"ES": (2.0, 1.0)}

    def test_a_short_position_survives_the_round_trip(self) -> None:
        positions = self._venue(
            [{"symbol": "ES", "netPos": -3, "netPrice": 5000}]
        ).positions()
        assert positions["ES"].quantity == -3.0
        assert positions["ES"].is_long is False

    def test_a_platform_only_position_is_reported(self) -> None:
        report = self._venue([]).reconcile(_Portfolio([_Position(ES, 2.0)]))
        assert report.platform_only == {"ES": 2.0}


class TestErrorSurface:
    def test_a_blocked_host_says_so_instead_of_blaming_the_credentials(self) -> None:
        """A 403 from the proxy looks like an auth failure and is not one.
        Chasing the wrong cause here costs an afternoon."""
        import urllib.error

        class _Blocked(TradovateVenue):
            def _request(self, *args: Any, **kwargs: Any) -> Any:
                raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

        venue = _Blocked(username="u", password="p", cid="c", secret="s")
        with pytest.raises(urllib.error.URLError):
            venue.authenticate()

    def test_the_default_host_is_demo(self) -> None:
        """Defaulting to live would put the burden of safety on remembering."""
        assert TradovateVenue(username="u", password="p", cid="c", secret="s").host == (
            DEMO_HOST
        )
