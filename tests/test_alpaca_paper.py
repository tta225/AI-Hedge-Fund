"""Alpaca paper venue. Offline — the HTTP layer is stubbed.

Two properties carry the weight here.

**It must never route to a live account.** The class is paper-only, and the one
bug that would matter most in this file is a misconfiguration that sends orders
somewhere real. `TestPaperGuard` asserts the refusal rather than trusting the
hostname.

**Reconciliation must not report clean when it is not.** A position the broker
holds and the platform does not is unmanaged — nothing sizes against it, stops
it out, or closes it. A reconciliation that misses that is worse than none,
because it manufactures confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from axiom.core.types import OrderStatus, OrderType, Side, get_instrument
from axiom.execution.alpaca_paper import AlpacaPaperVenue, ReconciliationReport
from axiom.execution.base import ExecutionError, Order

NOW = pd.Timestamp("2026-03-02 15:00", tz="UTC")
SPY = get_instrument("SPY")


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop this module from ever reaching a real broker account.

    `AlpacaPaperVenue(api_key=None)` deliberately falls back to the environment,
    which is right for the application and wrong for a test: on a machine with
    real keys set, a test asserting "no credentials" instead authenticates and
    queries a live paper account. That happened. Clearing the environment for
    the whole module makes the network unreachable by construction rather than
    by the discipline of remembering.
    """
    for name in (
        "APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class _StubVenue(AlpacaPaperVenue):
    """Serves canned API responses instead of reaching the network."""

    def __init__(self, responses: dict[tuple[str, str], Any] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(api_key="PKTEST", secret_key="secret", **kwargs)
        self.responses = responses or {}
        self.calls: list[tuple[str, str, Any]] = []

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        self.calls.append((method, path, body))
        for (m, prefix), value in self.responses.items():
            if m == method and path.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return value
        return {}


def _account(number: str = "PA123456", equity: float = 100_000.0) -> dict[str, Any]:
    return {"account_number": number, "status": "ACTIVE", "equity": str(equity)}


def _order(**kwargs: Any) -> Order:
    defaults: dict[str, Any] = {
        "instrument": SPY, "side": Side.BUY, "quantity": 10.0,
        "order_type": OrderType.MARKET, "strategy": "test",
    }
    return Order(**{**defaults, **kwargs})


class TestPaperGuard:
    def test_is_not_live(self) -> None:
        """The router's live refusal keys off this. It must stay False."""
        assert AlpacaPaperVenue.is_live is False

    def test_refuses_a_non_paper_host(self) -> None:
        venue = _StubVenue(
            {("GET", "/v2/account"): _account()},
            host="https://api.alpaca.markets",
        )
        with pytest.raises(ExecutionError, match="not the paper endpoint"):
            venue._ensure_paper()

    def test_refuses_an_account_without_a_paper_prefix(self) -> None:
        """A live account reachable at the paper host would be the worst bug here."""
        venue = _StubVenue({("GET", "/v2/account"): _account(number="87654321")})
        with pytest.raises(ExecutionError, match="does not look like a paper"):
            venue._ensure_paper()

    def test_accepts_a_paper_account(self) -> None:
        venue = _StubVenue({("GET", "/v2/account"): _account()})
        venue._ensure_paper()
        assert venue._verified_paper

    def test_the_paper_check_is_cached(self) -> None:
        venue = _StubVenue({("GET", "/v2/account"): _account()})
        venue._ensure_paper()
        venue._ensure_paper()
        assert sum(1 for c in venue.calls if c[1] == "/v2/account") == 1


class TestSubmit:
    def test_sends_the_order_and_records_the_broker_id(self) -> None:
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("POST", "/v2/orders"): {"id": "abc-123", "status": "new"},
        })
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.SUBMITTED
        assert venue._broker_ids[order.order_id] == "abc-123"
        assert order.created_at == NOW

    def test_brackets_travel_with_the_order(self) -> None:
        """Submitting legs afterwards leaves a window with no stop.

        Short, but real, and it is exactly the window a gap happens in.
        """
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("POST", "/v2/orders"): {"id": "abc", "status": "new"},
        })
        venue.submit(_order(stop_loss=95.0, take_profit=110.0), NOW)
        body = next(b for m, p, b in venue.calls if m == "POST")
        assert body["order_class"] == "bracket"
        assert body["stop_loss"]["stop_price"] == "95.0"
        assert body["take_profit"]["limit_price"] == "110.0"

    def test_a_broker_rejection_becomes_a_rejected_order_not_a_crash(self) -> None:
        """Rejections are information the signal funnel should report."""
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("POST", "/v2/orders"): ExecutionError("insufficient buying power"),
        })
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED
        assert "insufficient buying power" in order.reject_reason

    def test_missing_credentials_reject_before_any_request(self) -> None:
        venue = AlpacaPaperVenue(api_key=None, secret_key=None)
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED
        assert "credentials" in order.reject_reason

    def test_limit_and_stop_prices_are_forwarded(self) -> None:
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("POST", "/v2/orders"): {"id": "a", "status": "new"},
        })
        venue.submit(
            _order(order_type=OrderType.STOP_LIMIT, limit_price=101.0, stop_price=100.0),
            NOW,
        )
        body = next(b for m, p, b in venue.calls if m == "POST")
        assert body["type"] == "stop_limit"
        assert body["limit_price"] == "101.0" and body["stop_price"] == "100.0"


class TestPolling:
    def _submitted(self, order_response: dict[str, Any]) -> tuple[_StubVenue, Order]:
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("POST", "/v2/orders"): {"id": "x", "status": "new"},
            ("GET", "/v2/orders/x"): order_response,
        })
        return venue, venue.submit(_order(), NOW)

    def test_a_fill_is_emitted_once(self) -> None:
        venue, order = self._submitted(
            {"status": "filled", "filled_qty": "10", "filled_avg_price": "664.5"}
        )
        fills = venue.poll(NOW)
        assert len(fills) == 1
        assert fills[0].quantity == 10.0
        assert fills[0].price == 664.5
        assert order.status is OrderStatus.FILLED
        # Terminal orders stop being polled, so no duplicate fill.
        assert venue.poll(NOW) == []

    def test_a_partial_fill_emits_only_the_new_quantity(self) -> None:
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("POST", "/v2/orders"): {"id": "x", "status": "new"},
            ("GET", "/v2/orders/x"): {
                "status": "partially_filled", "filled_qty": "4",
                "filled_avg_price": "664.0",
            },
        })
        order = venue.submit(_order(), NOW)
        assert venue.poll(NOW)[0].quantity == 4.0
        assert order.status is OrderStatus.PARTIALLY_FILLED
        # Still 4 filled — nothing new, so nothing emitted.
        assert venue.poll(NOW) == []

    def test_a_rejected_order_emits_no_fill(self) -> None:
        venue, order = self._submitted({"status": "rejected", "filled_qty": "0"})
        assert venue.poll(NOW) == []
        assert order.status is OrderStatus.REJECTED

    def test_slippage_is_measured_against_the_reference_price(self) -> None:
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("POST", "/v2/orders"): {"id": "x", "status": "new"},
            ("GET", "/v2/orders/x"): {
                "status": "filled", "filled_qty": "10", "filled_avg_price": "665.0",
            },
        })
        venue.submit(_order(reference_price=664.0), NOW)
        assert venue.poll(NOW)[0].slippage == pytest.approx(1.0)


@dataclass
class _Position:
    instrument: Any
    quantity: float


@dataclass
class _Portfolio:
    open_positions: list[_Position]
    equity: float = 100_000.0


class TestReconciliation:
    def _venue(self, positions: list[dict[str, Any]]) -> _StubVenue:
        return _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("GET", "/v2/positions"): positions,
        })

    def test_agreement_is_clean(self) -> None:
        venue = self._venue([
            {"symbol": "SPY", "qty": "10", "avg_entry_price": "660",
             "market_value": "6640", "unrealized_pl": "40"}
        ])
        report = venue.reconcile(_Portfolio([_Position(SPY, 10.0)]))
        assert report.is_clean
        assert "✓" in report.render()

    def test_a_broker_only_position_is_flagged_as_unmanaged(self) -> None:
        """The failure that matters: nothing here will stop that position out."""
        venue = self._venue([
            {"symbol": "QQQ", "qty": "5", "avg_entry_price": "500",
             "market_value": "2500", "unrealized_pl": "0"}
        ])
        report = venue.reconcile(_Portfolio([]))
        assert not report.is_clean
        assert report.broker_only == {"QQQ": 5.0}
        assert "UNMANAGED" in report.render()

    def test_a_quantity_mismatch_is_reported_with_both_sides(self) -> None:
        venue = self._venue([
            {"symbol": "SPY", "qty": "7", "avg_entry_price": "660",
             "market_value": "4620", "unrealized_pl": "0"}
        ])
        report = venue.reconcile(_Portfolio([_Position(SPY, 10.0)]))
        assert report.mismatched == {"SPY": (10.0, 7.0)}
        assert not report.is_clean

    def test_a_platform_only_position_is_reported(self) -> None:
        report = self._venue([]).reconcile(_Portfolio([_Position(SPY, 10.0)]))
        assert report.platform_only == {"SPY": 10.0}
        assert not report.is_clean

    def test_equity_drift_is_measured(self) -> None:
        venue = self._venue([])
        report = venue.reconcile(_Portfolio([], equity=95_000.0))
        assert report.equity_drift == pytest.approx(5_000.0)

    def test_a_failed_reconciliation_never_renders_as_clean(self) -> None:
        report = ReconciliationReport(broker_only={"QQQ": 1.0})
        assert not report.is_clean
        assert "✓" not in report.render()
        assert "do not trade" in report.render()


class TestErrorSurface:
    def test_a_blocked_host_explains_itself(self) -> None:
        import urllib.error

        class _Blocked(AlpacaPaperVenue):
            def _request(self, method: str, path: str, body: Any = None) -> Any:
                raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

        venue = _Blocked(api_key="k", secret_key="s")
        with pytest.raises(urllib.error.URLError):
            venue.account()

    def test_check_reports_a_missing_credential_without_calling_out(self) -> None:
        assert "no Alpaca credentials" in AlpacaPaperVenue(
            api_key=None, secret_key=None
        ).check()

    def test_check_summarises_a_working_account(self) -> None:
        venue = _StubVenue({
            ("GET", "/v2/account"): _account(),
            ("GET", "/v2/positions"): [],
        })
        status = venue.check()
        assert status.startswith("✓")
        assert "PA123456" in status
