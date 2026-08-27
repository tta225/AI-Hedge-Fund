"""Tests for the Alpaca execution venue.

Every test drives a stubbed ``_request``. Nothing here touches the network, and
nothing here can reach an account — a test suite that could place an order is a
test suite that will eventually place one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from axiom.core.types import OrderStatus, OrderType, Side, TimeInForce, get_instrument
from axiom.execution.alpaca import (
    LIVE_URL,
    PAPER_URL,
    AlpacaVenue,
)
from axiom.execution.base import ExecutionError, Order

BTC = get_instrument("BTC-USD")
AAPL = get_instrument("AAPL")
T0 = pd.Timestamp("2026-01-01T00:00:00Z")


class _StubVenue(AlpacaVenue):
    """Records requests and replays canned responses."""

    def __init__(self, responses: list[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(api_key="k", secret_key="s", **kwargs)
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.responses = list(responses or [])
        self.raise_with: ExecutionError | None = None

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        self.calls.append((method, path, body))
        if self.raise_with is not None:
            raise self.raise_with
        return self.responses.pop(0) if self.responses else {}

    @property
    def last_body(self) -> dict[str, Any]:
        return self.calls[-1][2] or {}


def _order(**kwargs: Any) -> Order:
    defaults: dict[str, Any] = {
        "instrument": AAPL,
        "side": Side.BUY,
        "quantity": 10.0,
        "strategy": "test",
    }
    defaults.update(kwargs)
    return Order(**defaults)


class TestSafetyPosture:
    def test_paper_is_the_default(self) -> None:
        """Reaching real money must take a deliberate act, not a default."""
        venue = AlpacaVenue(api_key="k", secret_key="s")
        assert venue.paper is True
        assert venue.is_live is False
        assert venue.base_url == PAPER_URL

    def test_live_must_be_asked_for(self) -> None:
        venue = AlpacaVenue(api_key="k", secret_key="s", paper=False)
        assert venue.is_live is True
        assert venue.base_url == LIVE_URL

    def test_liveness_is_per_instance_not_per_class(self) -> None:
        """A live venue in one process must not mark a paper one live."""
        live = AlpacaVenue(api_key="k", secret_key="s", paper=False)
        paper = AlpacaVenue(api_key="k", secret_key="s")
        assert live.is_live and not paper.is_live

    def test_the_router_refuses_a_live_venue_outside_live_mode(self) -> None:
        from axiom.execution.router import OrderRouter
        from axiom.risk.manager import RiskManager

        live = AlpacaVenue(api_key="k", secret_key="s", paper=False)
        with pytest.raises(PermissionError, match="Refusing"):
            OrderRouter(venue=live, risk=RiskManager())

    def test_the_router_accepts_a_paper_venue(self) -> None:
        from axiom.execution.router import OrderRouter
        from axiom.risk.manager import RiskManager

        paper = AlpacaVenue(api_key="k", secret_key="s")
        assert OrderRouter(venue=paper, risk=RiskManager()) is not None

    def test_declares_venue_held_brackets(self) -> None:
        """The desk's require_brackets gate reads this."""
        assert AlpacaVenue.supports_brackets is True

    def test_missing_credentials_are_reported_not_guessed(self) -> None:
        # Set after construction: an empty string falls through to the
        # environment, and this box may legitimately have keys exported.
        venue = AlpacaVenue(api_key="k", secret_key="s")
        venue.api_key = None
        venue.secret_key = None
        assert not venue.is_available()
        assert "No Alpaca credentials" in venue.check_credentials()

    def test_an_unkeyed_venue_refuses_to_send(self) -> None:
        venue = AlpacaVenue(api_key="k", secret_key="s")
        venue.api_key = None
        with pytest.raises(ExecutionError, match="credentials not found"):
            venue.submit(_order(), T0)


class TestSymbolTranslation:
    def test_equities_pass_through(self) -> None:
        assert AlpacaVenue.venue_symbol(AAPL) == "AAPL"

    def test_crypto_becomes_slash_separated(self) -> None:
        """`BTC-USD` is rejected with a 400, and that looks like an outage."""
        assert AlpacaVenue.venue_symbol(BTC) == "BTC/USD"

    def test_already_slashed_is_untouched(self) -> None:
        from axiom.core.types import AssetClass, Instrument

        instrument = Instrument(symbol="ETH/USDT", asset_class=AssetClass.CRYPTO)
        assert AlpacaVenue.venue_symbol(instrument) == "ETH/USDT"

    def test_unseparated_pair_is_split_on_the_longest_quote(self) -> None:
        from axiom.core.types import AssetClass, Instrument

        instrument = Instrument(symbol="ETHUSDT", asset_class=AssetClass.CRYPTO)
        assert AlpacaVenue.venue_symbol(instrument) == "ETH/USDT"

    def test_round_trip_back_to_local_spelling(self) -> None:
        """Otherwise the reconciler sees a phantom and an unknown position."""
        assert AlpacaVenue.local_symbol("BTC/USD") == "BTC-USD"
        assert AlpacaVenue.local_symbol("AAPL") == "AAPL"


class TestOrderSubmission:
    def test_sends_the_basic_fields(self) -> None:
        venue = _StubVenue([{"id": "abc", "status": "accepted"}])
        venue.submit(_order(), T0)
        method, path, body = venue.calls[-1]
        assert (method, path) == ("POST", "/v2/orders")
        assert body is not None
        assert body["symbol"] == "AAPL"
        assert body["side"] == "buy"
        assert body["type"] == "market"
        assert body["qty"] == "10"

    def test_maps_the_venue_status_back(self) -> None:
        venue = _StubVenue([{"id": "abc", "status": "partially_filled",
                             "filled_qty": "4", "filled_avg_price": "101.5"}])
        order = venue.submit(_order(), T0)
        assert order.status is OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == pytest.approx(4.0)
        assert order.average_fill_price == pytest.approx(101.5)

    def test_an_unknown_status_is_treated_as_open(self) -> None:
        """Assuming an unrecognised terminal state abandons a live position."""
        venue = _StubVenue([{"id": "abc", "status": "some_new_alpaca_state"}])
        order = venue.submit(_order(), T0)
        assert order.status.is_open

    def test_records_the_venue_id_for_later_cancel(self) -> None:
        venue = _StubVenue([{"id": "venue-123", "status": "accepted"}])
        order = venue.submit(_order(), T0)
        assert venue._venue_ids[order.order_id] == "venue-123"

    def test_crypto_symbols_are_translated_on_the_wire(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(instrument=BTC, quantity=0.5), T0)
        assert venue.last_body["symbol"] == "BTC/USD"

    def test_fractional_quantities_survive(self) -> None:
        """Rounding a small crypto size to zero would silently drop the order."""
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(instrument=BTC, quantity=0.00042), T0)
        assert venue.last_body["qty"] == "0.00042"

    def test_limit_price_is_sent(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(order_type=OrderType.LIMIT, limit_price=99.5), T0)
        assert venue.last_body["limit_price"] == "99.5"

    def test_the_idempotency_tag_becomes_the_client_order_id(self) -> None:
        """This is what makes retrying an ambiguous submit safe."""
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(tag="decision-key-1"), T0)
        assert venue.last_body["client_order_id"] == "decision-key-1"

    def test_a_long_tag_is_truncated_rather_than_rejected(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(tag="x" * 400), T0)
        assert len(venue.last_body["client_order_id"]) == 128


class TestBrackets:
    def test_both_legs_produce_a_bracket(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(stop_loss=95.0, take_profit=110.0), T0)
        body = venue.last_body
        assert body["order_class"] == "bracket"
        assert body["stop_loss"] == {"stop_price": "95"}
        assert body["take_profit"] == {"limit_price": "110"}

    def test_a_lone_stop_still_rests_at_the_venue(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(stop_loss=95.0), T0)
        body = venue.last_body
        assert body["order_class"] == "oto"
        assert body["stop_loss"] == {"stop_price": "95"}
        assert "take_profit" not in body

    def test_a_lone_target_is_also_oto(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(take_profit=110.0), T0)
        assert venue.last_body["order_class"] == "oto"

    def test_no_legs_means_no_order_class(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(), T0)
        assert "order_class" not in venue.last_body

    def test_an_incompatible_tif_is_refused_not_rewritten(self) -> None:
        """Silently changing the caller's intent is worse than refusing."""
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        with pytest.raises(ExecutionError, match=r"must.*outlive the entry"):
            venue.submit(
                _order(stop_loss=95.0, time_in_force=TimeInForce.IOC), T0
            )

    def test_an_unbracketed_ioc_is_fine(self) -> None:
        venue = _StubVenue([{"id": "a", "status": "accepted"}])
        venue.submit(_order(time_in_force=TimeInForce.IOC), T0)
        assert venue.last_body["time_in_force"] == "ioc"


class TestCancel:
    def test_cancels_by_venue_id(self) -> None:
        venue = _StubVenue([{"id": "venue-9", "status": "accepted"}, {}])
        order = venue.submit(_order(), T0)
        venue.cancel(order)
        assert venue.calls[-1][:2] == ("DELETE", "/v2/orders/venue-9")
        assert order.status is OrderStatus.CANCELLED

    def test_an_already_gone_order_is_success(self) -> None:
        """Retrying a successful cancel must not look like a failure."""
        venue = _StubVenue([{"id": "venue-9", "status": "accepted"}])
        order = venue.submit(_order(), T0)
        venue.raise_with = ExecutionError("alpaca: HTTP 404 on DELETE /v2/orders/venue-9")
        assert venue.cancel(order).status is OrderStatus.CANCELLED

    def test_other_errors_still_raise(self) -> None:
        venue = _StubVenue([{"id": "venue-9", "status": "accepted"}])
        order = venue.submit(_order(), T0)
        venue.raise_with = ExecutionError("alpaca: HTTP 500 on DELETE")
        with pytest.raises(ExecutionError, match="500"):
            venue.cancel(order)

    def test_an_unknown_order_is_an_explicit_error(self) -> None:
        venue = _StubVenue()
        with pytest.raises(ExecutionError, match="no venue id known"):
            venue.cancel(_order())


class TestPositions:
    def test_returns_local_symbols(self) -> None:
        venue = _StubVenue([[{"symbol": "BTC/USD", "qty": "1.5", "side": "long"}]])
        assert venue.positions() == {"BTC-USD": pytest.approx(1.5)}

    def test_shorts_come_back_negative(self) -> None:
        venue = _StubVenue([[{"symbol": "AAPL", "qty": "-10", "side": "short"}]])
        assert venue.positions() == {"AAPL": pytest.approx(-10.0)}

    def test_a_positive_qty_marked_short_is_still_negative(self) -> None:
        """Some responses report magnitude plus a side; trust both."""
        venue = _StubVenue([[{"symbol": "AAPL", "qty": "10", "side": "short"}]])
        assert venue.positions() == {"AAPL": pytest.approx(-10.0)}

    def test_flat_symbols_are_omitted(self) -> None:
        venue = _StubVenue([[{"symbol": "AAPL", "qty": "0", "side": "long"}]])
        assert venue.positions() == {}

    def test_an_empty_account_is_an_empty_book(self) -> None:
        assert _StubVenue([[]]).positions() == {}

    def test_positions_reconcile_against_the_store(self) -> None:
        """The end-to-end property: same spelling on both sides."""
        from axiom.store import reconcile_positions

        venue = _StubVenue([[{"symbol": "BTC/USD", "qty": "1", "side": "long"}]])
        result = reconcile_positions({"BTC-USD": 1.0}, venue.positions())
        assert result.is_clean


class TestAccount:
    def test_parses_the_snapshot(self) -> None:
        venue = _StubVenue([{
            "equity": "100000", "cash": "50000", "buying_power": "200000",
            "trading_blocked": False, "pattern_day_trader": False,
            "currency": "USD",
        }])
        account = venue.account()
        assert account.equity == pytest.approx(100_000)
        assert account.is_tradable

    def test_a_blocked_account_is_not_tradable(self) -> None:
        venue = _StubVenue([{"equity": "1", "cash": "1", "buying_power": "0",
                             "trading_blocked": True}])
        assert not venue.account().is_tradable

    def test_check_credentials_names_the_environment(self) -> None:
        payload = {"equity": "1000", "cash": "1000", "buying_power": "1000",
                   "trading_blocked": False, "pattern_day_trader": False}
        assert "PAPER" in _StubVenue([payload]).check_credentials()
        assert "REAL MONEY" in _StubVenue([payload], paper=False).check_credentials()

    def test_check_credentials_surfaces_a_failure(self) -> None:
        venue = _StubVenue()
        venue.raise_with = ExecutionError("alpaca: HTTP 401")
        assert venue.check_credentials().startswith("✗")


class TestFills:
    def test_pages_until_the_venue_runs_out(self) -> None:
        """A truncated poll advances the watermark past fills it never saw."""
        full = [
            {"id": f"f{i}", "order_id": "o", "symbol": "AAPL", "side": "buy",
             "qty": "1", "price": "1", "transaction_time": "2026-01-01T15:00:00Z"}
            for i in range(100)
        ]
        tail = [{"id": "f100", "order_id": "o", "symbol": "AAPL", "side": "buy",
                 "qty": "1", "price": "1",
                 "transaction_time": "2026-01-01T15:00:00Z"}]
        venue = _StubVenue([full, tail])
        assert len(venue.fills_since(T0)) == 101
        assert "page_token=f99" in venue.calls[-1][1]

    def test_a_short_page_ends_the_walk(self) -> None:
        venue = _StubVenue([[{
            "id": "f", "order_id": "o", "symbol": "AAPL", "side": "buy",
            "qty": "1", "price": "1",
            "transaction_time": "2026-01-01T15:00:00Z",
        }]])
        venue.fills_since(T0)
        assert len(venue.calls) == 1

    def test_page_size_is_clamped_to_the_venue_maximum(self) -> None:
        """Alpaca rejects page_size above 100 with a 422."""
        venue = _StubVenue([[]])
        venue.fills_since(T0, limit=5000)
        assert "page_size=100" in venue.calls[-1][1]

    def test_parses_activities(self) -> None:
        venue = _StubVenue([[{
            "id": "fill-1", "order_id": "o-1", "symbol": "AAPL", "side": "buy",
            "qty": "5", "price": "150.25",
            "transaction_time": "2026-01-01T15:00:00Z",
        }]])
        fills = venue.fills_since(T0)
        assert len(fills) == 1
        assert fills[0].venue_fill_id == "fill-1"
        assert fills[0].fill.quantity == pytest.approx(5.0)
        assert fills[0].fill.price == pytest.approx(150.25)
        assert fills[0].fill.side is Side.BUY

    def test_the_venue_order_id_is_carried_for_attribution(self) -> None:
        """Without it a fill cannot be linked to the decision that caused it."""
        venue = _StubVenue([[{
            "id": "f", "order_id": "order-abc", "symbol": "AAPL", "side": "buy",
            "qty": "1", "price": "1", "transaction_time": "2026-01-01T15:00:00Z",
        }]])
        assert venue.fills_since(T0)[0].venue_order_id == "order-abc"

    def test_crypto_fills_come_back_in_local_spelling(self) -> None:
        venue = _StubVenue([[{
            "id": "f", "symbol": "BTC/USD", "side": "sell", "qty": "0.1",
            "price": "60000", "transaction_time": "2026-01-01T15:00:00Z",
        }]])
        assert venue.fills_since(T0)[0].fill.instrument.symbol == "BTC-USD"

    def test_zero_quantity_rows_are_skipped(self) -> None:
        venue = _StubVenue([[{"id": "f", "symbol": "AAPL", "side": "buy",
                              "qty": "0", "price": "1",
                              "transaction_time": "2026-01-01T15:00:00Z"}]])
        assert venue.fills_since(T0) == []

    def test_a_naive_timestamp_is_treated_as_utc(self) -> None:
        import urllib.parse

        venue = _StubVenue([[]])
        venue.fills_since(pd.Timestamp("2026-01-01T00:00:00"))
        assert "2026-01-01T00:00:00Z" in urllib.parse.unquote(venue.calls[-1][1])

    def test_fill_ids_let_the_store_deduplicate(self) -> None:
        """A poll overlapping a previous window must not double the position."""
        from axiom.store import Store

        row = {"id": "fill-1", "order_id": "o-1", "symbol": "AAPL",
               "side": "buy", "qty": "5", "price": "150",
               "transaction_time": "2026-01-01T15:00:00Z"}
        venue = _StubVenue([[row], [row]])
        with Store(":memory:") as store:
            order_id, _ = store.record_order(_order(), "k", T0)
            for _ in range(2):
                for venue_fill in venue.fills_since(T0):
                    store.record_fill(
                        order_id, venue_fill.fill, venue_fill.venue_fill_id
                    )
            assert store.positions()["AAPL"] == pytest.approx(5.0)


class TestWorkingOrders:
    def test_reads_from_the_venue_not_local_memory(self) -> None:
        """A restarted process has no local memory; the venue is the truth."""
        venue = _StubVenue([[{
            "id": "v1", "symbol": "AAPL", "side": "buy", "qty": "10",
            "type": "limit", "limit_price": "99.5", "status": "new",
            "filled_qty": "0", "client_order_id": "key-1",
        }]])
        orders = venue.working_orders()
        assert venue.calls[-1][0] == "GET"
        assert len(orders) == 1
        assert orders[0].instrument.symbol == "AAPL"
        assert orders[0].limit_price == pytest.approx(99.5)
        assert orders[0].status is OrderStatus.SUBMITTED

    def test_empty_is_empty(self) -> None:
        assert _StubVenue([[]]).working_orders() == []


class TestErrorMessages:
    @pytest.mark.parametrize(
        ("code", "fragment"),
        [
            (401, "APCA_API_KEY_ID"),
            (403, "paper keys do not work live"),
            (422, "already exists"),
            (422, "page size"),
            (429, "rate limited"),
        ],
    )
    def test_hints_explain_the_common_failures(self, code: int, fragment: str) -> None:
        assert fragment in AlpacaVenue._hint(code)

    def test_an_unknown_code_has_no_hint(self) -> None:
        assert AlpacaVenue._hint(500) == ""
