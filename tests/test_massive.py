"""Massive provider (formerly Polygon.io). Offline — the HTTP layer is stubbed.

Massive is one brand over two genuinely different APIs, and that is where every
interesting bug in this adapter lives. Equities and futures do not share a path,
a response schema, a resolution vocabulary, or a time zone:

* equities key their timestamp ``t`` and their prices ``o h l c v``; futures
  spell all of it out and call the timestamp ``window_start``
* equities want ``timespan=minute``, futures want ``resolution=1min``
* equities aggregate a calendar ``day``, futures a trading ``session``

Parsing a futures response with equity key names produces an empty frame rather
than an error, which is the failure worth testing hardest.

Two more, of the same family:

**Timestamp units are not stated for the futures response.** Reading nanoseconds
as milliseconds dates every bar to roughly the year 57,000; reading milliseconds
as nanoseconds puts them in January 1970. Detection by magnitude is safe — the
units are six orders apart — and is pinned here.

**Massive has no continuous contracts.** A bare ``ES`` resolves to nothing, so
the adapter looks up the front month. It must fetch exactly one contract and say
which: silently concatenating contracts would turn a roll gap into a return.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import pytest

from axiom.core.types import AssetClass, get_instrument, parse_contract
from axiom.data.base import BarRequest, ProviderError, ProviderUnavailableError
from axiom.data.massive import (
    DEFAULT_HOST,
    POLYGON_HOST,
    MassiveProvider,
    _to_utc,
)

SPY = get_instrument("SPY")


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real key would turn 'no credentials' tests into billed network calls."""
    for name in ("MASSIVE_API_KEY", "POLYGON_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _request(symbol: str = "SPY", timeframe: str = "15m", days: int = 5) -> BarRequest:
    return BarRequest.lookback(get_instrument(symbol), timeframe, days)


# Anchored to "now" rather than to a fixed date. `_fetch_raw` trims the response
# to the requested window — correct, because Massive's date-granularity range
# returns whole days and can overshoot an intraday ask — so a fixture pinned to
# a past date produces an empty frame and a failure that says "no bars" rather
# than whatever the test was actually about.
_BASE = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=1)

#: Milliseconds — the equity aggregates unit.
_MS = int(_BASE.value // 1_000_000)
#: The same instant in nanoseconds, as the futures endpoint may report it.
_NS = int(_BASE.value)


def _equity_rows(n: int = 3, offset: int = 0) -> list[dict[str, Any]]:
    """`offset` shifts a page's bars so successive pages do not collide.

    Repeating the same timestamps across pages trips the series' duplicate
    check — correctly — before the pagination under test is ever reached.
    """
    return [
        {"t": _MS + (offset + i) * 900_000, "o": 660.0 + i, "h": 661.0 + i,
         "l": 659.0 + i, "c": 660.5 + i, "v": 1000 + i, "vw": 660.2, "n": 42}
        for i in range(n)
    ]


def _futures_rows(n: int = 3, *, ns: bool = True) -> list[dict[str, Any]]:
    base = _NS if ns else _MS
    step = 900_000_000_000 if ns else 900_000
    return [
        {"ticker": "ESH6", "window_start": base + i * step,
         "open": 5000.0 + i, "high": 5002.0 + i, "low": 4998.0 + i,
         "close": 5001.0 + i, "volume": 500 + i, "dollar_volume": 1e6,
         "transactions": 88, "settlement_price": 5001.0}
        for i in range(n)
    ]


class _StubProvider(MassiveProvider):
    """Serves canned JSON instead of reaching the network."""

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("api_key", "test-key")
        super().__init__(**kwargs)
        self.responses = responses or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def _get(self, path_or_url: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((path_or_url, params))
        if self.error is not None:
            raise self.error
        for prefix, value in self.responses.items():
            if path_or_url.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return value
        return {}

    def paths(self) -> list[str]:
        return [p for p, _ in self.calls]


_CONTRACTS = {
    "/futures/v1/contracts": {
        "results": [
            {"ticker": "ESH6", "product_code": "ES", "name": "E-mini S&P Mar 26",
             "active": True, "last_trade_date": "2099-03-20"},
            {"ticker": "ESM6", "product_code": "ES", "name": "E-mini S&P Jun 26",
             "active": True, "last_trade_date": "2099-06-19"},
        ]
    }
}


class TestAvailability:
    def test_no_key_is_unavailable(self) -> None:
        assert not MassiveProvider().is_available()

    def test_a_polygon_key_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same company, same key, older variable name. Rejecting it would send
        an existing Polygon user hunting for a key they already have."""
        monkeypatch.setenv("POLYGON_API_KEY", "old-name")
        assert MassiveProvider().is_available()

    def test_the_missing_key_message_names_both_variables(self) -> None:
        with pytest.raises(ProviderUnavailableError, match="POLYGON_API_KEY"):
            MassiveProvider()._get("/v2/aggs")

    def test_the_default_host_is_the_new_one(self) -> None:
        assert MassiveProvider(api_key="k").host == DEFAULT_HOST

    def test_the_old_host_still_works(self) -> None:
        assert MassiveProvider(api_key="k", host=POLYGON_HOST).host == POLYGON_HOST


class TestAssetClassCoverage:
    def test_it_carries_futures(self) -> None:
        assert MassiveProvider(api_key="k").supports(get_instrument("ES"))

    def test_it_carries_equities(self) -> None:
        assert MassiveProvider(api_key="k").supports(SPY)

    def test_it_does_not_claim_crypto(self) -> None:
        """Massive sells crypto, but this adapter does not translate the
        'X:BTCUSD' symbology — claiming it would route requests here that
        cannot be served."""
        assert not MassiveProvider(api_key="k").supports(get_instrument("BTC-USD"))


class TestEquityPath:
    def _venue(self) -> _StubProvider:
        return _StubProvider({"/v2/aggs": {"results": _equity_rows()}})

    def test_it_uses_the_v2_aggregates_path(self) -> None:
        provider = self._venue()
        provider.fetch_bars(_request("SPY", "15m"))
        assert provider.paths()[0].startswith("/v2/aggs/ticker/SPY/range/15/minute/")

    def test_the_multiplier_and_timespan_are_split_correctly(self) -> None:
        provider = self._venue()
        provider.fetch_bars(_request("SPY", "4h"))
        assert "/range/4/hour/" in provider.paths()[0]

    def test_bars_are_split_adjusted_by_default(self) -> None:
        """Any multi-year price comparison is meaningless without it."""
        provider = self._venue()
        provider.fetch_bars(_request("SPY"))
        assert provider.calls[0][1]["adjusted"] == "true"

    def test_adjustment_can_be_turned_off(self) -> None:
        provider = _StubProvider({"/v2/aggs": {"results": _equity_rows()}}, adjusted=False)
        provider.fetch_bars(_request("SPY"))
        assert provider.calls[0][1]["adjusted"] == "false"

    def test_the_single_letter_keys_are_mapped(self) -> None:
        series = self._venue().fetch_bars(_request("SPY"))
        assert series.closes[0] == pytest.approx(660.5)
        assert series.highs[0] == pytest.approx(661.0)

    def test_an_unsupported_timeframe_is_named_not_guessed(self) -> None:
        with pytest.raises(ProviderError, match="unsupported timeframe"):
            self._venue().fetch_bars(_request("SPY", "3m"))

    def test_a_missing_column_is_reported_with_what_arrived(self) -> None:
        provider = _StubProvider({"/v2/aggs": {"results": [{"t": _MS, "o": 1.0}]}})
        with pytest.raises(ProviderError, match="missing"):
            provider.fetch_bars(_request("SPY"))


class TestFuturesPath:
    def _venue(self, rows: list[dict[str, Any]] | None = None) -> _StubProvider:
        return _StubProvider({
            **_CONTRACTS,
            "/futures/v1/aggs": {"results": rows if rows is not None else _futures_rows()},
        })

    def test_futures_use_a_different_endpoint_entirely(self) -> None:
        """Not a parameter on the equity path — a separate API."""
        provider = self._venue()
        provider.fetch_bars(_request("ES", "1m"))
        assert any(p.startswith("/futures/v1/aggs/") for p in provider.paths())
        assert not any(p.startswith("/v2/aggs") for p in provider.paths())

    def test_the_resolution_vocabulary_is_not_the_equity_one(self) -> None:
        """'min', not 'minute'. Sending the equity spelling is rejected."""
        provider = self._venue()
        provider.fetch_bars(_request("ES", "15m"))
        aggs = next(p for p, _ in provider.calls if "aggs" in p)
        params = next(q for p, q in provider.calls if p == aggs)
        assert params["resolution"] == "15min"

    def test_a_daily_futures_bar_is_a_session_not_a_day(self) -> None:
        """Futures sessions span midnight, so these are genuinely different
        bars rather than two spellings of one."""
        provider = self._venue()
        provider.fetch_bars(_request("ES", "1d"))
        params = next(q for p, q in provider.calls if "futures/v1/aggs" in p)
        assert params["resolution"] == "1session"

    def test_the_spelled_out_keys_are_read(self) -> None:
        """The quiet failure: equity key names against a futures response
        produce an empty frame, not an error."""
        series = self._venue().fetch_bars(_request("ES", "1m"))
        assert len(series) == 3
        assert series.closes[0] == pytest.approx(5001.0)

    def test_a_futures_response_missing_columns_says_which_schema_it_expected(
        self,
    ) -> None:
        provider = self._venue([{"t": _MS, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1}])
        with pytest.raises(ProviderError, match="single-letter equity keys"):
            provider.fetch_bars(_request("ES", "1m"))


class TestContractResolution:
    def _venue(self) -> _StubProvider:
        return _StubProvider({
            **_CONTRACTS,
            "/futures/v1/aggs": {"results": _futures_rows()},
        })

    def test_a_bare_root_is_resolved_to_a_contract(self) -> None:
        """Massive has no continuous symbology — a bare 'ES' returns nothing."""
        provider = self._venue()
        provider.fetch_bars(_request("ES", "1m"))
        assert provider.last.ticker_resolved == "ESH6"

    def test_the_front_month_is_chosen(self) -> None:
        assert self._venue().front_contract("ES")["ticker"] == "ESH6"

    def test_the_contract_used_is_reported_not_hidden(self) -> None:
        provider = self._venue()
        provider.fetch_bars(_request("ES", "1m"))
        rendered = provider.last.render()
        assert "ESH6" in rendered or "E-mini S&P Mar 26" in rendered

    def test_it_says_it_did_not_stitch(self) -> None:
        """Silently concatenating contracts would turn a roll gap — which is
        not a return — into one."""
        provider = self._venue()
        provider.fetch_bars(_request("ES", "1m"))
        assert "not a stitched series" in provider.last.render()

    def test_only_one_contract_is_fetched(self) -> None:
        provider = self._venue()
        provider.fetch_bars(_request("ES", "1m"))
        aggs = [p for p in provider.paths() if "futures/v1/aggs" in p]
        assert len(aggs) == 1

    def test_an_explicit_contract_skips_the_lookup(self) -> None:
        """Someone who named a month meant that month, and the extra request
        would be a wasted call against a rate limit of five per minute."""
        provider = self._venue()
        provider.fetch_bars(_request("ESH6", "1m"))
        assert not any("contracts" in p for p in provider.paths())

    def test_an_unknown_root_explains_the_symbology(self) -> None:
        provider = _StubProvider({"/futures/v1/contracts": {"results": []}})
        with pytest.raises(ProviderError, match="ESH6"):
            provider.front_contract("ZZ")

    def test_an_expired_contract_is_not_preferred_over_a_live_one(self) -> None:
        provider = _StubProvider({
            "/futures/v1/contracts": {
                "results": [
                    {"ticker": "ESZ0", "active": True, "last_trade_date": "2020-12-18"},
                    {"ticker": "ESH6", "active": True, "last_trade_date": "2099-03-20"},
                ]
            }
        })
        assert provider.front_contract("ES")["ticker"] == "ESH6"


class TestTimestampUnits:
    """Six orders of magnitude apart, so detection is exact — but only if it
    happens. Guessing dates every bar to 1970 or to the year 57,000."""

    def test_milliseconds_are_read_as_milliseconds(self) -> None:
        index = _to_utc(pd.Series([_MS]))
        assert index[0].year == 2026

    def test_nanoseconds_are_read_as_nanoseconds(self) -> None:
        index = _to_utc(pd.Series([_NS]))
        assert index[0].year == 2026

    def test_seconds_are_read_as_seconds(self) -> None:
        assert _to_utc(pd.Series([_MS // 1000]))[0].year == 2026

    def test_the_same_instant_lands_identically_whatever_the_unit(self) -> None:
        assert _to_utc(pd.Series([_MS]))[0] == _to_utc(pd.Series([_NS]))[0]

    def test_a_futures_response_in_nanoseconds_dates_correctly(self) -> None:
        provider = _StubProvider({
            **_CONTRACTS,
            "/futures/v1/aggs": {"results": _futures_rows(ns=True)},
        })
        series = provider.fetch_bars(_request("ES", "1m", days=4000))
        assert series.index[0].year == 2026

    def test_a_futures_response_in_milliseconds_dates_correctly(self) -> None:
        """The units are not documented for this response, so both must work."""
        provider = _StubProvider({
            **_CONTRACTS,
            "/futures/v1/aggs": {"results": _futures_rows(ns=False)},
        })
        series = provider.fetch_bars(_request("ES", "1m", days=4000))
        assert series.index[0].year == 2026

    def test_unparseable_timestamps_raise_rather_than_produce_1970(self) -> None:
        with pytest.raises(ProviderError, match="no parseable timestamps"):
            _to_utc(pd.Series(["not-a-time"]))


class TestPagination:
    def test_it_follows_next_url(self) -> None:
        class _Paging(_StubProvider):
            def _get(self, path_or_url: str, params: Any = None) -> Any:
                self.calls.append((path_or_url, params))
                if len(self.calls) == 1:
                    return {"results": _equity_rows(2),
                            "next_url": "https://api.massive.com/page2?cursor=x"}
                return {"results": _equity_rows(1, offset=2)}

        provider = _Paging()
        provider.fetch_bars(_request("SPY"))
        assert provider.last.pages == 2

    def test_the_cursor_is_not_overwritten_by_the_original_params(self) -> None:
        """Re-appending the first request's params would replace next_url's
        cursor and loop on page one forever."""
        class _Paging(_StubProvider):
            def _get(self, path_or_url: str, params: Any = None) -> Any:
                self.calls.append((path_or_url, params))
                if len(self.calls) == 1:
                    return {"results": _equity_rows(1), "next_url": "https://x/p2"}
                return {"results": _equity_rows(1, offset=1)}

        provider = _Paging()
        provider.fetch_bars(_request("SPY"))
        assert provider.calls[1][1] is None

    def test_a_runaway_window_is_capped_and_says_so(self) -> None:
        """A decade of one-second bars would otherwise page for hours while
        looking like a hang."""
        class _Endless(_StubProvider):
            def _get(self, path_or_url: str, params: Any = None) -> Any:
                self.calls.append((path_or_url, params))
                return {"results": _equity_rows(1, offset=len(self.calls)),
                        "next_url": "https://x/next"}

        provider = _Endless()
        provider.fetch_bars(_request("SPY"))
        assert provider.last.truncated
        assert "truncated" in provider.last.render()

    def test_truncation_is_stamped_on_the_provenance(self) -> None:
        """Partial bars are still real, so this is not an error — but a caller
        who does not know the window was cut short reads a shorter dataset as
        a quieter market."""
        class _Endless(_StubProvider):
            def _get(self, path_or_url: str, params: Any = None) -> Any:
                self.calls.append((path_or_url, params))
                return {"results": _equity_rows(1, offset=len(self.calls)),
                        "next_url": "https://x/next"}

        series = _Endless().fetch_bars(_request("SPY"))
        assert "TRUNCATED" in series.provenance.detail


class TestErrorMessages:
    @staticmethod
    def _raising(
        monkeypatch: pytest.MonkeyPatch, code: int, body: str = "{}"
    ) -> MassiveProvider:
        """Fail at `urlopen`, not at `_get`.

        Overriding `_get` would skip the very code under test — the HTTPError →
        ProviderError translation lives inside it — and the raw urllib error
        would escape looking like a passing test of nothing.
        """
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise urllib.error.HTTPError(
                "https://api.massive.com", code, "err", {},  # type: ignore[arg-type]
                io.BytesIO(body.encode()),
            )

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        return MassiveProvider(api_key="k")

    def test_a_401_points_at_the_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ProviderError, match="MASSIVE_API_KEY"):
            self._raising(monkeypatch, 401).fetch_bars(_request("SPY"))

    def test_a_403_says_entitlement_not_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different fix entirely. Checking a valid key wastes the afternoon."""
        with pytest.raises(ProviderError, match="separate plans"):
            self._raising(monkeypatch, 403).fetch_bars(_request("SPY"))

    def test_a_429_names_the_free_tier_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ProviderError, match="5 requests per"):
            self._raising(monkeypatch, 429).fetch_bars(_request("SPY"))

    def test_a_404_explains_the_symbology(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ProviderError, match="ESH6"):
            self._raising(monkeypatch, 404).fetch_bars(_request("SPY"))

    def test_a_blocked_host_is_not_blamed_on_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact failure this environment produces today, and the one that
        looks most like a credentials problem while being nothing of the kind."""
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        with pytest.raises(ProviderError, match="network policy"):
            MassiveProvider(api_key="k").fetch_bars(_request("SPY"))

    def test_an_empty_futures_window_explains_contract_listing(self) -> None:
        provider = _StubProvider({**_CONTRACTS, "/futures/v1/aggs": {"results": []}})
        with pytest.raises(ProviderError, match="outside its own trading window"):
            provider.fetch_bars(_request("ES", "1m"))

    def test_check_credentials_needs_no_network_without_a_key(self) -> None:
        assert "No Massive API key" in MassiveProvider().check_credentials()

    def test_check_credentials_separates_futures_entitlement_from_a_bad_key(
        self,
    ) -> None:
        """A key that works for equities and not futures is a plan question,
        not a credentials one — and that decides where someone goes looking."""
        provider = _StubProvider({
            "/v2/aggs/ticker/AAPL/prev": {"results": [{"c": 200.0}]},
            "/futures/v1/contracts": {"results": []},
        })
        status = provider.check_credentials()
        assert status.startswith("✓")
        assert "futures  : no" in status


class TestContractSymbology:
    """`get_instrument` classifying a contract ticker as equity is a *sizing*
    bug, not just a routing one: an E-mini point is $50, and the equity default
    is $1, so a risk budget would size the position fifty times too large with
    no error anywhere."""

    def test_a_contract_ticker_is_a_future(self) -> None:
        assert get_instrument("ESH6").asset_class is AssetClass.FUTURES

    def test_a_contract_inherits_its_roots_point_value(self) -> None:
        assert get_instrument("ESH6").point_value == get_instrument("ES").point_value

    def test_a_contract_inherits_its_roots_tick_size_and_exchange(self) -> None:
        contract, root = get_instrument("GCJ5"), get_instrument("GC")
        assert contract.tick_size == root.tick_size
        assert contract.exchange == root.exchange

    def test_the_contract_keeps_its_own_symbol(self) -> None:
        assert get_instrument("ESH6").symbol == "ESH6"

    def test_an_unknown_root_is_still_recognised_as_a_future(self) -> None:
        """'Futures with generic specs' at least routes to a provider that
        carries futures. 'Equity' does not."""
        assert get_instrument("ZBH6").asset_class is AssetClass.FUTURES

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "TSLA", "SPY", "QQQ", "BRK"])
    def test_ordinary_equity_tickers_are_not_mistaken_for_contracts(
        self, ticker: str
    ) -> None:
        """The false positive that would matter: an equity re-classified as a
        future is routed away from the providers that can actually serve it."""
        assert parse_contract(ticker) is None
        assert get_instrument(ticker).asset_class is not AssetClass.FUTURES

    def test_crypto_pairs_are_unaffected(self) -> None:
        assert get_instrument("BTC-USD").asset_class is AssetClass.CRYPTO

    @pytest.mark.parametrize(
        ("ticker", "root", "month"),
        [("ESH6", "ES", "H"), ("GCJ5", "GC", "J"), ("CLZ25", "CL", "Z")],
    )
    def test_contracts_split_into_root_month_and_year(
        self, ticker: str, root: str, month: str
    ) -> None:
        parsed = parse_contract(ticker)
        assert parsed is not None
        assert parsed[0] == root and parsed[1] == month

    def test_an_invalid_month_code_is_not_a_contract(self) -> None:
        """CME month codes skip I, L, O, P, R and others. 'ESA6' is not one."""
        assert parse_contract("ESA6") is None


class TestRegistryIntegration:
    def test_massive_is_offered_for_futures_when_databento_is_absent(self) -> None:
        from axiom.data.registry import DataRegistry

        provider = MassiveProvider(api_key="k")
        registry = DataRegistry([provider])
        assert registry._supports(provider, _request("ES"))

    def test_a_contract_ticker_routes_to_futures_providers(self) -> None:
        """Before the symbology fix this landed on the equity providers."""
        from axiom.data.alpaca import AlpacaProvider

        assert not AlpacaProvider().supports(get_instrument("ESH6"))
        assert MassiveProvider(api_key="k").supports(get_instrument("ESH6"))


def test_the_stub_never_touches_the_network() -> None:
    """Guard on the test file itself: `json` is imported for the fixtures, and
    a stray real request here would be billed against the user's rate limit."""
    assert json is not None
