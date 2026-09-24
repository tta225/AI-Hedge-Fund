"""Alpaca adapter. No network — the HTTP layer is stubbed.

The symbol-translation tests are the ones that earn their keep. Alpaca's crypto
endpoint accepts only ``BASE/QUOTE`` and answers ``BTC-USD`` with a 400, and the
registry swallows that: it catches the ``ProviderError``, falls through to
Coinbase, and returns bars. Nothing looks broken from the outside — Alpaca's
crypto data is just silently never used.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from axiom.core.provenance import DataKind
from axiom.core.timeframe import Timeframe
from axiom.core.types import AssetClass, Instrument, get_instrument
from axiom.data.alpaca import AlpacaProvider
from axiom.data.base import BarRequest, ProviderError

BTC = Instrument("BTC-USD", AssetClass.CRYPTO, "COINBASE", session_tz="UTC")


def _request(instrument: Instrument, timeframe: str = "1h") -> BarRequest:
    return BarRequest(
        instrument=instrument,
        timeframe=Timeframe.parse(timeframe),
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )


def _keyed() -> AlpacaProvider:
    return AlpacaProvider(api_key="key", secret_key="secret")


class TestVenueSymbol:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("BTC-USD", "BTC/USD"),
            ("btc-usd", "BTC/USD"),
            ("ETH-USDT", "ETH/USDT"),
            ("BTC/USD", "BTC/USD"),
            ("BTCUSD", "BTC/USD"),
            ("ETHUSDT", "ETH/USDT"),
            ("SOL", "SOL/USD"),
        ],
    )
    def test_crypto_is_written_the_way_alpaca_spells_it(
        self, symbol: str, expected: str
    ) -> None:
        instrument = Instrument(symbol.upper(), AssetClass.CRYPTO, session_tz="UTC")
        assert _keyed().venue_symbol(_request(instrument)) == expected

    @pytest.mark.parametrize("symbol", ["SPY", "QQQ", "AAPL"])
    def test_equities_pass_through_untouched(self, symbol: str) -> None:
        assert _keyed().venue_symbol(_request(get_instrument(symbol))) == symbol


class _StubProvider(AlpacaProvider):
    """Records the request Alpaca would have seen, and serves a fixed page."""

    def __init__(self, bars: dict[str, list[dict[str, object]]]) -> None:
        super().__init__(api_key="key", secret_key="secret")
        self.bars = bars
        self.urls: list[str] = []
        self.params: list[dict[str, str]] = []
        self.symbols: list[str] = []

    def _paginate(
        self, url: str, params: dict[str, str], symbol: str
    ) -> list[dict[str, object]]:
        self.urls.append(url)
        self.params.append(dict(params))
        self.symbols.append(symbol)
        return list(self.bars.get(symbol, []))


_PAGE = [
    {"t": "2026-01-01T00:00:00Z", "o": 95.0, "h": 110.0,
     "l": 90.0, "c": 105.0, "v": 42.0}
]


class TestCryptoRequests:
    def test_the_query_asks_for_the_slash_form(self) -> None:
        provider = _StubProvider({"BTC/USD": _PAGE})
        provider.fetch_bars(_request(BTC))
        assert provider.params[0]["symbols"] == "BTC/USD"

    def test_bars_are_read_from_the_slash_keyed_response(self) -> None:
        """Alpaca keys the payload by the symbol as sent.

        Querying one spelling and reading another returns an empty batch, which
        surfaces as 'no bars' rather than as the mismatch it is.
        """
        provider = _StubProvider({"BTC/USD": _PAGE})
        series = provider.fetch_bars(_request(BTC))
        assert len(series) == 1
        assert float(series.highs[0]) == 110.0
        assert float(series.lows[0]) == 90.0

    def test_crypto_hits_the_crypto_endpoint(self) -> None:
        provider = _StubProvider({"BTC/USD": _PAGE})
        provider.fetch_bars(_request(BTC))
        assert "crypto" in provider.urls[0]

    def test_crypto_sends_no_feed_or_adjustment(self) -> None:
        """Neither parameter exists on the crypto endpoint."""
        provider = _StubProvider({"BTC/USD": _PAGE})
        provider.fetch_bars(_request(BTC))
        assert "feed" not in provider.params[0]
        assert "adjustment" not in provider.params[0]

    def test_provenance_is_real_and_names_the_instrument(self) -> None:
        provider = _StubProvider({"BTC/USD": _PAGE})
        series = provider.fetch_bars(_request(BTC))
        assert series.provenance.kind is DataKind.REAL
        assert series.provenance.is_evidential
        assert "BTC-USD" in series.provenance.detail


class TestEquityRequests:
    def test_equity_hits_the_stocks_endpoint_with_the_plain_symbol(self) -> None:
        provider = _StubProvider({"SPY": _PAGE})
        provider.fetch_bars(_request(get_instrument("SPY")))
        assert "stocks" in provider.urls[0]
        assert provider.params[0]["symbols"] == "SPY"

    def test_equity_carries_feed_and_adjustment(self) -> None:
        provider = _StubProvider({"SPY": _PAGE})
        provider.fetch_bars(_request(get_instrument("SPY")))
        assert provider.params[0]["feed"] == "iex"
        assert provider.params[0]["adjustment"] == "split"


class TestRefusals:
    def test_rejects_an_unsupported_timeframe(self) -> None:
        with pytest.raises(ProviderError, match="unsupported timeframe"):
            _keyed().fetch_bars(_request(BTC, "3h"))

    def test_an_empty_window_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(ProviderError, match="no bars"):
            _StubProvider({}).fetch_bars(_request(BTC))
