"""Coinbase adapter. No network — the HTTP layer is stubbed.

The column-order test is the one that earns its keep. Coinbase returns
``[time, low, high, open, close, volume]``, which is not the conventional order,
and getting it wrong swaps every high and low. Nothing downstream would raise:
the series would validate, the ICT engine would run, and every level it produced
would be wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from axiom.core.provenance import DataKind
from axiom.core.timeframe import Timeframe
from axiom.core.types import AssetClass, Instrument, get_instrument
from axiom.data.base import BarRequest, ProviderError
from axiom.data.coinbase import CoinbaseProvider

BTC = Instrument("BTC-USD", AssetClass.CRYPTO, "COINBASE", session_tz="UTC")


class TestProductId:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("BTC-USD", "BTC-USD"),
            ("btc-usd", "BTC-USD"),
            ("BTC", "BTC-USD"),
            ("BTCUSD", "BTC-USD"),
            ("ETHUSDT", "ETH-USDT"),
            ("SOL", "SOL-USD"),
        ],
    )
    def test_resolves(self, symbol: str, expected: str) -> None:
        assert CoinbaseProvider().product_id(symbol) == expected

    def test_explicit_map_wins(self) -> None:
        provider = CoinbaseProvider(product_map={"XBT": "BTC-USD"})
        assert provider.product_id("XBT") == "BTC-USD"


class TestRefusals:
    def test_rejects_non_crypto_without_a_request(self) -> None:
        """Falls through fast so a registry lookup for SPY costs no round trip."""
        request = BarRequest.lookback(get_instrument("SPY"), "1h", days=5)
        with pytest.raises(ProviderError, match="crypto only"):
            CoinbaseProvider().fetch_bars(request)

    def test_rejects_an_unsupported_timeframe(self) -> None:
        request = BarRequest.lookback(BTC, "4h", days=5)
        with pytest.raises(ProviderError, match="unsupported timeframe"):
            CoinbaseProvider().fetch_bars(request)

    def test_names_the_supported_timeframes(self) -> None:
        request = BarRequest.lookback(BTC, "4h", days=5)
        with pytest.raises(ProviderError, match="15m"):
            CoinbaseProvider().fetch_bars(request)


class _StubProvider(CoinbaseProvider):
    """Serves a fixed page, then empties — as the real endpoint does at the edge."""

    def __init__(self, pages: list[list[list[float]]]) -> None:
        super().__init__()
        self.pages = list(pages)
        self.calls = 0

    def _get(self, url: str) -> list[list[float]]:
        self.calls += 1
        return self.pages.pop(0) if self.pages else []


class TestParsing:
    def _request(self) -> BarRequest:
        return BarRequest(
            instrument=BTC,
            timeframe=Timeframe.parse("1h"),
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 2, tzinfo=UTC),
        )

    def test_maps_the_unconventional_column_order(self) -> None:
        # [time, low, high, open, close, volume] — low BEFORE high.
        page = [[1767225600, 90.0, 110.0, 95.0, 105.0, 42.0]]
        series = _StubProvider([page]).fetch_bars(self._request())
        assert float(series.highs[0]) == 110.0
        assert float(series.lows[0]) == 90.0
        assert float(series.opens[0]) == 95.0
        assert float(series.closes[0]) == 105.0
        assert float(series.volumes[0]) == 42.0

    def test_drops_duplicate_timestamps_across_pages(self) -> None:
        """Pages overlap at their boundaries.

        A duplicated bar double-counts volume and can fabricate an equal high,
        which the liquidity detector reads as a pool that never existed.
        """
        bar = [1767225600, 90.0, 110.0, 95.0, 105.0, 42.0]
        later = [1767229200, 91.0, 111.0, 96.0, 106.0, 43.0]
        series = _StubProvider([[bar, later], [bar]]).fetch_bars(self._request())
        assert len(series) == 2

    def test_bars_come_back_sorted(self) -> None:
        newest = [1767229200, 91.0, 111.0, 96.0, 106.0, 43.0]
        oldest = [1767225600, 90.0, 110.0, 95.0, 105.0, 42.0]
        series = _StubProvider([[newest, oldest]]).fetch_bars(self._request())
        assert series.index[0] < series.index[1]

    def test_provenance_is_real_and_names_the_venue(self) -> None:
        page = [[1767225600, 90.0, 110.0, 95.0, 105.0, 42.0]]
        series = _StubProvider([page]).fetch_bars(self._request())
        assert series.provenance.kind is DataKind.REAL
        assert series.provenance.is_evidential
        assert "coinbase" in series.provenance.detail

    def test_an_empty_window_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(ProviderError, match="no candles"):
            _StubProvider([]).fetch_bars(self._request())


class TestInstrumentResolution:
    def test_unknown_dash_pair_resolves_as_crypto(self) -> None:
        """Asset class decides which providers serve a symbol — not cosmetic."""
        assert get_instrument("DOGE-USD").asset_class is AssetClass.CRYPTO

    def test_unknown_plain_symbol_stays_equity(self) -> None:
        assert get_instrument("AAPL").asset_class is AssetClass.EQUITY

    def test_crypto_uses_a_utc_session(self) -> None:
        assert get_instrument("BTC-USD").session_tz == "UTC"


class TestReachabilityProbe:
    """`is_available` is unconditionally true, so only a probe knows the truth.

    On a restricted network those two answers disagree, and the disagreement is
    exactly what `data-check` exists to surface.
    """

    def test_available_does_not_imply_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.error

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

        monkeypatch.setattr("urllib.request.urlopen", refuse)
        provider = CoinbaseProvider()

        assert provider.is_available() is True
        assert provider.check_reachable().startswith("✗")
        assert "403" in provider.check_reachable()

    def test_reports_reachable_when_the_endpoint_answers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Response:
            def read(self) -> bytes:
                return b'{"id": "BTC-USD", "status": "online"}'

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response())
        assert CoinbaseProvider().check_reachable().startswith("✓")

    def test_an_http_error_names_the_status_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.error

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise urllib.error.HTTPError("url", 503, "unavailable", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr("urllib.request.urlopen", refuse)
        assert "503" in CoinbaseProvider().check_reachable()
