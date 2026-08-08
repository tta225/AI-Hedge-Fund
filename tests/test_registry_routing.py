"""Provider routing by asset class.

This file exists because of one bug, and the bug did not raise.

Ticker symbols are not unique across asset classes. ``ES`` is the E-mini S&P on
CME **and** Eversource Energy on the NYSE. ``CL`` is crude oil **and**
Colgate-Palmolive. ``GC`` is gold **and** Gabelli Equity Trust. An equity
provider asked for ``ES`` therefore does not fail — it returns a utility trading
near $71, a full year of clean bars, and a backtest that runs to completion and
prints a Sharpe ratio.

Every other safety mechanism in this platform is downstream of that and cannot
help. Provenance says REAL, and it is real. The integrity checks pass, because
the data is not corrupt. Walk-forward validation happily validates. The result
is a correct measurement of the wrong instrument, and nothing anywhere says so.

The only place to catch it is before the request is made, by asking whether the
provider carries that asset class at all. These tests pin that.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import AssetClass, get_instrument
from axiom.data.alpaca import AlpacaProvider
from axiom.data.base import BarRequest, BaseProvider, ProviderError
from axiom.data.coinbase import CoinbaseProvider
from axiom.data.databento import DatabentoProvider
from axiom.data.providers import CSVProvider, YFinanceProvider
from axiom.data.registry import DataRegistry, default_registry

ES = get_instrument("ES")
SPY = get_instrument("SPY")
BTC = get_instrument("BTC-USD")


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-03-02", periods=5, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0},
        index=index,
    )


class _Recording(BaseProvider):
    """A provider that answers everything and remembers being asked."""

    def __init__(self, name: str, classes: frozenset[AssetClass] | None = None) -> None:
        self.name = name
        if classes is not None:
            self.asset_classes = classes
        self.asked: list[str] = []

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        self.asked.append(request.instrument.symbol)
        return _frame()

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance.real(self.name)


def _request(instrument: Any = ES) -> BarRequest:
    return BarRequest.lookback(instrument, "15m", 5)


class TestDeclaredCoverage:
    """What each shipped provider claims it can serve."""

    def test_alpaca_does_not_claim_futures(self) -> None:
        """The specific instance of the bug: Alpaca answers 'ES' with a utility."""
        assert not AlpacaProvider().supports(ES)

    def test_alpaca_serves_equities_and_crypto(self) -> None:
        alpaca = AlpacaProvider()
        assert alpaca.supports(SPY) and alpaca.supports(BTC)

    def test_coinbase_serves_only_crypto(self) -> None:
        coinbase = CoinbaseProvider()
        assert coinbase.supports(BTC)
        assert not coinbase.supports(SPY)
        assert not coinbase.supports(ES)

    def test_yfinance_does_not_claim_futures(self) -> None:
        """Yahoo does carry futures, but under suffixed symbology ('ES=F').
        A bare 'ES' resolves to Eversource there too."""
        assert not YFinanceProvider().supports(ES)

    def test_databento_is_the_one_that_carries_futures(self) -> None:
        assert DatabentoProvider(api_key="db-x").supports(ES)

    def test_a_local_cache_is_not_second_guessed(self) -> None:
        """A CSV the user placed themselves is theirs to name. Declaring asset
        classes for it would refuse files that are perfectly correct."""
        assert CSVProvider("data/cache").supports(ES)


class TestRouting:
    def test_a_futures_request_skips_equity_providers_entirely(self) -> None:
        """Skipped, not tried-and-failed: they would have succeeded."""
        equity = _Recording("equity-only", frozenset({AssetClass.EQUITY}))
        futures = _Recording("futures", frozenset({AssetClass.FUTURES}))
        registry = DataRegistry([equity, futures])

        registry.fetch_bars(_request(ES))
        assert equity.asked == []
        assert futures.asked == ["ES"]

    def test_an_equity_request_still_reaches_the_equity_provider(self) -> None:
        equity = _Recording("equity-only", frozenset({AssetClass.EQUITY}))
        registry = DataRegistry([_Recording("crypto", frozenset({AssetClass.CRYPTO})), equity])
        registry.fetch_bars(_request(SPY))
        assert equity.asked == ["SPY"]

    def test_an_undeclared_provider_is_still_tried(self) -> None:
        """Empty means 'all'. Existing providers must not be narrowed by
        accident just because the attribute was added."""
        anything = _Recording("anything")
        DataRegistry([anything]).fetch_bars(_request(ES))
        assert anything.asked == ["ES"]

    def test_a_provider_without_the_method_at_all_is_still_tried(self) -> None:
        """The protocol grew a method. A third-party provider written against
        the old one should keep working rather than silently never be called."""

        class _Legacy:
            name = "legacy"

            def is_available(self) -> bool:
                return True

            def fetch_bars(self, request: BarRequest) -> OHLCVSeries:
                return OHLCVSeries.from_dataframe(
                    _frame(), request.instrument, request.timeframe,
                    Provenance.real("legacy"),
                )

        series = DataRegistry([_Legacy()]).fetch_bars(_request(ES))
        assert series.provenance.source == "legacy"


class TestFailureMessages:
    def test_the_skip_reason_names_the_asset_class(self) -> None:
        registry = DataRegistry([_Recording("equity", frozenset({AssetClass.EQUITY}))])
        with pytest.raises(ProviderError, match="does not carry futures"):
            registry.fetch_bars(_request(ES))

    def test_a_futures_request_with_no_futures_provider_says_what_to_install(
        self,
    ) -> None:
        """A dead end with no next step is where someone concludes the platform
        cannot do futures at all."""
        registry = DataRegistry([_Recording("equity", frozenset({AssetClass.EQUITY}))])
        with pytest.raises(ProviderError, match="DATABENTO_API_KEY"):
            registry.fetch_bars(_request(ES))

    def test_an_equity_failure_does_not_suggest_databento(self) -> None:
        """Advice attached to every failure is advice nobody reads."""
        registry = DataRegistry([_Recording("crypto", frozenset({AssetClass.CRYPTO}))])
        with pytest.raises(ProviderError) as caught:
            registry.fetch_bars(_request(SPY))
        assert "DATABENTO_API_KEY" not in str(caught.value)

    def test_an_empty_registry_says_so(self) -> None:
        with pytest.raises(ProviderError, match="no data providers registered"):
            DataRegistry().fetch_bars(_request(SPY))


class TestDefaultRegistry:
    def test_no_default_provider_will_answer_a_futures_symbol_wrongly(self) -> None:
        """The regression guard. If any provider in the default set starts
        claiming futures without being able to serve them, this fails.
        """
        registry = default_registry()
        claimants = [
            p.name for p in registry.providers
            if getattr(p, "supports", lambda i: True)(ES)
        ]
        assert claimants == ["databento"]

    def test_futures_routing_survives_databento_being_uninstalled(self) -> None:
        """The common case for this user today. It must produce the install
        instruction, not bars from a utility stock."""
        registry = default_registry()
        with pytest.raises(ProviderError) as caught:
            registry.fetch_bars(_request(ES))
        message = str(caught.value)
        assert "does not carry futures" in message
        assert "FUTURES_SETUP" in message

    def test_equity_routing_is_unchanged_by_the_futures_work(self) -> None:
        registry = default_registry()
        equity_capable = [
            p.name for p in registry.providers
            if getattr(p, "supports", lambda i: True)(SPY)
        ]
        assert "alpaca" in equity_capable
        assert "yfinance" in equity_capable

    def test_crypto_routing_is_unchanged(self) -> None:
        registry = default_registry()
        crypto_capable = [
            p.name for p in registry.providers
            if getattr(p, "supports", lambda i: True)(BTC)
        ]
        assert "coinbase" in crypto_capable
        assert "alpaca" in crypto_capable
