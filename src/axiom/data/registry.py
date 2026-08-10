"""Provider registry with ordered fallback."""

from __future__ import annotations

from axiom.core.series import OHLCVSeries
from axiom.core.timeframe import Timeframe
from axiom.core.types import AssetClass, Instrument
from axiom.data.alpaca import AlpacaProvider
from axiom.data.base import BarRequest, MarketDataProvider, ProviderError
from axiom.data.coinbase import CoinbaseProvider
from axiom.data.databento import DatabentoProvider
from axiom.data.massive import MassiveProvider
from axiom.data.providers import CSVProvider, YFinanceProvider
from axiom.data.synthetic import SyntheticProvider


class DataRegistry:
    """Tries providers in priority order and returns the first success.

    Synthetic is never included implicitly: falling back to generated bars when
    a real feed fails is exactly how a fabricated backtest happens. Add it only
    by explicit registration.
    """

    def __init__(self, providers: list[MarketDataProvider] | None = None) -> None:
        self._providers: list[MarketDataProvider] = list(providers or [])

    def register(self, provider: MarketDataProvider, *, first: bool = False) -> None:
        if first:
            self._providers.insert(0, provider)
        else:
            self._providers.append(provider)

    @property
    def providers(self) -> tuple[MarketDataProvider, ...]:
        return tuple(self._providers)

    def available(self) -> tuple[MarketDataProvider, ...]:
        return tuple(p for p in self._providers if p.is_available())

    @staticmethod
    def _supports(provider: MarketDataProvider, request: BarRequest) -> bool:
        supports = getattr(provider, "supports", None)
        return True if supports is None else bool(supports(request.instrument))

    def fetch_bars(self, request: BarRequest) -> OHLCVSeries:
        """Try each provider in turn, skipping those that cannot serve the ask.

        The asset-class check is not an optimisation. Fallback-on-failure only
        protects against providers that *fail*, and the dangerous case here does
        not fail: ticker symbols are not unique across asset classes, so an
        equity provider asked for ``ES`` returns Eversource Energy — a real
        utility at a real price — and the backtest runs to completion on the
        wrong instrument. Nothing downstream can detect that, because there is
        nothing wrong with the data. It is only wrong about what it is.
        """
        if not self._providers:
            raise ProviderError("no data providers registered")
        failures: list[str] = []
        for provider in self._providers:
            if not self._supports(provider, request):
                failures.append(
                    f"{provider.name}: does not carry "
                    f"{request.instrument.asset_class.value}"
                )
                continue
            if not provider.is_available():
                failures.append(f"{provider.name}: unavailable")
                continue
            try:
                return provider.fetch_bars(request)
            except ProviderError as exc:
                failures.append(f"{provider.name}: {exc}")

        hint = ""
        if request.instrument.asset_class is AssetClass.FUTURES:
            hint = (
                "\n\nNo registered provider carries futures. Two do: Databento "
                '(`python3 -m pip install -e ".[databento]"`, DATABENTO_API_KEY) '
                "and Massive (MASSIVE_API_KEY, no install needed). "
                "See docs/FUTURES_SETUP.md."
            )
        raise ProviderError(
            f"all providers failed for {request.instrument.symbol} {request.timeframe}\n  - "
            + "\n  - ".join(failures)
            + hint
        )

    def fetch(
        self,
        instrument: Instrument,
        timeframe: str | Timeframe,
        *,
        days: int = 30,
    ) -> OHLCVSeries:
        return self.fetch_bars(BarRequest.lookback(instrument, timeframe, days))


def default_registry(data_root: str | None = None) -> DataRegistry:
    """Registry wired to the available adapters, best first.

    Order: local CSV (fastest, fully controlled) → Databento → Massive → Alpaca
    → Coinbase (keyless, crypto only) → yfinance (unofficial endpoint, last
    resort). Every one is skipped silently when unkeyed. Synthetic is
    deliberately absent; see :class:`DataRegistry`.

    The two futures sources sit at the top because they are the only ones that
    can be *right* about a futures symbol, and each provider now declares the
    asset classes it serves — so an equity request skips straight past them at
    no cost.

    Databento outranks Massive for futures on one property: it has continuous
    contract symbology with real roll rules, so a multi-year series is a series
    rather than a splice. Massive addresses individual contracts, so this
    adapter fetches a single front month and refuses to stitch. For a window
    inside one contract they are equivalent; across a roll they are not.
    """
    registry = DataRegistry()
    if data_root:
        registry.register(CSVProvider(data_root))
    registry.register(DatabentoProvider())
    registry.register(MassiveProvider())
    registry.register(AlpacaProvider())
    registry.register(CoinbaseProvider())
    registry.register(YFinanceProvider())
    return registry


def research_registry(seed: int = 7, start_price: float = 5000.0) -> DataRegistry:
    """Offline registry backed solely by the synthetic generator.

    Everything it returns is stamped SYNTHETIC and will be rejected by any path
    that reports performance.
    """
    return DataRegistry([SyntheticProvider(seed=seed, start_price=start_price)])
