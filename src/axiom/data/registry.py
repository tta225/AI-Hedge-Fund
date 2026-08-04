"""Provider registry with ordered fallback."""

from __future__ import annotations

from axiom.core.series import OHLCVSeries
from axiom.core.timeframe import Timeframe
from axiom.core.types import Instrument
from axiom.data.base import BarRequest, MarketDataProvider, ProviderError
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

    def fetch_bars(self, request: BarRequest) -> OHLCVSeries:
        if not self._providers:
            raise ProviderError("no data providers registered")
        failures: list[str] = []
        for provider in self._providers:
            if not provider.is_available():
                failures.append(f"{provider.name}: unavailable")
                continue
            try:
                return provider.fetch_bars(request)
            except ProviderError as exc:
                failures.append(f"{provider.name}: {exc}")
        raise ProviderError(
            f"all providers failed for {request.instrument.symbol} {request.timeframe}\n  - "
            + "\n  - ".join(failures)
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
    """Registry wired to the free adapters: local CSV first, then yfinance."""
    registry = DataRegistry()
    if data_root:
        registry.register(CSVProvider(data_root))
    registry.register(YFinanceProvider())
    return registry


def research_registry(seed: int = 7, start_price: float = 5000.0) -> DataRegistry:
    """Offline registry backed solely by the synthetic generator.

    Everything it returns is stamped SYNTHETIC and will be rejected by any path
    that reports performance.
    """
    return DataRegistry([SyntheticProvider(seed=seed, start_price=start_price)])
