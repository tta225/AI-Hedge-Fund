"""Market data provider abstraction.

Every provider returns an :class:`~axiom.core.series.OHLCVSeries` carrying its
own :class:`~axiom.core.provenance.Provenance`. Swapping yfinance for Databento
is an adapter change; nothing downstream of this interface moves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

import pandas as pd

from axiom.core.series import OHLCVSeries
from axiom.core.timeframe import Timeframe
from axiom.core.types import Instrument


class ProviderError(RuntimeError):
    """A provider could not satisfy a request."""


class ProviderUnavailableError(ProviderError):
    """The provider's dependency or credentials are missing."""


@dataclass(frozen=True, slots=True)
class BarRequest:
    """A request for historical bars."""

    instrument: Instrument
    timeframe: Timeframe
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("BarRequest start/end must be timezone-aware")
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must precede end {self.end}")

    @classmethod
    def lookback(
        cls,
        instrument: Instrument,
        timeframe: str | Timeframe,
        days: int,
        end: datetime | None = None,
    ) -> BarRequest:
        stop = end or datetime.now(UTC)
        return cls(
            instrument=instrument,
            timeframe=Timeframe.parse(timeframe),
            start=stop - timedelta(days=days),
            end=stop,
        )

    @property
    def expected_bars(self) -> int:
        """Upper bound on bar count, ignoring session gaps. Sizing hint only."""
        span = (self.end - self.start).total_seconds()
        return int(span // self.timeframe.seconds)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Structural interface every data source satisfies."""

    name: str

    def is_available(self) -> bool:
        """False when the adapter's dependency or credentials are missing."""
        ...

    def fetch_bars(self, request: BarRequest) -> OHLCVSeries:
        """Return validated bars, or raise :class:`ProviderError`."""
        ...


class BaseProvider(ABC):
    """Convenience base implementing the common request plumbing."""

    name: str = "base"

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        """Return a raw OHLCV frame indexed by timestamp."""

    @abstractmethod
    def _provenance(self, request: BarRequest):  # type: ignore[no-untyped-def]
        """Return the provenance stamp for a satisfied request."""

    def fetch_bars(self, request: BarRequest) -> OHLCVSeries:
        if not self.is_available():
            raise ProviderUnavailableError(f"provider {self.name!r} is not available")
        raw = self._fetch_raw(request)
        if raw is None or len(raw) == 0:
            raise ProviderError(
                f"{self.name}: no bars for {request.instrument.symbol} "
                f"{request.timeframe} between {request.start.date()} and {request.end.date()}"
            )
        return OHLCVSeries.from_dataframe(
            raw, request.instrument, request.timeframe, self._provenance(request)
        )

    def fetch(
        self,
        instrument: Instrument,
        timeframe: str | Timeframe,
        *,
        days: int = 30,
        end: datetime | None = None,
    ) -> OHLCVSeries:
        """Ergonomic wrapper around :meth:`fetch_bars`."""
        return self.fetch_bars(BarRequest.lookback(instrument, timeframe, days, end))
