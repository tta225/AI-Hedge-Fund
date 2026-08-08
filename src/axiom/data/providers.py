"""Concrete market data adapters.

Each adapter imports its dependency lazily so the core platform installs and
runs with none of them present.

Licensing note
--------------
:class:`OpenBBProvider` wraps an **AGPL-3.0** package. It is isolated here, is
not imported unless explicitly constructed, and is not a declared core
dependency. Enabling it in a distributed or network-served product has copyleft
consequences for the surrounding code — see ``docs/LICENSING.md``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from axiom.core.provenance import Provenance
from axiom.core.types import AssetClass
from axiom.data.base import BarRequest, BaseProvider, ProviderError, ProviderUnavailableError

# yfinance interval strings, keyed by our timeframe label.
_YF_INTERVALS = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "1d": "1d", "1w": "1wk",
}


class CSVProvider(BaseProvider):
    """Reads bars from local CSV files — the offline path for vendor exports.

    Files are looked up as ``{root}/{SYMBOL}_{timeframe}.csv`` and must contain
    a timestamp column plus open/high/low/close (volume optional).
    """

    name = "csv"

    def __init__(self, root: str | Path, timestamp_column: str = "timestamp") -> None:
        self.root = Path(root)
        self.timestamp_column = timestamp_column

    def is_available(self) -> bool:
        return self.root.is_dir()

    def path_for(self, request: BarRequest) -> Path:
        return self.root / f"{request.instrument.symbol}_{request.timeframe}.csv"

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance.real(self.name, detail=str(self.path_for(request)))

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        path = self.path_for(request)
        if not path.exists():
            raise ProviderError(f"{self.name}: no such file {path}")
        frame = pd.read_csv(path)
        cols = {c.lower(): c for c in frame.columns}
        ts_col = cols.get(self.timestamp_column.lower()) or cols.get("date") or frame.columns[0]
        frame = frame.set_index(pd.to_datetime(frame[ts_col], utc=True)).drop(columns=[ts_col])
        return frame.loc[str(request.start) : str(request.end)]

    @staticmethod
    def write(series, path: str | Path) -> Path:  # type: ignore[no-untyped-def]
        """Persist an :class:`OHLCVSeries` in the format this provider reads."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = series.df.copy()
        frame.index.name = "timestamp"
        frame.to_csv(out, quoting=csv.QUOTE_MINIMAL)
        return out


class YFinanceProvider(BaseProvider):
    """Free equity/ETF/index/FX bars via Yahoo Finance.

    Fine for research and prototyping. Intraday history is capped (roughly 60
    days at ``1m``), bars are unadjusted-close by default, and the endpoint is
    unofficial — do not route production decisions through it.
    """

    name = "yfinance"
    #: Yahoo does carry futures, but under its own suffixed symbology
    #: ("ES=F"). A bare "ES" resolves to Eversource Energy instead, so
    #: futures are excluded rather than silently mis-resolved.
    asset_classes = frozenset({AssetClass.EQUITY, AssetClass.INDEX, AssetClass.FX})

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False
        return True

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance.real(
            self.name, detail=f"{request.instrument.symbol}@{request.timeframe}"
        )

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - guarded by is_available
            raise ProviderUnavailableError("yfinance is not installed") from exc

        interval = _YF_INTERVALS.get(request.timeframe.label)
        if interval is None:
            raise ProviderError(
                f"{self.name}: unsupported timeframe {request.timeframe}; "
                f"supported: {sorted(_YF_INTERVALS)}"
            )
        raw = yf.download(
            request.instrument.symbol,
            start=request.start,
            end=request.end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
        )
        if raw is None or raw.empty:
            raise ProviderError(f"{self.name}: empty response for {request.instrument.symbol}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower() for c in raw.columns]
        return raw[["open", "high", "low", "close", "volume"]]


class OpenBBProvider(BaseProvider):
    """Optional adapter over the OpenBB Platform (**AGPL-3.0**).

    Constructing this class is an explicit, auditable act. The rest of AXIOM
    never imports it, so a proprietary build simply never touches AGPL code.
    """

    name = "openbb"

    def __init__(self, acknowledge_agpl: bool = False) -> None:
        if not acknowledge_agpl:
            raise ProviderError(
                "OpenBB is licensed AGPL-3.0. Linking it into a distributed or "
                "network-served product places copyleft obligations on the "
                "surrounding code. Pass acknowledge_agpl=True to proceed for "
                "internal research use, and read docs/LICENSING.md first."
            )
        self._acknowledged = True

    def is_available(self) -> bool:
        try:
            import openbb  # noqa: F401
        except ImportError:
            return False
        return True

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance.real(self.name, detail=f"AGPL-3.0 source; {request.instrument.symbol}")

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        try:
            from openbb import obb
        except ImportError as exc:  # pragma: no cover - guarded by is_available
            raise ProviderUnavailableError("openbb is not installed") from exc

        result = obb.equity.price.historical(
            symbol=request.instrument.symbol,
            start_date=request.start.date().isoformat(),
            end_date=request.end.date().isoformat(),
            interval=request.timeframe.label,
        )
        frame = result.to_dataframe()
        frame.columns = [str(c).lower() for c in frame.columns]
        return frame
