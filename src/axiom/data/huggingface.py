"""Hugging Face dataset adapter.

Loads OHLCV bars from a dataset on the Hub into a validated
:class:`~axiom.core.series.OHLCVSeries`, so your own training data flows through
exactly the same validation, provenance, and backtesting path as any vendor feed.

Requirements
------------
* ``pip install 'axiom[huggingface]'``
* ``HF_TOKEN`` in the environment for private datasets (create one at
  https://huggingface.co/settings/tokens with *read* scope). Public datasets
  need no token.

Provenance
----------
Data from the Hub is stamped ``DataKind.REAL`` by default, because it is
presumed to be observed market data. **If your dataset is generated,
augmented, or partly synthetic, pass ``kind=DataKind.SYNTHETIC``** — otherwise
the platform will treat backtests over it as evidence. The provenance system is
only as honest as what it is told at the boundary.
"""

from __future__ import annotations

import pandas as pd

from axiom.core.credentials import get_credential
from axiom.core.provenance import DataKind, Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.timeframe import Timeframe
from axiom.core.types import Instrument, get_instrument
from axiom.data.base import BarRequest, BaseProvider, ProviderError, ProviderUnavailableError

#: Column names commonly used for the timestamp in market-data datasets.
_TIMESTAMP_CANDIDATES = (
    "timestamp", "time", "date", "datetime", "open_time", "ts", "t",
)

#: Aliases mapped onto the canonical OHLCV column names.
_COLUMN_ALIASES: dict[str, str] = {
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    "adj close": "close", "adj_close": "close", "vol": "volume",
    "price": "close", "last": "close",
}


class HuggingFaceProvider(BaseProvider):
    """Reads OHLCV bars from a Hugging Face dataset."""

    name = "huggingface"

    def __init__(
        self,
        dataset: str,
        *,
        split: str = "train",
        config: str | None = None,
        token: str | None = None,
        symbol_column: str | None = None,
        timestamp_column: str | None = None,
        kind: DataKind = DataKind.REAL,
        column_map: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            dataset: Hub repo id, e.g. ``"your-username/es-futures-1m"``.
            split: dataset split to load.
            config: dataset configuration name, if the dataset defines several.
            token: HF access token. Falls back to ``HF_TOKEN`` then
                ``HUGGINGFACE_TOKEN`` in the environment.
            symbol_column: column to filter on when one dataset holds many
                instruments.
            timestamp_column: override for timestamp detection.
            kind: provenance kind. Leave ``REAL`` only if these are genuinely
                observed market bars.
            column_map: explicit ``{source: canonical}`` mapping, applied before
                the built-in aliases. Use when the dataset's names are unusual.
        """
        self.dataset = dataset
        self.split = split
        self.config = config
        self.token = token or get_credential("HF_TOKEN", "HUGGINGFACE_TOKEN")
        self.symbol_column = symbol_column
        self.timestamp_column = timestamp_column
        self.kind = kind
        self.column_map = column_map or {}
        self._frame: pd.DataFrame | None = None

    def is_available(self) -> bool:
        try:
            import datasets  # noqa: F401
        except ImportError:
            return False
        return True

    def _provenance(self, request: BarRequest) -> Provenance:
        detail = f"{self.dataset}:{self.split}"
        if self.config:
            detail += f"/{self.config}"
        return Provenance(source=self.name, kind=self.kind, detail=detail)

    def load(self) -> pd.DataFrame:
        """Fetch and normalise the dataset once, then cache it."""
        if self._frame is not None:
            return self._frame

        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - guarded by is_available
            raise ProviderUnavailableError(
                "the huggingface extra is not installed "
                "(pip install 'axiom[huggingface]')"
            ) from exc

        try:
            data = load_dataset(
                self.dataset,
                self.config,
                split=self.split,
                token=self.token,
            )
        except Exception as exc:
            hint = (
                " (no HF_TOKEN found — private datasets need one)"
                if not self.token
                else ""
            )
            raise ProviderError(
                f"{self.name}: could not load {self.dataset!r}{hint}: {exc}"
            ) from exc

        frame = data.to_pandas()
        self._frame = self._normalise_columns(frame)
        return self._frame

    def _normalise_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        if self.column_map:
            out = out.rename(
                columns={k.lower(): v for k, v in self.column_map.items()}
            )
        out = out.rename(columns=_COLUMN_ALIASES)

        ts_col = self.timestamp_column.lower() if self.timestamp_column else None
        if ts_col is None:
            ts_col = next((c for c in _TIMESTAMP_CANDIDATES if c in out.columns), None)
        if ts_col is None:
            raise ProviderError(
                f"{self.name}: no timestamp column found in {sorted(out.columns)}. "
                "Pass timestamp_column= explicitly."
            )

        stamps = out[ts_col]
        if pd.api.types.is_numeric_dtype(stamps):
            # Epoch seconds, milliseconds, or nanoseconds — inferred from range.
            largest = float(stamps.max())
            unit = "s" if largest < 1e11 else "ms" if largest < 1e14 else "ns"
            index = pd.to_datetime(stamps, unit=unit, utc=True)
        else:
            index = pd.to_datetime(stamps, utc=True, format="mixed")

        out = out.set_index(pd.DatetimeIndex(index)).drop(columns=[ts_col])
        return out.sort_index()

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        frame = self.load()

        if self.symbol_column and self.symbol_column.lower() in frame.columns:
            column = frame[self.symbol_column.lower()].astype(str).str.upper()
            frame = frame[column == request.instrument.symbol.upper()]
            if frame.empty:
                available = sorted(set(column.unique()))[:20]
                raise ProviderError(
                    f"{self.name}: no rows for {request.instrument.symbol}; "
                    f"dataset contains {available}"
                )

        window = frame.loc[str(request.start) : str(request.end)]
        if window.empty:
            raise ProviderError(
                f"{self.name}: dataset covers "
                f"{frame.index[0]} to {frame.index[-1]}, which does not overlap "
                f"the requested {request.start} to {request.end}"
            )
        return window

    def inspect(self) -> str:
        """Summarise the dataset — run this first to confirm the schema."""
        frame = self.load()
        lines = [
            f"dataset : {self.dataset} ({self.split})",
            f"rows    : {len(frame):,}",
            f"span    : {frame.index[0]} → {frame.index[-1]}",
            f"columns : {sorted(frame.columns)}",
        ]
        missing = [c for c in ("open", "high", "low", "close") if c not in frame.columns]
        if missing:
            lines.append(
                f"MISSING : {missing} — pass column_map= to map your names onto these"
            )
        return "\n".join(lines)

    def series(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        *,
        instrument: Instrument | None = None,
    ) -> OHLCVSeries:
        """Load the whole dataset for one symbol, ignoring date bounds."""
        frame = self.load()
        target = instrument or get_instrument(symbol)
        if self.symbol_column and self.symbol_column.lower() in frame.columns:
            column = frame[self.symbol_column.lower()].astype(str).str.upper()
            frame = frame[column == symbol.upper()].drop(
                columns=[self.symbol_column.lower()]
            )
        detail = f"{self.dataset}:{self.split}"
        return OHLCVSeries.from_dataframe(
            frame,
            target,
            timeframe,
            Provenance(source=self.name, kind=self.kind, detail=detail),
        )


def describe_setup() -> str:
    """Instructions for wiring a Hugging Face dataset into AXIOM."""
    return (
        "1. pip install 'axiom[huggingface]'\n"
        "2. export HF_TOKEN=hf_...            # only for private datasets\n"
        "3. provider = HuggingFaceProvider('your-username/your-dataset')\n"
        "4. print(provider.inspect())          # confirm the schema first\n"
        "5. series = provider.series('ES', '1h')\n"
        "\n"
        "If the dataset is generated or augmented rather than observed market\n"
        "data, construct it with kind=DataKind.SYNTHETIC so the platform refuses\n"
        "to treat backtests over it as evidence."
    )
