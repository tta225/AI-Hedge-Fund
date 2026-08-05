"""OHLCV series — the single container every analytic in AXIOM consumes.

Invariants enforced at construction:

* index is a tz-aware ``DatetimeIndex`` normalised to UTC, strictly increasing,
  with no duplicates and no NaT;
* columns ``open/high/low/close/volume`` exist, are float64, and are finite;
* each row satisfies ``low <= min(open, close) <= max(open, close) <= high``.

Analytics downstream may therefore assume clean data and index positionally
without defensive checks. Bad data fails loudly at the boundary instead of
quietly producing a plausible-looking backtest.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

import numpy as np
import pandas as pd

from axiom.core.provenance import Provenance
from axiom.core.timeframe import Timeframe
from axiom.core.types import Bar, Instrument

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class DataIntegrityError(ValueError):
    """Raised when candidate bars violate an OHLCV invariant."""


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalise a raw OHLCV frame. Returns a new frame."""
    if df.empty:
        raise DataIntegrityError("cannot build an OHLCVSeries from an empty frame")

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    if "volume" not in out.columns:
        out["volume"] = 0.0

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise DataIntegrityError(
            f"missing required column(s) {missing}; got {sorted(out.columns)}"
        )
    out = out.loc[:, list(OHLCV_COLUMNS)]

    if not isinstance(out.index, pd.DatetimeIndex):
        raise DataIntegrityError(f"index must be a DatetimeIndex, got {type(out.index).__name__}")
    if out.index.hasnans:
        raise DataIntegrityError("index contains NaT")
    out.index = (
        out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    )
    out.index.name = "timestamp"

    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    dupes = out.index.duplicated()
    if dupes.any():
        raise DataIntegrityError(
            f"{int(dupes.sum())} duplicate timestamp(s), first at {out.index[dupes][0]}"
        )

    out = out.astype("float64")
    if not np.isfinite(out.to_numpy()).all():
        bad = out.index[~np.isfinite(out.to_numpy()).all(axis=1)]
        raise DataIntegrityError(f"{len(bad)} row(s) contain NaN/inf, first at {bad[0]}")

    hi, lo = out["high"].to_numpy(), out["low"].to_numpy()
    op, cl = out["open"].to_numpy(), out["close"].to_numpy()
    broken = (lo > hi) | (op > hi) | (op < lo) | (cl > hi) | (cl < lo)
    if broken.any():
        first = out.index[broken][0]
        raise DataIntegrityError(
            f"{int(broken.sum())} row(s) violate low<=open,close<=high; first at {first}"
        )
    if (out["volume"].to_numpy() < 0).any():
        raise DataIntegrityError("negative volume")

    return out


@dataclass(frozen=True, slots=True)
class OHLCVSeries:
    """Immutable, validated OHLCV bars for one instrument and timeframe.

    Column arrays and ATR are memoised. This is not micro-optimisation: the
    obvious implementation calls ``df["close"].to_numpy()`` on every access and
    recomputes the whole ATR array on every call, so a strategy reading
    ``context.closes`` and ``context.atr()` once per bar turns an O(n) backtest
    into an O(n²) one. On a year of hourly data that is the difference between
    two seconds and several minutes per run — and a parameter sweep multiplies
    it by the number of candidates.

    The cache is safe precisely because the series is immutable: the underlying
    frame is validated at construction and never mutated afterwards.
    """

    instrument: Instrument
    timeframe: Timeframe
    df: pd.DataFrame
    provenance: Provenance
    #: Memoised column arrays and indicators. Excluded from equality.
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def _column(self, name: str) -> np.ndarray:
        cached: np.ndarray | None = self._cache.get(name)
        if cached is None:
            cached = np.ascontiguousarray(self.df[name].to_numpy(dtype="float64"))
            cached.flags.writeable = False
            self._cache[name] = cached
        return cached

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        instrument: Instrument,
        timeframe: str | Timeframe,
        provenance: Provenance,
    ) -> Self:
        return cls(
            instrument=instrument,
            timeframe=Timeframe.parse(timeframe),
            df=_normalise_frame(df),
            provenance=provenance,
        )

    @classmethod
    def from_bars(
        cls,
        bars: list[Bar],
        instrument: Instrument,
        timeframe: str | Timeframe,
        provenance: Provenance,
    ) -> Self:
        frame = pd.DataFrame(
            {
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            },
            index=pd.DatetimeIndex([b.timestamp for b in bars]),
        )
        return cls.from_dataframe(frame, instrument, timeframe, provenance)

    # --- array access -------------------------------------------------------
    # Cheap views; analytics work on these rather than on pandas row lookups.

    @property
    def opens(self) -> np.ndarray:
        return self._column("open")

    @property
    def highs(self) -> np.ndarray:
        return self._column("high")

    @property
    def lows(self) -> np.ndarray:
        return self._column("low")

    @property
    def closes(self) -> np.ndarray:
        return self._column("close")

    @property
    def volumes(self) -> np.ndarray:
        return self._column("volume")

    @property
    def index(self) -> pd.DatetimeIndex:
        assert isinstance(self.df.index, pd.DatetimeIndex)
        return self.df.index

    @property
    def start(self) -> pd.Timestamp:
        return self.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.index[-1]

    def __len__(self) -> int:
        return len(self.df)

    def timestamp_at(self, i: int) -> pd.Timestamp:
        return self.index[i]

    def bar_at(self, i: int) -> Bar:
        # Built from the cached arrays; `df.iloc[i]` constructs a Series per
        # call and dominates the backtest loop when used per bar.
        return Bar(
            timestamp=self.index[i].to_pydatetime(),
            open=float(self.opens[i]),
            high=float(self.highs[i]),
            low=float(self.lows[i]),
            close=float(self.closes[i]),
            volume=float(self.volumes[i]),
        )

    @property
    def last(self) -> Bar:
        return self.bar_at(len(self) - 1)

    def __iter__(self) -> Iterator[Bar]:
        for i in range(len(self)):
            yield self.bar_at(i)

    # --- reshaping ----------------------------------------------------------

    def head(self, n: int) -> OHLCVSeries:
        return self._with_frame(self.df.iloc[:n], f"head({n})")

    def tail(self, n: int) -> OHLCVSeries:
        return self._with_frame(self.df.iloc[-n:], f"tail({n})")

    def islice(self, start: int, stop: int | None = None) -> OHLCVSeries:
        """Positional slice. Used by the backtester to expose only past bars."""
        return self._with_frame(self.df.iloc[start:stop], f"islice({start},{stop})")

    def between(self, start: datetime | str, end: datetime | str) -> OHLCVSeries:
        lo = pd.Timestamp(start, tz="UTC") if not isinstance(start, pd.Timestamp) else start
        hi = pd.Timestamp(end, tz="UTC") if not isinstance(end, pd.Timestamp) else end
        return self._with_frame(self.df.loc[lo:hi], f"between({lo.date()},{hi.date()})")

    def resample(self, timeframe: str | Timeframe) -> OHLCVSeries:
        """Aggregate to a higher timeframe. Refuses to fabricate finer bars."""
        target = Timeframe.parse(timeframe)
        if target.seconds < self.timeframe.seconds:
            raise ValueError(
                f"cannot resample {self.timeframe} up to finer {target}: "
                "downsampling would invent bars that were never observed"
            )
        if target.seconds == self.timeframe.seconds:
            return self
        agg = self.df.resample(target.pandas_freq, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        agg = agg.dropna(subset=["open", "high", "low", "close"])
        return OHLCVSeries(
            instrument=self.instrument,
            timeframe=target,
            df=_normalise_frame(agg),
            provenance=self.provenance.derive(f"resample:{self.timeframe}->{target}"),
        )

    def _with_frame(self, frame: pd.DataFrame, transform: str) -> OHLCVSeries:
        return OHLCVSeries(
            instrument=self.instrument,
            timeframe=self.timeframe,
            df=frame,
            provenance=self.provenance.derive(transform),
        )

    # --- indicators ---------------------------------------------------------

    def true_range(self) -> np.ndarray:
        """Wilder's True Range. TR[0] falls back to the bar's own range."""
        memo: np.ndarray | None = self._cache.get("true_range")
        if memo is not None:
            return memo
        high, low, close = self.highs, self.lows, self.closes
        prev_close = np.concatenate(([close[0]], close[:-1]))
        tr = np.asarray(
            np.maximum.reduce(
                [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
            )
        )
        tr.flags.writeable = False
        self._cache["true_range"] = tr
        return tr

    def atr(self, period: int = 14) -> np.ndarray:
        """Wilder-smoothed ATR, seeded with the SMA of the first ``period`` TRs.

        Values before the seed index are the running mean, so early bars get a
        usable (if noisy) scale rather than NaN. Callers needing strict warm-up
        should slice off the first ``period`` bars.
        """
        if period < 1:
            raise ValueError("ATR period must be >= 1")
        key = f"atr:{period}"
        memo: np.ndarray | None = self._cache.get(key)
        if memo is not None:
            return memo
        tr = self.true_range()
        n = len(tr)
        out = np.empty(n, dtype="float64")
        if n <= period:
            short = np.asarray(np.cumsum(tr) / np.arange(1, n + 1))
            short.flags.writeable = False
            self._cache[key] = short
            return short
        out[:period] = np.cumsum(tr[:period]) / np.arange(1, period + 1)
        alpha = 1.0 / period
        acc = out[period - 1]
        for i in range(period, n):
            acc = acc + alpha * (tr[i] - acc)
            out[i] = acc
        out.flags.writeable = False
        self._cache[key] = out
        return out

    def describe(self) -> str:
        return (
            f"{self.instrument.symbol} {self.timeframe} · {len(self)} bars · "
            f"{self.start.isoformat()} → {self.end.isoformat()} · {self.provenance.label}"
        )
