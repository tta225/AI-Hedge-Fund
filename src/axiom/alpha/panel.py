"""A time-aligned cross-section of instruments.

The uploaded architecture carried its universe as a long-format table keyed by
``(timestamp, ticker)`` and filtered per name on every access. That is
convenient and quietly dangerous for two reasons.

*Alignment is assumed, not checked.* Two instruments with different bar counts
— a listing that started late, a venue that was down for an hour — concatenate
happily and then misalign silently. A factor model fitted on misaligned returns
produces residuals that are pure artefact.

*Recency is by position.* ``closes[-1]`` is the last bar in the frame, which
during a backtest is the last bar of the *whole history*, not of the bar being
scored. That is the lookahead leak, and it is one character wide.

:class:`Panel` closes both. Construction requires a common index, and every
accessor takes the bar index being scored and returns only history strictly
before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from axiom.core.provenance import Provenance, weakest_kind

if TYPE_CHECKING:
    from axiom.core.series import OHLCVSeries


@dataclass(frozen=True, slots=True)
class Panel:
    """Aligned closes and volumes for a universe of instruments.

    Attributes:
        symbols: column order, fixed for the panel's life.
        index: the common timestamp index.
        closes: ``(n_bars, n_symbols)`` close prices.
        volumes: ``(n_bars, n_symbols)`` volumes.
        provenance: the *weakest* provenance across constituents — one
            synthetic member makes the whole panel non-evidential, because a
            cross-sectional signal mixes every column into every output.
    """

    symbols: tuple[str, ...]
    index: pd.DatetimeIndex
    closes: np.ndarray
    volumes: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.closes.shape != (len(self.index), len(self.symbols)):
            raise ValueError(
                f"closes {self.closes.shape} does not match "
                f"({len(self.index)}, {len(self.symbols)})"
            )
        if self.volumes.shape != self.closes.shape:
            raise ValueError("volumes and closes must have the same shape")

    def __len__(self) -> int:
        return len(self.index)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def column(self, symbol: str) -> int:
        try:
            return self.symbols.index(symbol)
        except ValueError:
            raise KeyError(f"{symbol} is not in this panel") from None

    def history(self, index: int, lookback: int | None = None) -> np.ndarray:
        """Closes strictly before ``index``, most recent last.

        The exclusive upper bound is the whole point: a signal for bar ``t``
        may see ``t-1`` and earlier. Passing ``t`` itself is how a backtest
        learns the answer before it is asked.
        """
        start = 0 if lookback is None else max(0, index - lookback)
        return self.closes[start:index]

    def volume_history(self, index: int, lookback: int | None = None) -> np.ndarray:
        start = 0 if lookback is None else max(0, index - lookback)
        return self.volumes[start:index]

    def returns(self, index: int, lookback: int | None = None) -> np.ndarray:
        """Log returns over the history strictly before ``index``.

        Log rather than simple returns because these are summed across time and
        fed to a PCA that assumes additivity.
        """
        prices = self.history(index, None if lookback is None else lookback + 1)
        if len(prices) < 2:
            return np.empty((0, self.n_symbols))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.diff(np.log(np.where(prices > 0, prices, np.nan)), axis=0)

    def last_price(self, index: int) -> np.ndarray:
        """The most recent observed close — bar ``index - 1``."""
        if index < 1:
            raise IndexError("no observed bar before index 0")
        return np.asarray(self.closes[index - 1])

    @classmethod
    def from_series(cls, series: dict[str, OHLCVSeries]) -> Panel:
        """Build a panel from per-instrument series, on their common bars.

        Inner-joining on the index is deliberate. The alternative — forward
        filling to a union index — invents prices at timestamps where an
        instrument did not trade, and a factor model cannot tell an invented
        price from an observed one.
        """
        if not series:
            raise ValueError("cannot build a panel from no instruments")

        symbols = tuple(sorted(series))
        frames = {
            symbol: pd.DataFrame(
                {"close": series[symbol].closes, "volume": series[symbol].volumes},
                index=series[symbol].index,
            )
            for symbol in symbols
        }

        common = frames[symbols[0]].index
        for symbol in symbols[1:]:
            common = common.intersection(frames[symbol].index)
        if len(common) == 0:
            raise ValueError(
                "instruments share no common bars; they cannot be compared "
                "cross-sectionally"
            )
        common = common.sort_values()

        closes = np.column_stack(
            [frames[symbol].loc[common, "close"].to_numpy(dtype=float) for symbol in symbols]
        )
        volumes = np.column_stack(
            [frames[symbol].loc[common, "volume"].to_numpy(dtype=float) for symbol in symbols]
        )

        return cls(
            symbols=symbols,
            index=pd.DatetimeIndex(common),
            closes=closes,
            volumes=volumes,
            provenance=_weakest_provenance([series[s].provenance for s in symbols]),
        )


def _weakest_provenance(sources: list[Provenance]) -> Provenance:
    """Combine provenances by taking the least trustworthy.

    A panel is only as evidential as its worst column: cross-sectional signals
    mix every instrument into every output, so one synthetic member contaminates
    all of them. Taking the maximum would be how a synthetic member gets
    laundered into a track record.
    """
    kind = weakest_kind(p.kind for p in sources)
    weakest = next(p for p in sources if p.kind is kind)
    labels = ", ".join(sorted({p.source for p in sources}))
    return Provenance(
        source=f"panel[{labels}]",
        kind=kind,
        detail=f"{len(sources)} instruments, weakest={weakest.label}",
    )
