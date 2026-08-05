"""SMT divergence — Smart Money Technique.

Correlated instruments should make their extremes together. ES and NQ should
top together; EURUSD and GBPUSD should bottom together; a market and its inverse
(EURUSD vs DXY) should mirror. When one makes a new extreme and its correlate
refuses, the move is not supported across the complex, and ICT reads the
non-confirmation as a raid rather than genuine expansion.

The comparison is done on aligned timestamps only. Two instruments with
different session calendars must not be compared bar-index to bar-index — that
silently offsets the series and manufactures divergences that do not exist.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.models import SMTDivergence, SwingKind, SwingPoint

#: Inverse pairs, where a divergence means the two moved the *same* way.
INVERSE_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"EURUSD", "DXY"}),
        frozenset({"GBPUSD", "DXY"}),
        frozenset({"USDJPY", "EURUSD"}),
    }
)

#: Instruments conventionally read against one another for SMT.
CORRELATION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ES", "NQ", "YM", "RTY"),
    ("SPY", "QQQ", "DIA", "IWM"),
    ("EURUSD", "GBPUSD"),
    ("GC", "SI"),
)


def is_inverse(a: str, b: str) -> bool:
    return frozenset({a.upper(), b.upper()}) in INVERSE_PAIRS


def align(primary: OHLCVSeries, correlate: OHLCVSeries) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both series to their shared timestamps."""
    if primary.timeframe != correlate.timeframe:
        raise ValueError(
            f"SMT requires matching timeframes, got {primary.timeframe} and "
            f"{correlate.timeframe}"
        )
    shared = primary.index.intersection(correlate.index)
    if len(shared) == 0:
        raise ValueError(
            f"{primary.instrument.symbol} and {correlate.instrument.symbol} share no "
            "timestamps; check session calendars before reading SMT"
        )
    return primary.df.loc[shared], correlate.df.loc[shared]


def find_smt_divergences(
    primary: OHLCVSeries,
    correlate: OHLCVSeries,
    swings: list[SwingPoint],
    *,
    window: int = 3,
) -> list[SMTDivergence]:
    """Compare each confirmed pivot on ``primary`` against ``correlate``.

    A pivot registers as divergent when the primary makes a new extreme versus
    its previous same-kind pivot and the correlate does not follow.

    Args:
        window: bars either side of the pivot timestamp searched on the
            correlate, absorbing minor timestamp misalignment.
    """
    if not swings:
        return []

    primary_df, correlate_df = align(primary, correlate)
    if len(primary_df) < 2:
        return []

    inverse = is_inverse(primary.instrument.symbol, correlate.instrument.symbol)
    positions = {ts: i for i, ts in enumerate(primary_df.index)}

    p_high = primary_df["high"].to_numpy()
    p_low = primary_df["low"].to_numpy()
    c_high = correlate_df["high"].to_numpy()
    c_low = correlate_df["low"].to_numpy()

    out: list[SMTDivergence] = []
    for kind in (SwingKind.HIGH, SwingKind.LOW):
        pivots = [
            p for p in swings if p.kind is kind and primary.index[p.origin_index] in positions
        ]
        for prev, curr in pairwise(pivots):
            i_prev = positions[primary.index[prev.origin_index]]
            i_curr = positions[primary.index[curr.origin_index]]

            if kind is SwingKind.HIGH:
                primary_extreme = p_high[i_curr] > p_high[i_prev]
                c_ref = _extreme(c_high, i_prev, window, high=True)
                c_now = _extreme(c_high, i_curr, window, high=True)
                correlate_extreme = (c_now < c_ref) if inverse else (c_now > c_ref)
                divergence_dir = Direction.BEARISH
            else:
                primary_extreme = p_low[i_curr] < p_low[i_prev]
                c_ref = _extreme(c_low, i_prev, window, high=False)
                c_now = _extreme(c_low, i_curr, window, high=False)
                correlate_extreme = (c_now > c_ref) if inverse else (c_now < c_ref)
                divergence_dir = Direction.BULLISH

            if primary_extreme and not correlate_extreme:
                out.append(
                    SMTDivergence(
                        origin_index=curr.origin_index,
                        confirmed_index=curr.confirmed_index,
                        timestamp=curr.timestamp,
                        primary_symbol=primary.instrument.symbol,
                        correlate_symbol=correlate.instrument.symbol,
                        direction=divergence_dir,
                        primary_made_extreme=True,
                        primary_level=float(curr.price),
                        correlate_level=float(c_now),
                    )
                )

    out.sort(key=lambda d: d.confirmed_index)
    return out


def _extreme(values: np.ndarray, centre: int, window: int, *, high: bool) -> float:
    lo = max(centre - window, 0)
    hi = min(centre + window + 1, len(values))
    segment = values[lo:hi]
    return float(segment.max() if high else segment.min())
