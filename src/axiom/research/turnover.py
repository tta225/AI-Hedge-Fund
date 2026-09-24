"""Trade less without deciding less.

Every candidate in the cross-sectional campaign died on cost, and the pattern
was unambiguous: everything above 30× annual turnover lost money, and the only
two positive results turned over 7× and 15×. Direction was not the problem.
The book was rebuilt too often.

The naive fix — rebalance less frequently — throws away information along with
the trading. Waiting a month to act on a signal that changed a week ago is a
worse strategy, not a cheaper one. What is wanted is to act on the changes that
matter and ignore the ones that do not.

Two mechanisms do that, and they compose:

**Hysteresis.** Hold a name until it falls well out of favour, rather than
until it merely leaves the top decile. A name ranked 20th of 100 today and 21st
tomorrow has told you nothing, and trading on it pays a spread for noise. The
standard construction uses two thresholds: buy into the top ``entry_fraction``,
but only sell once a name drops out of a wider ``exit_fraction``. The gap
between them is the buffer, and it is where most of the saved turnover lives.

**Partial rebalancing.** Move a fraction of the way to the target rather than
all of it. This trades a small tracking error against a proportional reduction
in turnover, and is the right tool when the signal is noisy at the margin
rather than wrong.

Both are *cost* decisions, not *signal* decisions, which is why they live here
rather than in an agent. The agent's job is to rank; how expensive it is to act
on that ranking is a different question with a different answer per venue, per
size, and per cost model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Default buffer: enter the top 20%, exit only below the top 40%.
DEFAULT_ENTRY_FRACTION = 0.2
DEFAULT_EXIT_FRACTION = 0.4


@dataclass(frozen=True, slots=True)
class HysteresisBands:
    """Entry and exit thresholds for a ranked book.

    Args:
        entry_fraction: portion of the scored universe bought on each side.
        exit_fraction: portion outside which a held name is sold. Must be at
            least ``entry_fraction``; equal means no buffer, which reduces to
            the unbuffered construction and is a legitimate control.
    """

    entry_fraction: float = DEFAULT_ENTRY_FRACTION
    exit_fraction: float = DEFAULT_EXIT_FRACTION

    def __post_init__(self) -> None:
        if not 0.0 < self.entry_fraction <= 0.5:
            raise ValueError(
                f"entry_fraction must be in (0, 0.5], got {self.entry_fraction}"
            )
        if not self.entry_fraction <= self.exit_fraction <= 0.5:
            raise ValueError(
                f"exit_fraction {self.exit_fraction} must be between "
                f"entry_fraction {self.entry_fraction} and 0.5 — an exit band "
                "tighter than the entry band would sell names it just bought"
            )

    @property
    def buffer(self) -> float:
        """Width of the hold zone. Zero means no hysteresis."""
        return self.exit_fraction - self.entry_fraction


def hysteresis_weights(
    scores: dict[str, float],
    symbols: Sequence[str],
    current: np.ndarray,
    *,
    bands: HysteresisBands | None = None,
    gross: float = 1.0,
    long_short: bool = True,
) -> np.ndarray:
    """Target weights that keep held names until they fall out of the buffer.

    The construction, per side:

    1. Every name inside ``entry_fraction`` is wanted.
    2. Every name currently held that is still inside ``exit_fraction`` is
       kept, even if it has slipped out of the entry band.
    3. The resulting set is equal-weighted to ``gross``.

    Step 2 is the whole mechanism, and it has a consequence worth naming: the
    number of positions varies over time. Forcing a fixed count would mean
    dropping a held name to make room for a new one, which is exactly the trade
    the buffer exists to avoid.

    Args:
        scores: symbol → conviction. Missing symbols are unranked and unheld.
        symbols: panel column order; weights come back in it.
        current: weights currently held, same order.
        bands: entry and exit thresholds.
        gross: total absolute exposure.
        long_short: dollar-neutral when True.
    """
    bands = bands or HysteresisBands()
    current = np.asarray(current, dtype=float)
    if current.shape[0] != len(symbols):
        raise ValueError(
            f"{current.shape[0]} current weights against {len(symbols)} symbols"
        )

    scored = [(s, v) for s, v in scores.items() if s in symbols and np.isfinite(v)]
    weights = np.zeros(len(symbols), dtype=float)
    if len(scored) < 2:
        return weights

    ordered = sorted(scored, key=lambda kv: kv[1], reverse=True)
    index_of = {symbol: i for i, symbol in enumerate(symbols)}
    entry_count = max(1, int(len(ordered) * bands.entry_fraction))
    exit_count = max(entry_count, int(len(ordered) * bands.exit_fraction))

    def side(top: bool) -> list[str]:
        """Names to hold on one side: the entry band, plus what is still tolerated."""
        ranked = ordered if top else list(reversed(ordered))
        wanted = {s for s, _ in ranked[:entry_count]}
        tolerated = {s for s, _ in ranked[:exit_count]}

        def already_held(symbol: str) -> bool:
            weight = current[index_of[symbol]]
            return bool(weight > 0) if top else bool(weight < 0)

        return sorted(wanted | {s for s in tolerated if already_held(s)})

    if not long_short:
        chosen = side(top=True)
        for symbol in chosen:
            weights[index_of[symbol]] = gross / len(chosen)
        return weights

    longs = side(top=True)
    shorts = side(top=False)
    # A name cannot be both. If the buffer would keep it on one side while the
    # ranking wants it on the other, the ranking wins — holding a long in
    # something now ranked bottom-decile is not thrift, it is a stale position.
    overlap = set(longs) & set(shorts)
    if overlap:
        top_entry = {s for s, _ in ordered[:entry_count]}
        longs = [s for s in longs if s not in overlap or s in top_entry]
        shorts = [s for s in shorts if s not in overlap or s not in top_entry]

    for symbol in longs:
        weights[index_of[symbol]] += gross / (2 * max(len(longs), 1))
    for symbol in shorts:
        weights[index_of[symbol]] -= gross / (2 * max(len(shorts), 1))
    return weights


def partial_rebalance(
    current: np.ndarray, target: np.ndarray, fraction: float
) -> np.ndarray:
    """Move ``fraction`` of the way from ``current`` to ``target``.

    ``fraction=1.0`` is a full rebalance and the identity on ``target``.
    Smaller values trade tracking error for turnover, linearly in both: half
    the step is half the trading and half the convergence.

    Not a substitute for hysteresis. Partial rebalancing shrinks every trade
    including the ones worth making, while hysteresis suppresses only the
    trades that carry no information. They compose, and the buffer should be
    applied first.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != target.shape:
        raise ValueError("current and target must have the same shape")
    return np.asarray(current + fraction * (target - current), dtype=float)


def suppress_small_trades(
    current: np.ndarray, target: np.ndarray, threshold: float
) -> np.ndarray:
    """Leave a position alone when the required change is below ``threshold``.

    A last-mile filter. After hysteresis has decided *which* names to hold,
    equal-weighting still produces tiny adjustments every period as the count
    changes — trades whose expected benefit is below their own spread. This
    drops them.

    Applied per name rather than to the aggregate, because a book of many small
    adjustments and a book with one large one cost very different amounts and
    only the per-name view can tell them apart.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != target.shape:
        raise ValueError("current and target must have the same shape")
    delta = target - current
    return np.where(np.abs(delta) < threshold, current, target)
