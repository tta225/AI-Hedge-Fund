"""Label a bar by what would have happened to the trade, not by a fixed horizon.

The default in most financial ML is a fixed-horizon return: label bar ``t`` by
``sign(close[t+h] - close[t])``. It is convenient and it is wrong in a specific
way — it labels an outcome nobody could have realised. A position opened at
``t`` and stopped out on the second bar never sees ``t+h``. Training a
classifier on returns the strategy could not have collected teaches it to
predict a quantity that does not exist.

The triple-barrier method labels by the first of three things to happen:

* the **profit target** is hit — label ``+1``;
* the **stop** is hit — label ``-1``;
* the **holding period expires** — label by the sign of the return at that
  point, or ``0`` under :attr:`Barriers.zero_on_timeout`.

The barriers are set in units of local volatility rather than fixed points,
because a 1% move means something different in a quiet week than in a crash,
and a fixed threshold silently relabels the same market behaviour depending on
regime.

Path dependence is the entire point, so these functions look **forward** from
each bar by construction. That is not lookahead: a label is allowed to know the
future, because it is the thing being predicted. What must never leak forward
is a *feature*, and the purging in :mod:`axiom.ml.crossval` is what stops a
label's forward window from contaminating a training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries

#: Bars used to estimate local volatility when scaling the barriers.
DEFAULT_VOL_LOOKBACK = 50
#: Barrier widths in units of that volatility.
DEFAULT_PROFIT_SIGMA = 2.0
DEFAULT_STOP_SIGMA = 2.0
#: Bars a position may stay open before the vertical barrier closes it.
DEFAULT_MAX_HOLD = 20


@dataclass(frozen=True, slots=True)
class Barriers:
    """Barrier geometry, in volatility units.

    Args:
        profit_sigma: upper barrier distance. Zero disables it.
        stop_sigma: lower barrier distance. Zero disables it.
        max_hold: bars before the vertical barrier fires.
        zero_on_timeout: when True a timed-out trade is labelled ``0`` rather
            than by the sign of its return. Use it when the question is "does
            this setup reach its target" — a trade that drifted nowhere is a
            different animal from one that worked, and forcing it to ``±1``
            teaches the classifier to predict noise.
    """

    profit_sigma: float = DEFAULT_PROFIT_SIGMA
    stop_sigma: float = DEFAULT_STOP_SIGMA
    max_hold: int = DEFAULT_MAX_HOLD
    zero_on_timeout: bool = False

    def __post_init__(self) -> None:
        if self.profit_sigma < 0 or self.stop_sigma < 0:
            raise ValueError("barrier widths must be non-negative")
        if self.profit_sigma == 0 and self.stop_sigma == 0:
            raise ValueError(
                "at least one horizontal barrier must be enabled, or this is "
                "just a fixed-horizon label with extra steps"
            )
        if self.max_hold < 1:
            raise ValueError("max_hold must be at least 1 bar")


@dataclass(frozen=True, slots=True)
class Label:
    """One labelled observation."""

    index: int
    timestamp: pd.Timestamp
    #: ``+1`` target hit, ``-1`` stop hit, ``0`` timed out (when configured).
    label: int
    #: Bar at which the label resolved. The span ``index..touch_index`` is what
    #: must be purged from any training set that tests near this observation.
    touch_index: int
    #: Return realised at the touch, signed by ``side``.
    ret: float
    #: Which barrier fired.
    reason: str
    #: Direction the trade was taken in: ``+1`` long, ``-1`` short.
    side: int = 1


@dataclass(slots=True)
class LabelSet:
    """Labels for one series, plus the provenance they inherit."""

    labels: list[Label]
    provenance: Provenance
    barriers: Barriers

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def is_evidential(self) -> bool:
        """Labels are only as evidential as the bars they were cut from."""
        return self.provenance.is_evidential

    @property
    def y(self) -> np.ndarray:
        return np.array([label.label for label in self.labels], dtype=int)

    @property
    def indices(self) -> np.ndarray:
        return np.array([label.index for label in self.labels], dtype=int)

    @property
    def touch_indices(self) -> np.ndarray:
        """Resolution bar per observation — the input purging needs."""
        return np.array([label.touch_index for label in self.labels], dtype=int)

    @property
    def returns(self) -> np.ndarray:
        return np.array([label.ret for label in self.labels], dtype=float)

    def balance(self) -> dict[int, int]:
        """Class counts. A wildly skewed set is a modelling problem, not a bug."""
        counts: dict[int, int] = {}
        for label in self.labels:
            counts[label.label] = counts.get(label.label, 0) + 1
        return dict(sorted(counts.items()))

    def describe(self) -> str:
        balance = self.balance()
        total = max(len(self.labels), 1)
        parts = ", ".join(f"{k:+d}: {v} ({v / total:.0%})" for k, v in balance.items())
        return (
            f"{len(self.labels)} labels [{parts}] "
            f"barriers={self.barriers.profit_sigma:g}σ/"
            f"{self.barriers.stop_sigma:g}σ/{self.barriers.max_hold}bars "
            f"provenance={self.provenance.label}"
        )


def volatility_target(
    series: OHLCVSeries, lookback: int = DEFAULT_VOL_LOOKBACK
) -> np.ndarray:
    """Trailing return volatility per bar, for scaling the barriers.

    Causal by construction: the value at bar ``i`` uses returns strictly before
    ``i``. Barrier *widths* must not see the future even though barrier
    *outcomes* do — otherwise the label is easier to hit exactly when the model
    would have known the move was coming.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    closes = np.asarray(series.closes, dtype=float)
    out = np.full(len(closes), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(np.where(closes > 0, closes, np.nan)))
    for i in range(lookback + 1, len(closes)):
        window = returns[i - 1 - lookback : i - 1]
        finite = window[np.isfinite(window)]
        if finite.size >= 2:
            out[i] = float(np.std(finite, ddof=1))
    return out


def triple_barrier_labels(
    series: OHLCVSeries,
    *,
    barriers: Barriers | None = None,
    events: np.ndarray | list[int] | None = None,
    sides: np.ndarray | list[int] | None = None,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
) -> LabelSet:
    """Label events by whichever barrier price touches first.

    Args:
        series: the bars. Intrabar highs and lows are used for the touch test,
            not closes — a stop that traded through mid-bar was hit, and
            checking closes only is how a backtest gets to keep losers it
            actually lost.
        barriers: geometry. Defaults to 2σ/2σ over 20 bars.
        events: bar indices to label. Defaults to every bar with a usable
            volatility estimate. Pass a strategy's entry bars to label only the
            setups it would actually have taken.
        sides: ``+1``/``-1`` per event, for labelling shorts correctly. When
            given, returns and barriers are oriented by side, so ``+1`` always
            means "the trade worked". Required for meta-labelling.
        vol_lookback: bars behind each event used to size the barriers.

    Returns:
        A :class:`LabelSet` carrying the series' provenance.
    """
    barriers = barriers or Barriers()
    highs = np.asarray(series.highs, dtype=float)
    lows = np.asarray(series.lows, dtype=float)
    closes = np.asarray(series.closes, dtype=float)
    sigma = volatility_target(series, vol_lookback)
    n = len(closes)

    if events is None:
        candidates = np.flatnonzero(np.isfinite(sigma))
    else:
        candidates = np.asarray(events, dtype=int)
        if candidates.size and (candidates.min() < 0 or candidates.max() >= n):
            raise IndexError("event index outside the series")

    if sides is None:
        side_lookup = {int(i): 1 for i in candidates}
    else:
        side_array = np.asarray(sides, dtype=int)
        if side_array.shape != candidates.shape:
            raise ValueError(
                f"sides has {side_array.shape} entries but there are "
                f"{candidates.shape} events"
            )
        if not np.isin(side_array, (-1, 1)).all():
            raise ValueError("sides must be +1 or -1")
        side_lookup = {int(i): int(s) for i, s in zip(candidates, side_array, strict=True)}

    out: list[Label] = []
    for raw_index in candidates:
        i = int(raw_index)
        scale = sigma[i]
        # The last bar cannot be labelled: there is no forward path to observe.
        if not np.isfinite(scale) or scale <= 0 or i >= n - 1:
            continue

        side = side_lookup[i]
        entry = closes[i]
        if entry <= 0:
            continue

        # Barriers are log-return distances converted to prices, so the two
        # sides are symmetric in return space rather than in price space.
        # Both levels are expressed relative to the *trade*: profit is above
        # for a long and below for a short. Mirroring the prices without also
        # mirroring which one counts as a win is how a short that was stopped
        # out gets labelled +1.
        profit_move = np.exp(side * barriers.profit_sigma * scale)
        stop_move = np.exp(-side * barriers.stop_sigma * scale)
        profit_level = entry * profit_move if barriers.profit_sigma else None
        stop_level = entry * stop_move if barriers.stop_sigma else None

        vertical = min(i + barriers.max_hold, n - 1)
        label, touch, reason = _walk_forward(
            highs, lows, i, vertical, profit_level, stop_level, side, barriers
        )
        ret = side * float(np.log(closes[touch] / entry))
        out.append(
            Label(
                index=i,
                timestamp=series.timestamp_at(i),
                label=label,
                touch_index=touch,
                ret=ret,
                reason=reason,
                side=side,
            )
        )

    return LabelSet(labels=out, provenance=series.provenance, barriers=barriers)


def _walk_forward(
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    vertical: int,
    profit_level: float | None,
    stop_level: float | None,
    side: int,
    barriers: Barriers,
) -> tuple[int, int, str]:
    """First barrier touched, walking bar by bar from ``start + 1``.

    ``profit_level`` and ``stop_level`` are relative to the trade, not to the
    chart: for a short the profit level sits *below* entry and is touched by a
    bar's low, while its stop sits above and is touched by a bar's high. The
    returned label is therefore ``+1`` for "this trade worked" on both sides.

    When a single bar touches both barriers the **stop is assumed first**. Bar
    data cannot say which came first within the bar, and assuming the target
    would be assuming the favourable case every time — which is exactly the
    bias that makes backtests look good and live trading look surprising.
    """
    for j in range(start + 1, vertical + 1):
        if side > 0:
            hit_profit = profit_level is not None and highs[j] >= profit_level
            hit_stop = stop_level is not None and lows[j] <= stop_level
        else:
            hit_profit = profit_level is not None and lows[j] <= profit_level
            hit_stop = stop_level is not None and highs[j] >= stop_level
        if hit_profit and hit_stop:
            return (-1, j, "both_barriers_same_bar_assumed_stop")
        if hit_stop:
            return (-1, j, "stop")
        if hit_profit:
            return (1, j, "target")

    if barriers.zero_on_timeout:
        return (0, vertical, "timeout")
    # Sign of the realised move, with a flat close counted as a loss rather
    # than a win for the same reason as above.
    drift = side * (highs[vertical] + lows[vertical]) / 2.0
    reference = side * (highs[start] + lows[start]) / 2.0
    return (1 if drift > reference else -1, vertical, "timeout")


def meta_label(labels: LabelSet, predicted_sides: np.ndarray | list[int]) -> LabelSet:
    """Turn a directional label set into a "should I take this trade" set.

    Meta-labelling separates two questions a single classifier normally
    conflates: *which way* and *whether to bet at all*. A primary model (a
    strategy, or a first-stage classifier) supplies the side; this relabels
    each event ``1`` when that side was right and ``0`` when it was not.

    The value is asymmetric and worth being precise about. A meta-model can
    only ever *filter* — it raises precision by declining trades, and it cannot
    invent a winner the primary model never proposed. That makes it a
    genuinely safe place to apply machine learning: the worst a broken meta-model
    can do is trade less.

    Args:
        labels: output of :func:`triple_barrier_labels`.
        predicted_sides: the primary model's side per label, ``+1``/``-1``.
    """
    sides = np.asarray(predicted_sides, dtype=int)
    if sides.shape[0] != len(labels.labels):
        raise ValueError(
            f"{sides.shape[0]} predicted sides for {len(labels.labels)} labels"
        )
    if sides.size and not np.isin(sides, (-1, 1)).all():
        raise ValueError("predicted sides must be +1 or -1")

    relabelled = [
        Label(
            index=label.index,
            timestamp=label.timestamp,
            # 1 = the primary model's call was correct and the trade paid.
            label=int(label.label == side and label.label != 0),
            touch_index=label.touch_index,
            ret=label.ret,
            reason=label.reason,
            side=int(side),
        )
        for label, side in zip(labels.labels, sides, strict=True)
    ]
    return LabelSet(
        labels=relabelled, provenance=labels.provenance, barriers=labels.barriers
    )
