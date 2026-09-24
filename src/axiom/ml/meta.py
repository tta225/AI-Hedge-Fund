"""Wrap a strategy in a model that decides whether to take each trade.

This is where :mod:`axiom.ml` meets :mod:`axiom.strategy`, and it is the only
place in this codebase where a learned model is allowed near a trading
decision. The restriction is deliberate.

A meta-model **cannot invent a trade**. It sees only signals the primary
strategy already produced, and its output is binary: take it, or skip it. The
worst a completely broken meta-model can do is trade less. Compare that with a
model that chooses direction, where a broken one takes the wrong side with
confidence, and the asymmetry is the whole argument for putting ML here rather
than in the signal itself.

The honest framing of what this can and cannot do:

* It can raise **precision** — a higher fraction of taken trades work.
* It cannot raise **recall**, and it will lower it. Fewer trades means fewer
  observations, wider error bars, and a longer time to know whether the filter
  helped at all.
* It cannot rescue a strategy with no edge. Filtering a coin flip leaves a
  smaller sample of coin flips. If the primary strategy's expectancy is zero,
  the honest outcome here is a filter that declines a lot and changes nothing —
  and :meth:`MetaLabelFilter.report` is written to say so plainly.

Training is done by :func:`train_meta_filter`, which uses triple-barrier labels
for the outcome, purged walk-forward folds so the reported skill is not
measuring leakage, and uniqueness weights so overlapping labels do not count
as independent evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import ICTEngine
from axiom.ml.classifier import LogisticClassifier
from axiom.ml.crossval import PurgedWalkForward
from axiom.ml.features import sample_uniqueness
from axiom.ml.labelling import Barriers, triple_barrier_labels
from axiom.portfolio.positions import Portfolio
from axiom.strategy.base import Signal, Strategy, StrategyContext

#: Probability below which a signal is declined. 0.5 is the neutral point; a
#: higher threshold trades less and demands more conviction.
DEFAULT_THRESHOLD = 0.5
#: Minimum labelled signals before a filter is allowed to be fitted at all.
#: Below this the model is fitting noise and the filter is worse than nothing.
MIN_TRAINING_SIGNALS = 50


def default_features(context: StrategyContext, signal: Signal) -> dict[str, float]:
    """Features describing the *context* of a signal, not its direction.

    Direction is the primary strategy's job and is deliberately excluded — a
    meta-model that learns direction is no longer a filter. What it is allowed
    to see is the state of the market around the setup: how volatile, how
    stretched, how far the trade is reaching, what time it is.

    Every value is computed from bars at or before ``context.index``, so
    nothing here can see past the decision point.
    """
    atr = context.atr()
    price = context.price
    closes = context.series.closes
    i = context.index

    lookback = closes[max(0, i - 50) : i + 1]
    position_in_range = (
        float((price - lookback.min()) / (lookback.max() - lookback.min()))
        if len(lookback) > 1 and lookback.max() > lookback.min()
        else 0.5
    )
    recent = closes[max(0, i - 20) : i + 1]
    momentum = float(price / recent[0] - 1.0) if len(recent) > 1 and recent[0] > 0 else 0.0

    return {
        "atr_pct": float(atr / price) if price > 0 else 0.0,
        "risk_atr": float(signal.risk_points / atr) if atr > 0 else 0.0,
        "reward_risk": float(signal.reward_risk or 0.0),
        "position_in_range": position_in_range,
        "momentum_20": momentum,
        "hour_utc": float(context.timestamp.hour),
        # Signed so the model can learn that a setup works better with or
        # against the prevailing structure, without learning direction itself.
        "bias_agrees": float(context.bias is signal.direction),
    }


FeatureFn = Callable[[StrategyContext, Signal], dict[str, float]]


@dataclass(slots=True)
class MetaTrainingReport:
    """What the meta-model learned, and whether it is worth using."""

    n_signals: int
    n_taken: int
    base_rate: float
    #: Out-of-sample accuracy of the filter across purged walk-forward folds.
    oos_accuracy: float
    #: Precision on trades the filter *would have taken*, out of sample. This
    #: is the number that matters: it is the win rate of the filtered strategy.
    oos_precision: float
    feature_names: list[str] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    folds: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def lift(self) -> float:
        """Precision minus base rate: what filtering actually bought."""
        return self.oos_precision - self.base_rate

    @property
    def lift_standard_error(self) -> float:
        """Standard error of the precision estimate, as a proportion.

        ``sqrt(p(1-p)/n)`` on the trades the filter actually took. This is the
        number that makes a lift interpretable: on 340 taken trades at 55%
        precision it is about 2.7 percentage points, so a 2-point lift is
        indistinguishable from zero however encouraging it looks.
        """
        if self.n_taken < 2:
            return float("inf")
        p = self.oos_precision
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / self.n_taken))

    @property
    def is_useful(self) -> bool:
        """Whether the filter earned its reduction in sample size.

        Three conditions, all necessary:

        * the lift clears **two standard errors**, so it is distinguishable
          from noise rather than merely positive — a bar that positive-lift
          alone does not clear about half the time by construction;
        * the filter still takes a meaningful share of signals, since one that
          declines 98% of trades can post a wonderful precision on the two it
          keeps and tell you nothing;
        * enough folds actually ran.
        """
        return (
            self.lift > 2.0 * self.lift_standard_error
            and self.n_taken >= max(20, 0.1 * self.n_signals)
            and self.folds > 0
        )

    def render(self) -> str:
        verdict = (
            "USEFUL" if self.is_useful
            else "NOT USEFUL — do not deploy this filter"
        )
        lines = [
            f"Meta-label filter: {verdict}",
            f"  signals labelled     : {self.n_signals}",
            f"  base rate (unfiltered): {self.base_rate:.1%}",
            f"  would take           : {self.n_taken} "
            f"({self.n_taken / max(self.n_signals, 1):.0%} of signals)",
            f"  OOS precision        : {self.oos_precision:.1%}",
            f"  lift over base rate  : {self.lift:+.1%} "
            f"(±{self.lift_standard_error:.1%} std err; needs "
            f">{2 * self.lift_standard_error:.1%} to count)",
            f"  OOS accuracy         : {self.oos_accuracy:.1%}",
            f"  purged folds         : {self.folds}",
        ]
        if self.feature_names and self.coefficients:
            lines.append("  coefficients (standardised):")
            order = np.argsort(-np.abs(np.array(self.coefficients)))
            lines += [
                f"    {self.feature_names[i]:<20} {self.coefficients[i]:+.4f}"
                for i in order
            ]
        lines += [f"  ! {note}" for note in self.notes]
        return "\n".join(lines)


class MetaLabelFilter(Strategy):
    """A primary strategy, plus a model that vetoes low-probability setups.

    Passes through :meth:`~axiom.strategy.base.Strategy.should_exit` unchanged.
    The filter's authority is over *entries only* — a model that can also close
    positions is a model that can decide to stop protecting one.

    Args:
        primary: the strategy generating signals.
        model: a fitted classifier. Unfitted, every signal passes through and
            the wrapper is a no-op, which is the correct behaviour before
            training rather than declining everything.
        threshold: minimum probability to take a trade.
        feature_fn: builds the feature dict. Must not encode direction.
    """

    def __init__(
        self,
        primary: Strategy,
        model: LogisticClassifier | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        feature_fn: FeatureFn = default_features,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        super().__init__(primary=primary.name, threshold=threshold)
        self.primary = primary
        self.model = model
        self.threshold = threshold
        self.feature_fn = feature_fn
        self.feature_names = list(feature_names) if feature_names else None
        self.name = f"meta[{primary.name}]"
        self.requires_ict = primary.requires_ict
        #: Signals seen and declined, for reporting after a run.
        self.seen = 0
        self.declined = 0

    def evaluate(self, context: StrategyContext) -> Signal | None:
        signal = self.primary.evaluate(context)
        if signal is None:
            return None
        self.seen += 1

        if self.model is None or not self.model.is_fitted:
            return signal

        features = self.feature_fn(context, signal)
        if self.feature_names is None:
            self.feature_names = list(features)
        row = np.array([[features[name] for name in self.feature_names]])
        probability = float(self.model.predict_proba(row)[0])

        if probability < self.threshold:
            self.declined += 1
            return None

        # Confidence carries the model's probability so downstream ranking sees
        # it. The signal's levels are untouched: the filter's authority is over
        # whether to trade, not where to place the stop.
        return Signal(
            direction=signal.direction,
            entry=signal.entry,
            stop=signal.stop,
            targets=signal.targets,
            confidence=probability,
            rationale=f"{signal.rationale} | meta p={probability:.2f}",
            order_type=signal.order_type,
            tags=(*signal.tags, "meta_approved"),
        )

    def should_exit(self, context: StrategyContext) -> str | None:
        """Delegated untouched — the filter has no say over exits."""
        return self.primary.should_exit(context)


def collect_signals(
    strategy: Strategy,
    series: OHLCVSeries,
    *,
    feature_fn: FeatureFn = default_features,
    warmup: int = 100,
) -> tuple[list[int], list[int], list[dict[str, float]]]:
    """Replay a strategy over a series and record where it fired.

    Returns ``(indices, sides, features)``. The portfolio is held flat
    throughout so the strategy is asked at every bar rather than being silenced
    while a position is open — this is a survey of the strategy's *opinions*,
    which is what the meta-model needs, not a simulation of its trading.
    """
    state = ICTEngine().analyse(series) if strategy.requires_ict else None
    if state is None:
        from axiom.ict.models import ICTState

        state = ICTState(
            symbol=series.instrument.symbol,
            timeframe=str(series.timeframe),
            as_of=series.index[-1],
            index=len(series) - 1,
        )

    portfolio = Portfolio(starting_cash=1_000_000)
    indices: list[int] = []
    sides: list[int] = []
    rows: list[dict[str, float]] = []

    for i in range(warmup, len(series)):
        context = StrategyContext(
            series=series,
            index=i,
            ict=state,
            portfolio=portfolio,
            timestamp=series.index[i],
        )
        signal = strategy.evaluate(context)
        if signal is None:
            continue
        indices.append(i)
        sides.append(1 if signal.direction is Direction.BULLISH else -1)
        rows.append(feature_fn(context, signal))

    return indices, sides, rows


def train_meta_filter(
    primary: Strategy,
    series: OHLCVSeries,
    *,
    barriers: Barriers | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    feature_fn: FeatureFn = default_features,
    n_folds: int = 4,
    warmup: int = 100,
) -> tuple[MetaLabelFilter, MetaTrainingReport]:
    """Fit a meta-model on a strategy's own historical signals.

    The evaluation is purged walk-forward: each fold trains only on signals
    whose label windows resolved before the test window opened. Without that,
    overlapping triple-barrier labels put the answer to the test set into the
    training set and the reported precision is fiction.

    Raises:
        ValueError: if the series is not evidential, or produced too few
            signals to fit anything meaningful.
    """
    if not series.provenance.is_evidential:
        raise ValueError(
            f"refusing to train on {series.provenance.kind.value} data — a "
            "filter fitted to generated bars describes the generator"
        )

    indices, sides, rows = collect_signals(
        primary, series, feature_fn=feature_fn, warmup=warmup
    )
    if len(indices) < MIN_TRAINING_SIGNALS:
        raise ValueError(
            f"{primary.name} produced {len(indices)} signals on this series; "
            f"at least {MIN_TRAINING_SIGNALS} are needed before a meta-model "
            "is anything but noise"
        )

    directional = triple_barrier_labels(
        series,
        barriers=barriers or Barriers(),
        events=np.array(indices),
        sides=np.array(sides),
    )
    if len(directional) < MIN_TRAINING_SIGNALS:
        raise ValueError(
            f"only {len(directional)} of {len(indices)} signals could be "
            "labelled; the rest lacked a forward path or a volatility estimate"
        )

    # No call to `meta_label` here, deliberately. These labels were built with
    # `sides`, so the barriers are already oriented to the trade and +1 already
    # means "the primary model's call worked". Passing them through
    # `meta_label` would compare an outcome against a direction — two different
    # spaces — and silently invert every short. `meta_label` now refuses that.
    labelled_positions = {int(i): k for k, i in enumerate(directional.indices)}
    features = np.array(
        [[rows[labelled_positions[int(i)]][name] for name in rows[0]]
         for i in directional.indices]
    ) if rows else np.empty((0, 0))
    names = list(rows[0]) if rows else []
    y = (directional.y == 1).astype(int)

    starts = directional.indices
    ends = directional.touch_indices
    weights = sample_uniqueness(starts, ends)

    accuracies: list[float] = []
    precisions: list[float] = []
    taken = 0
    folds = 0
    for train, test in PurgedWalkForward(n_splits=n_folds, min_train=30).split(starts, ends):
        if train.size < 20 or test.size == 0:
            continue
        model = LogisticClassifier().fit(features[train], y[train], weights[train])
        probabilities = model.predict_proba(features[test])
        predicted = (probabilities >= threshold).astype(int)
        accuracies.append(float((predicted == y[test]).mean()))
        if predicted.sum() > 0:
            precisions.append(float(y[test][predicted == 1].mean()))
            taken += int(predicted.sum())
        folds += 1

    notes: list[str] = []
    if folds == 0:
        notes.append(
            "no usable folds — purging removed every training set, which means "
            "label windows are long relative to the number of signals"
        )
    if not precisions:
        notes.append(
            "the filter declined every out-of-sample signal at this threshold"
        )

    final = LogisticClassifier().fit(features, y, weights)
    report = MetaTrainingReport(
        n_signals=len(y),
        n_taken=taken,
        base_rate=float(y.mean()) if y.size else 0.0,
        oos_accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
        oos_precision=float(np.mean(precisions)) if precisions else 0.0,
        feature_names=names,
        coefficients=list(final.coefficients) if final.coefficients is not None else [],
        folds=folds,
        notes=notes,
    )
    filtered = MetaLabelFilter(
        primary, final, threshold=threshold, feature_fn=feature_fn, feature_names=names
    )
    return filtered, report
