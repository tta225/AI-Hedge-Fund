"""Tests for the classifier and the meta-label filter."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import make_series

from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.models import ICTState
from axiom.ml.classifier import LogisticClassifier, NotFittedError
from axiom.ml.labelling import Barriers, meta_label, triple_barrier_labels
from axiom.ml.meta import (
    MetaLabelFilter,
    MetaTrainingReport,
    collect_signals,
    default_features,
    train_meta_filter,
)
from axiom.portfolio.positions import Portfolio
from axiom.strategy.base import Signal, Strategy, StrategyContext


def _separable(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    features = np.column_stack([labels + rng.normal(0, 0.4, n), rng.normal(size=n)])
    return features, labels


class TestLogisticClassifier:
    def test_learns_a_separable_problem(self) -> None:
        features, labels = _separable()
        model = LogisticClassifier().fit(features, labels)
        assert (model.predict(features) == labels).mean() > 0.8

    def test_ranks_signal_above_noise(self) -> None:
        features, labels = _separable()
        model = LogisticClassifier().fit(features, labels)
        assert model.coefficients is not None
        assert abs(model.coefficients[0]) > abs(model.coefficients[1])

    def test_probabilities_are_bounded(self) -> None:
        features, labels = _separable()
        probabilities = LogisticClassifier().fit(features, labels).predict_proba(features)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()

    def test_extreme_inputs_do_not_overflow(self) -> None:
        """Outlier bars are exactly where a risk system must not produce NaN."""
        features, labels = _separable()
        model = LogisticClassifier().fit(features, labels)
        extreme = model.predict_proba(np.array([[1e12, 1e12], [-1e12, -1e12]]))
        assert np.isfinite(extreme).all()

    def test_a_constant_column_does_not_poison_the_fit(self) -> None:
        """Zero variance would divide by zero and NaN every coefficient."""
        rng = np.random.default_rng(1)
        n = 200
        labels = rng.integers(0, 2, n)
        features = np.column_stack([labels + rng.normal(0, 0.4, n), np.ones(n)])
        model = LogisticClassifier().fit(features, labels)
        assert model.coefficients is not None
        assert np.isfinite(model.coefficients).all()

    def test_sample_weights_are_actually_applied(self) -> None:
        """Ignoring them silently undoes the uniqueness correction."""
        features, labels = _separable(n=200)
        unweighted = LogisticClassifier().fit(features, labels)
        weights = np.where(labels == 1, 10.0, 0.1)
        weighted = LogisticClassifier().fit(features, labels, weights)
        assert unweighted.intercept != pytest.approx(weighted.intercept)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            LogisticClassifier().predict_proba(np.zeros((1, 2)))

    def test_rejects_non_binary_labels(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            LogisticClassifier().fit(np.zeros((4, 2)), np.array([-1, 1, -1, 1]))

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            LogisticClassifier().fit(np.zeros((4, 2)), np.array([0, 1]))

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            LogisticClassifier().fit(
                np.zeros((2, 2)), np.array([0, 1]), np.array([-1.0, 1.0])
            )

    def test_rejects_empty_sample(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            LogisticClassifier().fit(np.zeros((0, 2)), np.array([]))

    def test_feature_count_is_enforced_at_predict(self) -> None:
        features, labels = _separable(n=100)
        model = LogisticClassifier().fit(features, labels)
        with pytest.raises(ValueError, match="expected 2 features"):
            model.predict_proba(np.zeros((1, 5)))

    def test_test_data_uses_training_statistics(self) -> None:
        """Standardising the test set on its own stats leaks its distribution."""
        features, labels = _separable(n=200)
        model = LogisticClassifier().fit(features, labels)
        single = model.predict_proba(features[:1])
        batch = model.predict_proba(features)
        assert single[0] == pytest.approx(batch[0])

    def test_explain_is_readable(self) -> None:
        features, labels = _separable(n=100)
        model = LogisticClassifier().fit(features, labels)
        assert "intercept" in model.explain(["a", "b"])

    def test_explain_before_fit(self) -> None:
        assert LogisticClassifier().explain() == "not fitted"


class TestMetaLabelGuard:
    def test_refuses_labels_that_already_encode_the_trade(self) -> None:
        """Re-meta-labelling side-oriented labels inverts every short."""
        series = _rising()
        events = np.arange(60, 150)
        sided = triple_barrier_labels(
            series, events=events, sides=-np.ones(len(events), dtype=int)
        )
        with pytest.raises(ValueError, match="already encode"):
            meta_label(sided, np.ones(len(sided), dtype=int))

    def test_allows_directional_labels(self) -> None:
        series = _rising()
        directional = triple_barrier_labels(series)
        result = meta_label(directional, np.ones(len(directional), dtype=int))
        assert set(result.y) <= {0, 1}


def _rising(n: int = 400, step: float = 0.3, *, evidential: bool = False) -> OHLCVSeries:
    """A rising tape. `evidential` stamps it real, which training requires.

    `OHLCVSeries` is frozen, so provenance is chosen at construction rather
    than patched afterwards — which is the point of making it frozen.
    """
    rng = np.random.default_rng(5)
    rows = []
    price = 100.0
    for _ in range(n):
        price += step + rng.normal(0, 0.6)
        rows.append((price, price + 0.8, price - 0.8, price))
    series = make_series(rows, timeframe="1h")
    if not evidential:
        return series
    return OHLCVSeries.from_dataframe(
        series.df, series.instrument, "1h", Provenance.real("test_exchange")
    )


class _EveryBar(Strategy):
    """Fires on every bar, so signal collection has something to survey."""

    name = "every_bar"
    requires_ict = False

    def evaluate(self, context: StrategyContext) -> Signal | None:
        price = context.price
        return Signal(
            direction=Direction.BULLISH,
            entry=price,
            stop=price * 0.98,
            targets=(price * 1.04,),
        )


class _Silent(Strategy):
    name = "silent"
    requires_ict = False

    def evaluate(self, context: StrategyContext) -> Signal | None:
        return None


def _context(index: int = 200) -> StrategyContext:
    series = _rising()
    return StrategyContext(
        series=series,
        index=index,
        ict=ICTState(symbol="ES", timeframe="1h", as_of=series.index[index], index=index),
        portfolio=Portfolio(starting_cash=1_000_000),
        timestamp=series.index[index],
    )


class TestDefaultFeatures:
    def test_produces_finite_values(self) -> None:
        context = _context()
        signal = _EveryBar().evaluate(context)
        assert signal is not None
        features = default_features(context, signal)
        assert all(np.isfinite(v) for v in features.values())

    def test_does_not_encode_direction(self) -> None:
        """A meta-model that learns direction is no longer a filter."""
        context = _context()
        long_signal = _EveryBar().evaluate(context)
        assert long_signal is not None
        price = context.price
        short_signal = Signal(
            direction=Direction.BEARISH,
            entry=price,
            stop=price * 1.02,
            targets=(price * 0.96,),
        )
        long_features = default_features(context, long_signal)
        short_features = default_features(context, short_signal)
        # Only `bias_agrees` may differ, and only because the bias itself is
        # directional — the geometry features must be identical.
        for name in ("atr_pct", "risk_atr", "reward_risk", "position_in_range",
                     "momentum_20", "hour_utc"):
            assert long_features[name] == pytest.approx(short_features[name])


class TestCollectSignals:
    def test_surveys_every_opinion_not_just_traded_ones(self) -> None:
        series = _rising()
        indices, sides, rows = collect_signals(_EveryBar(), series, warmup=100)
        assert len(indices) == len(series) - 100
        assert len(indices) == len(sides) == len(rows)

    def test_a_silent_strategy_yields_nothing(self) -> None:
        indices, _, _ = collect_signals(_Silent(), _rising(), warmup=100)
        assert indices == []


class TestMetaLabelFilter:
    def test_an_unfitted_filter_passes_everything_through(self) -> None:
        """Before training the wrapper must be a no-op, not a veto."""
        wrapped = MetaLabelFilter(_EveryBar())
        assert wrapped.evaluate(_context()) is not None

    def test_a_confident_model_lets_signals_through(self) -> None:
        wrapped, _ = _fitted_filter(threshold=0.01)
        assert wrapped.evaluate(_context()) is not None

    def test_a_strict_threshold_declines(self) -> None:
        wrapped, _ = _fitted_filter(threshold=0.99)
        assert wrapped.evaluate(_context()) is None

    def test_declines_are_counted(self) -> None:
        wrapped, _ = _fitted_filter(threshold=0.99)
        wrapped.evaluate(_context())
        assert wrapped.seen == 1 and wrapped.declined == 1

    def test_levels_are_never_modified(self) -> None:
        """The filter decides whether to trade, not where the stop goes."""
        wrapped, _ = _fitted_filter(threshold=0.01)
        context = _context()
        original = _EveryBar().evaluate(context)
        filtered = wrapped.evaluate(context)
        assert original is not None and filtered is not None
        assert filtered.entry == original.entry
        assert filtered.stop == original.stop
        assert filtered.targets == original.targets
        assert filtered.direction is original.direction

    def test_confidence_carries_the_probability(self) -> None:
        wrapped, _ = _fitted_filter(threshold=0.01)
        signal = wrapped.evaluate(_context())
        assert signal is not None
        assert 0.0 <= signal.confidence <= 1.0
        assert "meta_approved" in signal.tags

    def test_a_silent_primary_stays_silent(self) -> None:
        assert MetaLabelFilter(_Silent()).evaluate(_context()) is None

    def test_exits_are_delegated_untouched(self) -> None:
        """A model that can close positions can decide to stop protecting one."""
        class _WantsOut(_EveryBar):
            def should_exit(self, context: StrategyContext) -> str | None:
                return "primary says so"

        wrapped = MetaLabelFilter(_WantsOut())
        assert wrapped.should_exit(_context()) == "primary says so"

    def test_rejects_an_impossible_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            MetaLabelFilter(_EveryBar(), threshold=1.5)

    def test_inherits_the_primary_ict_requirement(self) -> None:
        from axiom.strategy.sweep_continuation import SweepContinuationStrategy

        assert MetaLabelFilter(SweepContinuationStrategy()).requires_ict is True
        assert MetaLabelFilter(_EveryBar()).requires_ict is False


def _fitted_filter(threshold: float) -> tuple[MetaLabelFilter, MetaTrainingReport]:
    return train_meta_filter(_EveryBar(), _rising(evidential=True), threshold=threshold)


class TestTraining:
    def test_refuses_non_evidential_data(self) -> None:
        """A filter fitted to generated bars describes the generator."""
        with pytest.raises(ValueError, match="refusing to train"):
            train_meta_filter(_EveryBar(), _rising())

    def test_refuses_too_few_signals(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            train_meta_filter(_Silent(), _rising(n=150, evidential=True))

    def test_produces_a_fitted_filter(self) -> None:
        wrapped, report = _fitted_filter(threshold=0.5)
        assert wrapped.model is not None and wrapped.model.is_fitted
        assert report.n_signals > 0

    def test_report_renders(self) -> None:
        _, report = _fitted_filter(threshold=0.5)
        assert "Meta-label filter" in report.render()
        assert "std err" in report.render()

    def test_barriers_are_configurable(self) -> None:
        _, report = train_meta_filter(
            _EveryBar(), _rising(evidential=True), barriers=Barriers(1.0, 1.0, 5)
        )
        assert report.n_signals > 0


class TestUsefulnessBar:
    @staticmethod
    def _report(**kwargs: object) -> MetaTrainingReport:
        base: dict[str, object] = {
            "n_signals": 500, "n_taken": 300, "base_rate": 0.50,
            "oos_accuracy": 0.55, "oos_precision": 0.55, "folds": 4,
        }
        base.update(kwargs)
        return MetaTrainingReport(**base)  # type: ignore[arg-type]

    def test_a_lift_inside_one_standard_error_is_not_useful(self) -> None:
        """+2pp on 340 trades is noise, however encouraging it looks."""
        report = self._report(base_rate=0.526, oos_precision=0.546, n_taken=340)
        assert report.lift > 0
        assert not report.is_useful

    def test_a_large_lift_is_useful(self) -> None:
        report = self._report(base_rate=0.50, oos_precision=0.70, n_taken=300)
        assert report.is_useful

    def test_a_filter_that_declines_almost_everything_is_not_useful(self) -> None:
        """Wonderful precision on two trades tells you nothing."""
        report = self._report(base_rate=0.50, oos_precision=1.00, n_taken=2)
        assert not report.is_useful

    def test_zero_folds_is_not_useful(self) -> None:
        report = self._report(base_rate=0.50, oos_precision=0.90, folds=0)
        assert not report.is_useful

    def test_negative_lift_is_not_useful(self) -> None:
        assert not self._report(base_rate=0.60, oos_precision=0.55).is_useful

    def test_standard_error_shrinks_with_sample(self) -> None:
        small = self._report(n_taken=50).lift_standard_error
        large = self._report(n_taken=5000).lift_standard_error
        assert small > large
