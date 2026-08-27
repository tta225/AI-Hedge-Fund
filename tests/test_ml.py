"""Tests for the financial-ML package: labelling, purging, features, importance."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import make_series

from axiom.core.series import OHLCVSeries
from axiom.ml import (
    Barriers,
    PurgedKFold,
    PurgedWalkForward,
    embargo_size,
    frac_diff_weights,
    fractional_difference,
    mean_decrease_accuracy,
    meta_label,
    minimum_stationary_d,
    purged_train_indices,
    sample_uniqueness,
    time_decay_weights,
    triple_barrier_labels,
    volatility_target,
)


def _ramp(n: int = 200, step: float = 0.5, noise: float = 0.2) -> OHLCVSeries:
    """A rising tape with enough wiggle for a volatility estimate to exist."""
    rng = np.random.default_rng(7)
    rows = []
    price = 100.0
    for _ in range(n):
        price += step + rng.normal(0, noise)
        rows.append((price, price + 0.5, price - 0.5, price))
    return make_series(rows, timeframe="1h")


class TestVolatilityTarget:
    def test_is_causal(self) -> None:
        """A barrier width that sees the future makes the label easier to hit."""
        series = _ramp()
        full = volatility_target(series, 50)
        truncated = volatility_target(series.head(120), 50)
        n = len(truncated)
        both = np.isfinite(full[:n]) & np.isfinite(truncated)
        assert both.any()
        assert full[:n][both] == pytest.approx(truncated[both])

    def test_warmup_is_nan(self) -> None:
        vol = volatility_target(_ramp(), 50)
        assert np.isnan(vol[:50]).all()

    def test_rejects_degenerate_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            volatility_target(_ramp(), 1)


class TestBarriers:
    def test_rejects_two_disabled_horizontals(self) -> None:
        """Without a horizontal barrier this is a fixed-horizon label."""
        with pytest.raises(ValueError, match="at least one horizontal"):
            Barriers(profit_sigma=0.0, stop_sigma=0.0)

    def test_rejects_negative_widths(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Barriers(profit_sigma=-1.0)

    def test_rejects_zero_hold(self) -> None:
        with pytest.raises(ValueError, match="max_hold"):
            Barriers(max_hold=0)


class TestTripleBarrier:
    def test_labels_are_signed_and_resolve(self) -> None:
        labels = triple_barrier_labels(_ramp())
        assert len(labels) > 0
        assert set(labels.y) <= {-1, 1}
        assert (labels.touch_indices > labels.indices).all()

    def test_touch_never_exceeds_the_vertical_barrier(self) -> None:
        barriers = Barriers(max_hold=10)
        labels = triple_barrier_labels(_ramp(), barriers=barriers)
        assert (labels.touch_indices - labels.indices <= 10).all()

    def test_timeout_can_be_labelled_zero(self) -> None:
        """A trade that went nowhere is a third outcome, not a coin flip."""
        # Very wide barriers so nothing is touched horizontally.
        barriers = Barriers(profit_sigma=99.0, stop_sigma=99.0, max_hold=3,
                            zero_on_timeout=True)
        labels = triple_barrier_labels(_ramp(), barriers=barriers)
        assert set(labels.y) == {0}

    def test_a_rising_tape_hits_targets(self) -> None:
        labels = triple_barrier_labels(_ramp(step=1.0, noise=0.05))
        assert (labels.y == 1).sum() > (labels.y == -1).sum()

    def test_ambiguous_bar_resolves_against_the_trade(self) -> None:
        """Assuming the target on a both-barriers bar is free money."""
        labels = triple_barrier_labels(_ramp())
        ambiguous = [x for x in labels.labels if "both_barriers" in x.reason]
        assert all(x.label == -1 for x in ambiguous)

    def test_honours_an_event_subset(self) -> None:
        events = np.array([60, 80, 100])
        labels = triple_barrier_labels(_ramp(), events=events)
        assert set(labels.indices) <= set(events)

    def test_rejects_out_of_range_events(self) -> None:
        with pytest.raises(IndexError):
            triple_barrier_labels(_ramp(n=100), events=np.array([500]))

    def test_short_side_mirrors_the_barriers(self) -> None:
        """`+1` must mean 'the trade worked' on both sides."""
        series = _ramp(step=1.0, noise=0.05)
        events = np.arange(60, 150)
        longs = triple_barrier_labels(series, events=events, sides=np.ones(len(events), int))
        shorts = triple_barrier_labels(series, events=events, sides=-np.ones(len(events), int))
        # A rising tape rewards longs and punishes shorts.
        assert (longs.y == 1).mean() > (shorts.y == 1).mean()

    def test_rejects_mismatched_sides(self) -> None:
        with pytest.raises(ValueError, match="sides"):
            triple_barrier_labels(_ramp(), events=np.array([60, 70]), sides=np.array([1]))

    def test_inherits_provenance(self) -> None:
        """Labels are only as evidential as the bars they were cut from."""
        labels = triple_barrier_labels(_ramp())
        assert labels.provenance is _ramp().provenance or not labels.is_evidential

    def test_describe_reports_balance(self) -> None:
        assert "labels" in triple_barrier_labels(_ramp()).describe()


class TestMetaLabelling:
    def test_marks_correct_calls(self) -> None:
        labels = triple_barrier_labels(_ramp())
        perfect = meta_label(labels, labels.y)
        assert set(perfect.y) == {1}

    def test_marks_wrong_calls(self) -> None:
        labels = triple_barrier_labels(_ramp())
        inverted = meta_label(labels, -labels.y)
        assert set(inverted.y) == {0}

    def test_output_is_binary(self) -> None:
        labels = triple_barrier_labels(_ramp())
        sides = np.ones(len(labels), dtype=int)
        assert set(meta_label(labels, sides).y) <= {0, 1}

    def test_rejects_length_mismatch(self) -> None:
        labels = triple_barrier_labels(_ramp())
        with pytest.raises(ValueError, match="predicted sides"):
            meta_label(labels, np.ones(3, dtype=int))


class TestPurging:
    def test_overlapping_labels_are_purged(self) -> None:
        """The leak this module exists to stop."""
        starts = np.arange(20)
        ends = starts + 5
        train = purged_train_indices(starts, ends, np.array([10, 11, 12]))
        # Anything whose window touches bars 10..17 must be gone.
        assert not any(starts[i] <= 17 and ends[i] >= 10 for i in train)

    def test_disjoint_labels_survive(self) -> None:
        starts = np.array([0, 50])
        ends = np.array([1, 51])
        train = purged_train_indices(starts, ends, np.array([1]))
        assert train.tolist() == [0]

    def test_embargo_removes_more(self) -> None:
        starts = np.arange(40)
        ends = starts + 2
        without = purged_train_indices(starts, ends, np.array([20]), embargo=0)
        with_embargo = purged_train_indices(starts, ends, np.array([20]), embargo=5)
        assert len(with_embargo) < len(without)

    def test_test_indices_never_appear_in_train(self) -> None:
        starts, ends = np.arange(30), np.arange(30) + 3
        test = np.array([5, 6, 7])
        assert np.intersect1d(purged_train_indices(starts, ends, test), test).size == 0

    def test_rejects_backwards_labels(self) -> None:
        with pytest.raises(ValueError, match="resolve before"):
            purged_train_indices(np.array([5]), np.array([1]), np.array([0]))

    def test_embargo_size_scales(self) -> None:
        assert embargo_size(1000, 0.01) == 10
        assert embargo_size(1000, 0.0) == 0

    def test_embargo_pct_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="embargo_pct"):
            embargo_size(100, 1.5)


class TestPurgedKFold:
    def test_folds_are_disjoint_from_their_training_sets(self) -> None:
        cv = PurgedKFold(n_splits=4)
        starts, ends = np.arange(100), np.arange(100) + 10
        for train, test in cv.split(starts, ends):
            assert np.intersect1d(train, test).size == 0

    def test_every_observation_is_tested_once(self) -> None:
        cv = PurgedKFold(n_splits=5)
        starts, ends = np.arange(50), np.arange(50) + 2
        tested = np.concatenate([test for _, test in cv.split(starts, ends)])
        assert sorted(tested) == list(range(50))

    def test_longer_labels_purge_more(self) -> None:
        """The cost of purging scales with how long labels stay open."""
        starts = np.arange(100)
        short = PurgedKFold(n_splits=4).purge_ratio(starts, starts + 1)
        long = PurgedKFold(n_splits=4).purge_ratio(starts, starts + 20)
        assert long > short

    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="cannot fill"):
            list(PurgedKFold(n_splits=10).split(np.arange(3), np.arange(3)))

    def test_rejects_single_split(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            PurgedKFold(n_splits=1)


class TestPurgedWalkForward:
    def test_training_always_precedes_testing(self) -> None:
        """The deployment question: the model only ever has the past."""
        cv = PurgedWalkForward(n_splits=3, min_train=40)
        starts, ends = np.arange(100), np.arange(100) + 3
        for train, test in cv.split(starts, ends):
            if train.size:
                assert train.max() < test.min()

    def test_purges_labels_still_open_at_the_test_boundary(self) -> None:
        cv = PurgedWalkForward(n_splits=2, min_train=40)
        starts = np.arange(100)
        for train, test in cv.split(starts, starts + 15):
            if train.size:
                assert ends_clear(starts, train, test)

    def test_rejects_impossible_geometry(self) -> None:
        with pytest.raises(ValueError, match="cannot support"):
            list(PurgedWalkForward(n_splits=5, min_train=95).split(
                np.arange(96), np.arange(96)
            ))


def ends_clear(starts: np.ndarray, train: np.ndarray, test: np.ndarray) -> bool:
    return bool((starts[train] < starts[test].min()).all())


class TestFractionalDifference:
    def test_d_one_is_first_differencing(self) -> None:
        weights = frac_diff_weights(1.0, 6)
        assert weights[:2].tolist() == [1.0, -1.0]
        assert weights[2:] == pytest.approx(np.zeros(4))

    def test_d_zero_is_identity(self) -> None:
        values = np.arange(10, dtype=float)
        assert fractional_difference(values, 0.0) == pytest.approx(values)

    def test_fractional_d_keeps_a_long_tail(self) -> None:
        """The tail is the retained memory — that is the whole point."""
        weights = frac_diff_weights(0.5, 20)
        assert np.abs(weights[10:]).sum() > 0

    def test_rejects_negative_order(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            fractional_difference(np.arange(10, dtype=float), -0.5)

    def test_rejects_two_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            fractional_difference(np.zeros((5, 5)), 0.5)

    def test_warmup_is_nan(self) -> None:
        values = np.cumsum(np.random.default_rng(1).normal(size=300)) + 100
        out = fractional_difference(values, 0.9, max_width=50)
        assert np.isnan(out[:49]).all()
        assert np.isfinite(out[49:]).any()

    def test_finds_a_d_below_one_for_a_random_walk(self) -> None:
        """If the answer were always 1.0 the method would buy nothing."""
        values = np.cumsum(np.random.default_rng(3).normal(size=2000)) + 100
        d, _ = minimum_stationary_d(values)
        assert d is not None
        assert 0.0 < d < 1.0

    def test_reports_failure_rather_than_guessing(self) -> None:
        d, out = minimum_stationary_d(np.arange(30, dtype=float),
                                      candidates=np.array([0.0]))
        assert d is None
        assert out.size == 30


class TestSampleUniqueness:
    def test_disjoint_labels_are_fully_unique(self) -> None:
        u = sample_uniqueness(np.array([0, 10]), np.array([2, 12]))
        assert u == pytest.approx([1.0, 1.0])

    def test_identical_labels_split_their_weight(self) -> None:
        u = sample_uniqueness(np.array([0, 0, 0]), np.array([5, 5, 5]))
        assert u == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_uniqueness_is_bounded(self) -> None:
        starts = np.arange(50)
        u = sample_uniqueness(starts, starts + 10)
        assert (u > 0).all() and (u <= 1).all()

    def test_more_overlap_means_less_uniqueness(self) -> None:
        starts = np.arange(50)
        assert sample_uniqueness(starts, starts + 20).mean() < (
            sample_uniqueness(starts, starts + 2).mean()
        )

    def test_empty_input(self) -> None:
        assert sample_uniqueness(np.array([]), np.array([])).size == 0

    def test_rejects_backwards_windows(self) -> None:
        with pytest.raises(ValueError, match="resolve before"):
            sample_uniqueness(np.array([5]), np.array([1]))


class TestTimeDecay:
    def test_disabled_by_default_weight(self) -> None:
        u = np.ones(10)
        assert time_decay_weights(u, last_weight=1.0) == pytest.approx(np.ones(10))

    def test_recent_observations_weigh_more(self) -> None:
        weights = time_decay_weights(np.ones(100), last_weight=0.2)
        # The newest observation always carries full weight; the oldest
        # approaches `last_weight` (exactly it only in the continuous limit,
        # since the first observation already consumes one step of the ramp).
        assert weights[-1] == pytest.approx(1.0)
        assert weights[0] == pytest.approx(0.2, abs=0.01)

    def test_weights_never_go_negative(self) -> None:
        assert (time_decay_weights(np.ones(20), last_weight=-0.5) >= 0).all()

    def test_rejects_weights_above_one(self) -> None:
        with pytest.raises(ValueError, match="last_weight"):
            time_decay_weights(np.ones(5), last_weight=2.0)


class TestMeanDecreaseAccuracy:
    @staticmethod
    def _fit_predict(
        x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        """Threshold on column 0 — deliberately trivial and dependency-free."""
        mask = y_train == 1
        if not mask.any() or mask.all():
            return np.full(len(x_test), int(y_train[0]) if len(y_train) else 1)
        split = (x_train[mask, 0].mean() + x_train[~mask, 0].mean()) / 2
        above_is_positive = x_train[mask, 0].mean() > split
        above = x_test[:, 0] > split
        return np.where(above == above_is_positive, 1, -1)

    def test_ranks_the_informative_feature_first(self) -> None:
        rng = np.random.default_rng(0)
        n = 300
        labels = rng.choice([-1, 1], size=n)
        signal = labels + rng.normal(0, 0.3, size=n)
        noise = rng.normal(0, 1, size=n)
        features = np.column_stack([signal, noise])

        starts = np.arange(n)
        importances = mean_decrease_accuracy(
            features,
            labels,
            PurgedKFold(n_splits=3).split(starts, starts + 2),
            fit_predict=self._fit_predict,
            feature_names=["signal", "noise"],
            repeats=2,
        )
        assert importances[0].name == "signal"
        assert importances[0].importance > importances[1].importance

    def test_pure_noise_scores_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        labels = rng.choice([-1, 1], size=n)
        features = rng.normal(size=(n, 2))
        starts = np.arange(n)
        importances = mean_decrease_accuracy(
            features, labels,
            PurgedKFold(n_splits=3).split(starts, starts + 1),
            fit_predict=self._fit_predict, repeats=2,
        )
        assert all(abs(f.importance) < 0.25 for f in importances)

    def test_rejects_mismatched_shapes(self) -> None:
        starts = np.arange(10)
        with pytest.raises(ValueError, match="feature rows"):
            mean_decrease_accuracy(
                np.zeros((10, 2)), np.zeros(5),
                PurgedKFold(n_splits=2).split(starts, starts),
                fit_predict=self._fit_predict,
            )

    def test_rejects_bad_names(self) -> None:
        starts = np.arange(20)
        with pytest.raises(ValueError, match="names"):
            mean_decrease_accuracy(
                np.zeros((20, 2)), np.ones(20),
                PurgedKFold(n_splits=2).split(starts, starts),
                fit_predict=self._fit_predict, feature_names=["only_one"],
            )

    def test_is_reproducible(self) -> None:
        rng = np.random.default_rng(2)
        n = 150
        labels = rng.choice([-1, 1], size=n)
        features = np.column_stack([labels + rng.normal(0, 0.5, n), rng.normal(size=n)])
        starts = np.arange(n)
        kwargs = {"fit_predict": self._fit_predict, "repeats": 2, "seed": 42}
        first = mean_decrease_accuracy(
            features, labels, PurgedKFold(n_splits=3).split(starts, starts + 1), **kwargs
        )
        second = mean_decrease_accuracy(
            features, labels, PurgedKFold(n_splits=3).split(starts, starts + 1), **kwargs
        )
        assert [f.importance for f in first] == [f.importance for f in second]
