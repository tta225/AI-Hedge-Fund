"""What a feature is worth, measured by breaking it.

Tree-model feature importance (``feature_importances_``, mean decrease in
impurity) is computed *in sample* and is biased in a way that matters here: it
inflates high-cardinality and noisy features, because a continuous variable
offers more places to split and will always find one that helps on training
data. In a financial dataset, where most features are continuous and most
relationships are weak, MDI reliably ranks noise above signal.

Mean decrease accuracy asks a better question. Fit the model, score it out of
sample, then **shuffle one feature** and score again. The drop is that
feature's contribution. A feature the model relied on causes a large drop; a
feature it ignored causes none; a feature that was actively misleading can
cause a *negative* drop, meaning the model scored better without it.

Two things make this implementation specific to finance rather than generic:

*It uses purged splits.* The scoring folds come from
:mod:`axiom.ml.crossval`, so the out-of-sample score is not contaminated by
overlapping labels. MDA computed on leaked folds measures a feature's
usefulness for memorising, which is exactly backwards.

*It shuffles within the fold.* Permuting across the whole sample would break
the time ordering as well as the feature, conflating "this feature matters"
with "temporal structure matters".
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

#: Shuffles per feature per fold. More is smoother and linearly more expensive.
DEFAULT_REPEATS = 3


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    """Importance of one feature, with its dispersion across folds."""

    name: str
    #: Mean score drop when the feature is shuffled. Higher is more important.
    importance: float
    #: Standard deviation of that drop across folds and repeats.
    std: float
    n_observations: int

    @property
    def is_significant(self) -> bool:
        """Whether the drop is larger than its own variability.

        A crude bar deliberately: with a handful of folds there is no basis for
        a real test, and anything fancier would imply a precision this sample
        size cannot support. Treat it as a filter, not a p-value.
        """
        return self.importance > self.std > 0

    def __str__(self) -> str:
        marker = "*" if self.is_significant else " "
        return f"{marker} {self.name:<28} {self.importance:+.4f} ± {self.std:.4f}"


def mean_decrease_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    splits: Iterator[tuple[np.ndarray, np.ndarray]],
    *,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    feature_names: Sequence[str] | None = None,
    sample_weights: np.ndarray | None = None,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
) -> list[FeatureImportance]:
    """Rank features by out-of-sample damage done when each is shuffled.

    Args:
        features: ``(n_samples, n_features)``.
        labels: ``(n_samples,)`` class labels.
        splits: purged ``(train, test)`` folds — pass
            ``PurgedKFold(...).split(starts, ends)``. Consumed once, so it is
            materialised internally.
        fit_predict: ``(X_train, y_train, X_test) -> predictions``. Injected
            rather than hard-wired to a library so this module stays free of
            any ML dependency, and so the same routine can rank features for a
            gradient booster, a linear model, or a hand-written rule.
        feature_names: column names. Defaults to ``f0..fn``.
        sample_weights: per-sample weights, e.g. from
            :func:`~axiom.ml.features.sample_uniqueness`. Applied to scoring.
        repeats: shuffles per feature per fold.
        seed: RNG seed, so a ranking is reproducible.

    Returns:
        Importances, most important first.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("features must be 2-D (n_samples, n_features)")
    if len(features) != len(labels):
        raise ValueError(
            f"{len(features)} feature rows but {len(labels)} labels"
        )
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    n_features = features.shape[1]
    names = (
        [f"f{i}" for i in range(n_features)]
        if feature_names is None
        else list(feature_names)
    )
    if len(names) != n_features:
        raise ValueError(f"{len(names)} names for {n_features} features")

    weights = (
        np.ones(len(labels))
        if sample_weights is None
        else np.asarray(sample_weights, dtype=float)
    )
    folds = [(np.asarray(tr), np.asarray(te)) for tr, te in splits]
    folds = [(tr, te) for tr, te in folds if tr.size > 0 and te.size > 0]
    if not folds:
        raise ValueError(
            "no usable folds — purging may have emptied every training set, "
            "which happens when label windows are long relative to the sample"
        )

    rng = np.random.default_rng(seed)
    drops: list[list[float]] = [[] for _ in range(n_features)]
    scored = 0

    for train, test in folds:
        baseline_predictions = fit_predict(
            features[train], labels[train], features[test]
        )
        baseline = _weighted_accuracy(
            labels[test], baseline_predictions, weights[test]
        )
        scored += len(test)

        for column in range(n_features):
            for _ in range(repeats):
                shuffled = features[test].copy()
                # Permute within the fold only: shuffling across the whole
                # sample would break time ordering as well as the feature.
                shuffled[:, column] = rng.permutation(shuffled[:, column])
                predictions = fit_predict(features[train], labels[train], shuffled)
                score = _weighted_accuracy(labels[test], predictions, weights[test])
                drops[column].append(baseline - score)

    importances = [
        FeatureImportance(
            name=names[column],
            importance=float(np.mean(drops[column])),
            std=float(np.std(drops[column], ddof=1)) if len(drops[column]) > 1 else 0.0,
            n_observations=scored,
        )
        for column in range(n_features)
    ]
    return sorted(importances, key=lambda f: f.importance, reverse=True)


def _weighted_accuracy(
    truth: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted fraction correct."""
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return float(np.sum(weights * (truth == predicted)) / total)


def render_importances(importances: Sequence[FeatureImportance]) -> str:
    """Human-readable ranking, significant features marked."""
    if not importances:
        return "no features scored"
    lines = [
        f"Mean decrease accuracy over {importances[0].n_observations} "
        "out-of-sample observations",
        "(* = drop exceeds its own dispersion; negative = model did better "
        "without the feature)",
        "",
    ]
    lines += [str(f) for f in importances]
    return "\n".join(lines)
