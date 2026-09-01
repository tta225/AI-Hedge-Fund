"""Cross-validation that survives overlapping labels.

Standard K-fold assumes observations are independent. Financial labels are not,
and the violation is severe rather than technical.

A triple-barrier label at bar 100 may not resolve until bar 118. Its outcome is
a function of bars 100–118. If bar 110 lands in the test fold and bar 100 in
the training fold, the training set contains an observation whose label *is
partly the answer to the test question*. The model does not need to generalise;
it can memorise. Cross-validated accuracy goes up, live accuracy does not, and
nothing in the fold structure reveals the problem.

Two corrections, both from López de Prado (2018):

**Purging.** Drop from the training set every observation whose label window
overlaps the test window at all. This is the correction that matters, and it
requires knowing each label's resolution bar — which is why
:class:`~axiom.ml.labelling.Label` records ``touch_index``.

**Embargo.** Drop a further band of training observations immediately *after*
the test window. Purging handles direct overlap; the embargo handles serial
correlation, where a training observation that starts just after the test set
still carries information about it through autocorrelated features. The
embargo is the cheap insurance against the leak purging cannot see.

The cost is real: purging with long label windows can remove a large fraction
of the training data. That is the correct price. A model trained on leaked data
is not a model, and a smaller honest training set beats a larger dishonest one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

#: Fraction of the sample embargoed after each test window.
DEFAULT_EMBARGO_PCT = 0.01


def embargo_size(n_samples: int, embargo_pct: float = DEFAULT_EMBARGO_PCT) -> int:
    """Number of observations to embargo after a test window."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if not 0.0 <= embargo_pct < 1.0:
        raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")
    return int(n_samples * embargo_pct)


def purged_train_indices(
    starts: np.ndarray,
    ends: np.ndarray,
    test_indices: np.ndarray,
    *,
    embargo: int = 0,
    n_samples: int | None = None,
) -> np.ndarray:
    """Training indices whose label windows do not touch the test window.

    Args:
        starts: bar at which each observation's label window opens.
        ends: bar at which it resolves (``touch_index``). Must be >= ``starts``.
        test_indices: positions (into ``starts``/``ends``) held out for testing.
        embargo: observations to additionally drop after the test window.
        n_samples: total observation count. Defaults to ``len(starts)``.

    Returns:
        Sorted positions safe to train on.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have the same length")
    if np.any(ends < starts):
        raise ValueError("a label cannot resolve before it opens")
    total = len(starts) if n_samples is None else n_samples

    test_indices = np.asarray(test_indices, dtype=np.int64)
    if test_indices.size == 0:
        return np.arange(total, dtype=np.int64)

    # The test window in *bar* space, not position space: it spans from the
    # earliest test label's start to the latest test label's resolution. Using
    # positions here would miss a long-horizon label that resolves well beyond
    # the last test position.
    test_start = int(starts[test_indices].min())
    test_end = int(ends[test_indices].max())
    embargo_end = test_end + max(embargo, 0)

    # Two intervals overlap unless one ends before the other begins.
    overlaps = (starts <= embargo_end) & (ends >= test_start)
    keep = ~overlaps
    keep[test_indices] = False
    return np.flatnonzero(keep).astype(np.int64)


@dataclass(frozen=True, slots=True)
class _Fold:
    train: np.ndarray
    test: np.ndarray


class PurgedKFold:
    """K-fold over contiguous blocks, with purging and an embargo.

    Folds are **contiguous in time**, never shuffled. Shuffling financial
    observations destroys the very adjacency the purge is trying to reason
    about, and a shuffled fold cannot be purged coherently because its test
    window is the whole sample.

    Args:
        n_splits: number of folds.
        embargo_pct: fraction of the sample embargoed after each test window.

    Example:
        >>> import numpy as np
        >>> cv = PurgedKFold(n_splits=3)
        >>> starts = np.arange(30)
        >>> ends = starts + 5          # each label resolves 5 bars later
        >>> folds = list(cv.split(starts, ends))
        >>> len(folds)
        3
        >>> all(np.intersect1d(tr, te).size == 0 for tr, te in folds)
        True
    """

    def __init__(self, n_splits: int = 5, *, embargo_pct: float = DEFAULT_EMBARGO_PCT) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` per fold."""
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        n = len(starts)
        if n < self.n_splits:
            raise ValueError(f"{n} observations cannot fill {self.n_splits} folds")

        embargo = embargo_size(n, self.embargo_pct)
        for test_indices in np.array_split(np.arange(n, dtype=np.int64), self.n_splits):
            train = purged_train_indices(
                starts, ends, test_indices, embargo=embargo, n_samples=n
            )
            yield train, test_indices

    def purge_ratio(self, starts: np.ndarray, ends: np.ndarray) -> float:
        """Fraction of training data the purge removes, averaged over folds.

        Worth checking before trusting a result. A ratio near 1 means the label
        windows are so long relative to the sample that almost nothing survives
        purging, and whatever the model learned came from a handful of
        observations.
        """
        n = len(np.asarray(starts))
        lost = [
            1.0 - len(train) / max(n - len(test), 1)
            for train, test in self.split(starts, ends)
        ]
        return float(np.mean(lost)) if lost else 0.0


class PurgedWalkForward:
    """Walk-forward folds with purging — training strictly precedes testing.

    :class:`PurgedKFold` trains on data from *both* sides of the test window,
    which is legitimate for measuring whether a relationship exists but does
    not describe deployment: in production the model has only the past. This
    splitter answers the deployment question instead, and adds the purge that
    a plain walk-forward lacks.

    Prefer this for anything that will be traded. Prefer :class:`PurgedKFold`
    when the question is scientific rather than operational — feature
    importance, say, where using both sides of the window is more efficient and
    the model is never deployed.
    """

    def __init__(
        self,
        n_splits: int = 4,
        *,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
        min_train: int = 50,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if min_train < 1:
            raise ValueError("min_train must be at least 1")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.min_train = min_train

    def split(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        n = len(starts)
        test_size = (n - self.min_train) // self.n_splits
        if test_size < 1:
            raise ValueError(
                f"{n} observations cannot support {self.n_splits} folds after a "
                f"{self.min_train}-observation minimum training window"
            )

        embargo = embargo_size(n, self.embargo_pct)
        for fold in range(self.n_splits):
            train_end = self.min_train + fold * test_size
            test_indices = np.arange(
                train_end, min(train_end + test_size, n), dtype=np.int64
            )
            if test_indices.size == 0:
                continue
            # Only the past is available, and only the part of the past whose
            # labels resolved before the test window opened.
            test_start_bar = int(starts[test_indices].min())
            candidate = np.arange(train_end, dtype=np.int64)
            train = candidate[ends[candidate] < test_start_bar - embargo]
            yield train, test_indices
