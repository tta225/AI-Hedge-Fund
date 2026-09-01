"""Stationarity without amnesia, and weights that account for overlap.

Two problems, both of which quietly ruin financial ML models.

**The stationarity/memory dilemma.** Models want stationary inputs. Prices are
not stationary, so everyone differences them into returns — and integer
differencing is a sledgehammer. ``d=1`` makes the series stationary and
destroys essentially all of its memory: today's return says nothing about the
price level, the distance from a year's high, or where the series sits in its
own history. Every long-horizon relationship is deleted to fix a statistical
technicality.

Fractional differentiation takes the derivative to a **real-valued order**.
Somewhere between ``d=0`` (raw prices, maximum memory, non-stationary) and
``d=1`` (returns, no memory, stationary) sits the smallest ``d`` that passes a
stationarity test. That series keeps most of the level information *and* is
usable. :func:`minimum_stationary_d` finds it.

**Overlapping labels are not independent observations.** A triple-barrier
label spanning bars 100–118 and another spanning 110–125 share fifteen bars of
outcome. Counting both as full observations overweights whatever happened in
the overlap, and in a busy period a single move can dominate the training set
through dozens of near-duplicate labels. :func:`sample_uniqueness` computes how
much of each label is its own, giving weights that fix it.
"""

from __future__ import annotations

import numpy as np

#: Weights below this are dropped from the fractional-difference kernel.
DEFAULT_WEIGHT_THRESHOLD = 1e-5
#: ADF-style threshold. See :func:`is_stationary` for what this is and is not.
DEFAULT_STATIONARITY_THRESHOLD = -2.86


def frac_diff_weights(d: float, size: int) -> np.ndarray:
    """Binomial weights for fractional differencing of order ``d``.

    ``w[0] = 1`` and ``w[k] = -w[k-1] * (d - k + 1) / k``, most recent first.

    At integer ``d`` the series terminates: ``d=1`` gives ``[1, -1, 0, 0, ...]``,
    which is exactly first differencing. At fractional ``d`` it never
    terminates, decaying slowly — and that long tail *is* the retained memory.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    weights = np.ones(size, dtype=float)
    for k in range(1, size):
        weights[k] = -weights[k - 1] * (d - k + 1) / k
    return weights


def fractional_difference(
    values: np.ndarray,
    d: float,
    *,
    threshold: float = DEFAULT_WEIGHT_THRESHOLD,
    max_width: int | None = None,
) -> np.ndarray:
    """Fractionally difference a series, fixed-width window.

    The kernel is truncated where weights fall below ``threshold``, which keeps
    every output point computed from the same number of terms. The alternative
    — an expanding window — gives early points a different effective ``d`` from
    later ones, so the feature's meaning drifts across the sample and the model
    learns partly to detect position in the series.

    The first ``width - 1`` outputs are NaN. They have no complete window, and
    filling them with a partial one is the drift this function exists to avoid.

    **The kernel is long, and that is the cost of the memory.** At the default
    threshold the width runs to roughly 125 terms at ``d=0.9``, 470 at
    ``d=0.65``, and 3,400 at ``d=0.2`` — so a low ``d`` on a short series can
    leave almost nothing usable. Check how many finite outputs come back before
    building features on them, and cap with ``max_width`` when the warm-up
    matters more than the tail of the kernel.

    Args:
        values: the series, oldest first. Prices, or logs of them.
        d: differencing order. ``0`` returns the input; ``1`` is first
            differencing.
        threshold: kernel truncation cutoff.
        max_width: hard cap on kernel length.

    Returns:
        The differenced series, same length, NaN-padded at the front.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("fractional_difference expects a 1-D series")
    if d < 0:
        raise ValueError(f"d must be non-negative, got {d}")
    if d == 0:
        return values.copy()

    span = len(values) if max_width is None else min(len(values), max_width)
    weights = frac_diff_weights(d, max(span, 1))
    keep = np.flatnonzero(np.abs(weights) >= threshold)
    width = int(keep[-1]) + 1 if keep.size else 1
    weights = weights[:width]

    out = np.full(len(values), np.nan)
    if len(values) < width:
        return out
    # weights[0] applies to the most recent point, so reverse for the dot.
    kernel = weights[::-1]
    for i in range(width - 1, len(values)):
        window = values[i - width + 1 : i + 1]
        if np.isfinite(window).all():
            out[i] = float(np.dot(kernel, window))
    return out


def _adf_statistic(values: np.ndarray) -> float:
    """Augmented Dickey-Fuller t-statistic, lag 1, no trend term.

    Regresses ``Δy[t]`` on ``y[t-1]`` and ``Δy[t-1]`` and returns the
    t-statistic on the ``y[t-1]`` coefficient. More negative means more
    evidence against a unit root.

    Hand-rolled rather than taken from statsmodels: this is the only test the
    package needs, it is twenty lines of least squares, and a heavyweight
    dependency for one statistic is a poor trade. The consequence is that the
    critical value is hard-coded rather than interpolated from a table, so
    treat the output as a *ranking* device for choosing ``d``, not as a
    publication-grade hypothesis test.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return float("nan")

    delta = np.diff(values)
    y = delta[1:]
    lagged_level = values[1:-1]
    lagged_delta = delta[:-1]
    design = np.column_stack([lagged_level, lagged_delta, np.ones(len(y))])

    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")

    residuals = y - design @ beta
    dof = len(y) - design.shape[1]
    if dof <= 0:
        return float("nan")
    sigma_squared = float(residuals @ residuals) / dof
    try:
        covariance = np.linalg.inv(design.T @ design) * sigma_squared
    except np.linalg.LinAlgError:
        return float("nan")
    standard_error = float(np.sqrt(covariance[0, 0]))
    if standard_error <= 0:
        return float("nan")
    return float(beta[0] / standard_error)


def is_stationary(
    values: np.ndarray, threshold: float = DEFAULT_STATIONARITY_THRESHOLD
) -> bool:
    """Whether a series clears the ADF threshold. See :func:`_adf_statistic`."""
    stat = _adf_statistic(values)
    return bool(np.isfinite(stat) and stat < threshold)


def minimum_stationary_d(
    values: np.ndarray,
    *,
    candidates: np.ndarray | None = None,
    threshold: float = DEFAULT_STATIONARITY_THRESHOLD,
    weight_threshold: float = DEFAULT_WEIGHT_THRESHOLD,
) -> tuple[float | None, np.ndarray]:
    """Smallest ``d`` whose differenced series passes the stationarity test.

    Smallest, not any — the whole point is to give up as little memory as
    possible. Searching upward and stopping at the first pass is the method.

    Returns:
        ``(d, differenced)``. ``d`` is None when nothing in ``candidates``
        passes, and ``differenced`` is then the last attempt, so a caller can
        inspect why rather than getting an empty result.
    """
    grid = np.arange(0.0, 1.01, 0.05) if candidates is None else np.asarray(candidates)
    differenced = np.asarray(values, dtype=float)
    for d in grid:
        differenced = fractional_difference(
            values, float(d), threshold=weight_threshold
        )
        if is_stationary(differenced, threshold):
            return float(d), differenced
    return None, differenced


def sample_uniqueness(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Average uniqueness of each label, in ``(0, 1]``.

    For every bar, count how many label windows span it. A label's uniqueness
    is the mean of ``1 / concurrency`` over the bars it spans: a label sharing
    every one of its bars with three others scores ~0.25.

    Used as sample weights, this stops a busy stretch of overlapping labels
    from counting as many independent observations when it is really one event
    seen repeatedly.

    Args:
        starts: opening bar of each label window.
        ends: resolving bar, inclusive.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have the same length")
    if starts.size == 0:
        return np.array([])
    if np.any(ends < starts):
        raise ValueError("a label cannot resolve before it opens")

    span = int(ends.max()) + 1
    # Difference array, so concurrency over all bars costs one pass.
    deltas = np.zeros(span + 1, dtype=np.int64)
    np.add.at(deltas, starts, 1)
    np.add.at(deltas, ends + 1, -1)
    concurrency = np.cumsum(deltas)[:span]

    out = np.empty(len(starts), dtype=float)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        window = concurrency[start : end + 1]
        # Concurrency is at least 1 across a label's own span by construction.
        out[i] = float(np.mean(1.0 / np.maximum(window, 1)))
    return out


def time_decay_weights(
    uniqueness: np.ndarray, *, last_weight: float = 1.0
) -> np.ndarray:
    """Linearly decay sample weights so recent observations count for more.

    Args:
        uniqueness: per-sample uniqueness from :func:`sample_uniqueness`,
            oldest first.
        last_weight: weight of the *oldest* observation relative to the newest.
            ``1.0`` disables decay. ``0.0`` fades the oldest to nothing.
            Negative values erase the oldest fraction of the sample entirely —
            López de Prado's convention, and a blunt instrument: use it only
            when there is a reason to believe old data is actively misleading
            rather than merely stale.
    """
    uniqueness = np.asarray(uniqueness, dtype=float)
    if uniqueness.size == 0:
        return np.array([])
    if last_weight > 1.0:
        raise ValueError("last_weight must be <= 1")

    cumulative = uniqueness.cumsum()
    total = cumulative[-1]
    if total <= 0:
        return np.ones_like(uniqueness)

    slope = (
        (1.0 - last_weight) / total
        if last_weight >= 0
        else 1.0 / ((last_weight + 1) * total)
    )
    intercept = 1.0 - slope * total
    weights = intercept + slope * cumulative
    return np.asarray(np.clip(weights, 0.0, None), dtype=float)
