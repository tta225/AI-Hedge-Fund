"""Statistics for judging a strategy search honestly.

The problem this module exists to solve
---------------------------------------
Search 500 strategy/parameter combinations, keep the best Sharpe, and you will
find something impressive **even if every candidate is pure noise**. With 500
independent trials on a random series, the expected maximum Sharpe is around
2.5-3.0 purely from selection. Reporting that number as though one strategy had
been tested is not a small overstatement; it is the entire result.

Three corrections are implemented, following Bailey & López de Prado:

:func:`expected_max_sharpe`
    what the best of *N* trials looks like when none of them has any edge. The
    null hypothesis a search result must beat.

:func:`deflated_sharpe_ratio`
    the probability the observed Sharpe is genuinely above that null, adjusting
    for trial count, sample length, skewness, and kurtosis. Below ~0.95 the
    winner is not distinguishable from the best of a set of coin flips.

:func:`probability_of_backtest_overfitting`
    combinatorially-symmetric cross-validation. Splits the sample, checks how
    often the in-sample winner underperforms the median out of sample. Above
    ~0.5 the *selection procedure itself* is not producing generalising picks.

References:
  Bailey & López de Prado (2014), *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality*.
  Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest
  Overfitting*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

#: Euler-Mascheroni constant, used in the expected-maximum order statistic.
EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_max_sharpe(n_trials: int, variance_of_sharpes: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` when the true Sharpe is zero.

    This is the bar a search winner must clear. It grows with the number of
    trials — which is precisely why "we tested 500 strategies and this was the
    best" is weaker evidence than testing one and having it work.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    sigma = math.sqrt(max(variance_of_sharpes, 0.0))
    if sigma == 0:
        return 0.0
    # Expected maximum of N standard normals, to second order.
    q1 = _norm_ppf(1.0 - 1.0 / n_trials)
    q2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * ((1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    n_trials: int,
    variance_of_sharpes: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the observed Sharpe reflects genuine skill, in ``[0, 1]``.

    Args:
        observed_sharpe: annualised Sharpe of the selected strategy.
        n_observations: return observations backing it. Short samples are
            penalised — the same Sharpe over 40 days means far less than over
            4 years.
        n_trials: how many candidates the search considered. **Count every one**,
            including those discarded early; each is a hypothesis tested.
        variance_of_sharpes: variance of Sharpe across the trials.
        skewness / kurtosis: of the return series. Negative skew and fat tails
            both inflate a naive Sharpe, so both reduce the deflated value.

    Returns:
        A probability. Convention treats ``>= 0.95`` as evidence of skill.
        Values near 0.5 mean the result is indistinguishable from the best of
        a set of coin flips.
    """
    if n_observations < 3:
        return float("nan")

    threshold = expected_max_sharpe(n_trials, variance_of_sharpes)
    excess = observed_sharpe - threshold

    # Standard error of an estimated Sharpe under non-normal returns.
    denominator = 1.0 - skewness * observed_sharpe + \
        (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    if denominator <= 0:
        return float("nan")
    standard_error = math.sqrt(denominator / (n_observations - 1))
    if standard_error <= 0:
        return float("nan")
    return _norm_cdf(excess / standard_error)


@dataclass(slots=True)
class PBOResult:
    """Outcome of the combinatorially-symmetric cross-validation."""

    probability: float
    n_splits: int
    #: Out-of-sample ranks of the in-sample winner, normalised to [0, 1].
    logits: list[float]

    @property
    def is_overfit(self) -> bool:
        return self.probability > 0.5

    @property
    def verdict(self) -> str:
        if math.isnan(self.probability):
            return "insufficient data to assess overfitting"
        if self.probability > 0.5:
            return (
                f"PBO {self.probability:.0%} — the selection procedure is "
                "overfitting: the in-sample winner more often than not "
                "underperforms the median out of sample"
            )
        if self.probability > 0.25:
            return f"PBO {self.probability:.0%} — marginal; treat picks with caution"
        return f"PBO {self.probability:.0%} — selection generalises acceptably"


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, *, n_partitions: int = 8
) -> PBOResult:
    """Probability that the search's *winner selection* fails out of sample.

    Args:
        returns_matrix: ``(n_observations, n_strategies)`` of aligned returns,
            one column per candidate the search evaluated.
        n_partitions: even number of equal time slices. All balanced
            in-sample/out-of-sample combinations are evaluated.

    This measures something different from the deflated Sharpe. DSR asks "is
    this number real?"; PBO asks "does picking the best in-sample actually find
    strategies that work?". A search can produce a high Sharpe and still have a
    terrible PBO, which means the whole procedure is unreliable regardless of
    what it happened to return this time.
    """
    x = np.asarray(returns_matrix, dtype="float64")
    if x.ndim != 2 or x.shape[1] < 2:
        return PBOResult(float("nan"), 0, [])
    if n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even")

    n_obs = x.shape[0]
    if n_obs < n_partitions * 2:
        return PBOResult(float("nan"), 0, [])

    slices = np.array_split(np.arange(n_obs), n_partitions)
    half = n_partitions // 2
    logits: list[float] = []

    for combo in combinations(range(n_partitions), half):
        in_idx = np.concatenate([slices[i] for i in combo])
        out_idx = np.concatenate(
            [slices[i] for i in range(n_partitions) if i not in combo]
        )
        in_sample, out_sample = x[in_idx], x[out_idx]

        in_perf = _sharpe_columns(in_sample)
        out_perf = _sharpe_columns(out_sample)
        if not np.isfinite(in_perf).any() or not np.isfinite(out_perf).any():
            continue

        best = int(np.nanargmax(in_perf))
        # Where does the in-sample winner rank out of sample?
        finite = out_perf[np.isfinite(out_perf)]
        if finite.size < 2 or not np.isfinite(out_perf[best]):
            continue
        rank = float((finite <= out_perf[best]).sum()) / finite.size
        logits.append(rank)

    if not logits:
        return PBOResult(float("nan"), 0, [])

    # Overfitting = the winner lands in the bottom half out of sample.
    probability = float(np.mean([1.0 if r <= 0.5 else 0.0 for r in logits]))
    return PBOResult(probability=probability, n_splits=len(logits), logits=logits)


def _sharpe_columns(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    sigma = x.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigma > 0, mean / sigma, np.nan)


def parameter_stability(scores: dict[tuple[float, ...], float]) -> float:
    """How smooth the performance surface is across neighbouring parameters.

    Returns a value in ``[0, 1]``: 1 means neighbours perform alike, 0 means the
    surface is a spike field.

    A strategy that works at ``lookback=20`` and fails at 19 and 21 has not
    found a market regularity; it has found one arrangement of the noise. This
    is often a more damning signal than any single fit statistic, because it
    cannot be explained away by sample size.
    """
    if len(scores) < 3:
        return float("nan")
    keys = list(scores)
    values = np.array([scores[k] for k in keys], dtype="float64")
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return float("nan")
    values = values[finite]
    keys = [k for k, ok in zip(keys, finite, strict=True) if ok]

    spread = values.std(ddof=1)
    if spread == 0:
        return 1.0

    coords = np.array(keys, dtype="float64")
    ranges = coords.max(axis=0) - coords.min(axis=0)
    ranges[ranges == 0] = 1.0
    normalised = coords / ranges

    diffs: list[float] = []
    for i in range(len(keys)):
        distances = np.linalg.norm(normalised - normalised[i], axis=1)
        distances[i] = np.inf
        nearest = int(np.argmin(distances))
        diffs.append(abs(values[i] - values[nearest]))

    # Neighbour gap relative to overall spread; 0 gap -> perfectly stable.
    return float(max(0.0, 1.0 - np.mean(diffs) / (2.0 * spread)))
