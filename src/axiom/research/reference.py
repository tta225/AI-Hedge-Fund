"""What a real edge looks like, so an unreal one is recognisable.

Every honesty control in :mod:`axiom.research.statistics` is *internal*: the
deflated Sharpe asks whether a result beats the best of N coin flips drawn from
its own trial set, and PBO asks whether the selection procedure generalises.
Both can pass while the number itself is absurd, because neither has any idea
what a plausible Sharpe ratio is.

This module supplies that missing outside view.

The reference class
-------------------
``paperswithbacktest/awesome-systematic-trading`` surveyed 61 systematic
strategies from the published academic literature — Fama-French factor
territory, papers cited thousands of times — and backtested each
independently. The distribution is the useful part:

===========================  ========
best observed                   0.892
75th percentile                 0.559
median                          0.354
25th percentile                 0.142
worst observed                 −0.272
above Sharpe 1.0                    0
below Sharpe 0.5                  69%
negative                          15%
===========================  ========

Nothing in the surveyed literature clears 1.0. Fifteen percent of the famous
anomalies are *negative* when someone else runs them. That is the base rate for
"a published, peer-reviewed, widely-replicated systematic edge".

How to use it, and how not to
-----------------------------
This is a **prior, not a verdict**. The reference class is low-frequency,
capacity-bearing, long-horizon strategies. A market-making or latency
strategy legitimately lives outside it and can carry a Sharpe of 3+ honestly,
because it is being paid for a different thing entirely. So
:func:`assess_plausibility` flags and explains; it never rejects.

What it is genuinely good at is catching the common case, where a backtest
reports a Sharpe far above anything in the literature *at the same frequency
and capacity*, and the explanation is lookahead, survivorship, an unmodelled
cost, or a short sample — not a discovery. The single most useful output is
the confidence interval, because a Sharpe estimated from a few dozen trades
over one year has an error bar wide enough to swallow the entire table above,
and quoting the point estimate hides that.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from axiom.research.statistics import _norm_ppf

#: Source of the reference distribution. Recorded so the prior is auditable
#: rather than folkloric — a number nobody can trace is not a control.
REFERENCE_SOURCE = (
    "paperswithbacktest/awesome-systematic-trading, 61 published systematic "
    "strategies independently backtested"
)
REFERENCE_SAMPLE_SIZE = 61
#: Best Sharpe anywhere in the surveyed literature (Overnight Seasonality in
#: Bitcoin, an intraday crypto effect). The best *equity* anomaly is 0.835.
REFERENCE_MAX_SHARPE = 0.892
REFERENCE_P75_SHARPE = 0.559
REFERENCE_MEDIAN_SHARPE = 0.354
REFERENCE_P25_SHARPE = 0.142
REFERENCE_MIN_SHARPE = -0.272
REFERENCE_MEAN_SHARPE = 0.338
REFERENCE_STDEV_SHARPE = 0.283
#: Fraction of the surveyed anomalies that backtest negative.
REFERENCE_NEGATIVE_FRACTION = 0.15


def sharpe_standard_error(sharpe: float, n_observations: int) -> float:
    """Standard error of a Sharpe estimate under IID returns.

    ``SE ≈ sqrt((1 + S²/2) / n)``, the Lo (2002) result. The dependence on
    ``n`` is the entire point: a Sharpe from 250 daily observations carries an
    error bar of roughly ±0.13, and one from 40 observations carries ±0.33.
    Point estimates from short samples are not comparable to point estimates
    from thirty-year studies, and quoting them side by side is how a noisy
    backtest comes to look like a factor.

    The IID assumption is optimistic — autocorrelated returns inflate the true
    error — so this is a floor on the uncertainty, not an estimate of it.
    """
    if n_observations < 2:
        raise ValueError("need at least 2 observations to estimate an error")
    return math.sqrt((1.0 + 0.5 * sharpe * sharpe) / n_observations)


def sharpe_confidence_interval(
    sharpe: float, n_observations: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided interval for a Sharpe estimate."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    half_width = _norm_ppf(0.5 + confidence / 2.0) * sharpe_standard_error(
        sharpe, n_observations
    )
    return sharpe - half_width, sharpe + half_width


def reference_percentile(sharpe: float) -> float:
    """Where ``sharpe`` sits in the published distribution, in [0, 1].

    Linear interpolation between the recorded quantiles. Deliberately crude —
    five knots from a 61-point sample does not support anything finer, and a
    smooth fit would imply a precision the source cannot back.
    """
    knots = [
        (REFERENCE_MIN_SHARPE, 0.0),
        (REFERENCE_P25_SHARPE, 0.25),
        (REFERENCE_MEDIAN_SHARPE, 0.50),
        (REFERENCE_P75_SHARPE, 0.75),
        (REFERENCE_MAX_SHARPE, 1.0),
    ]
    if sharpe <= knots[0][0]:
        return 0.0
    if sharpe >= knots[-1][0]:
        return 1.0
    for (lo_s, lo_p), (hi_s, hi_p) in itertools.pairwise(knots):
        if lo_s <= sharpe <= hi_s:
            span = hi_s - lo_s
            return lo_p if span <= 0 else lo_p + (sharpe - lo_s) / span * (hi_p - lo_p)
    return 1.0


@dataclass(frozen=True, slots=True)
class PlausibilityCheck:
    """An outside view on a reported Sharpe ratio."""

    sharpe: float
    n_observations: int
    percentile: float
    ci_low: float
    ci_high: float
    exceeds_best_published: bool
    #: True when the interval includes the published median, i.e. the result is
    #: statistically indistinguishable from a thoroughly ordinary anomaly.
    indistinguishable_from_median: bool
    verdict: str

    @property
    def is_suspicious(self) -> bool:
        """Worth explaining before it is worth believing."""
        return self.exceeds_best_published


def assess_plausibility(sharpe: float, n_observations: int) -> PlausibilityCheck:
    """Compare a backtest Sharpe against the published reference class.

    Args:
        sharpe: the annualised Sharpe to assess.
        n_observations: number of return observations behind it. Folds pooled
            across a walk-forward count once each; trade count is *not* the
            right number unless one return is recorded per trade.

    Returns:
        A :class:`PlausibilityCheck` whose ``verdict`` is written to be pasted
        into a research note.
    """
    ci_low, ci_high = sharpe_confidence_interval(sharpe, n_observations)
    percentile = reference_percentile(sharpe)
    exceeds = sharpe > REFERENCE_MAX_SHARPE
    indistinguishable = ci_low <= REFERENCE_MEDIAN_SHARPE <= ci_high

    if exceeds:
        verdict = (
            f"ABOVE THE LITERATURE — a Sharpe of {sharpe:.2f} exceeds the best "
            f"of {REFERENCE_SAMPLE_SIZE} published systematic strategies "
            f"({REFERENCE_MAX_SHARPE:.2f}), none of which clears 1.0. That is "
            "not impossible — high-frequency and capacity-constrained "
            "strategies sit outside this reference class legitimately — but at "
            "low frequency it is far more often lookahead, survivorship, an "
            "unmodelled cost, or too short a sample. The claim needs an "
            f"explanation. Its own 95% interval is [{ci_low:.2f}, {ci_high:.2f}]."
        )
    elif indistinguishable:
        verdict = (
            f"ORDINARY — a Sharpe of {sharpe:.2f} sits at the "
            f"{percentile:.0%} percentile of the published distribution, and "
            f"its 95% interval [{ci_low:.2f}, {ci_high:.2f}] contains the "
            f"published median of {REFERENCE_MEDIAN_SHARPE:.2f}. On this "
            "sample it is not distinguishable from a thoroughly average "
            "anomaly, and 15% of those backtest negative when someone else "
            "runs them."
        )
    elif percentile >= 0.75 and ci_low > REFERENCE_MEDIAN_SHARPE:
        # Both conditions are required. Statistical separation from the median
        # is cheap at large n — a Sharpe of 0.40 on ten years of daily bars
        # clears it while sitting at the 56th percentile, which is ordinary,
        # not strong. The percentile says how good; the interval says how sure.
        verdict = (
            f"STRONG FOR THE CLASS — a Sharpe of {sharpe:.2f} is at the "
            f"{percentile:.0%} percentile and its 95% interval "
            f"[{ci_low:.2f}, {ci_high:.2f}] clears the published median. "
            "Within the range the literature says is real."
        )
    elif ci_high < REFERENCE_MEDIAN_SHARPE:
        verdict = (
            f"BELOW THE CLASS — a Sharpe of {sharpe:.2f} sits at the "
            f"{percentile:.0%} percentile, and its 95% interval "
            f"[{ci_low:.2f}, {ci_high:.2f}] falls entirely below the published "
            f"median of {REFERENCE_MEDIAN_SHARPE:.2f}."
        )
    else:
        verdict = (
            f"ORDINARY — a Sharpe of {sharpe:.2f} sits at the "
            f"{percentile:.0%} percentile of the published distribution, with "
            f"a 95% interval of [{ci_low:.2f}, {ci_high:.2f}]. Comfortably "
            "inside the range the literature calls unremarkable."
        )

    return PlausibilityCheck(
        sharpe=sharpe,
        n_observations=n_observations,
        percentile=percentile,
        ci_low=ci_low,
        ci_high=ci_high,
        exceeds_best_published=exceeds,
        indistinguishable_from_median=indistinguishable,
        verdict=verdict,
    )
