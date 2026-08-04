"""Causal market regime detection.

Wraps :class:`~axiom.quant.hmm.GaussianHMM` into something a backtest can
actually use, which means closing **both** lookahead surfaces:

1. **inference** — only the forward filter is used, never smoothing or Viterbi;
2. **fitting** — parameters are re-estimated on an expanding window of *past*
   data only. Fitting once on the whole history and then decoding it is the
   subtler leak: the model's means and transitions encode what the whole period
   looked like, so even a causal filter is informed by the future.

The cost is real — the first ``min_train`` bars have no regime at all, and the
model lags a genuine regime change by design. That lag is not a defect to
engineer away; it is the honest amount of evidence required before you can know
the regime changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from axiom.core.series import OHLCVSeries
from axiom.quant.hmm import GaussianHMM, HMMConvergenceError

#: Label used before enough history exists to fit anything.
UNKNOWN_REGIME = -1


class RegimeLabel(str, Enum):
    """Semantic reading of a fitted state."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    #: Elevated volatility without direction — where trend rules get shredded.
    VOLATILE = "volatile"
    #: Quiet, low-volatility drift.
    QUIET = "quiet"
    UNKNOWN = "unknown"

    @property
    def is_tradable_trend(self) -> bool:
        return self in (RegimeLabel.TRENDING_UP, RegimeLabel.TRENDING_DOWN)


@dataclass(slots=True)
class RegimeSeries:
    """Per-bar causal regime assignment for one series."""

    states: np.ndarray
    #: ``(n, k)`` causal posterior. Row sums to 1; all-zero during warmup.
    probabilities: np.ndarray
    labels: dict[int, RegimeLabel]
    n_states: int
    #: First bar with a usable regime.
    warmup_end: int
    #: Bars at which the model was refitted, for audit.
    refit_points: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.states)

    def state_at(self, index: int) -> int:
        return int(self.states[index])

    def label_at(self, index: int) -> RegimeLabel:
        return self.labels.get(self.state_at(index), RegimeLabel.UNKNOWN)

    def confidence_at(self, index: int) -> float:
        """Posterior mass on the assigned state — 1/k means no information."""
        if index < self.warmup_end:
            return 0.0
        return float(self.probabilities[index].max())

    def is_known_at(self, index: int) -> bool:
        return index >= self.warmup_end and self.states[index] != UNKNOWN_REGIME

    def occupancy(self) -> dict[RegimeLabel, float]:
        """Fraction of known bars spent in each regime."""
        known = self.states[self.states != UNKNOWN_REGIME]
        if known.size == 0:
            return {}
        out: dict[RegimeLabel, float] = {}
        for state, count in zip(*np.unique(known, return_counts=True), strict=True):
            label = self.labels.get(int(state), RegimeLabel.UNKNOWN)
            out[label] = out.get(label, 0.0) + count / known.size
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def summary(self) -> str:
        parts = [f"{lab.value}={share:.0%}" for lab, share in self.occupancy().items()]
        return (
            f"{self.n_states}-state regime · warmup {self.warmup_end} bars · "
            f"{len(self.refit_points)} refits · " + " ".join(parts)
        )


class RegimeModel:
    """Expanding-window HMM regime detection over an OHLCV series."""

    def __init__(
        self,
        n_states: int = 3,
        *,
        min_train: int = 750,
        refit_every: int = 500,
        max_lookback: int = 3000,
        n_iter: int = 60,
        seed: int = 0,
    ) -> None:
        """
        Args:
            min_train: bars required before any regime is emitted.
            refit_every: bars between parameter re-estimations. Refitting every
                bar is ideal and far too slow; this is the practical compromise.
                Between refits the *filter* still updates every bar, so the
                regime estimate is current even when the parameters are not.
            max_lookback: cap on training window length. Bounds cost and lets
                the model forget structural history that no longer applies.
        """
        if min_train < n_states * 20:
            raise ValueError(
                f"min_train={min_train} is too small to fit {n_states} states"
            )
        self.n_states = n_states
        self.min_train = min_train
        self.refit_every = refit_every
        self.max_lookback = max_lookback
        self.n_iter = n_iter
        self.seed = seed

    @staticmethod
    def features(series: OHLCVSeries) -> np.ndarray:
        """``(n, 2)`` observation matrix: log return and absolute return.

        Return captures direction; absolute return captures volatility. Two
        features is deliberate — every extra one multiplies the parameters per
        state and financial data does not support many.
        """
        closes = series.closes
        log_returns = np.zeros(len(closes))
        log_returns[1:] = np.diff(np.log(np.maximum(closes, 1e-12)))
        return np.column_stack([log_returns, np.abs(log_returns)])

    def _label_states(self, model: GaussianHMM) -> dict[int, RegimeLabel]:
        """Map fitted state indices onto semantic labels.

        States arrive sorted by mean return. Volatility relative to the
        cross-state median separates a directional trend from noise.
        """
        assert model.means is not None
        means = model.means
        returns, vols = means[:, 0], means[:, 1]
        median_vol = float(np.median(vols))

        labels: dict[int, RegimeLabel] = {}
        for i in range(len(means)):
            high_vol = vols[i] > median_vol * 1.25
            if high_vol and abs(returns[i]) < vols[i] * 0.25:
                labels[i] = RegimeLabel.VOLATILE
            elif returns[i] > 0 and abs(returns[i]) >= vols[i] * 0.05:
                labels[i] = RegimeLabel.TRENDING_UP
            elif returns[i] < 0 and abs(returns[i]) >= vols[i] * 0.05:
                labels[i] = RegimeLabel.TRENDING_DOWN
            else:
                labels[i] = RegimeLabel.QUIET
        return labels

    def fit_causal(self, series: OHLCVSeries) -> RegimeSeries:
        """Produce a per-bar regime series with no lookahead anywhere.

        Walks forward. At each refit point the model is re-estimated on bars
        strictly *before* that point, then used to filter forward until the next
        refit. Nothing at bar ``t`` depends on any bar after ``t``.
        """
        x = self.features(series)
        n = len(x)
        states = np.full(n, UNKNOWN_REGIME, dtype=int)
        probabilities = np.zeros((n, self.n_states))
        labels: dict[int, RegimeLabel] = {}
        refit_points: list[int] = []

        if n <= self.min_train:
            return RegimeSeries(
                states=states, probabilities=probabilities, labels={},
                n_states=self.n_states, warmup_end=n,
            )

        model: GaussianHMM | None = None
        next_refit = self.min_train
        alpha: np.ndarray | None = None

        for t in range(self.min_train, n):
            if t >= next_refit:
                start = max(0, t - self.max_lookback)
                # Strictly `[:t]` — bar t itself is not in the training set.
                train = x[start:t]
                candidate = GaussianHMM(
                    self.n_states, n_iter=self.n_iter, seed=self.seed
                )
                try:
                    candidate.fit(train)
                    model = candidate
                    labels = self._label_states(candidate)
                    refit_points.append(t)
                    # New parameters mean the old alpha is in the wrong basis
                    # (state indices are re-sorted on every fit), so the filter
                    # restarts from the prior.
                    alpha = None
                except HMMConvergenceError:
                    # Keep the previous model rather than emitting garbage; if
                    # there is no previous model the bar stays UNKNOWN.
                    pass
                next_refit = t + self.refit_every

            if model is None:
                continue

            # Incremental forward step: O(k²) per bar. Re-filtering the whole
            # window each bar is O(n·window) overall and does not finish on a
            # year of hourly data.
            alpha = (
                model.filter_init(x[t]) if alpha is None
                else model.filter_step(alpha, x[t])
            )
            probabilities[t] = alpha
            states[t] = int(alpha.argmax())

        return RegimeSeries(
            states=states,
            probabilities=probabilities,
            labels=labels,
            n_states=self.n_states,
            warmup_end=self.min_train,
            refit_points=tuple(refit_points),
        )

    def fit_full_history(self, series: OHLCVSeries) -> RegimeSeries:
        """Fit once on everything and decode with Viterbi. **Research only.**

        Every value depends on the whole sample. Useful for charting what the
        regimes were, and for measuring how much the causal version gives up.
        Never valid as a backtest input.
        """
        x = self.features(series)
        model = GaussianHMM(self.n_states, n_iter=self.n_iter, seed=self.seed)
        model.fit(x)
        path = model.viterbi(x)
        return RegimeSeries(
            states=path,
            probabilities=model.smooth(x),
            labels=self._label_states(model),
            n_states=self.n_states,
            warmup_end=0,
            refit_points=(0,),
        )
