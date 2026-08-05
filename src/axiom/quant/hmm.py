"""Gaussian hidden Markov model for market regime detection.

Markets are not one process. They alternate between states with genuinely
different statistics — quiet trending periods where momentum works, and violent
choppy periods where the same rules get shredded. An HMM models that directly:
a latent discrete state evolving as a Markov chain, with each state emitting
observations from its own Gaussian.

Implemented here rather than pulled from ``hmmlearn`` for two reasons: it keeps
the platform's core dependency-free, and — more importantly — it lets the
**causal** inference path be the default, which is the whole ballgame for
backtesting.

Causality: the point almost every tutorial gets wrong
-----------------------------------------------------
There are three ways to infer states, and they are not interchangeable:

:meth:`~GaussianHMM.filter`
    ``P(state_t | x_1..x_t)`` — the forward pass alone. Uses only the past.
    **This is the only one a backtest or a live strategy may use.**

:meth:`~GaussianHMM.smooth`
    ``P(state_t | x_1..x_T)`` — forward-backward over the *whole* sequence.
    Every value depends on the future.

:meth:`~GaussianHMM.viterbi`
    the single most likely state path over the whole sequence. Also depends on
    the future, and the standard choice in published examples.

Decoding a regime with Viterbi over the full history and then backtesting
against it produces a strategy that knew in March which regime March would turn
out to be. The equity curve is spectacular and entirely fictional. The
non-causal methods are provided for research and charting, and both refuse to
be used by :class:`~axiom.quant.regime.RegimeModel` in a backtest context.

Fitting is also a lookahead surface: parameters must be estimated on data that
precedes the period being decoded. :class:`~axiom.quant.regime.RegimeModel`
handles that with an expanding-window refit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Floor applied to variances so a degenerate state cannot divide by zero.
MIN_VARIANCE = 1e-10
_LOG_2PI = float(np.log(2.0 * np.pi))


class HMMConvergenceError(RuntimeError):
    """Fitting failed to produce a usable model."""


@dataclass(slots=True)
class HMMFitResult:
    """Diagnostics from a fit. Worth reading before trusting the states."""

    log_likelihood: float
    iterations: int
    converged: bool
    #: Log-likelihood at each iteration, for inspecting the EM trajectory.
    history: list[float] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "converged" if self.converged else "hit iteration cap"
        return (
            f"{status} after {self.iterations} iterations, "
            f"log-likelihood {self.log_likelihood:,.2f}"
        )


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM fitted by Baum-Welch (EM).

    Diagonal rather than full covariance is deliberate: with the handful of
    features that make sense here, full covariance adds ``d(d-1)/2`` parameters
    per state and mostly buys overfitting on financial data.
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        n_iter: int = 200,
        tol: float = 1e-5,
        seed: int = 0,
        min_variance: float = MIN_VARIANCE,
    ) -> None:
        if n_states < 2:
            raise ValueError(f"n_states must be at least 2, got {n_states}")
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.min_variance = min_variance

        self.start_prob: np.ndarray | None = None
        self.transitions: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.variances: np.ndarray | None = None
        self.fit_result: HMMFitResult | None = None

    # --- fitting ------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.transitions is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise HMMConvergenceError("model is not fitted; call fit() first")

    def _initialise(self, x: np.ndarray) -> None:
        """Seed EM by splitting on the first feature's quantiles.

        Deterministic given the data, so a fit is reproducible — which matters
        when a backtest refits hundreds of times and you need to be able to
        reproduce a specific one.
        """
        k = self.n_states
        order = np.argsort(x[:, 0])
        chunks = np.array_split(order, k)

        self.means = np.vstack([x[c].mean(axis=0) for c in chunks])
        self.variances = np.vstack(
            [np.maximum(x[c].var(axis=0), self.min_variance) for c in chunks]
        )
        # Persistent chain: regimes last, so the diagonal starts dominant.
        self.transitions = np.full((k, k), 0.1 / max(k - 1, 1))
        np.fill_diagonal(self.transitions, 0.9)
        self.start_prob = np.full(k, 1.0 / k)

    def _log_emissions(self, x: np.ndarray) -> np.ndarray:
        """``(n, k)`` log N(x_t | mu_k, diag(var_k))."""
        assert self.means is not None and self.variances is not None
        var = np.maximum(self.variances, self.min_variance)
        # (n, k, d)
        diff = x[:, None, :] - self.means[None, :, :]
        quad = np.sum(diff**2 / var[None, :, :], axis=2)
        logdet = np.sum(np.log(var), axis=1)
        d = x.shape[1]
        return np.asarray(-0.5 * (quad + logdet[None, :] + d * _LOG_2PI))

    @staticmethod
    def _forward_pass(
        log_b: np.ndarray, start: np.ndarray, trans: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Scaled forward recursion.

        Returns ``(alpha, scale, log_likelihood)`` where ``alpha[t]`` is the
        normalised ``P(state_t | x_1..x_t)`` — already the causal filter.
        Scaling keeps the recursion in linear space without underflowing over
        long series, which is what a naive implementation dies from.
        """
        n, k = log_b.shape
        alpha = np.zeros((n, k))
        scale = np.zeros(n)

        b0 = np.exp(log_b[0] - log_b[0].max())
        a = start * b0
        total = a.sum()
        if total <= 0:
            raise HMMConvergenceError("degenerate emissions at t=0")
        alpha[0] = a / total
        scale[0] = np.log(total) + log_b[0].max()

        for t in range(1, n):
            shift = log_b[t].max()
            b = np.exp(log_b[t] - shift)
            a = (alpha[t - 1] @ trans) * b
            total = a.sum()
            if total <= 0:
                raise HMMConvergenceError(f"degenerate emissions at t={t}")
            alpha[t] = a / total
            scale[t] = np.log(total) + shift

        return alpha, scale, float(scale.sum())

    @staticmethod
    def _backward_pass(log_b: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """Backward recursion, normalised per step.

        Each step is renormalised rather than divided by the forward scaling
        factors. The two differ by a constant per timestep, which cancels when
        gamma is normalised, so this is equivalent and avoids carrying the
        scale array through.
        """
        n, k = log_b.shape
        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            shift = log_b[t + 1].max()
            b = np.exp(log_b[t + 1] - shift)
            raw = trans @ (b * beta[t + 1])
            norm = raw.sum()
            beta[t] = raw / norm if norm > 0 else np.full(k, 1.0 / k)
        return beta

    def fit(self, x: np.ndarray) -> HMMFitResult:
        """Estimate parameters by Baum-Welch on ``x`` of shape ``(n, d)``."""
        x = np.asarray(x, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] < self.n_states * 10:
            raise HMMConvergenceError(
                f"{x.shape[0]} observations is too few to fit {self.n_states} states; "
                "each state needs enough data to estimate a mean and variance"
            )
        if not np.isfinite(x).all():
            raise HMMConvergenceError("observations contain NaN or inf")

        self._initialise(x)
        k = self.n_states
        history: list[float] = []
        previous = -np.inf
        converged = False
        iteration = 0

        for iteration in range(1, self.n_iter + 1):  # noqa: B007
            assert self.transitions is not None and self.start_prob is not None
            log_b = self._log_emissions(x)
            alpha, _scale, loglik = self._forward_pass(
                log_b, self.start_prob, self.transitions
            )
            beta = self._backward_pass(log_b, self.transitions)

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            # Expected transition counts, vectorised over time.
            #
            # The textbook form is a Python loop over t accumulating a (k, k)
            # outer product. At ~3,000 observations x 60 EM iterations x a
            # refit every 500 bars, that loop dominates the entire runtime.
            # Building the (n-1, k, k) tensor once and reducing costs a few
            # hundred kilobytes and removes the loop entirely.
            shift = log_b[1:].max(axis=1, keepdims=True)
            b_next = np.exp(log_b[1:] - shift) * beta[1:]          # (n-1, k)
            step = (
                alpha[:-1][:, :, None]
                * self.transitions[None, :, :]
                * b_next[:, None, :]
            )                                                       # (n-1, k, k)
            totals = step.sum(axis=(1, 2), keepdims=True)
            # `out=` is required, not cosmetic: without it the entries where the
            # condition is False hold uninitialised memory, which then gets
            # summed into xi. Zeroing them is the intended behaviour — a
            # timestep with no probability mass contributes no transitions.
            normalised = np.divide(
                step, totals, out=np.zeros_like(step), where=totals > 0
            )
            xi = normalised.sum(axis=0)

            self.start_prob = gamma[0] / gamma[0].sum()
            row_sums = xi.sum(axis=1, keepdims=True)
            self.transitions = np.where(row_sums > 0, xi / np.maximum(row_sums, 1e-300),
                                        1.0 / k)

            weights = gamma.sum(axis=0)
            self.means = (gamma.T @ x) / np.maximum(weights[:, None], 1e-300)
            diff = x[:, None, :] - self.means[None, :, :]
            self.variances = np.maximum(
                np.einsum("nk,nkd->kd", gamma, diff**2)
                / np.maximum(weights[:, None], 1e-300),
                self.min_variance,
            )

            history.append(loglik)
            if abs(loglik - previous) < self.tol * max(abs(previous), 1.0):
                converged = True
                previous = loglik
                break
            previous = loglik

        self.fit_result = HMMFitResult(
            log_likelihood=float(previous),
            iterations=iteration,
            converged=converged,
            history=history,
        )
        self._order_states()
        return self.fit_result

    def _order_states(self) -> None:
        """Sort states by mean of the first feature, ascending.

        EM labels states arbitrarily, so two fits on overlapping windows can
        swap "state 0" and "state 2" with no change in the model. A strategy
        keyed to a state index would silently invert. Canonical ordering makes
        the labels stable across refits.
        """
        assert self.means is not None and self.variances is not None
        assert self.transitions is not None and self.start_prob is not None
        order = np.argsort(self.means[:, 0])
        self.means = self.means[order]
        self.variances = self.variances[order]
        self.start_prob = self.start_prob[order]
        self.transitions = self.transitions[np.ix_(order, order)]

    # --- inference ----------------------------------------------------------

    def filter(self, x: np.ndarray) -> np.ndarray:
        """**Causal** posterior ``P(state_t | x_1..x_t)``, shape ``(n, k)``.

        The only inference method safe for backtests and live trading: value
        ``t`` depends on observations up to ``t`` and nothing later.
        """
        self._require_fitted()
        x = np.atleast_2d(np.asarray(x, dtype="float64"))
        if x.shape[0] == 1 and x.shape[1] != (self.means.shape[1]):  # type: ignore[union-attr]
            x = x.T
        assert self.start_prob is not None and self.transitions is not None
        log_b = self._log_emissions(x)
        alpha, _, _ = self._forward_pass(log_b, self.start_prob, self.transitions)
        return alpha

    def filter_states(self, x: np.ndarray) -> np.ndarray:
        """Causal most-likely state at each step, shape ``(n,)``."""
        return np.asarray(self.filter(x).argmax(axis=1))

    def filter_init(self, x_t: np.ndarray) -> np.ndarray:
        """Start an incremental causal filter from a single observation."""
        self._require_fitted()
        assert self.start_prob is not None
        log_b = self._log_emissions(np.atleast_2d(x_t))[0]
        a = self.start_prob * np.exp(log_b - log_b.max())
        total = a.sum()
        return np.asarray(a / total if total > 0 else self.start_prob)

    def filter_step(self, alpha_prev: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """Advance the causal filter by one observation.

        Mathematically identical to re-running :meth:`filter` over the whole
        history, but O(k²) per bar instead of O(n·k²). Walking a long series
        by repeatedly re-filtering a window is quadratic and unusable — this
        is what makes per-bar regime inference tractable in a backtest.
        """
        self._require_fitted()
        assert self.transitions is not None
        log_b = self._log_emissions(np.atleast_2d(x_t))[0]
        a = (alpha_prev @ self.transitions) * np.exp(log_b - log_b.max())
        total = a.sum()
        if total <= 0:
            # Emissions vanished under this model; fall back to the prior
            # rather than propagating NaN through the rest of the series.
            return np.asarray(alpha_prev @ self.transitions)
        return np.asarray(a / total)

    def predict_next(self, x: np.ndarray) -> np.ndarray:
        """Causal one-step-ahead state distribution ``P(state_t+1 | x_1..x_t)``."""
        self._require_fitted()
        assert self.transitions is not None
        return np.asarray(self.filter(x)[-1] @ self.transitions)

    def smooth(self, x: np.ndarray) -> np.ndarray:
        """**Non-causal** posterior using the whole sequence.

        For charting and research only. Using this in a backtest leaks the
        future into every state estimate.
        """
        self._require_fitted()
        assert self.start_prob is not None and self.transitions is not None
        x = np.asarray(x, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        log_b = self._log_emissions(x)
        alpha, _, _ = self._forward_pass(log_b, self.start_prob, self.transitions)
        beta = self._backward_pass(log_b, self.transitions)
        gamma = alpha * beta
        return np.asarray(gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300))

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """**Non-causal** most likely state path over the whole sequence.

        For charting and research only — see the module docstring.
        """
        self._require_fitted()
        assert self.start_prob is not None and self.transitions is not None
        x = np.asarray(x, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        log_b = self._log_emissions(x)
        n, k = log_b.shape
        log_t = np.log(np.maximum(self.transitions, 1e-300))

        delta = np.log(np.maximum(self.start_prob, 1e-300)) + log_b[0]
        psi = np.zeros((n, k), dtype=int)
        for t in range(1, n):
            scores = delta[:, None] + log_t
            psi[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + log_b[t]

        path = np.zeros(n, dtype=int)
        path[-1] = int(delta.argmax())
        for t in range(n - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def score(self, x: np.ndarray) -> float:
        """Log-likelihood of ``x`` under the fitted model."""
        self._require_fitted()
        assert self.start_prob is not None and self.transitions is not None
        x = np.asarray(x, dtype="float64")
        if x.ndim == 1:
            x = x[:, None]
        _, _, loglik = self._forward_pass(
            self._log_emissions(x), self.start_prob, self.transitions
        )
        return loglik

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected bars spent in each state, ``1 / (1 - p_ii)``.

        A quick sanity check: durations of 1-2 bars mean the model is fitting
        noise, not regimes.
        """
        self._require_fitted()
        assert self.transitions is not None
        diag = np.clip(np.diag(self.transitions), 0.0, 1.0 - 1e-12)
        return np.asarray(1.0 / (1.0 - diag))

    def describe(self) -> str:
        self._require_fitted()
        assert self.means is not None
        lines = [f"GaussianHMM({self.n_states} states)"]
        if self.fit_result is not None:
            lines.append(f"  {self.fit_result.summary}")
        for i, (mu, dur) in enumerate(zip(self.means, self.expected_durations, strict=True)):
            rendered = ", ".join(f"{m:+.5f}" for m in mu)
            lines.append(f"  state {i}: mean=({rendered}) expected duration={dur:.1f} bars")
        return "\n".join(lines)
