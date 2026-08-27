"""A small, honest classifier with no third-party dependency.

The point of this module is not to compete with gradient boosting. It is to
make the rest of :mod:`axiom.ml` *usable* without dragging scikit-learn,
LightGBM or torch into the install, and to be simple enough that when a
meta-model declines a trade a human can read off exactly why.

Logistic regression is the right default for meta-labelling specifically:

* The output is a calibrated-ish probability, and meta-labelling needs a
  threshold on a probability rather than a hard class.
* Coefficients are interpretable. "It declined because the volatility feature
  is strongly negative" is a sentence you can act on; "tree 47 voted no" is not.
* It cannot overfit as spectacularly as a deep model on a few hundred labelled
  events, which is the sample size real meta-labelling problems actually have.

Fitted by gradient descent on the L2-regularised log-likelihood, with sample
weights — because :func:`~axiom.ml.features.sample_uniqueness` produces weights
that must actually be applied, and a fitter that ignores them silently undoes
the correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Ridge penalty. Non-zero by default: financial features are collinear, and an
#: unregularised fit on collinear inputs produces huge offsetting coefficients
#: that look decisive and generalise terribly.
DEFAULT_L2 = 1.0
DEFAULT_EPOCHS = 500
DEFAULT_LEARNING_RATE = 0.1


class NotFittedError(RuntimeError):
    """Predict was called before fit."""


@dataclass(slots=True)
class LogisticClassifier:
    """Binary logistic regression, gradient descent, weighted.

    Args:
        l2: ridge penalty on the coefficients. The intercept is not penalised —
            shrinking it would bias the base rate, which for an imbalanced
            meta-label set is the one thing worth getting right.
        epochs: full-batch gradient steps.
        learning_rate: step size on standardised features.
        seed: unused by the fit itself (it is deterministic) but recorded so a
            model artefact carries the same fields as a stochastic one.
    """

    l2: float = DEFAULT_L2
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    seed: int = 0
    coefficients: np.ndarray | None = field(default=None, repr=False)
    intercept: float = 0.0
    #: Standardisation constants from the training set, applied at predict time.
    _mean: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self.coefficients is not None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> LogisticClassifier:
        """Fit on ``(n_samples, n_features)`` against 0/1 labels.

        Features are standardised using **training** statistics only, which are
        then reused at predict time. Standardising the test set on its own
        statistics is a leak: it tells the model where the test distribution
        sits.
        """
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if features.ndim != 2:
            raise ValueError("features must be 2-D (n_samples, n_features)")
        if len(features) != len(labels):
            raise ValueError(f"{len(features)} rows but {len(labels)} labels")
        if not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("labels must be 0 or 1; use meta_label to produce them")
        if len(features) == 0:
            raise ValueError("cannot fit on an empty sample")

        weights = (
            np.ones(len(labels))
            if sample_weights is None
            else np.asarray(sample_weights, dtype=float)
        )
        if len(weights) != len(labels):
            raise ValueError("sample_weights must match the number of labels")
        if np.any(weights < 0):
            raise ValueError("sample weights must be non-negative")

        self._mean = features.mean(axis=0)
        scale = features.std(axis=0)
        # A constant column has zero variance; dividing by it produces NaN and
        # poisons every downstream coefficient, so it is left unscaled and its
        # coefficient simply learns nothing.
        self._scale = np.where(scale > 0, scale, 1.0)
        standardised = (features - self._mean) / self._scale

        n_features = standardised.shape[1]
        coefficients = np.zeros(n_features)
        intercept = 0.0
        weight_total = float(weights.sum())
        if weight_total <= 0:
            raise ValueError("sample weights sum to zero")

        for _ in range(self.epochs):
            predictions = _sigmoid(standardised @ coefficients + intercept)
            residual = (predictions - labels) * weights
            gradient = standardised.T @ residual / weight_total
            gradient += self.l2 * coefficients / max(len(labels), 1)
            intercept_gradient = float(residual.sum()) / weight_total
            coefficients -= self.learning_rate * gradient
            intercept -= self.learning_rate * intercept_gradient

        self.coefficients = coefficients
        self.intercept = intercept
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probability of class 1."""
        if self.coefficients is None or self._mean is None or self._scale is None:
            raise NotFittedError("fit the classifier before calling predict_proba")
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != len(self.coefficients):
            raise ValueError(
                f"expected {len(self.coefficients)} features, got {features.shape[1]}"
            )
        standardised = (features - self._mean) / self._scale
        return _sigmoid(standardised @ self.coefficients + self.intercept)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(int)

    def explain(self, feature_names: list[str] | None = None) -> str:
        """Coefficients, largest magnitude first.

        On standardised features these are directly comparable: a coefficient
        of +0.4 moves the log-odds by 0.4 per standard deviation of that
        feature, whatever its natural units.
        """
        if self.coefficients is None:
            return "not fitted"
        names = feature_names or [f"f{i}" for i in range(len(self.coefficients))]
        order = np.argsort(-np.abs(self.coefficients))
        lines = [f"intercept {self.intercept:+.4f} (log-odds of the base rate)"]
        lines += [
            f"  {names[i]:<28} {self.coefficients[i]:+.4f}"
            for i in order
        ]
        return "\n".join(lines)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    ``1/(1+exp(-x))`` overflows for large negative ``x``. The piecewise form
    keeps every exponent negative, which matters here because standardised
    financial features routinely produce large activations on outlier bars —
    exactly the bars a risk system must not silently produce NaN on.
    """
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out
