"""The contract every deep model must satisfy before it can trade.

The interesting code here is not the neural network — that part is standard and
was already fine in the uploaded architecture. It is the boundary: what has to
be true before a network's output is allowed to influence a position.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from axiom.core.provenance import Provenance


class ModelNotFittedError(RuntimeError):
    """A prediction was requested from a model that has never been trained.

    Raised rather than returning zeros or a random draw. The uploaded
    orchestrator fed its RL agent ``torch.randn(50)`` and blended the output
    into a trading signal at weight 0.15 — noise entering a position with no
    trace. An exception is the only response that cannot be silently averaged.
    """


def torch_available() -> bool:
    """True when the optional deep-learning extra is installed."""
    return importlib.util.find_spec("torch") is not None


def require_torch() -> Any:
    """Import torch, or explain how to get it.

    Returns the module so call sites need not import it at module scope, which
    is what keeps the core platform installable without a 2GB dependency.
    """
    if not torch_available():
        raise ImportError(
            "this model needs PyTorch, which is an optional extra. "
            "Install it with: pip install 'axiom[deep]'"
        )
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What a fit actually achieved, on data it was not fitted to.

    Every field here exists to answer one question — *should this model's
    output be believed?* — and the answer has to come from held-out data.
    In-sample error is a statement about memorisation.
    """

    #: Bars used for fitting.
    train_samples: int
    #: Bars held out, strictly after the training window.
    validation_samples: int
    #: Mean squared error on the held-out window.
    validation_mse: float
    #: MSE of predicting the training mean on the same held-out window. The
    #: honest baseline: a model that cannot beat it has learned nothing, and
    #: without it a small MSE is unreadable — it may just be a quiet market.
    baseline_mse: float
    #: Fraction of held-out bars whose direction was called correctly.
    directional_accuracy: float
    epochs: int
    fitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    provenance: Provenance | None = None

    @property
    def skill(self) -> float:
        """Fraction of baseline error removed. Negative means worse than the mean.

        This — not the raw loss — is what the alpha adapter converts into
        confidence, because it is scale-free and has a meaningful zero.
        """
        if self.baseline_mse <= 0:
            return 0.0
        return float(1.0 - self.validation_mse / self.baseline_mse)

    @property
    def has_skill(self) -> bool:
        """True only when the model beat the mean *and* called direction better
        than a coin. Both, because either alone is routinely achievable by
        accident on financial data."""
        return self.skill > 0.0 and self.directional_accuracy > 0.5

    def render(self) -> str:
        verdict = "usable" if self.has_skill else "NO DEMONSTRATED SKILL"
        return (
            f"train {self.train_samples:,} / val {self.validation_samples:,} bars, "
            f"{self.epochs} epochs\n"
            f"  val MSE {self.validation_mse:.6g} vs baseline {self.baseline_mse:.6g}\n"
            f"  skill {self.skill:+.3f}  directional {self.directional_accuracy:.1%}  "
            f"→ {verdict}"
        )


class DeepModel(ABC):
    """Base class for sequence models that forecast forward returns.

    Subclasses implement :meth:`_build`, :meth:`_fit_epochs`, and
    :meth:`_forward`. This class owns the parts that must not vary between
    models: the causal train/validation split, the skill baseline, and the
    refusal to predict before fitting.
    """

    name: str = "deep_model"

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._model: Any = None
        self._report: TrainingReport | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._report is not None

    @property
    def report(self) -> TrainingReport | None:
        return self._report

    @abstractmethod
    def _build(self, n_features: int) -> Any:
        """Construct the underlying network."""

    @abstractmethod
    def _fit_epochs(
        self,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
    ) -> None:
        """Run the training loop in place."""

    @abstractmethod
    def _forward(self, model: Any, x: np.ndarray) -> np.ndarray:
        """Predict for a batch of already-windowed inputs."""

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int = 20,
        validation_fraction: float = 0.2,
        provenance: Provenance | None = None,
    ) -> TrainingReport:
        """Fit, then measure on a strictly later window.

        The split is chronological and never shuffled. Shuffling a time series
        before splitting puts tomorrow in the training set and yesterday in
        validation, which produces excellent validation scores and a model that
        cannot forecast anything — it is the deep-learning equivalent of the
        stat-arb leak this codebase closes elsewhere.
        """
        if features.ndim != 3:
            raise ValueError(
                f"expected (samples, timesteps, features), got {features.shape}"
            )
        if len(features) != len(targets):
            raise ValueError(
                f"{len(features)} feature windows but {len(targets)} targets"
            )
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")

        split = int(len(features) * (1.0 - validation_fraction))
        if split < 1 or split >= len(features):
            raise ValueError(
                f"{len(features)} samples cannot be split at "
                f"{validation_fraction:.0%} into a usable train/validation pair"
            )

        x_train, x_validation = features[:split], features[split:]
        y_train, y_validation = targets[:split], targets[split:]

        model = self._build(features.shape[2])
        self._fit_epochs(model, x_train, y_train, epochs)

        predictions = self._forward(model, x_validation)
        validation_mse = float(np.mean((predictions - y_validation) ** 2))
        # The baseline uses the *training* mean, not the validation mean: using
        # the latter would leak the held-out window's average into the
        # comparison and flatter every model that predicts near zero.
        baseline_mse = float(np.mean((y_validation - float(np.mean(y_train))) ** 2))
        directional = float(np.mean(np.sign(predictions) == np.sign(y_validation)))

        self._model = model
        self._report = TrainingReport(
            train_samples=len(x_train),
            validation_samples=len(x_validation),
            validation_mse=validation_mse,
            baseline_mse=baseline_mse,
            directional_accuracy=directional,
            epochs=epochs,
            provenance=provenance,
        )
        return self._report

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Forecast forward returns for already-windowed inputs.

        Raises:
            ModelNotFittedError: if :meth:`fit` has not run. See the class
                docstring for why this raises rather than returning zeros.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"{self.name} has not been fitted. A prediction from an "
                "untrained network is noise, and noise blended into an "
                "ensemble is indistinguishable from a view."
            )
        if features.ndim != 3:
            raise ValueError(
                f"expected (samples, timesteps, features), got {features.shape}"
            )
        return self._forward(self._model, features)


def make_windows(
    series: np.ndarray, window: int, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a feature matrix into supervised windows and forward-return targets.

    Args:
        series: ``(n_bars, n_features)``. Column 0 must be the log return being
            forecast; the target is its forward sum over ``horizon`` bars.
        window: timesteps per sample.
        horizon: bars ahead to predict.

    Returns:
        ``(windows, targets)`` where ``windows[i]`` ends at bar ``i + window -
        1`` and ``targets[i]`` covers bars ``i + window`` through
        ``i + window + horizon - 1``. The target begins strictly after the
        window ends — an off-by-one here is a lookahead leak that improves
        every metric and cannot be traded.
    """
    if series.ndim != 2:
        raise ValueError(f"expected (n_bars, n_features), got {series.shape}")
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must both be at least 1")

    n_samples = len(series) - window - horizon + 1
    if n_samples < 1:
        return np.empty((0, window, series.shape[1])), np.empty(0)

    windows = np.stack([series[i : i + window] for i in range(n_samples)])
    targets = np.array(
        [series[i + window : i + window + horizon, 0].sum() for i in range(n_samples)]
    )
    return windows, targets
