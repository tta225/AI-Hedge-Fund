"""Adapter: a trained sequence model as a cross-sectional alpha source.

This is the boundary the uploaded architecture did not have. Its orchestrator
took each model's raw output, multiplied by a static weight, and averaged —
including a reinforcement-learning agent fed ``torch.randn(50)`` as its state
vector, whose output was therefore a random number entering a trading decision
at weight 0.15.

Three rules are enforced here, and they are the reason this file exists rather
than the models emitting signals directly:

*An unfitted model emits nothing.* Not zero, not a neutral signal — nothing.
A neutral signal is a claim that the model looked and found no edge.

*A model with no demonstrated skill emits nothing.* Beating the training mean
on held-out data and calling direction better than a coin are both required.
A network that fits and cannot forecast is the normal outcome on financial
data, and the ensemble must not be asked to average it with sources that work.

*Confidence comes from validation skill, not from output magnitude.* A
network's activation is not a probability. Scaling belief by how confidently
the last layer fired is how a saturated tanh becomes 99% conviction.
"""

from __future__ import annotations

import numpy as np

from axiom.alpha.base import AlphaAgent, AlphaSignal
from axiom.alpha.panel import Panel
from axiom.models.base import DeepModel, ModelNotFittedError


class DeepModelAgent(AlphaAgent):
    """Wraps one fitted :class:`~axiom.models.base.DeepModel` per instrument.

    Args:
        models: fitted model per symbol. A symbol with no model, an unfitted
            model, or a model without demonstrated skill simply produces no
            signal for that name — which the ensemble reads as reduced
            coverage and lowers its confidence accordingly.
        window: timesteps the models were trained on. Must match, or the
            network is being asked to read a shape it never saw.
        scale: divisor mapping a predicted return onto the ``[-1, 1]`` signal
            scale. Expressed in return units, so 0.02 means "a predicted 2%
            move is a full-strength signal".
    """

    agent_id = "deep_models"

    def __init__(
        self,
        models: dict[str, DeepModel],
        *,
        window: int = 32,
        scale: float = 0.02,
        horizon_bars: int = 1,
        config: dict[str, object] | None = None,
    ) -> None:
        super().__init__(config)
        if scale <= 0:
            raise ValueError("scale must be positive; it divides the prediction")
        self.models = dict(models)
        self.window = window
        self.scale = scale
        self.horizon = horizon_bars
        self.min_history = window + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        signals: list[AlphaSignal] = []
        for symbol, model in self.models.items():
            if symbol not in panel.symbols:
                continue
            report = model.report
            # Both gates, before any inference runs. An unskilled model's
            # prediction is not worth the forward pass, and more importantly is
            # not worth the risk that someone later reads it as a view.
            if not model.is_fitted or report is None or not report.has_skill:
                continue

            features = self._features(panel, index, symbol)
            if features is None:
                continue

            try:
                prediction = float(model.predict(features[np.newaxis, ...])[0])
            except ModelNotFittedError:
                # Defensive: is_fitted was checked above, but a model swapped
                # out from under us must degrade to silence, never to zero.
                continue

            if not np.isfinite(prediction):
                continue

            signals.append(
                AlphaSignal(
                    symbol=symbol,
                    signal=float(np.clip(prediction / self.scale, -1.0, 1.0)),
                    # Skill is on (-inf, 1]; only its positive part is belief,
                    # and it is capped well below 1 because held-out skill on a
                    # single split is itself a noisy estimate.
                    confidence=float(np.clip(report.skill, 0.0, 0.9)),
                    expected_return=prediction,
                    horizon_bars=self.horizon,
                    strategy=model.name,
                    agent_id=self.agent_id,
                    as_of_index=index,
                    provenance=panel.provenance,
                    features_used=("log_return", "abs_return", "relative_volume"),
                    metadata={
                        "model": model.name,
                        "validation_skill": round(report.skill, 4),
                        "directional_accuracy": round(report.directional_accuracy, 4),
                        "validation_samples": report.validation_samples,
                    },
                )
            )
        return signals

    def _features(self, panel: Panel, index: int, symbol: str) -> np.ndarray | None:
        """The feature window for one instrument, ending strictly before ``index``."""
        column = panel.column(symbol)
        prices = panel.history(index, self.window + 2)[:, column]
        volumes = panel.volume_history(index, self.window + 2)[:, column]
        if len(prices) < self.window + 2 or np.any(prices <= 0):
            return None

        log_returns = np.diff(np.log(prices))
        baseline = np.mean(volumes[1:])
        relative_volume = volumes[1:] / baseline if baseline > 0 else np.ones_like(log_returns)

        matrix = np.column_stack(
            [log_returns, np.abs(log_returns), relative_volume[: len(log_returns)]]
        )
        if len(matrix) < self.window or not np.all(np.isfinite(matrix)):
            return None
        return matrix[-self.window :]


def build_training_windows(
    panel: Panel, symbol: str, *, window: int = 32, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Supervised windows for one instrument, from the whole panel history.

    Intended for offline fitting only. It deliberately uses every bar, because
    a training set is allowed to see its own past and future — the causality
    contract binds *inference*, and
    :meth:`~axiom.models.base.DeepModel.fit` enforces a chronological split so
    the validation window is still strictly later than the training one.
    """
    from axiom.models.base import make_windows

    column = panel.column(symbol)
    prices = panel.closes[:, column]
    volumes = panel.volumes[:, column]
    if len(prices) < window + horizon + 2 or np.any(prices <= 0):
        return np.empty((0, window, 3)), np.empty(0)

    log_returns = np.diff(np.log(prices))
    baseline = np.mean(volumes[1:])
    relative = volumes[1:] / baseline if baseline > 0 else np.ones_like(log_returns)
    matrix = np.column_stack(
        [log_returns, np.abs(log_returns), relative[: len(log_returns)]]
    )
    return make_windows(matrix, window, horizon)
