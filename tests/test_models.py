"""The contract deep models must satisfy before their output can trade.

No PyTorch required. Everything tested here is the boundary — the split, the
skill baseline, the refusal to predict unfitted, and the adapter's gates — and
that boundary is where the uploaded architecture blended a random vector into a
trading signal at weight 0.15. The networks themselves are exercised only when
the optional extra happens to be installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.alpha.deep import DeepModelAgent, build_training_windows
from axiom.alpha.ensemble import AlphaEnsemble
from axiom.alpha.panel import Panel
from axiom.core.provenance import Provenance
from axiom.models.base import (
    DeepModel,
    ModelNotFittedError,
    TrainingReport,
    make_windows,
    torch_available,
)


class _LinearStub(DeepModel):
    """A trivially fittable model: no torch, exercises the whole contract."""

    name = "stub"

    def __init__(self, *, gain: float = 1.0, **config: object) -> None:
        super().__init__(**config)
        self.gain = gain
        self.epochs_run = 0

    def _build(self, n_features: int) -> object:
        return {"n_features": n_features}

    def _fit_epochs(
        self, model: object, x_train: np.ndarray, y_train: np.ndarray, epochs: int
    ) -> None:
        self.epochs_run = epochs

    def _forward(self, model: object, x: np.ndarray) -> np.ndarray:
        # Predict the last observed return, scaled. gain=0 predicts nothing.
        return x[:, -1, 0] * self.gain


def _data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    matrix = np.column_stack([returns, np.abs(returns), rng.lognormal(0, 0.1, n)])
    return make_windows(matrix, window=16, horizon=1)


class TestWindowing:
    def test_the_target_starts_strictly_after_the_window(self) -> None:
        """An off-by-one here improves every metric and cannot be traded."""
        series = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        windows, targets = make_windows(series, window=3, horizon=1)
        assert np.array_equal(windows[0], series[0:3])
        assert targets[0] == series[3, 0]

    def test_a_multi_bar_horizon_sums_forward_returns(self) -> None:
        series = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        _, targets = make_windows(series, window=3, horizon=2)
        assert targets[0] == series[3, 0] + series[4, 0]

    def test_windows_and_targets_are_the_same_length(self) -> None:
        windows, targets = make_windows(np.zeros((50, 2)), window=10, horizon=3)
        assert len(windows) == len(targets) == 50 - 10 - 3 + 1

    def test_too_little_data_yields_nothing(self) -> None:
        windows, targets = make_windows(np.zeros((3, 2)), window=10)
        assert len(windows) == 0 and len(targets) == 0

    def test_a_one_dimensional_series_is_refused(self) -> None:
        with pytest.raises(ValueError, match="n_bars, n_features"):
            make_windows(np.zeros(50), window=10)


class TestFitContract:
    def test_the_split_is_chronological(self) -> None:
        """Shuffling before splitting puts tomorrow in the training set."""
        features, targets = _data()
        model = _LinearStub()
        report = model.fit(features, targets, epochs=1, validation_fraction=0.25)
        assert report.train_samples == int(len(features) * 0.75)
        assert report.validation_samples == len(features) - report.train_samples

    def test_the_baseline_uses_the_training_mean(self) -> None:
        """Using the validation mean would leak the held-out average."""
        features, targets = _data()
        report = _LinearStub().fit(features, targets, epochs=1)
        split = report.train_samples
        expected = float(np.mean((targets[split:] - float(np.mean(targets[:split]))) ** 2))
        assert report.baseline_mse == pytest.approx(expected)

    def test_a_mismatched_feature_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="samples, timesteps, features"):
            _LinearStub().fit(np.zeros((10, 5)), np.zeros(10))

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="targets"):
            _LinearStub().fit(np.zeros((10, 4, 2)), np.zeros(9))

    def test_an_impossible_split_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be split"):
            _LinearStub().fit(np.zeros((2, 4, 2)), np.zeros(2), validation_fraction=0.99)

    def test_an_out_of_range_validation_fraction_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly between"):
            _LinearStub().fit(np.zeros((10, 4, 2)), np.zeros(10), validation_fraction=1.0)


class TestPredictionRefusal:
    def test_an_unfitted_model_raises(self) -> None:
        """Zeros here would be averaged into an ensemble as a neutral view."""
        with pytest.raises(ModelNotFittedError, match="has not been fitted"):
            _LinearStub().predict(np.zeros((1, 16, 3)))

    def test_a_fitted_model_predicts(self) -> None:
        features, targets = _data()
        model = _LinearStub()
        model.fit(features, targets, epochs=1)
        assert model.predict(features[:5]).shape == (5,)


class TestSkill:
    def _report(self, **kwargs: float) -> TrainingReport:
        defaults: dict[str, float] = {
            "train_samples": 100, "validation_samples": 20,
            "validation_mse": 1.0, "baseline_mse": 2.0,
            "directional_accuracy": 0.55, "epochs": 5,
        }
        defaults.update(kwargs)
        return TrainingReport(**defaults)  # type: ignore[arg-type]

    def test_skill_is_the_fraction_of_baseline_error_removed(self) -> None:
        assert self._report(validation_mse=1.0, baseline_mse=2.0).skill == pytest.approx(0.5)

    def test_a_model_worse_than_the_mean_has_negative_skill(self) -> None:
        assert self._report(validation_mse=3.0, baseline_mse=2.0).skill < 0

    def test_skill_alone_is_not_enough(self) -> None:
        """Beating the mean while calling direction worse than a coin is not
        a tradable model — it is a model that hedges its magnitude well."""
        assert not self._report(directional_accuracy=0.45).has_skill

    def test_direction_alone_is_not_enough(self) -> None:
        assert not self._report(
            validation_mse=3.0, baseline_mse=2.0, directional_accuracy=0.7
        ).has_skill

    def test_both_together_are_enough(self) -> None:
        assert self._report(
            validation_mse=1.0, baseline_mse=2.0, directional_accuracy=0.6
        ).has_skill

    def test_a_zero_baseline_reports_no_skill_rather_than_dividing(self) -> None:
        assert self._report(baseline_mse=0.0).skill == 0.0

    def test_the_render_names_the_verdict(self) -> None:
        assert "NO DEMONSTRATED SKILL" in self._report(directional_accuracy=0.4).render()


def _panel(n_bars: int = 300, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    symbols = ("AAA", "BBB")
    steps = rng.normal(0.0002, 0.01, size=(n_bars, len(symbols)))
    return Panel(
        symbols=symbols,
        index=pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC"),
        closes=100.0 * np.exp(np.cumsum(steps, axis=0)),
        volumes=rng.lognormal(10.0, 0.2, size=(n_bars, len(symbols))),
        provenance=Provenance.real("test"),
    )


class TestDeepModelAgent:
    def _fitted(self, *, gain: float, seed: int = 0) -> _LinearStub:
        model = _LinearStub(gain=gain)
        features, targets = _data(seed=seed)
        model.fit(features, targets, epochs=1)
        return model

    def test_an_unfitted_model_emits_nothing(self) -> None:
        agent = DeepModelAgent({"AAA": _LinearStub()}, window=16)
        assert agent.generate_checked(_panel(), 200) == []

    def test_a_model_without_skill_emits_nothing(self) -> None:
        """The normal outcome on financial data — it must not reach the blend."""
        model = self._fitted(gain=1.0)
        assert model.report is not None
        if model.report.has_skill:
            pytest.skip("stub happened to show skill on this seed")
        assert DeepModelAgent({"AAA": model}, window=16).generate_checked(_panel(), 200) == []

    def test_a_skilled_model_emits_a_signal(self) -> None:
        model = self._fitted(gain=1.0)
        object.__setattr__(
            model,
            "_report",
            TrainingReport(
                train_samples=100, validation_samples=40, validation_mse=0.5,
                baseline_mse=1.0, directional_accuracy=0.62, epochs=3,
            ),
        )
        signals = DeepModelAgent({"AAA": model}, window=16).generate_checked(_panel(), 200)
        assert len(signals) == 1
        assert signals[0].symbol == "AAA"

    def test_confidence_comes_from_validation_skill(self) -> None:
        """Not from the size of the network's output."""
        model = self._fitted(gain=1000.0)
        object.__setattr__(
            model,
            "_report",
            TrainingReport(
                train_samples=100, validation_samples=40, validation_mse=0.7,
                baseline_mse=1.0, directional_accuracy=0.6, epochs=3,
            ),
        )
        signals = DeepModelAgent({"AAA": model}, window=16).generate_checked(_panel(), 200)
        assert signals[0].confidence == pytest.approx(0.3, abs=1e-9)
        # A huge prediction saturates the signal but must not move belief.
        assert abs(signals[0].signal) == pytest.approx(1.0)

    def test_a_symbol_absent_from_the_panel_is_skipped(self) -> None:
        model = self._fitted(gain=1.0)
        agent = DeepModelAgent({"NOT_HERE": model}, window=16)
        assert agent.generate_checked(_panel(), 200) == []

    def test_a_non_positive_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="scale must be positive"):
            DeepModelAgent({}, scale=0.0)

    def test_it_participates_in_the_ensemble_as_reduced_coverage(self) -> None:
        """A silent model lowers confidence rather than adding a neutral vote."""
        from axiom.alpha.agents import MomentumAgent

        ensemble = AlphaEnsemble([MomentumAgent(), DeepModelAgent({}, window=16)])
        views = ensemble.run(_panel(), 250)
        assert views
        assert all(v.coverage == pytest.approx(0.5) for v in views)


class TestTrainingWindowBuilder:
    def test_it_produces_aligned_windows(self) -> None:
        windows, targets = build_training_windows(_panel(), "AAA", window=16)
        assert len(windows) == len(targets)
        assert windows.shape[1:] == (16, 3)

    def test_too_little_history_yields_nothing(self) -> None:
        windows, _ = build_training_windows(_panel(n_bars=10), "AAA", window=16)
        assert len(windows) == 0


class TestOptionalTorch:
    def test_availability_is_reported_honestly(self) -> None:
        assert isinstance(torch_available(), bool)

    @pytest.mark.skipif(not torch_available(), reason="torch extra not installed")
    def test_the_transformer_trains_and_predicts(self) -> None:
        from axiom.models.sequence import PriceTransformerModel

        features, targets = _data(n=200)
        model = PriceTransformerModel(d_model=32, n_heads=2, n_layers=1)
        model.fit(features, targets, epochs=1)
        assert model.predict(features[:4]).shape == (4,)

    @pytest.mark.skipif(torch_available(), reason="torch extra is installed")
    def test_a_missing_extra_explains_itself(self) -> None:
        from axiom.models.sequence import PriceTransformerModel

        with pytest.raises(ImportError, match=r"axiom\[deep\]"):
            PriceTransformerModel().fit(*_data(n=60), epochs=1)
