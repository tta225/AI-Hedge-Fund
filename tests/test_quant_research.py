"""HMM, regime detection, and the strategy-search honesty statistics.

The load-bearing tests here are the causality ones. An HMM decoded with Viterbi
over the full history, or gated on a full-history fit, produces a backtest that
knew the future — and it looks *excellent*, which is what makes it dangerous.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from axiom.core.series import OHLCVSeries
from axiom.core.sessions import NY_AM_KILLZONE
from axiom.ict.confluence import find_unicorns, power_of_three
from axiom.ict.engine import ICTEngine
from axiom.quant.hmm import GaussianHMM, HMMConvergenceError
from axiom.quant.regime import UNKNOWN_REGIME, RegimeLabel, RegimeModel, RegimeSeries
from axiom.research.lab import Candidate, StrategyLab, grid, walk_forward_splits
from axiom.research.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    parameter_stability,
    probability_of_backtest_overfitting,
)
from axiom.strategy.quant_strategies import (
    MeanReversionZScore,
    RegimeGatedStrategy,
    TimeSeriesMomentum,
)


def two_regime_data(n: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic tape with two genuinely different states."""
    rng = np.random.default_rng(seed)
    true = np.zeros(n, dtype=int)
    x = np.zeros((n, 2))
    state = 0
    for t in range(n):
        if rng.random() < (0.01 if state == 0 else 0.03):
            state = 1 - state
        true[t] = state
        r = rng.normal(0.0008, 0.004) if state == 0 else rng.normal(-0.0005, 0.020)
        x[t] = [r, abs(r)]
    return x, true


class TestGaussianHMM:
    def test_recovers_known_regimes(self) -> None:
        x, true = two_regime_data()
        model = GaussianHMM(2, seed=1)
        model.fit(x)
        states = model.filter_states(x)
        accuracy = max((states == true).mean(), (states == 1 - true).mean())
        assert accuracy > 0.8

    def test_fit_converges(self) -> None:
        x, _ = two_regime_data()
        result = GaussianHMM(2, seed=1).fit(x)
        assert result.converged
        assert result.log_likelihood > -np.inf

    def test_filter_is_causal(self) -> None:
        """The property the whole module exists to guarantee.

        Filtering a prefix must give the same answer as filtering everything and
        reading that position — i.e. later bars cannot change earlier states.
        """
        x, _ = two_regime_data()
        model = GaussianHMM(2, seed=1)
        model.fit(x)
        full = model.filter(x)
        for cut in (200, 500, 900):
            assert np.allclose(model.filter(x[:cut])[-1], full[cut - 1])

    def test_smoothing_is_not_causal(self) -> None:
        """Documents why smoothing must never be used in a backtest."""
        x, _ = two_regime_data()
        model = GaussianHMM(2, seed=1)
        model.fit(x)
        prefix = model.smooth(x[:600])[-1]
        full = model.smooth(x)[599]
        # Future bars change the estimate — that is the definition of a leak.
        assert not np.allclose(prefix, full, atol=1e-6)

    def test_incremental_filter_matches_batch(self) -> None:
        x, _ = two_regime_data(400)
        model = GaussianHMM(2, seed=1)
        model.fit(x)
        batch = model.filter(x)
        alpha = model.filter_init(x[0])
        incremental = [alpha]
        for t in range(1, len(x)):
            alpha = model.filter_step(alpha, x[t])
            incremental.append(alpha)
        assert np.allclose(np.array(incremental), batch, atol=1e-10)

    def test_states_are_canonically_ordered(self) -> None:
        """Otherwise a refit can silently swap state labels."""
        x, _ = two_regime_data()
        model = GaussianHMM(3, seed=1)
        model.fit(x)
        assert model.means is not None
        assert np.all(np.diff(model.means[:, 0]) >= 0)

    def test_expected_durations_are_positive(self) -> None:
        x, _ = two_regime_data()
        model = GaussianHMM(2, seed=1)
        model.fit(x)
        assert (model.expected_durations > 0).all()

    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(HMMConvergenceError, match="too few"):
            GaussianHMM(3).fit(np.zeros((10, 2)))

    def test_rejects_non_finite_input(self) -> None:
        x, _ = two_regime_data(300)
        x[5, 0] = np.nan
        with pytest.raises(HMMConvergenceError, match="NaN"):
            GaussianHMM(2).fit(x)

    def test_unfitted_model_refuses_inference(self) -> None:
        with pytest.raises(HMMConvergenceError, match="not fitted"):
            GaussianHMM(2).filter(np.zeros((10, 2)))


class TestRegimeModel:
    def test_causal_fit_leaves_a_warmup_gap(self, synthetic_series: OHLCVSeries) -> None:
        model = RegimeModel(n_states=2, min_train=400, refit_every=800)
        regimes = model.fit_causal(synthetic_series.head(2000))
        assert (regimes.states[:400] == UNKNOWN_REGIME).all()
        assert not regimes.is_known_at(0)

    def test_causal_fit_produces_regimes_after_warmup(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        model = RegimeModel(n_states=2, min_train=400, refit_every=800)
        regimes = model.fit_causal(synthetic_series.head(2000))
        assert regimes.is_known_at(1500)
        assert regimes.label_at(1500) is not RegimeLabel.UNKNOWN
        assert 0.0 < regimes.confidence_at(1500) <= 1.0

    def test_refits_use_only_prior_data(self, synthetic_series: OHLCVSeries) -> None:
        model = RegimeModel(n_states=2, min_train=400, refit_every=500)
        regimes = model.fit_causal(synthetic_series.head(2000))
        assert regimes.refit_points
        assert min(regimes.refit_points) >= 400

    def test_occupancy_sums_to_one(self, synthetic_series: OHLCVSeries) -> None:
        model = RegimeModel(n_states=2, min_train=400, refit_every=800)
        regimes = model.fit_causal(synthetic_series.head(2000))
        assert sum(regimes.occupancy().values()) == pytest.approx(1.0)

    def test_full_history_fit_is_marked_non_causal(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        model = RegimeModel(n_states=2, min_train=400)
        regimes = model.fit_full_history(synthetic_series.head(1500))
        assert regimes.warmup_end == 0
        assert regimes.refit_points == (0,)


class TestRegimeGating:
    def _causal(self, series: OHLCVSeries) -> RegimeSeries:
        return RegimeModel(n_states=2, min_train=400, refit_every=800).fit_causal(series)

    def test_rejects_a_full_history_regime_series(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The guard that stops the most attractive possible mistake."""
        leaky = RegimeModel(n_states=2, min_train=400).fit_full_history(
            synthetic_series.head(1500)
        )
        with pytest.raises(ValueError, match="leaks the future"):
            RegimeGatedStrategy(
                TimeSeriesMomentum(), leaky, (RegimeLabel.TRENDING_UP,)
            )

    def test_accepts_a_causal_regime_series(self, synthetic_series: OHLCVSeries) -> None:
        regimes = self._causal(synthetic_series.head(2000))
        gated = RegimeGatedStrategy(
            TimeSeriesMomentum(), regimes, (RegimeLabel.TRENDING_UP,)
        )
        assert "regime_gated" in gated.name

    def test_gate_never_emits_more_than_the_inner_strategy(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A gate is a filter; it must not manufacture signals."""
        from axiom.portfolio.positions import Portfolio
        from axiom.strategy.base import StrategyContext

        series = synthetic_series.head(2000)
        regimes = self._causal(series)
        inner = TimeSeriesMomentum(lookback=48)
        gated = RegimeGatedStrategy(
            inner, regimes, tuple(RegimeLabel), min_confidence=0.0
        )
        portfolio = Portfolio(starting_cash=1_000_000)
        state = ICTEngine().analyse(series)

        inner_count = gated_count = 0
        for i in range(600, 1200):
            context = StrategyContext(
                series=series, index=i, ict=state,
                portfolio=portfolio, timestamp=series.index[i],
            )
            inner_count += inner.evaluate(context) is not None
            gated_count += gated.evaluate(context) is not None
        assert gated_count <= inner_count


class TestOverfittingStatistics:
    def test_skill_bar_rises_with_trial_count(self) -> None:
        """The core insight: more trials means a higher bar, not a better result."""
        few = expected_max_sharpe(5, 0.25)
        many = expected_max_sharpe(500, 0.25)
        assert many > few > 0

    def test_single_trial_has_no_selection_bias(self) -> None:
        assert expected_max_sharpe(1, 0.25) == 0.0

    def test_deflated_sharpe_rewards_genuine_edge(self) -> None:
        strong = deflated_sharpe_ratio(
            2.5, n_observations=500, n_trials=10, variance_of_sharpes=0.25
        )
        assert strong > 0.95

    def test_deflated_sharpe_punishes_selection_from_many_trials(self) -> None:
        few = deflated_sharpe_ratio(
            1.5, n_observations=500, n_trials=5, variance_of_sharpes=0.25
        )
        many = deflated_sharpe_ratio(
            1.5, n_observations=500, n_trials=1000, variance_of_sharpes=0.25
        )
        assert many < few

    def test_deflated_sharpe_punishes_short_samples(self) -> None:
        long_sample = deflated_sharpe_ratio(
            2.0, n_observations=1000, n_trials=10, variance_of_sharpes=0.25
        )
        short_sample = deflated_sharpe_ratio(
            2.0, n_observations=30, n_trials=10, variance_of_sharpes=0.25
        )
        assert short_sample < long_sample

    def test_pure_noise_does_not_clear_the_bar(self) -> None:
        """The end-to-end guarantee: searching noise must not produce 'skill'."""
        rng = np.random.default_rng(4)
        n_trials, n_obs = 200, 500
        returns = rng.normal(0, 0.01, size=(n_obs, n_trials))
        sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1) * np.sqrt(252)
        best = int(np.argmax(sharpes))

        naive = sharpes[best]
        deflated = deflated_sharpe_ratio(
            float(naive),
            n_observations=n_obs,
            n_trials=n_trials,
            variance_of_sharpes=float(np.var(sharpes, ddof=1)),
        )
        # The naive winner looks impressive; the deflated score does not.
        assert naive > 1.5
        assert deflated < 0.95

    def test_pbo_detects_a_non_generalising_selection(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.normal(0, 0.01, size=(400, 20))
        result = probability_of_backtest_overfitting(matrix, n_partitions=6)
        # On pure noise, the in-sample winner should be a coin flip out of
        # sample — PBO near 0.5, definitively not near 0.
        assert 0.2 <= result.probability <= 0.8

    def test_pbo_handles_degenerate_input(self) -> None:
        assert np.isnan(
            probability_of_backtest_overfitting(np.zeros((10, 1))).probability
        )

    def test_parameter_stability_prefers_smooth_surfaces(self) -> None:
        smooth = {(float(i),): 1.0 + 0.01 * i for i in range(10)}
        spiky = {(float(i),): (5.0 if i == 5 else 0.0) for i in range(10)}
        assert parameter_stability(smooth) > parameter_stability(spiky)


class TestWalkForward:
    def test_test_windows_follow_their_training_window(self) -> None:
        for split in walk_forward_splits(5000, n_folds=4, warmup=100):
            assert split.test_start >= split.train_end
            assert split.test_end > split.test_start

    def test_test_windows_do_not_overlap(self) -> None:
        splits = walk_forward_splits(5000, n_folds=4, warmup=100)
        for a, b in pairwise(splits):
            assert b.test_start >= a.test_end

    def test_refuses_too_little_data(self) -> None:
        with pytest.raises(ValueError, match="too few"):
            walk_forward_splits(150, n_folds=4, warmup=100)

    def test_grid_is_the_cartesian_product(self) -> None:
        assert len(grid(a=[1, 2, 3], b=[1, 2])) == 6
        assert grid() == [{}]


class TestStrategyLab:
    def test_search_counts_every_candidate_as_a_trial(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from axiom.core.config import RiskSettings

        series = synthetic_series.head(3000)
        candidates = [
            Candidate("ts_momentum", lambda p=p: TimeSeriesMomentum(**p), p)
            for p in grid(lookback=[48, 96])
        ] + [
            Candidate("mean_reversion", lambda p=p: MeanReversionZScore(**p), p)
            for p in grid(lookback=[48])
        ]
        lab = StrategyLab(
            risk_settings=RiskSettings(
                account_equity=5_000_000, max_gross_exposure_pct=2000.0
            ),
            n_folds=2,
            warmup=200,
        )
        result = lab.search(series, candidates)
        assert result.n_trials == 3
        assert len(result.results) == 3

    def test_synthetic_search_refuses_to_conclude(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from axiom.core.config import RiskSettings

        lab = StrategyLab(
            risk_settings=RiskSettings(
                account_equity=5_000_000, max_gross_exposure_pct=2000.0
            ),
            n_folds=2,
            warmup=200,
        )
        result = lab.search(
            synthetic_series.head(3000),
            [Candidate("ts_momentum", TimeSeriesMomentum, {})],
        )
        assert "NO CONCLUSION" in result.verdict
        assert "SYNTHETIC" in result.render()

    def test_empty_candidate_set_is_refused(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        with pytest.raises(ValueError, match="no candidates"):
            StrategyLab().search(synthetic_series, [])


class TestUnicornModel:
    def test_unicorn_zone_is_the_intersection(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        setups = find_unicorns(
            synthetic_series, state.order_blocks, state.fair_value_gaps
        )
        assert setups
        for setup in setups:
            assert setup.top > setup.bottom
            assert 0.0 < setup.overlap_ratio <= 1.0 + 1e-9

    def test_unicorn_confirms_only_when_both_components_are_known(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        for setup in find_unicorns(
            synthetic_series, state.order_blocks, state.fair_value_gaps
        ):
            assert setup.confirmed_index >= setup.origin_index

    def test_power_of_three_is_causal(self, synthetic_series: OHLCVSeries) -> None:
        """A session still forming must be judged on what has happened."""
        index = len(synthetic_series) // 2
        prefix = synthetic_series.islice(0, index + 1)
        from_prefix = power_of_three(prefix, NY_AM_KILLZONE, index)
        from_full = power_of_three(synthetic_series, NY_AM_KILLZONE, index)
        if from_prefix is None or from_full is None:
            pytest.skip("no session window at this index")
        assert from_prefix.phase == from_full.phase
        assert from_prefix.session_high == from_full.session_high
