"""Tests for the cost model and the turnover-reduction machinery."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from axiom.execution.costs import (
    PARTICIPATION_WARN,
    FlatBpsCost,
    SpreadImpactCost,
    average_dollar_volume,
    trailing_volatility,
)
from axiom.research.turnover import (
    HysteresisBands,
    hysteresis_weights,
    partial_rebalance,
    suppress_small_trades,
)

SYMBOLS = tuple(f"S{i:02d}" for i in range(10))


def _adv(value: float = 1e8, n: int = 10) -> np.ndarray:
    return np.full(n, value)


def _vol(value: float = 0.02, n: int = 10) -> np.ndarray:
    return np.full(n, value)


class TestFlatBpsCost:
    def test_charges_per_unit_traded(self) -> None:
        cost = FlatBpsCost(10.0)
        deltas = np.array([0.5, -0.5])
        assert cost.charge(deltas, aum=1e6, adv_notional=_adv(n=2),
                           volatility=_vol(n=2)) == pytest.approx(10.0 / 10_000)

    def test_is_independent_of_size(self) -> None:
        """The defect the impact model exists to fix."""
        cost = FlatBpsCost(10.0)
        deltas = np.array([1.0])
        small = cost.charge(deltas, aum=1e5, adv_notional=_adv(n=1), volatility=_vol(n=1))
        huge = cost.charge(deltas, aum=1e12, adv_notional=_adv(n=1), volatility=_vol(n=1))
        assert small == huge

    def test_no_trade_is_free(self) -> None:
        assert FlatBpsCost(10.0).charge(
            np.zeros(3), aum=1e6, adv_notional=_adv(n=3), volatility=_vol(n=3)
        ) == 0.0

    def test_describes_itself_as_size_independent(self) -> None:
        assert "size-independent" in FlatBpsCost().describe()


class TestSpreadImpactCost:
    model = SpreadImpactCost()

    def test_a_tiny_trade_costs_about_the_spread(self) -> None:
        """With negligible participation, impact vanishes and spread remains."""
        deltas = np.array([1e-6])
        charged = self.model.charge(
            deltas, aum=1e6, adv_notional=_adv(1e10, 1), volatility=_vol(0.02, 1)
        )
        linear = 1e-6 * (self.model.half_spread_bps + self.model.commission_bps) / 1e4
        assert charged == pytest.approx(linear, rel=0.01)

    def test_cost_rises_with_capital(self) -> None:
        """The whole point: the same book is not equally tradable at every size."""
        deltas = np.array([1.0])
        small = self.model.charge(deltas, aum=1e6, adv_notional=_adv(n=1),
                                  volatility=_vol(n=1))
        large = self.model.charge(deltas, aum=1e10, adv_notional=_adv(n=1),
                                  volatility=_vol(n=1))
        assert large > small * 10

    def test_impact_follows_the_square_root_law(self) -> None:
        """A 100x larger trade pays 10x more per unit, not 100x."""
        bare = SpreadImpactCost(half_spread_bps=0.0, commission_bps=0.0)
        deltas = np.array([1.0])
        one = bare.charge(deltas, aum=1e6, adv_notional=_adv(n=1), volatility=_vol(n=1))
        hundred = bare.charge(
            deltas, aum=1e8, adv_notional=_adv(n=1), volatility=_vol(n=1)
        )
        assert hundred / one == pytest.approx(10.0, rel=0.01)

    def test_a_more_volatile_name_costs_more(self) -> None:
        deltas = np.array([1.0])
        calm = self.model.charge(deltas, aum=1e8, adv_notional=_adv(n=1),
                                 volatility=_vol(0.01, 1))
        wild = self.model.charge(deltas, aum=1e8, adv_notional=_adv(n=1),
                                 volatility=_vol(0.05, 1))
        assert wild > calm

    def test_a_less_liquid_name_costs_more(self) -> None:
        deltas = np.array([1.0])
        liquid = self.model.charge(deltas, aum=1e8, adv_notional=_adv(1e10, 1),
                                   volatility=_vol(n=1))
        thin = self.model.charge(deltas, aum=1e8, adv_notional=_adv(1e7, 1),
                                 volatility=_vol(n=1))
        assert thin > liquid

    def test_zero_volume_does_not_divide_by_zero(self) -> None:
        """A missing volume history must not produce an infinite cost."""
        charged = self.model.charge(
            np.array([1.0]), aum=1e8, adv_notional=np.array([0.0]),
            volatility=_vol(n=1),
        )
        assert np.isfinite(charged)

    def test_nan_inputs_are_survived(self) -> None:
        charged = self.model.charge(
            np.array([1.0]), aum=1e8, adv_notional=np.array([np.nan]),
            volatility=np.array([np.nan]),
        )
        assert np.isfinite(charged)

    def test_no_trade_is_free(self) -> None:
        assert self.model.charge(
            np.zeros(3), aum=1e8, adv_notional=_adv(n=3), volatility=_vol(n=3)
        ) == 0.0

    def test_zero_aum_is_refused(self) -> None:
        """Impact is undefined without capital; guessing would be worse."""
        with pytest.raises(ValueError, match="aum must be positive"):
            self.model.charge(np.array([1.0]), aum=0.0, adv_notional=_adv(n=1),
                              volatility=_vol(n=1))

    def test_participation_is_reported(self) -> None:
        share = self.model.participation(
            np.array([0.1]), aum=1e9, adv_notional=_adv(1e9, 1)
        )
        assert share[0] == pytest.approx(0.1)

    def test_participation_flags_extrapolation(self) -> None:
        """Above the warn level the square-root law is outside its fit."""
        share = self.model.participation(
            np.array([0.5]), aum=1e9, adv_notional=_adv(1e9, 1)
        )
        assert share[0] > PARTICIPATION_WARN

    def test_rejects_negative_parameters(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SpreadImpactCost(half_spread_bps=-1.0)
        with pytest.raises(ValueError, match="impact_coefficient"):
            SpreadImpactCost(impact_coefficient=-1.0)
        with pytest.raises(ValueError, match="min_adv_notional"):
            SpreadImpactCost(min_adv_notional=0.0)

    def test_describes_its_components(self) -> None:
        assert "participation" in SpreadImpactCost().describe()


class TestLiquidityEstimates:
    @staticmethod
    def _panel_arrays(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0))
        volumes = np.full((n, 3), 1e6)
        return closes, volumes

    def test_dollar_volume_uses_price_not_shares(self) -> None:
        """A hundred thousand shares of a $5 and a $500 stock are not alike."""
        closes = np.array([[10.0, 100.0]] * 30)
        volumes = np.full((30, 2), 1000.0)
        adv = average_dollar_volume(closes, volumes, 30)
        assert adv[1] == pytest.approx(adv[0] * 10)

    def test_it_is_causal(self) -> None:
        """Using the traded day's volume flatters exactly the large trades."""
        closes, volumes = self._panel_arrays()
        volumes[50] = 1e12  # a huge spike on the bar being scored
        assert average_dollar_volume(closes, volumes, 50).max() < 1e11

    def test_no_history_yields_zero(self) -> None:
        closes, volumes = self._panel_arrays()
        assert average_dollar_volume(closes, volumes, 0).sum() == 0.0

    def test_volatility_is_per_bar_not_annualised(self) -> None:
        """Annualising would overstate a single trade's impact ~16x."""
        closes, _ = self._panel_arrays(200)
        vol = trailing_volatility(closes, 150)
        assert 0.001 < vol.mean() < 0.05

    def test_volatility_is_causal(self) -> None:
        closes, _ = self._panel_arrays(200)
        closes[150] *= 3.0  # a shock on the scored bar
        assert trailing_volatility(closes, 150).max() < 0.2

    def test_short_history_is_zero_not_nan(self) -> None:
        closes, _ = self._panel_arrays()
        assert np.isfinite(trailing_volatility(closes, 1)).all()

    def test_rejects_degenerate_lookbacks(self) -> None:
        closes, volumes = self._panel_arrays()
        with pytest.raises(ValueError, match="lookback"):
            average_dollar_volume(closes, volumes, 50, lookback=0)
        with pytest.raises(ValueError, match="lookback"):
            trailing_volatility(closes, 50, lookback=1)


class TestHysteresisBands:
    def test_buffer_width(self) -> None:
        assert HysteresisBands(0.2, 0.4).buffer == pytest.approx(0.2)

    def test_equal_bands_mean_no_buffer(self) -> None:
        """A legitimate control, not an error."""
        assert HysteresisBands(0.2, 0.2).buffer == 0.0

    def test_an_exit_tighter_than_entry_is_refused(self) -> None:
        """It would sell names it had just bought."""
        with pytest.raises(ValueError, match="must be between"):
            HysteresisBands(0.4, 0.2)

    def test_rejects_an_impossible_entry(self) -> None:
        with pytest.raises(ValueError, match="entry_fraction"):
            HysteresisBands(0.9, 0.9)


class TestHysteresisWeights:
    scores: ClassVar[dict[str, float]] = {
        s: float(i) for i, s in enumerate(SYMBOLS)
    }

    def test_from_flat_it_takes_the_entry_band(self) -> None:
        weights = hysteresis_weights(
            self.scores, SYMBOLS, np.zeros(10), bands=HysteresisBands(0.2, 0.4)
        )
        assert weights[-1] > 0  # best scored is long
        assert weights[0] < 0  # worst scored is short

    def test_it_is_dollar_neutral(self) -> None:
        weights = hysteresis_weights(self.scores, SYMBOLS, np.zeros(10))
        assert weights.sum() == pytest.approx(0.0)

    def test_gross_exposure_is_respected(self) -> None:
        weights = hysteresis_weights(self.scores, SYMBOLS, np.zeros(10), gross=2.0)
        assert np.abs(weights).sum() == pytest.approx(2.0)

    def test_a_held_name_that_slipped_is_kept(self) -> None:
        """The whole mechanism: rank 20 to 21 is not a reason to trade."""
        held = np.zeros(10)
        held[7] = 0.25  # held, but now outside the top 20%
        bands = HysteresisBands(0.2, 0.5)
        weights = hysteresis_weights(self.scores, SYMBOLS, held, bands=bands)
        assert weights[7] > 0

    def test_a_held_name_outside_the_exit_band_is_dropped(self) -> None:
        held = np.zeros(10)
        held[2] = 0.25  # held, but ranked near the bottom
        weights = hysteresis_weights(
            self.scores, SYMBOLS, held, bands=HysteresisBands(0.2, 0.3)
        )
        assert weights[2] <= 0

    def test_buffering_reduces_turnover_against_no_buffer(self) -> None:
        """Measured directly: same scores, same holdings, less trading."""
        rng = np.random.default_rng(0)
        held = hysteresis_weights(self.scores, SYMBOLS, np.zeros(10))
        churned = {s: float(rng.normal()) for s in SYMBOLS}

        wide = hysteresis_weights(
            churned, SYMBOLS, held, bands=HysteresisBands(0.2, 0.5)
        )
        tight = hysteresis_weights(
            churned, SYMBOLS, held, bands=HysteresisBands(0.2, 0.2)
        )
        assert np.abs(wide - held).sum() <= np.abs(tight - held).sum()

    def test_a_name_is_never_both_long_and_short(self) -> None:
        """A stale long in a bottom-ranked name is not thrift."""
        held = np.zeros(10)
        held[0] = 0.25  # held long, but ranked worst
        weights = hysteresis_weights(
            self.scores, SYMBOLS, held, bands=HysteresisBands(0.2, 0.5)
        )
        assert weights[0] <= 0

    def test_long_only_has_no_shorts(self) -> None:
        weights = hysteresis_weights(
            self.scores, SYMBOLS, np.zeros(10), long_short=False
        )
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_too_few_scores_produce_no_book(self) -> None:
        assert not np.any(hysteresis_weights({"S00": 1.0}, SYMBOLS, np.zeros(10)))

    def test_mismatched_current_weights_are_refused(self) -> None:
        with pytest.raises(ValueError, match="current weights"):
            hysteresis_weights(self.scores, SYMBOLS, np.zeros(3))

    def test_non_finite_scores_are_ignored(self) -> None:
        scores = dict(self.scores)
        scores["S05"] = float("nan")
        weights = hysteresis_weights(scores, SYMBOLS, np.zeros(10))
        assert np.isfinite(weights).all()


class TestPartialRebalance:
    def test_full_fraction_reaches_the_target(self) -> None:
        current, target = np.zeros(3), np.array([1.0, -1.0, 0.5])
        assert partial_rebalance(current, target, 1.0) == pytest.approx(target)

    def test_half_goes_halfway(self) -> None:
        current, target = np.zeros(3), np.array([1.0, -1.0, 0.5])
        assert partial_rebalance(current, target, 0.5) == pytest.approx(target / 2)

    def test_it_reduces_turnover_proportionally(self) -> None:
        current, target = np.zeros(4), np.array([0.25, -0.25, 0.25, -0.25])
        full = np.abs(partial_rebalance(current, target, 1.0) - current).sum()
        half = np.abs(partial_rebalance(current, target, 0.5) - current).sum()
        assert half == pytest.approx(full / 2)

    def test_rejects_a_zero_fraction(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            partial_rebalance(np.zeros(2), np.zeros(2), 0.0)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            partial_rebalance(np.zeros(2), np.zeros(3), 0.5)


class TestSuppressSmallTrades:
    def test_a_small_change_is_left_alone(self) -> None:
        current = np.array([0.25, 0.25])
        target = np.array([0.26, 0.25])
        assert suppress_small_trades(current, target, 0.05) == pytest.approx(current)

    def test_a_large_change_goes_through(self) -> None:
        current = np.array([0.25, 0.25])
        target = np.array([0.50, 0.25])
        assert suppress_small_trades(current, target, 0.05)[0] == pytest.approx(0.50)

    def test_it_is_applied_per_name(self) -> None:
        """Many small trades and one large one cost very different amounts."""
        current = np.array([0.25, 0.25, 0.25])
        target = np.array([0.26, 0.50, 0.24])
        result = suppress_small_trades(current, target, 0.05)
        assert result[0] == pytest.approx(0.25)
        assert result[1] == pytest.approx(0.50)
        assert result[2] == pytest.approx(0.25)

    def test_a_zero_threshold_changes_nothing(self) -> None:
        current, target = np.zeros(3), np.array([0.1, 0.2, 0.3])
        assert suppress_small_trades(current, target, 0.0) == pytest.approx(target)

    def test_rejects_a_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            suppress_small_trades(np.zeros(2), np.zeros(2), -0.1)


class TestBacktestIntegration:
    """The properties that only appear once these are wired to the backtester."""

    @staticmethod
    def _panel(n_symbols: int = 20, n_bars: int = 600):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime, timedelta

        import pandas as pd

        from axiom.alpha.panel import Panel
        from axiom.core.provenance import Provenance

        rng = np.random.default_rng(3)
        closes = 100.0 * np.exp(
            np.cumsum(rng.normal(0, 0.01, size=(n_bars, n_symbols)), axis=0)
        )
        start = datetime(2020, 1, 1, tzinfo=UTC)
        return Panel(
            symbols=tuple(f"S{i:02d}" for i in range(n_symbols)),
            index=pd.DatetimeIndex(
                [start + i * timedelta(days=1) for i in range(n_bars)]
            ),
            closes=closes,
            volumes=np.full_like(closes, 1e7),
            provenance=Provenance.real("test"),
        )

    def test_a_buffer_lowers_turnover_end_to_end(self) -> None:
        from axiom.alpha.agents import MomentumAgent
        from axiom.research.panel_lab import backtest_panel

        panel = self._panel()
        plain = backtest_panel(
            [MomentumAgent({"lookback": 60})], panel, start=100, rebalance_every=5
        )
        buffered = backtest_panel(
            [MomentumAgent({"lookback": 60})], panel, start=100, rebalance_every=5,
            bands=HysteresisBands(0.2, 0.4),
        )
        assert buffered.annual_turnover < plain.annual_turnover

    def test_the_impact_model_charges_more_at_larger_size(self) -> None:
        from axiom.alpha.agents import MomentumAgent
        from axiom.research.panel_lab import backtest_panel

        panel = self._panel()
        common = {"start": 100, "rebalance_every": 5, "cost_model": SpreadImpactCost()}
        small = backtest_panel(
            [MomentumAgent({"lookback": 60})], panel, aum=1e6, **common  # type: ignore[arg-type]
        )
        large = backtest_panel(
            [MomentumAgent({"lookback": 60})], panel, aum=1e10, **common  # type: ignore[arg-type]
        )
        assert large.returns.sum() < small.returns.sum()

    def test_capacity_curve_decreases_with_size(self) -> None:
        from axiom.alpha.agents import MomentumAgent
        from axiom.research.panel_lab import capacity_curve

        panel = self._panel()
        curve = capacity_curve(
            [MomentumAgent({"lookback": 60})], panel,
            aums=[1e6, 1e9], start=100, rebalance_every=5,
        )
        assert curve[1e9].returns.sum() < curve[1e6].returns.sum()

    def test_partial_rebalancing_lowers_turnover(self) -> None:
        from axiom.alpha.agents import MomentumAgent
        from axiom.research.panel_lab import backtest_panel

        panel = self._panel()
        full = backtest_panel(
            [MomentumAgent({"lookback": 60})], panel, start=100, rebalance_every=5
        )
        partial = backtest_panel(
            [MomentumAgent({"lookback": 60})], panel, start=100, rebalance_every=5,
            rebalance_fraction=0.25,
        )
        assert partial.annual_turnover < full.annual_turnover
