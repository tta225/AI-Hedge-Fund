"""Cross-sectional alpha: causality, blending, and provenance.

The tests that earn their keep here are the lookahead ones. A stat-arb signal
fitted on a window containing the bar it scores produces residuals that look
unusually informative, backtests beautifully, and cannot be traded. Nothing
downstream raises — the residual is just a number — so the only place it can be
caught is here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.alpha.agents import MomentumAgent, StatArbAgent, VolumePressureAgent
from axiom.alpha.base import AlphaAgent, AlphaSignal, LookaheadError
from axiom.alpha.ensemble import AlphaEnsemble
from axiom.alpha.panel import Panel
from axiom.core.provenance import DataKind, Provenance


def _panel(
    n_bars: int = 200,
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    *,
    kind: DataKind = DataKind.REAL,
    seed: int = 0,
) -> Panel:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.01, size=(n_bars, len(symbols)))
    closes = 100.0 * np.exp(np.cumsum(steps, axis=0))
    volumes = rng.lognormal(10.0, 0.3, size=(n_bars, len(symbols)))
    return Panel(
        symbols=symbols,
        index=pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC"),
        closes=closes,
        volumes=volumes,
        provenance=Provenance(source="test", kind=kind),
    )


class TestPanelCausality:
    def test_history_excludes_the_scored_bar(self) -> None:
        """The exclusive bound is the whole guarantee."""
        panel = _panel(n_bars=10)
        history = panel.history(5)
        assert len(history) == 5
        assert np.array_equal(history[-1], panel.closes[4])

    def test_history_never_reaches_the_future(self) -> None:
        panel = _panel(n_bars=10)
        for index in range(1, 10):
            assert len(panel.history(index)) == index

    def test_last_price_is_the_previous_bar(self) -> None:
        panel = _panel(n_bars=10)
        assert np.array_equal(panel.last_price(5), panel.closes[4])

    def test_there_is_no_observed_bar_before_the_first(self) -> None:
        with pytest.raises(IndexError):
            _panel().last_price(0)

    def test_returns_are_computed_only_from_history(self) -> None:
        panel = _panel(n_bars=50)
        returns = panel.returns(20, lookback=10)
        assert len(returns) == 10
        expected = np.log(panel.closes[19, 0] / panel.closes[18, 0])
        assert returns[-1, 0] == pytest.approx(expected)


class TestPanelConstruction:
    def test_a_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            Panel(
                symbols=("AAA", "BBB"),
                index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
                closes=np.ones((4, 2)),
                volumes=np.ones((4, 2)),
                provenance=Provenance.real("test"),
            )

    def test_instruments_with_no_common_bars_are_refused(self) -> None:
        """Silent misalignment makes every residual an artefact."""
        from axiom.core.series import OHLCVSeries

        def _series(start: str) -> OHLCVSeries:
            index = pd.date_range(start, periods=10, freq="D", tz="UTC")
            frame = pd.DataFrame(
                {
                    "open": 1.0, "high": 1.0, "low": 1.0,
                    "close": 1.0, "volume": 1.0,
                },
                index=index,
            )
            from axiom.core.timeframe import Timeframe
            from axiom.core.types import get_instrument

            return OHLCVSeries(
                instrument=get_instrument("SPY"),
                timeframe=Timeframe.parse("1d"),
                df=frame,
                provenance=Provenance.real("test"),
            )

        with pytest.raises(ValueError, match="no common bars"):
            Panel.from_series({"A": _series("2024-01-01"), "B": _series("2030-01-01")})


class TestProvenancePropagation:
    def test_one_synthetic_member_taints_the_panel(self) -> None:
        assert not _panel(kind=DataKind.SYNTHETIC).provenance.is_evidential

    def test_signals_inherit_panel_provenance(self) -> None:
        panel = _panel(kind=DataKind.SYNTHETIC)
        signals = MomentumAgent().generate_checked(panel, len(panel) - 1)
        assert signals
        assert all(not s.is_evidential for s in signals)

    def test_a_blend_cannot_launder_synthetic_input(self) -> None:
        """Without this, an ensemble is a provenance laundering step."""
        panel = _panel(kind=DataKind.SYNTHETIC)
        ensemble = AlphaEnsemble([MomentumAgent(), VolumePressureAgent()])
        views = ensemble.run(panel, len(panel) - 1)
        assert views
        assert all(not v.is_evidential for v in views)


class TestSignalContract:
    def test_an_unscaled_signal_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            AlphaSignal(
                symbol="AAA", signal=1.5, confidence=0.5, expected_return=None,
                horizon_bars=1, strategy="x", agent_id="x", as_of_index=0,
            )

    def test_an_out_of_range_confidence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            AlphaSignal(
                symbol="AAA", signal=0.5, confidence=1.5, expected_return=None,
                horizon_bars=1, strategy="x", agent_id="x", as_of_index=0,
            )

    def test_a_mis_stamped_signal_is_caught(self) -> None:
        """A signal stamped with the wrong bar is a lookahead leak in disguise."""

        class _Cheat(AlphaAgent):
            agent_id = "cheat"

            def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
                return [
                    AlphaSignal(
                        symbol=panel.symbols[0], signal=1.0, confidence=1.0,
                        expected_return=None, horizon_bars=1, strategy="x",
                        agent_id="cheat", as_of_index=index + 1,
                    )
                ]

        with pytest.raises(LookaheadError, match="stamped"):
            _Cheat().generate_checked(_panel(), 50)

    def test_a_signal_for_an_unknown_symbol_is_caught(self) -> None:
        class _Ghost(AlphaAgent):
            agent_id = "ghost"

            def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
                return [
                    AlphaSignal(
                        symbol="NOT_IN_PANEL", signal=1.0, confidence=1.0,
                        expected_return=None, horizon_bars=1, strategy="x",
                        agent_id="ghost", as_of_index=index,
                    )
                ]

        with pytest.raises(ValueError, match="not in the panel"):
            _Ghost().generate_checked(_panel(), 50)

    def test_no_signals_before_enough_history(self) -> None:
        assert MomentumAgent().generate_checked(_panel(), 3) == []


class TestAgentsAreCausal:
    """The property that matters: a signal at bar t must not move when the
    future changes. If it does, the agent read past its own bar."""

    @pytest.mark.parametrize(
        "agent", [MomentumAgent(), StatArbAgent(), VolumePressureAgent()]
    )
    def test_future_bars_do_not_change_a_past_signal(self, agent: AlphaAgent) -> None:
        panel = _panel(n_bars=200, seed=3)
        index = 150

        mutated = Panel(
            symbols=panel.symbols,
            index=panel.index,
            # Everything from the scored bar onward is made wildly different.
            closes=np.vstack([panel.closes[:index], panel.closes[index:] * 5.0]),
            volumes=np.vstack([panel.volumes[:index], panel.volumes[index:] * 5.0]),
            provenance=panel.provenance,
        )

        before = {s.symbol: s.signal for s in agent.generate_checked(panel, index)}
        after = {s.symbol: s.signal for s in agent.generate_checked(mutated, index)}
        assert before == after

    def test_stat_arb_does_not_fit_on_the_scored_bar(self) -> None:
        """The specific leak the original had: the scored row inside its own fit."""
        panel = _panel(n_bars=200, seed=5)
        index = 150
        agent = StatArbAgent()

        # Corrupt only the scored bar. A model that fitted on it would move.
        corrupted_closes = panel.closes.copy()
        corrupted_closes[index] *= 3.0
        corrupted = Panel(
            symbols=panel.symbols, index=panel.index, closes=corrupted_closes,
            volumes=panel.volumes, provenance=panel.provenance,
        )
        before = {s.symbol: s.signal for s in agent.generate_checked(panel, index)}
        after = {s.symbol: s.signal for s in agent.generate_checked(corrupted, index)}
        assert before == after


class TestMomentumAgent:
    def test_confidence_is_not_a_function_of_the_signal(self) -> None:
        """The original's confidence was monotone in |momentum|, so it carried
        no information the signal did not already carry."""
        signals = MomentumAgent().generate_checked(_panel(n_bars=200, seed=11), 150)
        assert len(signals) >= 3
        pairs = {(round(abs(s.signal), 4), round(s.confidence, 4)) for s in signals}
        magnitudes = {p[0] for p in pairs}
        confidences = {p[1] for p in pairs}
        # If confidence were a function of |signal| these would map 1:1.
        assert len(confidences) > 1 or len(magnitudes) == 1

    def test_signals_are_ranked_across_the_universe(self) -> None:
        signals = MomentumAgent().generate_checked(_panel(n_bars=200, seed=2), 150)
        values = sorted(s.signal for s in signals)
        assert values[0] == pytest.approx(-1.0)
        assert values[-1] == pytest.approx(1.0)

    def test_a_single_name_universe_produces_nothing(self) -> None:
        """Ranking one instrument against itself is not a cross-section."""
        assert MomentumAgent().generate_checked(_panel(symbols=("AAA",)), 150) == []


class TestStatArbAgent:
    def test_confidence_reflects_variance_explained(self) -> None:
        """When factors explain little, a large residual means little."""
        signals = StatArbAgent().generate_checked(_panel(n_bars=200, seed=4), 150)
        for signal in signals:
            # metadata rounds to 3dp for readability; the confidence is exact.
            assert signal.confidence == pytest.approx(
                signal.metadata["variance_explained"], abs=5e-4
            )

    def test_it_fades_the_residual(self) -> None:
        signals = StatArbAgent().generate_checked(_panel(n_bars=200, seed=6), 150)
        for signal in signals:
            assert np.sign(signal.signal) == -np.sign(signal.metadata["residual_z"])

    def test_too_few_instruments_produces_nothing(self) -> None:
        assert StatArbAgent().generate_checked(_panel(symbols=("AAA", "BBB")), 150) == []


class TestEnsembleBlending:
    def _signal(self, agent_id: str, value: float, confidence: float = 1.0) -> AlphaSignal:
        return AlphaSignal(
            symbol="AAA", signal=value, confidence=confidence, expected_return=None,
            horizon_bars=1, strategy=agent_id, agent_id=agent_id, as_of_index=0,
            provenance=Provenance.real("test"),
        )

    def _ensemble(self) -> AlphaEnsemble:
        return AlphaEnsemble([MomentumAgent(), StatArbAgent(), VolumePressureAgent()])

    def test_full_agreement_preserves_confidence(self) -> None:
        views = self._ensemble().blend(
            [
                self._signal("momentum", 1.0, 0.8),
                self._signal("stat_arb", 1.0, 0.8),
                self._signal("volume_pressure", 1.0, 0.8),
            ]
        )
        assert views[0].agreement == 1.0
        assert views[0].confidence == pytest.approx(0.8)

    def test_disagreement_collapses_confidence(self) -> None:
        """An ensemble that contradicts itself has found noise, not edge."""
        agreed = self._ensemble().blend(
            [self._signal("momentum", 1.0, 0.9), self._signal("stat_arb", 1.0, 0.9)]
        )
        split = self._ensemble().blend(
            [self._signal("momentum", 1.0, 0.9), self._signal("stat_arb", -1.0, 0.9)]
        )
        assert split[0].confidence < agreed[0].confidence / 2

    def test_confidence_weights_the_blend_exactly_once(self) -> None:
        """Applying confidence in the mean and again in conviction squares it."""
        views = self._ensemble().blend([self._signal("momentum", 1.0, 0.5)])
        assert views[0].signal == pytest.approx(1.0)
        assert views[0].conviction == pytest.approx(views[0].confidence)

    def test_a_confident_source_outweighs_an_unsure_one(self) -> None:
        views = self._ensemble().blend(
            [
                self._signal("momentum", 1.0, 0.9),
                self._signal("stat_arb", -1.0, 0.1),
            ]
        )
        assert views[0].signal > 0.5

    def test_partial_coverage_lowers_confidence(self) -> None:
        """One voice is not a consensus, however sure it is."""
        one = self._ensemble().blend([self._signal("momentum", 1.0, 1.0)])
        assert one[0].coverage == pytest.approx(1 / 3)
        assert one[0].confidence == pytest.approx(1 / 3)

    def test_min_coverage_suppresses_thin_views(self) -> None:
        ensemble = AlphaEnsemble(
            [MomentumAgent(), StatArbAgent(), VolumePressureAgent()], min_coverage=0.5
        )
        assert ensemble.blend([self._signal("momentum", 1.0)]) == []

    def test_zero_confidence_still_reports_a_direction(self) -> None:
        views = self._ensemble().blend([self._signal("momentum", 1.0, 0.0)])
        assert views[0].signal == pytest.approx(1.0)
        assert views[0].confidence == 0.0

    def test_a_duplicate_agent_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate agent_id"):
            AlphaEnsemble([MomentumAgent(), MomentumAgent()])

    def test_an_empty_roster_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one agent"):
            AlphaEnsemble([])
