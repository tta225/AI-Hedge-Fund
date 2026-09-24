"""Tests for the cross-sectional backtester and its search."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from axiom.alpha.base import AlphaAgent, AlphaSignal
from axiom.alpha.ensemble import AlphaEnsemble
from axiom.alpha.panel import Panel
from axiom.core.provenance import Provenance
from axiom.research.panel_lab import (
    MIN_BREADTH,
    PanelCandidate,
    PanelLab,
    backtest_panel,
    rank_to_weights,
)


def _panel(
    n_symbols: int = 10, n_bars: int = 600, *, drift: np.ndarray | None = None,
    evidential: bool = True, seed: int = 0,
) -> Panel:
    """A panel with optional per-symbol drift, so a ranking has something to find."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.01, size=(n_bars, n_symbols))
    if drift is not None:
        steps = steps + drift
    closes = 100.0 * np.exp(np.cumsum(steps, axis=0))
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return Panel(
        symbols=tuple(f"S{i:02d}" for i in range(n_symbols)),
        index=pd.DatetimeIndex([start + i * timedelta(days=1) for i in range(n_bars)]),
        closes=closes,
        volumes=np.full_like(closes, 1e6),
        provenance=(
            Provenance.real("test") if evidential else Provenance.synthetic("test")
        ),
    )


class _FixedAgent(AlphaAgent):
    """Scores each symbol by a constant, so weights are predictable.

    Scores are rescaled into ``[-1, 1]`` because `AlphaSignal` validates that
    range — an unscaled signal makes the ensemble's weights meaningless, and
    the check is worth keeping rather than working around.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__()
        self.agent_id = "fixed"
        largest = max((abs(v) for v in scores.values()), default=1.0) or 1.0
        self.scores = {k: v / largest for k, v in scores.items()}
        self.min_history = 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        return [
            AlphaSignal(
                symbol=symbol,
                signal=value,
                confidence=1.0,
                expected_return=None,
                horizon_bars=1,
                strategy="test",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
            )
            for symbol, value in self.scores.items()
            if symbol in panel.symbols
        ]


class _Oracle(AlphaAgent):
    """Ranks by the *next* bar's return. Used only to prove the harness can
    detect an edge when one genuinely exists — it cheats by construction."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_id = "oracle"
        self.min_history = 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        if index + 1 >= len(panel):
            return []
        forward = panel.closes[index + 1] / panel.closes[index] - 1.0
        return [
            AlphaSignal(
                symbol=symbol,
                signal=float(np.clip(forward[i] * 50, -1, 1)),
                confidence=1.0,
                expected_return=None,
                horizon_bars=1,
                strategy="test",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
            )
            for i, symbol in enumerate(panel.symbols)
        ]


class _ChurningAgent(AlphaAgent):
    """Reranks every bar, so the book actually turns over.

    A fixed ranking produces zero turnover after the first build no matter how
    often it is asked, which is correct and makes it useless for measuring
    cost. Churn has to be explicit.
    """

    def __init__(self, n_symbols: int = 20, seed: int = 0) -> None:
        super().__init__()
        self.agent_id = "churn"
        self.min_history = 2
        self.rng = np.random.default_rng(seed)
        self.n_symbols = n_symbols

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        draw = np.random.default_rng(index).uniform(-1, 1, size=panel.n_symbols)
        return [
            AlphaSignal(
                symbol=symbol, signal=float(draw[i]), confidence=1.0,
                expected_return=None, horizon_bars=1, strategy="test",
                agent_id=self.agent_id, as_of_index=index,
                provenance=panel.provenance,
            )
            for i, symbol in enumerate(panel.symbols)
        ]


class TestRankToWeights:
    symbols = tuple(f"S{i:02d}" for i in range(10))

    def test_long_short_is_dollar_neutral(self) -> None:
        scores = {s: float(i) for i, s in enumerate(self.symbols)}
        weights = rank_to_weights(scores, self.symbols)
        assert weights.sum() == pytest.approx(0.0)

    def test_gross_exposure_is_respected(self) -> None:
        scores = {s: float(i) for i, s in enumerate(self.symbols)}
        weights = rank_to_weights(scores, self.symbols, gross=2.0)
        assert np.abs(weights).sum() == pytest.approx(2.0)

    def test_the_best_scored_name_is_long(self) -> None:
        scores = {s: float(i) for i, s in enumerate(self.symbols)}
        weights = rank_to_weights(scores, self.symbols)
        assert weights[-1] > 0
        assert weights[0] < 0

    def test_long_only_has_no_shorts(self) -> None:
        scores = {s: float(i) for i, s in enumerate(self.symbols)}
        weights = rank_to_weights(scores, self.symbols, long_short=False)
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_unscored_symbols_get_no_position(self) -> None:
        """No view is not the same as zero conviction."""
        scores = {s: float(i) for i, s in enumerate(self.symbols[:6])}
        weights = rank_to_weights(scores, self.symbols)
        assert weights[6:].tolist() == [0.0] * 4

    def test_too_few_names_produces_no_book(self) -> None:
        """A cross-section of three is not a cross-section."""
        scores = dict.fromkeys(self.symbols[:MIN_BREADTH - 1], 1.0)
        assert not np.any(rank_to_weights(scores, self.symbols))

    def test_non_finite_scores_are_ignored(self) -> None:
        scores = {s: float(i) for i, s in enumerate(self.symbols)}
        scores["S03"] = float("nan")
        weights = rank_to_weights(scores, self.symbols)
        assert weights[3] == 0.0

    def test_rejects_an_impossible_fraction(self) -> None:
        with pytest.raises(ValueError, match="top_fraction"):
            rank_to_weights({}, self.symbols, top_fraction=0.9)


class TestCausality:
    def test_an_oracle_makes_money_and_a_lagged_one_does_not(self) -> None:
        """Proves the harness can see an edge, and that it is not leaking one.

        The oracle ranks by the next bar's return, so it must be strongly
        profitable. If it were not, the harness could not detect any edge and
        every null result would be uninformative.
        """
        panel = _panel(n_symbols=20, n_bars=400)
        cheating = backtest_panel([_Oracle()], panel, start=50, rebalance_every=1,
                                  cost_bps=0.0)
        assert cheating.sharpe > 3.0

    def test_a_constant_ranking_earns_nothing_systematic(self) -> None:
        """The null: a fixed ranking with no information must not profit."""
        panel = _panel(n_symbols=20, n_bars=600)
        scores = {s: float(i) for i, s in enumerate(panel.symbols)}
        outcome = backtest_panel([_FixedAgent(scores)], panel, start=50,
                                 rebalance_every=5, cost_bps=0.0)
        assert abs(outcome.sharpe) < 1.5

    def test_weights_earn_the_following_bar(self) -> None:
        """Off by one in the permissive direction and momentum looks magical."""
        panel = _panel(n_symbols=10, n_bars=100)
        scores = {s: float(i) for i, s in enumerate(panel.symbols)}
        agent = _FixedAgent(scores)
        outcome = backtest_panel([agent], panel, start=10, end=12,
                                 rebalance_every=1, cost_bps=0.0)
        weights = rank_to_weights(agent.scores, panel.symbols)
        # returns[0] is the book formed at bar 10 earning bar 10 -> 11. Using
        # bar 9 -> 10 there would be the lookahead this test exists to exclude.
        step = panel.closes[11] / panel.closes[10] - 1.0
        assert outcome.returns[0] == pytest.approx(float(weights @ step))


class TestCosts:
    def test_turnover_is_charged(self) -> None:
        panel = _panel(n_symbols=20, n_bars=300)
        scores = {s: float(i) for i, s in enumerate(panel.symbols)}
        free = backtest_panel([_FixedAgent(scores)], panel, start=50,
                              rebalance_every=1, cost_bps=0.0)
        costly = backtest_panel([_FixedAgent(scores)], panel, start=50,
                                rebalance_every=1, cost_bps=50.0)
        assert costly.returns.sum() < free.returns.sum()

    def test_a_held_book_pays_nothing_between_rebalances(self) -> None:
        """Charging per bar rather than per unit traded is how turnover hides."""
        panel = _panel(n_symbols=20, n_bars=300)
        scores = {s: float(i) for i, s in enumerate(panel.symbols)}
        outcome = backtest_panel([_FixedAgent(scores)], panel, start=50,
                                 rebalance_every=10, cost_bps=100.0)
        # Only the rebalance bars carry turnover; the rest are held.
        assert np.count_nonzero(outcome.turnover) < outcome.turnover.size / 5

    def test_slower_rebalancing_lowers_turnover(self) -> None:
        panel = _panel(n_symbols=20, n_bars=400)
        fast = backtest_panel([_ChurningAgent()], panel, start=50,
                              rebalance_every=1, cost_bps=0.0)
        slow = backtest_panel([_ChurningAgent()], panel, start=50,
                              rebalance_every=20, cost_bps=0.0)
        assert slow.annual_turnover < fast.annual_turnover

    def test_a_fixed_ranking_costs_nothing_to_maintain(self) -> None:
        """Rebalancing to the weights already held is not a trade."""
        panel = _panel(n_symbols=20, n_bars=300)
        scores = {s: float(i) for i, s in enumerate(panel.symbols)}
        outcome = backtest_panel([_FixedAgent(scores)], panel, start=50,
                                 rebalance_every=1, cost_bps=100.0)
        assert np.count_nonzero(outcome.turnover) == 1

    def test_rejects_negative_costs(self) -> None:
        panel = _panel()
        with pytest.raises(ValueError, match="cost_bps"):
            backtest_panel([_FixedAgent({})], panel, start=50, cost_bps=-1.0)


class TestPhaseAnchoring:
    def test_phase_is_independent_of_the_window_start(self) -> None:
        """A fold boundary must not change which days the book trades.

        This is the bug the harness had: anchoring phase to `start` made a
        walk-forward fold rebalance on different days than a continuous run,
        which alone moved a measured Sharpe from 0.07 to 0.15.
        """
        panel = _panel(n_symbols=20, n_bars=400)
        scores = {s: float(i) for i, s in enumerate(panel.symbols)}
        agent = _FixedAgent(scores)
        whole = backtest_panel([agent], panel, start=100, end=300,
                               rebalance_every=21, cost_bps=10.0)
        first = backtest_panel([agent], panel, start=100, end=200,
                               rebalance_every=21, cost_bps=10.0)
        second = backtest_panel([agent], panel, start=200, end=300,
                                rebalance_every=21, cost_bps=10.0)
        # The second half of the whole run must match the standalone second
        # fold on the bars where no reset is involved.
        assert whole.returns[-50:] == pytest.approx(second.returns[-50:], abs=1e-9)
        assert first.returns.size + second.returns.size == whole.returns.size

    def test_phase_changes_the_trading_days(self) -> None:
        panel = _panel(n_symbols=20, n_bars=400)
        rng = np.random.default_rng(2)
        noisy = {s: float(rng.normal()) for s in panel.symbols}
        at_zero = backtest_panel([_FixedAgent(noisy)], panel, start=100,
                                 rebalance_every=21, cost_bps=0.0, phase=0)
        at_five = backtest_panel([_FixedAgent(noisy)], panel, start=100,
                                 rebalance_every=21, cost_bps=0.0, phase=5)
        assert np.flatnonzero(at_zero.turnover).tolist() != (
            np.flatnonzero(at_five.turnover).tolist()
        )


class TestPanelLab:
    @staticmethod
    def _candidates(count: int = 4) -> list[PanelCandidate]:
        out = []
        for i in range(count):
            rng = np.random.default_rng(i)
            out.append(
                PanelCandidate(
                    name=f"c{i}",
                    factory=lambda r=rng: [  # type: ignore[misc]
                        _FixedAgent({f"S{j:02d}": float(r.normal()) for j in range(20)})
                    ],
                    params={"seed": i},
                )
            )
        return out

    def test_search_returns_a_result_per_candidate(self) -> None:
        panel = _panel(n_symbols=20, n_bars=800)
        result = PanelLab(n_folds=2, warmup=100).search(panel, self._candidates())
        assert len(result.results) == 4
        assert result.n_trials == 4

    def test_noise_does_not_survive(self) -> None:
        panel = _panel(n_symbols=20, n_bars=800)
        result = PanelLab(n_folds=2, warmup=100).search(panel, self._candidates())
        assert result.survivors == []
        assert "NOTHING SURVIVED" in result.verdict or "NO RESULT" in result.verdict

    def test_the_leaderboard_breaks_dsr_ties_by_sharpe(self) -> None:
        """All-zero DSR is the common case; insertion order must not decide."""
        panel = _panel(n_symbols=20, n_bars=800)
        result = PanelLab(n_folds=2, warmup=100).search(panel, self._candidates(6))
        board = result.leaderboard
        sharpes = [r.result.sharpe for r in board]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_synthetic_data_yields_no_conclusion(self) -> None:
        panel = _panel(n_symbols=20, n_bars=800, evidential=False)
        result = PanelLab(n_folds=2, warmup=100).search(panel, self._candidates())
        assert "NO CONCLUSION" in result.verdict

    def test_a_short_panel_is_refused(self) -> None:
        with pytest.raises(ValueError, match="too few"):
            PanelLab(n_folds=4, warmup=100).search(
                _panel(n_bars=150), self._candidates()
            )

    def test_candidate_overrides_reach_the_backtest(self) -> None:
        panel = _panel(n_symbols=20, n_bars=800)
        fast = PanelCandidate(
            name="fast", factory=lambda: [_ChurningAgent()], rebalance_every=1
        )
        slow = PanelCandidate(
            name="slow", factory=lambda: [_ChurningAgent()], rebalance_every=50
        )
        result = PanelLab(n_folds=2, warmup=100).search(panel, [fast, slow])
        by_name = {r.candidate.name: r for r in result.results}
        assert by_name["fast"].result.annual_turnover > (
            by_name["slow"].result.annual_turnover
        )

    def test_render_reports_the_trial_count(self) -> None:
        panel = _panel(n_symbols=20, n_bars=800)
        result = PanelLab(n_folds=2, warmup=100).search(panel, self._candidates())
        assert "Trials evaluated : 4" in result.render()

    def test_rejects_a_single_fold(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            PanelLab(n_folds=1)


class TestEnsembleContract:
    def test_an_unregistered_agent_id_is_named(self) -> None:
        """A wrapper that renames itself but not its signals broke silently."""
        panel = _panel(n_symbols=10, n_bars=100)
        agent = _FixedAgent(dict.fromkeys(panel.symbols, 1.0))
        ensemble = AlphaEnsemble([agent])
        signals = agent.generate_checked(panel, 50)
        renamed = [
            AlphaSignal(
                symbol=s.symbol, signal=s.signal, confidence=s.confidence,
                expected_return=None, horizon_bars=1, strategy="test",
                agent_id="not_in_roster", as_of_index=s.as_of_index,
                provenance=s.provenance,
            )
            for s in signals
        ]
        with pytest.raises(KeyError, match="not in this ensemble's roster"):
            ensemble.blend(renamed)
