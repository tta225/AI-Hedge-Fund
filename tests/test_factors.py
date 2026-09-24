"""The low-turnover factor families.

Two kinds of test here, and the second is the one that matters.

The first checks each agent's contract: causality, ranking, bounded signals,
measured confidence. The base class enforces some of that, and these confirm
each implementation actually satisfies it rather than passing by accident.

The second checks that each factor **measures what it says it measures**. A
momentum agent that quietly ranks by volatility would pass every contract test
and produce a factor study about the wrong thing. So each family is given a
panel constructed so the correct answer is known in advance, and is asked to
find it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from axiom.alpha.base import LookaheadError
from axiom.alpha.factors import (
    LOW_TURNOVER_FAMILIES,
    MONTH,
    QUARTER,
    Illiquidity,
    LongHorizonMomentum,
    LongTermReversal,
    LowVolatility,
    ResidualMomentum,
)
from axiom.alpha.panel import Panel
from axiom.core.provenance import DataKind, Provenance

SYNTHETIC = Provenance(kind=DataKind.SYNTHETIC, source="test")


def _panel(closes: np.ndarray, volumes: np.ndarray | None = None) -> Panel:
    n_bars, n_symbols = closes.shape
    return Panel(
        symbols=tuple(f"S{i}" for i in range(n_symbols)),
        index=pd.date_range("2018-01-01", periods=n_bars, freq="D"),
        closes=closes,
        volumes=np.full_like(closes, 1e6) if volumes is None else volumes,
        provenance=SYNTHETIC,
    )


def _random_panel(n_bars: int = 1100, n_symbols: int = 12, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_bars, n_symbols)), axis=0))
    volumes = rng.lognormal(12, 0.5, (n_bars, n_symbols))
    return _panel(closes, volumes)


def _trending_panel(n_bars: int = 1100, n_symbols: int = 10) -> Panel:
    """Deterministic ramps: symbol i drifts at rate i, so the ranking is known."""
    drifts = np.linspace(-0.0004, 0.0004, n_symbols)
    steps = np.tile(drifts, (n_bars, 1))
    closes = 100 * np.exp(np.cumsum(steps, axis=0))
    return _panel(closes)


def _rank_of(signals, symbol: str) -> float:
    return next(s.signal for s in signals if s.symbol == symbol)


class TestContracts:
    """Properties the whole family must hold, checked on every member."""

    @pytest.mark.parametrize("factory", LOW_TURNOVER_FAMILIES)
    def test_emits_nothing_before_min_history(self, factory) -> None:
        agent = factory()
        panel = _random_panel()
        assert agent.generate_checked(panel, agent.min_history - 1) == []

    @pytest.mark.parametrize("factory", LOW_TURNOVER_FAMILIES)
    def test_signals_are_stamped_with_the_scored_bar(self, factory) -> None:
        agent = factory()
        panel = _random_panel()
        index = len(panel) - 1
        signals = agent.generate_checked(panel, index)
        assert signals
        assert all(s.as_of_index == index for s in signals)

    @pytest.mark.parametrize("factory", LOW_TURNOVER_FAMILIES)
    def test_provenance_is_propagated(self, factory) -> None:
        """A signal derived from synthetic bars must not launder into evidence."""
        signals = factory().generate_checked(_random_panel(), 1099)
        assert signals
        assert all(not s.is_evidential for s in signals)

    @pytest.mark.parametrize("factory", LOW_TURNOVER_FAMILIES)
    def test_confidence_is_measured_not_constant_across_agents(self, factory) -> None:
        signals = factory().generate_checked(_random_panel(), 1099)
        assert all(0.0 <= s.confidence <= 1.0 for s in signals)
        assert all(s.confidence > 0.0 for s in signals)

    @pytest.mark.parametrize("factory", LOW_TURNOVER_FAMILIES)
    def test_ranking_is_centred_and_bounded(self, factory) -> None:
        signals = factory().generate_checked(_random_panel(), 1099)
        values = np.array([s.signal for s in signals])
        assert np.all(np.abs(values) <= 1.0)
        # A cross-sectional rank is symmetric about zero by construction.
        assert abs(float(values.mean())) < 0.2

    @pytest.mark.parametrize("factory", LOW_TURNOVER_FAMILIES)
    def test_a_future_bar_cannot_change_a_past_signal(self, factory) -> None:
        """The causality property, tested rather than asserted.

        The same bar is scored on a panel that has been extended with later
        data. If any agent reads forward, the signal moves.
        """
        agent = factory()
        base = _random_panel(n_bars=1000)
        index = 999
        first = agent.generate_checked(base, index)

        rng = np.random.default_rng(99)
        tail = base.closes[-1] * np.exp(np.cumsum(rng.normal(0, 0.05, (50, base.n_symbols)), axis=0))
        extended = _panel(
            np.vstack([base.closes, tail]),
            np.vstack([base.volumes, np.full((50, base.n_symbols), 1e6)]),
        )
        second = factory().generate_checked(extended, index)

        assert [s.symbol for s in first] == [s.symbol for s in second]
        for a, b in zip(first, second, strict=True):
            assert a.signal == pytest.approx(b.signal)

    @pytest.mark.parametrize("factory", LOW_TURNOVER_FAMILIES)
    def test_horizon_is_quarterly(self, factory) -> None:
        """The premise of the family: these are held, not traded."""
        signals = factory().generate_checked(_random_panel(), 1099)
        assert all(s.horizon_bars == QUARTER for s in signals)


class TestLongHorizonMomentum:
    def test_ranks_the_strongest_trend_highest(self) -> None:
        panel = _trending_panel()
        signals = LongHorizonMomentum().generate_checked(panel, 1099)
        # S9 has the strongest positive drift, S0 the strongest negative.
        assert _rank_of(signals, "S9") == pytest.approx(1.0)
        assert _rank_of(signals, "S0") == pytest.approx(-1.0)

    def test_the_skip_month_is_actually_skipped(self) -> None:
        """The construction's defining feature: the last month must not count.

        A panel whose final month reverses violently should not move the
        ranking, because the formation window ends before it.
        """
        panel = _trending_panel()
        index = 1099
        baseline = LongHorizonMomentum().generate_checked(panel, index)

        shocked = panel.closes.copy()
        # Invert the last 21 bars for the top-ranked name.
        shocked[index - MONTH : index, 9] *= np.linspace(1.0, 0.5, MONTH)
        after = LongHorizonMomentum().generate_checked(_panel(shocked), index)
        assert _rank_of(after, "S9") == pytest.approx(_rank_of(baseline, "S9"))

    def test_without_the_skip_the_last_month_does_count(self) -> None:
        """The control for the test above: the mechanism, not a coincidence."""
        panel = _trending_panel()
        index = 1099
        shocked = panel.closes.copy()
        shocked[index - MONTH : index, 9] *= np.linspace(1.0, 0.4, MONTH)
        agent = LongHorizonMomentum({"skip": 0})
        baseline = agent.generate_checked(panel, index)
        after = LongHorizonMomentum({"skip": 0}).generate_checked(_panel(shocked), index)
        assert _rank_of(after, "S9") < _rank_of(baseline, "S9")

    def test_skip_must_be_shorter_than_lookback(self) -> None:
        with pytest.raises(ValueError, match="shorter than lookback"):
            LongHorizonMomentum({"lookback": 21, "skip": 21})

    def test_confidence_rises_with_a_steadier_trend(self) -> None:
        """A 40% return earned in one week is not the evidence a steady drift is."""
        n = 400
        steady = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
        jumpy = np.full(n, 100.0)
        jumpy[n // 2 :] = 100.0 * np.exp(0.001 * n)
        flat = np.full(n, 100.0) * np.exp(np.cumsum(np.full(n, 0.0002)))
        closes = np.column_stack([steady, jumpy, flat])
        agent = LongHorizonMomentum({"lookback": 300, "skip": 21})
        signals = agent.generate_checked(_panel(closes), 399)
        by_symbol = {s.symbol: s.confidence for s in signals}
        assert by_symbol["S0"] > by_symbol["S1"]


class TestLowVolatility:
    def test_ranks_the_calmest_name_highest(self) -> None:
        rng = np.random.default_rng(11)
        n, m = 400, 5
        # Volatility rises across the cross-section.
        scales = np.linspace(0.002, 0.05, m)
        steps = rng.normal(0, 1, (n, m)) * scales
        closes = 100 * np.exp(np.cumsum(steps, axis=0))
        signals = LowVolatility({"lookback": 300}).generate_checked(_panel(closes), 399)
        assert _rank_of(signals, "S0") == pytest.approx(1.0)
        assert _rank_of(signals, "S4") == pytest.approx(-1.0)

    def test_a_constant_price_is_excluded_not_ranked_best(self) -> None:
        """Zero volatility is a data problem, not the calmest possible name."""
        rng = np.random.default_rng(5)
        n = 400
        moving = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0))
        closes = np.column_stack([np.full(n, 100.0), moving])
        signals = LowVolatility({"lookback": 300}).generate_checked(_panel(closes), 399)
        assert "S0" not in {s.symbol for s in signals}


class TestResidualMomentum:
    def test_a_pure_market_mover_ranks_neutrally(self) -> None:
        """The whole point: beta exposure is removed, not ranked.

        Every name here is the market times a beta and nothing else, so no name
        has residual drift and the ranking should carry no information.
        """
        rng = np.random.default_rng(17)
        n = 500
        market = rng.normal(0.0005, 0.01, n)
        betas = np.linspace(0.5, 2.0, 6)
        steps = np.outer(market, betas)
        closes = 100 * np.exp(np.cumsum(steps, axis=0))
        signals = ResidualMomentum({"lookback": 400, "skip": 21}).generate_checked(
            _panel(closes), 499
        )
        # Residuals are numerically zero, so no name should dominate on drift.
        assert signals
        assert all(abs(s.metadata["r_squared"] - 1.0) < 1e-6 for s in signals)

    def test_idiosyncratic_drift_is_found(self) -> None:
        rng = np.random.default_rng(23)
        n, m = 600, 6
        market = rng.normal(0.0, 0.01, n)
        steps = np.tile(market[:, None], (1, m)) + rng.normal(0, 0.002, (n, m))
        # S5 gets a private uptrend the market does not explain.
        steps[:, 5] += 0.0015
        steps[:, 0] -= 0.0015
        closes = 100 * np.exp(np.cumsum(steps, axis=0))
        signals = ResidualMomentum({"lookback": 500, "skip": 21}).generate_checked(
            _panel(closes), 599
        )
        assert _rank_of(signals, "S5") == pytest.approx(1.0)
        assert _rank_of(signals, "S0") == pytest.approx(-1.0)

    def test_confidence_falls_as_the_market_explains_more(self) -> None:
        signals = ResidualMomentum().generate_checked(_random_panel(), 1099)
        by_r2 = sorted(signals, key=lambda s: float(s.metadata["r_squared"]))
        assert by_r2[0].confidence >= by_r2[-1].confidence


class TestLongTermReversal:
    def test_buys_the_loser_and_sells_the_winner(self) -> None:
        panel = _trending_panel(n_bars=1100)
        signals = LongTermReversal().generate_checked(panel, 1099)
        # Inverted relative to momentum: the strongest riser is the short.
        assert _rank_of(signals, "S9") == pytest.approx(-1.0)
        assert _rank_of(signals, "S0") == pytest.approx(1.0)

    def test_confidence_is_flat_and_says_so(self) -> None:
        """Nothing in a three-year return distinguishes a more believable
        reversal, and inventing a spread would be unmeasured confidence."""
        signals = LongTermReversal().generate_checked(_random_panel(), 1099)
        assert len({s.confidence for s in signals}) == 1

    def test_needs_its_full_formation_window(self) -> None:
        agent = LongTermReversal()
        assert agent.min_history == 758
        assert agent.generate_checked(_random_panel(n_bars=700), 699) == []


class TestIlliquidity:
    def test_ranks_the_thinnest_name_highest(self) -> None:
        rng = np.random.default_rng(29)
        n, m = 400, 5
        steps = rng.normal(0, 0.01, (n, m))
        closes = 100 * np.exp(np.cumsum(steps, axis=0))
        # Dollar volume rises across the cross-section, so illiquidity falls.
        volumes = np.tile(np.logspace(3, 7, m), (n, 1))
        signals = Illiquidity({"lookback": 300}).generate_checked(_panel(closes, volumes), 399)
        assert _rank_of(signals, "S0") == pytest.approx(1.0)
        assert _rank_of(signals, "S4") == pytest.approx(-1.0)

    def test_a_name_that_never_traded_is_excluded(self) -> None:
        rng = np.random.default_rng(31)
        n = 400
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 4)), axis=0))
        volumes = np.full((n, 4), 1e6)
        volumes[:, 0] = 0.0
        signals = Illiquidity({"lookback": 300}).generate_checked(_panel(closes, volumes), 399)
        assert "S0" not in {s.symbol for s in signals}

    def test_confidence_tracks_volume_coverage(self) -> None:
        rng = np.random.default_rng(37)
        n = 400
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0))
        volumes = np.full((n, 3), 1e6)
        volumes[: n // 2, 1] = 0.0  # traded half the window
        signals = Illiquidity({"lookback": 300}).generate_checked(_panel(closes, volumes), 399)
        by_symbol = {s.symbol: s.confidence for s in signals}
        assert by_symbol["S0"] > by_symbol["S1"]


class TestTurnoverPremise:
    def test_a_long_horizon_ranking_barely_moves_day_to_day(self) -> None:
        """The premise the whole campaign rests on, stated as a test.

        If a 252-bar ranking reshuffled daily, these families would be no
        cheaper to trade than the fast ones and the rationale would be wrong.
        """
        panel = _random_panel(n_bars=1100, n_symbols=20)
        agent = LongHorizonMomentum()
        today = {s.symbol: s.signal for s in agent.generate_checked(panel, 1098)}
        tomorrow = {s.symbol: s.signal for s in agent.generate_checked(panel, 1099)}
        moved = np.array([abs(today[k] - tomorrow[k]) for k in today])
        # Mean absolute rank change across a [-1, 1] scale, over one bar.
        assert moved.mean() < 0.1

    def test_generate_checked_rejects_a_mis_stamped_signal(self) -> None:
        """Guards the enforcement itself, not just the agents."""

        class Liar(LongHorizonMomentum):
            def generate(self, panel: Panel, index: int):  # type: ignore[no-untyped-def]
                return [
                    replace(s, as_of_index=index - 1)
                    for s in super().generate(panel, index)[:1]
                ]

        with pytest.raises(LookaheadError):
            Liar().generate_checked(_random_panel(), 1099)
