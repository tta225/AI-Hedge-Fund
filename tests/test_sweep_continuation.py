"""Tests for the continuation reading of a liquidity sweep.

These tests hand-build the ``LiquiditySweep`` objects rather than letting the
detector find them. The strategy's contract is "given a sweep with these
properties, do this" — pinning it to whatever the detector happens to produce
on a fixture would test the detector instead, and would break every time the
detection thresholds moved.
"""

from __future__ import annotations

import pandas as pd
import pytest
from tests.conftest import make_series

from axiom.core.types import Direction
from axiom.ict.models import (
    ICTState,
    LiquidityKind,
    LiquidityPool,
    LiquiditySweep,
    StructureEvent,
)
from axiom.portfolio.positions import Portfolio
from axiom.strategy.base import StrategyContext
from axiom.strategy.sweep_continuation import (
    SweepContinuationStrategy,
    SweepReversalControl,
)


def _flat_tape(n: int = 60, price: float = 100.0) -> list[tuple[float, float, float, float]]:
    """A tape with a constant 1.0-point range, so ATR is exactly 1.0."""
    return [(price, price + 0.5, price - 0.5, price) for _ in range(n)]


def _context(
    *,
    sweeps: list[LiquiditySweep],
    pools: list[LiquidityPool] | None = None,
    index: int = 50,
    bias: Direction = Direction.NEUTRAL,
    rows: list[tuple[float, float, float, float]] | None = None,
) -> StrategyContext:
    series = make_series(rows or _flat_tape())
    # `StrategyContext.bias` is derived from the last confirmed structure event,
    # not read off `ICTState.bias` — so a bias fixture has to supply the event
    # that produces it, or the gate under test never sees anything.
    events = (
        []
        if bias is Direction.NEUTRAL
        else [
            StructureEvent(
                origin_index=index - 5,
                confirmed_index=index - 5,
                timestamp=series.index[index - 5],
                direction=bias,
            )
        ]
    )
    state = ICTState(
        symbol=series.instrument.symbol,
        timeframe="15m",
        as_of=series.index[index],
        index=index,
        structure_events=events,
        liquidity_pools=pools or [],
        sweeps=sweeps,
        bias=bias,
    )
    return StrategyContext(
        series=series,
        index=index,
        ict=state,
        portfolio=Portfolio(starting_cash=100_000),
        timestamp=series.index[index],
    )


def _sweep(
    *,
    kind: LiquidityKind = LiquidityKind.BUYSIDE,
    pool_price: float = 99.5,
    penetration_atr: float = 0.5,
    closed_back_inside: bool = False,
    confirmed_index: int = 49,
) -> LiquiditySweep:
    return LiquiditySweep(
        origin_index=confirmed_index,
        confirmed_index=confirmed_index,
        timestamp=pd.Timestamp("2025-03-03T13:00:00Z"),
        pool_price=pool_price,
        kind=kind,
        penetration_atr=penetration_atr,
        closed_back_inside=closed_back_inside,
    )


def _pool(
    price: float, kind: LiquidityKind, *, confirmed_index: int = 10
) -> LiquidityPool:
    return LiquidityPool(
        origin_index=confirmed_index,
        confirmed_index=confirmed_index,
        timestamp=pd.Timestamp("2025-03-03T13:00:00Z"),
        price=price,
        kind=kind,
    )


class TestDirection:
    """The whole point of the module: it trades *with* the raid."""

    def test_buyside_sweep_goes_long(self) -> None:
        """A raid above highs is read as a breakout, not exhaustion."""
        context = _context(sweeps=[_sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)])
        signal = SweepContinuationStrategy().evaluate(context)
        assert signal is not None
        assert signal.direction is Direction.BULLISH

    def test_sellside_sweep_goes_short(self) -> None:
        context = _context(sweeps=[_sweep(kind=LiquidityKind.SELLSIDE, pool_price=100.5)])
        signal = SweepContinuationStrategy().evaluate(context)
        assert signal is not None
        assert signal.direction is Direction.BEARISH

    def test_direction_is_the_opposite_of_the_ict_reading(self) -> None:
        """Guards the one-word bug that would silently make this the ICT strategy."""
        for kind in LiquidityKind:
            pool_price = 99.5 if kind is LiquidityKind.BUYSIDE else 100.5
            context = _context(sweeps=[_sweep(kind=kind, pool_price=pool_price)])
            signal = SweepContinuationStrategy().evaluate(context)
            assert signal is not None
            sweep = context.ict.sweeps[0]
            assert signal.direction is not sweep.reversal_direction
            assert signal.direction is sweep.kind.direction


class TestStopPlacement:
    def test_long_stop_sits_below_the_swept_pool(self) -> None:
        """The thesis is 'the level broke and held', so the level failing kills it."""
        context = _context(sweeps=[_sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)])
        signal = SweepContinuationStrategy(stop_buffer_atr=0.25).evaluate(context)
        assert signal is not None
        # ATR is 1.0 on the flat tape by construction.
        assert signal.stop == pytest.approx(99.5 - 0.25)

    def test_short_stop_sits_above_the_swept_pool(self) -> None:
        context = _context(sweeps=[_sweep(kind=LiquidityKind.SELLSIDE, pool_price=100.5)])
        signal = SweepContinuationStrategy(stop_buffer_atr=0.25).evaluate(context)
        assert signal is not None
        assert signal.stop == pytest.approx(100.5 + 0.25)

    def test_no_signal_when_price_has_already_fallen_back_through_the_stop(self) -> None:
        """A long whose stop is above the current price is not a trade."""
        context = _context(sweeps=[_sweep(kind=LiquidityKind.BUYSIDE, pool_price=105.0)])
        assert SweepContinuationStrategy().evaluate(context) is None

    def test_no_signal_when_short_stop_is_below_entry(self) -> None:
        context = _context(sweeps=[_sweep(kind=LiquidityKind.SELLSIDE, pool_price=95.0)])
        assert SweepContinuationStrategy().evaluate(context) is None


class TestTargeting:
    def test_prefers_the_next_unswept_pool_ahead(self) -> None:
        context = _context(
            sweeps=[_sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)],
            pools=[_pool(103.0, LiquidityKind.BUYSIDE)],
        )
        signal = SweepContinuationStrategy().evaluate(context)
        assert signal is not None
        assert signal.primary_target == pytest.approx(103.0)

    def test_falls_back_to_an_atr_extension_when_no_pool_lies_ahead(self) -> None:
        """Continuation makes no claim about a draw, so a missing pool is not fatal."""
        context = _context(sweeps=[_sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)])
        signal = SweepContinuationStrategy(target_atr=2.0).evaluate(context)
        assert signal is not None
        assert signal.primary_target == pytest.approx(102.0)

    def test_reward_risk_gate_rejects_a_target_that_does_not_pay(self) -> None:
        context = _context(
            sweeps=[_sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)],
            pools=[_pool(100.5, LiquidityKind.BUYSIDE)],
        )
        # risk = 100 - 99.25 = 0.75, reward = 0.5 → RR 0.67.
        assert SweepContinuationStrategy(min_rr=1.2).evaluate(context) is None
        # The same setup passes once the bar is lowered, proving RR is the gate
        # and not some unrelated filter.
        assert SweepContinuationStrategy(min_rr=0.2).evaluate(context) is not None


class TestGates:
    def test_require_hold_excludes_sweeps_that_closed_back_inside(self) -> None:
        """Closing back inside is ICT's reversal signature, not continuation."""
        sweep = _sweep(closed_back_inside=True)
        assert SweepContinuationStrategy(require_hold=True).evaluate(
            _context(sweeps=[sweep])
        ) is None
        assert SweepContinuationStrategy(require_hold=False).evaluate(
            _context(sweeps=[sweep])
        ) is not None

    def test_marginal_pokes_are_ignored(self) -> None:
        context = _context(sweeps=[_sweep(penetration_atr=0.01)])
        assert SweepContinuationStrategy(min_penetration_atr=0.10).evaluate(context) is None

    def test_violent_excursions_are_ignored(self) -> None:
        context = _context(sweeps=[_sweep(penetration_atr=5.0)])
        assert SweepContinuationStrategy(max_penetration_atr=2.0).evaluate(context) is None

    def test_entry_window_expires(self) -> None:
        sweep = _sweep(confirmed_index=40)  # 10 bars before index 50
        assert SweepContinuationStrategy(entry_window=3).evaluate(
            _context(sweeps=[sweep])
        ) is None
        assert SweepContinuationStrategy(entry_window=20).evaluate(
            _context(sweeps=[sweep])
        ) is not None

    def test_unconfirmed_sweeps_are_invisible(self) -> None:
        """Lookahead guard: a sweep confirmed after this bar cannot be traded."""
        context = _context(sweeps=[_sweep(confirmed_index=55)], index=50)
        assert SweepContinuationStrategy(entry_window=100).evaluate(context) is None

    def test_bias_alignment_gate(self) -> None:
        sweep = _sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)
        strategy = SweepContinuationStrategy(require_bias_alignment=True)
        assert strategy.evaluate(
            _context(sweeps=[sweep], bias=Direction.BEARISH)
        ) is None
        assert strategy.evaluate(
            _context(sweeps=[sweep], bias=Direction.BULLISH)
        ) is not None

    def test_no_signal_while_a_position_is_open(self) -> None:
        context = _context(sweeps=[_sweep()])
        context.portfolio.position(context.instrument).quantity = 1.0
        assert SweepContinuationStrategy().evaluate(context) is None

    def test_most_recent_qualifying_sweep_wins(self) -> None:
        old = _sweep(kind=LiquidityKind.SELLSIDE, pool_price=100.5, confirmed_index=48)
        new = _sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5, confirmed_index=50)
        signal = SweepContinuationStrategy(entry_window=10).evaluate(
            _context(sweeps=[old, new])
        )
        assert signal is not None
        assert signal.direction is Direction.BULLISH


class TestParameterExposure:
    def test_every_gate_is_recorded_for_the_lab(self) -> None:
        """``StrategyLab`` reports ``params``; a gate missing from it is invisible."""
        strategy = SweepContinuationStrategy()
        expected = {
            "entry_window",
            "require_hold",
            "min_penetration_atr",
            "max_penetration_atr",
            "require_bias_alignment",
            "min_rr",
            "stop_buffer_atr",
            "target_atr",
        }
        assert expected <= set(strategy.params)

    def test_describe_renders_the_configuration(self) -> None:
        assert "entry_window=5" in SweepContinuationStrategy(entry_window=5).describe()


class TestReversalControl:
    """The control must differ from the treatment in direction alone."""

    def test_direction_is_flipped(self) -> None:
        context_args = {"sweeps": [_sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)]}
        treatment = SweepContinuationStrategy().evaluate(_context(**context_args))
        control = SweepReversalControl().evaluate(_context(**context_args))
        assert treatment is not None and control is not None
        assert control.direction is treatment.direction.opposite

    def test_risk_and_reward_distances_are_preserved(self) -> None:
        """Same bet size, same payoff, opposite side — otherwise it is not a control."""
        context_args = {
            "sweeps": [_sweep(kind=LiquidityKind.BUYSIDE, pool_price=99.5)],
            "pools": [_pool(103.0, LiquidityKind.BUYSIDE)],
        }
        treatment = SweepContinuationStrategy().evaluate(_context(**context_args))
        control = SweepReversalControl().evaluate(_context(**context_args))
        assert treatment is not None and control is not None
        assert control.entry == pytest.approx(treatment.entry)
        assert control.risk_points == pytest.approx(treatment.risk_points)
        assert abs(control.primary_target - control.entry) == pytest.approx(
            abs(treatment.primary_target - treatment.entry)
        )

    def test_control_is_silent_wherever_the_treatment_is(self) -> None:
        """Identical population, so a difference in results is about direction."""
        context = _context(sweeps=[_sweep(penetration_atr=5.0)])
        assert SweepContinuationStrategy().evaluate(context) is None
        assert SweepReversalControl().evaluate(context) is None

    def test_control_accepts_the_same_parameters(self) -> None:
        control = SweepReversalControl(entry_window=7, require_hold=False)
        assert control.params["entry_window"] == 7
        assert control._inner.require_hold is False


class TestRegistration:
    def test_both_are_reachable_from_the_cli(self) -> None:
        from axiom.cli import STRATEGIES

        assert STRATEGIES["sweep-continuation"] is SweepContinuationStrategy
        assert STRATEGIES["sweep-reversal-control"] is SweepReversalControl
