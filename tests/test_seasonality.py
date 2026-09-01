"""Tests for the time-of-day seasonality strategy and its control."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from tests.conftest import make_series

from axiom.core.types import Direction
from axiom.ict.models import ICTState
from axiom.portfolio.positions import Portfolio
from axiom.strategy.base import StrategyContext
from axiom.strategy.seasonality import (
    PUBLISHED_ENTRY_HOUR,
    SeasonalityControl,
    TimeOfDaySeasonality,
)


def _context(index: int, *, hold_from: pd.Timestamp | None = None) -> StrategyContext:
    """A 1h tape starting at 00:00 UTC, so bar ``i`` is hour ``i % 24``."""
    rows = [(100.0, 100.5, 99.5, 100.0) for _ in range(72)]
    series = make_series(
        rows, timeframe="1h", start=datetime(2025, 3, 3, 0, 0, tzinfo=UTC)
    )
    portfolio = Portfolio(starting_cash=100_000)
    if hold_from is not None:
        position = portfolio.position(series.instrument)
        position.quantity = 1.0
        position.opened_at = hold_from
    return StrategyContext(
        series=series,
        index=index,
        ict=ICTState(
            symbol=series.instrument.symbol,
            timeframe="1h",
            as_of=series.index[index],
            index=index,
        ),
        portfolio=portfolio,
        timestamp=series.index[index],
    )


class TestEntryTiming:
    def test_fires_only_at_the_nominated_hour(self) -> None:
        strategy = TimeOfDaySeasonality(entry_hour=22)
        fired = [i for i in range(24, 48) if strategy.evaluate(_context(i)) is not None]
        assert fired == [46]  # bar 46 == 22:00 UTC on the second day

    def test_every_hour_is_reachable(self) -> None:
        """A 24-hour sweep is the study design, so no hour may be unreachable."""
        for hour in range(24):
            strategy = TimeOfDaySeasonality(entry_hour=hour)
            fired = [
                i for i in range(24, 48) if strategy.evaluate(_context(i)) is not None
            ]
            assert len(fired) == 1, f"hour {hour} fired {len(fired)} times"

    def test_defaults_to_the_published_hour(self) -> None:
        assert TimeOfDaySeasonality().entry_hour == PUBLISHED_ENTRY_HOUR

    def test_no_entry_while_a_position_is_open(self) -> None:
        context = _context(46, hold_from=pd.Timestamp("2025-03-04T20:00:00Z"))
        assert TimeOfDaySeasonality(entry_hour=22).evaluate(context) is None


class TestDirectionAndLevels:
    def test_long_by_default_with_stop_below(self) -> None:
        signal = TimeOfDaySeasonality(entry_hour=22).evaluate(_context(46))
        assert signal is not None
        assert signal.direction is Direction.BULLISH
        assert signal.stop < signal.entry < signal.primary_target

    def test_short_side_mirrors(self) -> None:
        strategy = TimeOfDaySeasonality(entry_hour=22, direction=Direction.BEARISH)
        signal = strategy.evaluate(_context(46))
        assert signal is not None
        assert signal.direction is Direction.BEARISH
        assert signal.stop > signal.entry > signal.primary_target

    def test_stop_is_wide_so_the_clock_is_what_is_measured(self) -> None:
        """A tight stop would turn this into a test of stop placement."""
        signal = TimeOfDaySeasonality(entry_hour=22, stop_atr=3.0).evaluate(_context(46))
        assert signal is not None
        atr = _context(46).atr()
        assert signal.risk_points == pytest.approx(3.0 * atr)


class TestHoldingPeriod:
    def test_holds_until_the_period_elapses(self) -> None:
        strategy = TimeOfDaySeasonality(entry_hour=22, hold_hours=2)
        opened = pd.Timestamp("2025-03-04T22:00:00Z")
        # Bar 47 is 23:00 — one hour in, still holding.
        assert strategy.should_exit(_context(47, hold_from=opened)) is None
        # Bar 48 is 00:00 — two hours in, done.
        assert strategy.should_exit(_context(48, hold_from=opened)) is not None

    def test_no_exit_when_flat(self) -> None:
        assert TimeOfDaySeasonality().should_exit(_context(46)) is None


class TestValidation:
    @pytest.mark.parametrize("hour", [-1, 24, 99])
    def test_rejects_impossible_hours(self, hour: int) -> None:
        with pytest.raises(ValueError, match="entry_hour"):
            TimeOfDaySeasonality(entry_hour=hour)

    def test_rejects_zero_hold(self) -> None:
        with pytest.raises(ValueError, match="hold_hours"):
            TimeOfDaySeasonality(hold_hours=0)

    def test_rejects_neutral_direction(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            TimeOfDaySeasonality(direction=Direction.NEUTRAL)


class TestControl:
    def test_shifts_the_hour(self) -> None:
        control = SeasonalityControl(entry_hour=22, shift_hours=12)
        assert control.entry_hour == 10

    def test_shift_wraps_around_midnight(self) -> None:
        assert SeasonalityControl(entry_hour=22, shift_hours=4).entry_hour == 2

    def test_params_record_the_hour_actually_traded(self) -> None:
        """A leaderboard label that misstates what ran is worse than no label."""
        control = SeasonalityControl(entry_hour=22, shift_hours=12)
        assert control.params["entry_hour"] == 10
        assert control.params["shift_hours"] == 12

    def test_control_differs_from_treatment_in_hour_alone(self) -> None:
        treatment = TimeOfDaySeasonality(entry_hour=22, hold_hours=2, stop_atr=3.0)
        control = SeasonalityControl(entry_hour=22, shift_hours=12, hold_hours=2, stop_atr=3.0)
        assert control.hold_hours == treatment.hold_hours
        assert control.stop_atr == treatment.stop_atr
        assert control.direction is treatment.direction
        assert control.entry_hour != treatment.entry_hour


class TestIntegration:
    def test_skips_ict_analysis(self) -> None:
        """Pure calendar arithmetic; paying for ICT on a 24-way sweep is waste."""
        assert TimeOfDaySeasonality.requires_ict is False

    def test_registered_in_the_cli(self) -> None:
        from axiom.cli import STRATEGIES

        assert STRATEGIES["seasonality"] is TimeOfDaySeasonality
        assert STRATEGIES["seasonality-control"] is SeasonalityControl
