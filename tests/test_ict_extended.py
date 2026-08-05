"""Rejection blocks, turtle soup, CRT, and IPDA.

Each concept is tested against the criteria in the ICT Knowledge Library, plus
the anti-lookahead property specific to it. CRT's is the sharpest: its reference
range comes from a higher timeframe, and reading an HTF candle before it closes
puts the traded move inside its own reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.crt import find_crt_setups
from axiom.ict.ipda import compute_ipda_levels, ipda_bias, ipda_level_series
from axiom.ict.liquidity import find_liquidity_pools
from axiom.ict.models import LiquidityKind
from axiom.ict.rejection import active_rejection_blocks, find_rejection_blocks
from axiom.ict.swings import find_swings
from axiom.ict.turtle_soup import find_turtle_soups, recent_turtle_soups


@pytest.fixture
def pools(synthetic_series: OHLCVSeries):  # type: ignore[no-untyped-def]
    swings = find_swings(synthetic_series, strength=2)
    return find_liquidity_pools(synthetic_series, swings)


class TestRejectionBlocks:
    def test_wick_dominates_the_candle(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        blocks = find_rejection_blocks(synthetic_series, pools)
        assert blocks
        for block in blocks:
            assert block.wick_ratio >= 0.60

    def test_zone_is_the_wick_not_the_body(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        """The defining difference from an order block."""
        opens, closes = synthetic_series.opens, synthetic_series.closes
        highs, lows = synthetic_series.highs, synthetic_series.lows
        for block in find_rejection_blocks(synthetic_series, pools):
            i = block.origin_index
            body_top = max(opens[i], closes[i])
            body_bottom = min(opens[i], closes[i])
            if block.direction is Direction.BULLISH:
                # Zone spans body bottom down to the low — the rejected span.
                assert block.top == pytest.approx(body_bottom)
                assert block.bottom == pytest.approx(lows[i])
            else:
                assert block.bottom == pytest.approx(body_top)
                assert block.top == pytest.approx(highs[i])

    def test_confirmation_lags_by_one_bar(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        """Criterion 4 needs the next candle, so it cannot be known sooner."""
        for block in find_rejection_blocks(synthetic_series, pools):
            assert block.confirmed_index == block.origin_index + 1

    def test_requires_a_known_liquidity_level(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Without levels there is nothing to reject *at*."""
        assert find_rejection_blocks(synthetic_series, []) == []

    def test_level_is_reached_by_the_wick(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        atr = synthetic_series.atr(14)
        for block in find_rejection_blocks(synthetic_series, pools):
            tolerance = 0.25 * atr[block.origin_index]
            assert abs(block.wick_tip - block.level) <= tolerance + 1e-9

    def test_protective_edge_is_the_wick_tip(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        for block in find_rejection_blocks(synthetic_series, pools):
            assert block.protective_edge == block.wick_tip

    def test_active_filter_respects_the_index(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        blocks = find_rejection_blocks(synthetic_series, pools)
        cutoff = len(synthetic_series) // 2
        for block in active_rejection_blocks(blocks, cutoff):
            assert block.confirmed_index <= cutoff


class TestTurtleSoup:
    def test_reclaim_happens_within_the_allowed_window(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        soups = find_turtle_soups(synthetic_series, pools, reclaim_bars=3)
        assert soups
        for soup in soups:
            assert 0 <= soup.reclaim_bars <= 3

    def test_price_actually_violated_the_level(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        highs, lows = synthetic_series.highs, synthetic_series.lows
        for soup in find_turtle_soups(synthetic_series, pools):
            i = soup.violation_index
            if soup.kind is LiquidityKind.BUYSIDE:
                assert highs[i] > soup.level
            else:
                assert lows[i] < soup.level

    def test_close_returned_inside_the_level(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        closes = synthetic_series.closes
        for soup in find_turtle_soups(synthetic_series, pools):
            if soup.kind is LiquidityKind.BUYSIDE:
                assert closes[soup.reclaim_index] < soup.level
            else:
                assert closes[soup.reclaim_index] > soup.level

    def test_displacement_is_required(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        """This is what separates a turtle soup from a bare sweep."""
        lenient = find_turtle_soups(
            synthetic_series, pools, min_displacement_atr=0.0
        )
        strict = find_turtle_soups(
            synthetic_series, pools, min_displacement_atr=1.5
        )
        assert len(strict) < len(lenient)
        for soup in strict:
            assert soup.displacement_atr >= 1.5

    def test_direction_opposes_the_liquidity_taken(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        for soup in find_turtle_soups(synthetic_series, pools):
            if soup.kind is LiquidityKind.BUYSIDE:
                assert soup.direction is Direction.BEARISH
            else:
                assert soup.direction is Direction.BULLISH

    def test_confirmation_follows_the_violation(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        for soup in find_turtle_soups(synthetic_series, pools):
            assert soup.violation_index <= soup.reclaim_index < soup.confirmed_index

    def test_recent_filter_bounds_age(
        self, synthetic_series: OHLCVSeries, pools
    ) -> None:
        soups = find_turtle_soups(synthetic_series, pools)
        cutoff = len(synthetic_series) - 1
        for soup in recent_turtle_soups(soups, cutoff, within=10):
            assert cutoff - soup.confirmed_index <= 10


class TestCRT:
    def test_reference_candle_must_have_closed_first(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The decisive CRT test.

        A sweep may only be scored against a higher-timeframe candle that has
        already closed. Reading a forming HTF candle's high/low puts the traded
        move inside its own reference range and fabricates the setup.
        """
        setups = find_crt_setups(synthetic_series, "4h")
        assert setups
        for setup in setups:
            assert setup.origin_index >= setup.reference_close_index

    def test_reference_range_excludes_the_sweep_bar(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A sweep exceeds the reference bound; if the bar were inside the
        reference candle, the bound would already contain it."""
        highs, lows = synthetic_series.highs, synthetic_series.lows
        for setup in find_crt_setups(synthetic_series, "4h"):
            i = setup.origin_index
            if setup.direction is Direction.BEARISH:
                assert highs[i] > setup.reference_high
            else:
                assert lows[i] < setup.reference_low

    def test_target_is_the_opposite_bound(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        for setup in find_crt_setups(synthetic_series, "4h"):
            if setup.direction is Direction.BEARISH:
                assert setup.target == setup.reference_low
                assert setup.swept_bound == setup.reference_high
            else:
                assert setup.target == setup.reference_high
                assert setup.swept_bound == setup.reference_low

    def test_confirmed_at_the_reclaim(self, synthetic_series: OHLCVSeries) -> None:
        closes = synthetic_series.closes
        for setup in find_crt_setups(synthetic_series, "4h"):
            assert setup.confirmed_index >= setup.origin_index
            if setup.direction is Direction.BEARISH:
                assert closes[setup.confirmed_index] < setup.swept_bound
            else:
                assert closes[setup.confirmed_index] > setup.swept_bound

    def test_refuses_a_lower_reference_timeframe(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        with pytest.raises(ValueError, match="must be higher"):
            find_crt_setups(synthetic_series, "5m")

    def test_reward_risk_needs_a_buffer_to_be_meaningful(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A bare-penetration sweep gives an unactionable R:R without one."""
        setups = find_crt_setups(synthetic_series, "4h")
        atr = synthetic_series.atr(14)
        buffered = [
            s.reward_risk(stop_buffer=0.5 * atr[s.origin_index]) for s in setups
        ]
        raw = [s.reward_risk() for s in setups]
        assert max(b for b in buffered if b) < max(r for r in raw if r)

    def test_time_anchor_filter_narrows_results(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from datetime import time

        unfiltered = find_crt_setups(synthetic_series, "4h")
        filtered = find_crt_setups(
            synthetic_series, "4h", time_anchors=(time(3, 0), time(9, 0))
        )
        assert len(filtered) < len(unfiltered)


class TestIPDA:
    def test_reports_all_three_canonical_windows(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = compute_ipda_levels(synthetic_series, len(synthetic_series) - 1)
        assert {lv.window_days for lv in state.levels} == {20, 40, 60}
        assert len(state.levels) == 6  # high and low for each

    def test_windows_nest_correctly(self, synthetic_series: OHLCVSeries) -> None:
        """A longer lookback can only be wider, never narrower."""
        state = compute_ipda_levels(synthetic_series, len(synthetic_series) - 1)
        h20 = state.level(20, high=True)
        h60 = state.level(60, high=True)
        low20 = state.level(20, high=False)
        low60 = state.level(60, high=False)
        assert h20 and h60 and low20 and low60
        assert h60.price >= h20.price
        assert low60.price <= low20.price

    def test_uses_wick_extremes_not_closes(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = compute_ipda_levels(synthetic_series, len(synthetic_series) - 1)
        high = state.level(20, high=True)
        assert high is not None
        assert high.price == synthetic_series.highs[high.formed_index]

    def test_is_causal(self, synthetic_series: OHLCVSeries) -> None:
        """Levels at bar i must not move when later bars are added."""
        cutoff = len(synthetic_series) // 2
        from_prefix = compute_ipda_levels(
            synthetic_series.islice(0, cutoff + 1), cutoff
        )
        from_full = compute_ipda_levels(synthetic_series, cutoff)
        for window in (20, 40, 60):
            a = from_prefix.level(window, high=True)
            b = from_full.level(window, high=True)
            if a and b:
                assert a.price == pytest.approx(b.price)

    def test_extremes_are_never_ahead_of_the_bar(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        index = len(synthetic_series) // 3
        state = compute_ipda_levels(synthetic_series, index)
        for level in state.levels:
            assert level.formed_index <= index

    def test_bias_is_contrarian_to_position(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = compute_ipda_levels(synthetic_series, len(synthetic_series) - 1)
        position = state.range_position(20)
        assert position is not None
        bias = ipda_bias(state)
        if position > 0.7:
            assert bias is Direction.BEARISH
        elif position < 0.3:
            assert bias is Direction.BULLISH

    def test_nearest_draw_is_on_the_correct_side(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = compute_ipda_levels(synthetic_series, len(synthetic_series) - 1)
        up = state.nearest_draw(Direction.BULLISH)
        down = state.nearest_draw(Direction.BEARISH)
        if up:
            assert up.price > state.price
        if down:
            assert down.price < state.price

    def test_level_series_matches_pointwise_computation(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The vectorised path must agree with the per-bar one."""
        series = synthetic_series.head(3000)
        rolled = ipda_level_series(series, 20, high=True)
        for index in (1200, 2000, 2900):
            state = compute_ipda_levels(series, index)
            level = state.level(20, high=True)
            assert level is not None
            assert rolled[index] == pytest.approx(level.price)

    def test_rejects_out_of_range_index(self, synthetic_series: OHLCVSeries) -> None:
        with pytest.raises(IndexError):
            compute_ipda_levels(synthetic_series, len(synthetic_series) + 5)

    def test_counts_trading_days_not_calendar_days(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A 20-calendar-day window would cover ~14 trading days."""
        from axiom.core.sessions import trading_day

        index = len(synthetic_series) - 1
        state = compute_ipda_levels(synthetic_series, index)
        days = trading_day(synthetic_series.index[: index + 1])
        assert state.days_available == len(np.unique(days))
