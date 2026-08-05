"""ICT engine correctness, on handcrafted geometry.

The most important tests here are in :class:`TestLookaheadDiscipline`. Every
other property can be wrong and cost you a mediocre strategy; a lookahead leak
produces a beautiful equity curve that does not exist.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_series

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import ICTEngine, confluence_score
from axiom.ict.imbalance import find_fair_value_gaps, unmitigated_gaps
from axiom.ict.liquidity import find_liquidity_pools, find_sweeps
from axiom.ict.models import LiquidityKind, StructureEventType, SwingKind
from axiom.ict.orderblocks import find_order_blocks
from axiom.ict.ranges import build_dealing_range, premium_discount_bias
from axiom.ict.structure import detect_structure
from axiom.ict.swings import find_swings


class TestSwings:
    def test_finds_an_obvious_swing_high(self) -> None:
        rows = [
            (10, 11, 9, 10), (10, 12, 10, 11), (11, 20, 11, 19),  # 2: the peak
            (19, 19.2, 14, 15), (15, 16, 13, 14),
        ]
        series = make_series([(r[0], r[1], r[2], r[3]) for r in rows])
        swings = find_swings(series, strength=2)
        highs = [s for s in swings if s.kind is SwingKind.HIGH]
        assert any(s.origin_index == 2 and s.price == 20 for s in highs)

    def test_confirmation_lags_by_strength(self) -> None:
        series = make_series([
            (10, 11, 9, 10), (10, 12, 10, 11), (11, 20, 11, 19),
            (19, 19.2, 14, 15), (15, 16, 13, 14),
        ])
        swing = next(
            s for s in find_swings(series, strength=2)
            if s.kind is SwingKind.HIGH and s.origin_index == 2
        )
        # A pivot at bar 2 cannot be known until bar 4.
        assert swing.confirmed_index == 4
        assert swing.confirmation_lag == 2
        assert not swing.is_known_at(3)
        assert swing.is_known_at(4)

    def test_returns_nothing_when_series_too_short(self) -> None:
        assert find_swings(make_series([(1, 2, 0.5, 1.5)] * 3), strength=3) == []

    def test_rejects_zero_strength(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            find_swings(make_series([(1, 2, 0.5, 1.5)] * 10), strength=0)


class TestFairValueGaps:
    def test_detects_a_bullish_gap(self) -> None:
        # bar0 high 101, bar2 low 105 -> unfilled band 101..105
        series = make_series([
            (100, 101, 99, 100.5),
            (100.5, 106, 100.4, 105.5),
            (105.5, 108, 105, 107),
            (107, 107.5, 106, 106.5),
        ])
        gaps = find_fair_value_gaps(series, min_gap_atr=0.0)
        bull = [g for g in gaps if g.direction is Direction.BULLISH]
        assert any(g.bottom == 101 and g.top == 105 for g in bull)

    def test_detects_a_bearish_gap(self) -> None:
        series = make_series([
            (110, 111, 109, 109.5),
            (109.5, 109.6, 104, 104.5),
            (104.5, 105, 102, 103),
            (103, 104, 102.5, 103.5),
        ])
        gaps = find_fair_value_gaps(series, min_gap_atr=0.0)
        bear = [g for g in gaps if g.direction is Direction.BEARISH]
        assert any(g.top == 109 and g.bottom == 105 for g in bear)

    def test_gap_is_confirmed_on_the_third_bar(self) -> None:
        series = make_series([
            (100, 101, 99, 100.5),
            (100.5, 106, 100.4, 105.5),
            (105.5, 108, 105, 107),
            (107, 107.5, 106, 106.5),
        ])
        gap = find_fair_value_gaps(series, min_gap_atr=0.0)[0]
        # Pattern spans bars 0-2; it is only visible once bar 2 closes.
        assert gap.origin_index == 1
        assert gap.confirmed_index == 2

    def test_midpoint_is_consequent_encroachment(self) -> None:
        series = make_series([
            (100, 101, 99, 100.5),
            (100.5, 106, 100.4, 105.5),
            (105.5, 108, 105, 107),
            (107, 107.5, 106, 106.5),
        ])
        gap = find_fair_value_gaps(series, min_gap_atr=0.0)[0]
        assert gap.midpoint == pytest.approx(103.0)

    def test_mitigation_is_recorded_when_price_returns(self) -> None:
        series = make_series([
            (100, 101, 99, 100.5),
            (100.5, 106, 100.4, 105.5),
            (105.5, 108, 105, 107),
            (107, 107.5, 100, 100.5),  # trades fully back through the gap
        ])
        gap = find_fair_value_gaps(series, min_gap_atr=0.0)[0]
        assert gap.mitigated_index == 3
        assert gap.fill_ratio == pytest.approx(1.0)

    def test_tiny_gaps_are_filtered_by_atr_threshold(self) -> None:
        series = make_series([(100 + i, 102 + i, 99 + i, 101 + i) for i in range(30)])
        assert len(find_fair_value_gaps(series, min_gap_atr=5.0)) == 0


class TestStructure:
    def test_first_break_establishes_trend_as_bos(self, uptrend_series: OHLCVSeries) -> None:
        swings = find_swings(uptrend_series, strength=2)
        events = detect_structure(uptrend_series, swings)
        assert events
        assert events[0].event_type is StructureEventType.BOS

    def test_break_direction_matches_price_action(self, uptrend_series: OHLCVSeries) -> None:
        swings = find_swings(uptrend_series, strength=2)
        events = detect_structure(uptrend_series, swings)
        assert any(e.direction is Direction.BULLISH for e in events)

    def test_events_never_reference_unconfirmed_swings(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        by_origin = {s.origin_index: s for s in swings}
        for event in detect_structure(synthetic_series, swings):
            swing = by_origin.get(event.broken_swing_index)  # type: ignore[arg-type]
            if swing is not None:
                assert swing.confirmed_index <= event.confirmed_index

    def test_no_events_without_swings(self) -> None:
        assert detect_structure(make_series([(1, 2, 0.5, 1.5)] * 5), []) == []


class TestOrderBlocks:
    def test_block_is_confirmed_at_the_structure_break_not_the_candle(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        blocks = find_order_blocks(synthetic_series, events, [])
        assert blocks
        # The origin candle always precedes the break that reveals it.
        assert all(b.origin_index < b.confirmed_index for b in blocks)

    def test_bullish_block_originates_from_a_down_close_candle(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        blocks = find_order_blocks(synthetic_series, events, [])
        opens, closes = synthetic_series.opens, synthetic_series.closes
        for block in blocks:
            i = block.origin_index
            if block.direction is Direction.BULLISH:
                assert closes[i] < opens[i]
            else:
                assert closes[i] > opens[i]

    def test_zone_is_the_candle_body_not_its_range(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Canonical: the order block zone is the body, open to close."""
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        blocks = find_order_blocks(synthetic_series, events, [])
        assert blocks
        for block in blocks:
            assert block.top == max(block.open_price, block.close_price)
            assert block.bottom == min(block.open_price, block.close_price)
            # The body always sits inside the candle.
            assert block.candle_low <= block.bottom
            assert block.top <= block.candle_high

    def test_mean_threshold_is_the_body_midpoint(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        for block in find_order_blocks(synthetic_series, events, []):
            assert block.mean_threshold == pytest.approx(
                (block.open_price + block.close_price) / 2
            )

    def test_protective_edge_uses_the_full_candle(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A stop at the body edge sits inside the candle's own wick."""
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        for block in find_order_blocks(synthetic_series, events, []):
            if block.direction is Direction.BULLISH:
                assert block.protective_edge == block.candle_low
                assert block.entry_edge == block.top
            else:
                assert block.protective_edge == block.candle_high

    def test_candle_range_variant_is_available(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        ranged = find_order_blocks(
            synthetic_series, events, [], use_candle_range=True
        )
        assert ranged
        for block in ranged:
            assert block.top == block.candle_high
            assert block.bottom == block.candle_low

    def test_pivot_anchoring_is_flagged(self, synthetic_series: OHLCVSeries) -> None:
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        blocks = find_order_blocks(synthetic_series, events, [], swings=swings)
        assert any(b.anchored_at_pivot for b in blocks)

    def test_breaker_inverts_polarity(self, synthetic_series: OHLCVSeries) -> None:
        swings = find_swings(synthetic_series, strength=2)
        events = detect_structure(synthetic_series, swings)
        broken = [
            b for b in find_order_blocks(synthetic_series, events, []) if b.is_breaker
        ]
        assert broken, "expected at least one failed block in a long sample"
        for block in broken:
            assert block.effective_direction is block.direction.opposite


class TestLiquidity:
    def test_swing_highs_become_buyside_pools(self, uptrend_series: OHLCVSeries) -> None:
        swings = find_swings(uptrend_series, strength=2)
        pools = find_liquidity_pools(uptrend_series, swings)
        assert any(p.kind is LiquidityKind.BUYSIDE for p in pools)
        assert any(p.kind is LiquidityKind.SELLSIDE for p in pools)

    def test_equal_highs_collapse_into_one_pool(self) -> None:
        # Two near-identical peaks should register as a single equal-highs pool.
        rows = [
            (10, 11, 9, 10), (10, 12, 10, 11), (11, 20, 11, 15),
            (15, 16, 13, 14), (14, 15, 12, 13),
            (13, 14, 12, 13), (13, 20.02, 13, 15),
            (15, 16, 13, 14), (14, 15, 12, 13),
        ]
        series = make_series(rows)
        swings = find_swings(series, strength=2)
        pools = find_liquidity_pools(series, swings, equality_atr=2.0)
        clusters = [p for p in pools if p.kind is LiquidityKind.BUYSIDE and p.is_equal_cluster]
        assert clusters
        assert clusters[0].size >= 2

    def test_pool_price_sits_at_the_extreme_of_its_members(self) -> None:
        rows = [
            (10, 11, 9, 10), (10, 12, 10, 11), (11, 20, 11, 15),
            (15, 16, 13, 14), (14, 15, 12, 13),
            (13, 14, 12, 13), (13, 20.02, 13, 15),
            (15, 16, 13, 14), (14, 15, 12, 13),
        ]
        series = make_series(rows)
        swings = find_swings(series, strength=2)
        pools = find_liquidity_pools(series, swings, equality_atr=2.0)
        cluster = next(
            p for p in pools if p.kind is LiquidityKind.BUYSIDE and p.is_equal_cluster
        )
        # Buy stops rest above the highest equal high, not at their mean.
        assert cluster.price == pytest.approx(20.02)

    def test_sweep_requires_closing_back_inside(self, synthetic_series: OHLCVSeries) -> None:
        swings = find_swings(synthetic_series, strength=2)
        pools = find_liquidity_pools(synthetic_series, swings)
        sweeps = find_sweeps(synthetic_series, pools, require_close_back=True)
        assert sweeps
        assert all(s.closed_back_inside for s in sweeps)

    def test_sweep_reversal_direction_opposes_the_raid(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        pools = find_liquidity_pools(synthetic_series, swings)
        for sweep in find_sweeps(synthetic_series, pools):
            if sweep.kind is LiquidityKind.BUYSIDE:
                assert sweep.reversal_direction is Direction.BEARISH
            else:
                assert sweep.reversal_direction is Direction.BULLISH


class TestDealingRange:
    def test_premium_and_discount_split_at_equilibrium(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        dealing = build_dealing_range(synthetic_series, swings, len(synthetic_series) - 1)
        assert dealing is not None
        assert dealing.is_premium(dealing.high - dealing.height * 0.1)
        assert dealing.is_discount(dealing.low + dealing.height * 0.1)
        assert dealing.position_of(dealing.equilibrium) == pytest.approx(0.5)

    def test_ote_band_is_the_62_to_79_retracement(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        dealing = build_dealing_range(synthetic_series, swings, len(synthetic_series) - 1)
        assert dealing is not None
        low, high = dealing.optimal_trade_entry(Direction.BULLISH)
        assert low < high
        # For longs the band sits in the lower half — a discount entry.
        assert high <= dealing.equilibrium

    def test_positional_bias_is_contrarian_to_position(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        swings = find_swings(synthetic_series, strength=2)
        dealing = build_dealing_range(synthetic_series, swings, len(synthetic_series) - 1)
        assert dealing is not None
        assert premium_discount_bias(dealing, dealing.high) is Direction.BEARISH
        assert premium_discount_bias(dealing, dealing.low) is Direction.BULLISH


class TestLookaheadDiscipline:
    """The properties that keep backtests honest."""

    def test_every_feature_confirms_at_or_after_its_origin(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        groups = (
            state.swings, state.structure_events, state.fair_value_gaps,
            state.order_blocks, state.liquidity_pools, state.sweeps,
        )
        for group in groups:
            for feature in group:
                assert feature.confirmed_index >= feature.origin_index, feature

    def test_known_view_excludes_future_features(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        cutoff = len(synthetic_series) // 3
        known = state.known(cutoff)
        for group in (
            known.swings, known.structure_events, known.fair_value_gaps,
            known.order_blocks, known.liquidity_pools, known.sweeps,
        ):
            for feature in group:
                assert feature.confirmed_index <= cutoff

    def test_known_view_is_a_strict_subset(self, synthetic_series: OHLCVSeries) -> None:
        state = ICTEngine().analyse(synthetic_series)
        cutoff = len(synthetic_series) // 3
        known = state.known(cutoff)
        assert len(known.swings) < len(state.swings)
        assert len(known.fair_value_gaps) < len(state.fair_value_gaps)

    def test_analysing_a_prefix_matches_the_known_view(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The decisive test: analysing only past bars must agree with
        filtering a full-history analysis by ``confirmed_index``.

        If these diverge, the single-pass optimisation in the backtester is
        leaking future information into per-bar decisions.
        """
        engine = ICTEngine()
        cutoff = 900

        prefix = synthetic_series.islice(0, cutoff + 1)
        from_prefix = engine.analyse(prefix)
        from_full = engine.analyse(synthetic_series).known(cutoff)

        # Swings confirmed strictly inside the prefix must match exactly. The
        # final `strength` bars are excluded: a pivot there is unconfirmable in
        # the prefix by definition, which is correct behaviour, not a mismatch.
        margin = cutoff - engine.config.swing_strength
        prefix_swings = {
            (s.origin_index, s.kind, s.price)
            for s in from_prefix.swings if s.confirmed_index < margin
        }
        full_swings = {
            (s.origin_index, s.kind, s.price)
            for s in from_full.swings if s.confirmed_index < margin
        }
        assert prefix_swings == full_swings

    def test_active_accessors_report_state_as_of_the_bar(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A gap filled later must still count as active today."""
        state = ICTEngine().analyse(synthetic_series)
        cutoff = 500
        active = state.known(cutoff).active_fvgs()
        assert active, "expected live imbalances mid-tape"
        for gap in active:
            assert gap.confirmed_index <= cutoff
            assert gap.mitigated_index is None or gap.mitigated_index >= cutoff

    def test_unmitigated_helper_includes_the_touch_bar(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The bar that mitigates a gap is the bar an entry trades it."""
        gaps = find_fair_value_gaps(synthetic_series)
        mitigated = [g for g in gaps if g.mitigated_index is not None]
        assert mitigated
        gap = mitigated[0]
        at_touch = unmitigated_gaps(gaps, gap.mitigated_index)  # type: ignore[arg-type]
        assert gap in at_touch
        after = unmitigated_gaps(gaps, gap.mitigated_index + 1)  # type: ignore[operator]
        assert gap not in after


class TestEngine:
    def test_analyse_populates_every_feature_family(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        assert state.swings and state.structure_events
        assert state.fair_value_gaps and state.order_blocks
        assert state.liquidity_pools and state.sweeps
        assert state.dealing_range is not None

    def test_strict_config_is_more_selective(self, synthetic_series: OHLCVSeries) -> None:
        from axiom.ict.engine import STRICT_CONFIG

        default = ICTEngine().analyse(synthetic_series)
        strict = ICTEngine(STRICT_CONFIG).analyse(synthetic_series)
        assert len(strict.order_blocks) < len(default.order_blocks)
        assert len(strict.swings) < len(default.swings)

    def test_rejects_out_of_range_bar(self, synthetic_series: OHLCVSeries) -> None:
        with pytest.raises(IndexError):
            ICTEngine().analyse(synthetic_series, at=len(synthetic_series) + 10)

    def test_confluence_is_bounded_and_neutral_scores_zero(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        state = ICTEngine().analyse(synthetic_series)
        price = float(synthetic_series.closes[-1])
        assert confluence_score(state, price, Direction.NEUTRAL) == 0.0
        for direction in (Direction.BULLISH, Direction.BEARISH):
            assert 0.0 <= confluence_score(state, price, direction) <= 1.0
