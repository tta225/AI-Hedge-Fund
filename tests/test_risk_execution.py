"""Risk limits, sizing arithmetic, and the fill model.

These tests exist because every failure mode here loses money silently: an
oversized position, a stop that was never really placed, or a backtest that
filled on the bar that generated the signal.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from axiom.core.config import RiskSettings
from axiom.core.types import ES, EURUSD, SPY, Bar, Direction, OrderType, Side, TradingMode
from axiom.execution.base import Order
from axiom.execution.router import OrderRouter, emergency_flatten
from axiom.execution.simulated import FillModel, SimulatedVenue
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.risk.sizing import reward_risk_ratio, size_by_risk

TS = pd.Timestamp("2025-03-03 14:30", tz="UTC")


def bar(o: float, h: float, low: float, c: float, minutes: int = 0) -> Bar:
    return Bar(
        timestamp=datetime(2025, 3, 3, 14, 30, tzinfo=UTC) + pd.Timedelta(minutes=minutes),
        open=o, high=h, low=low, close=c, volume=100.0,
    )


class TestSizing:
    def test_size_scales_inversely_with_stop_distance(self) -> None:
        tight = size_by_risk(ES, 5000, 4995, 5000)   # 5 pt stop = $250/contract
        wide = size_by_risk(ES, 5000, 4980, 5000)    # 20 pt stop = $1000/contract
        assert tight.quantity == 20
        assert wide.quantity == 5

    def test_actual_risk_never_exceeds_budget(self) -> None:
        result = size_by_risk(ES, 5000, 4993, 1000)
        assert result.risk_cash <= 1000

    def test_futures_round_down_to_whole_contracts(self) -> None:
        result = size_by_risk(ES, 5000, 4990, 900)  # affords 1.8 contracts
        assert result.quantity == 1

    def test_fx_permits_fractional_size(self) -> None:
        result = size_by_risk(EURUSD, 1.1000, 1.0980, 500)
        assert result.quantity != int(result.quantity)

    def test_zero_stop_distance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="coincides with entry"):
            size_by_risk(ES, 5000, 5000, 1000)

    def test_sub_tick_stop_is_refused(self) -> None:
        with pytest.raises(ValueError, match="narrower than one tick"):
            size_by_risk(ES, 5000.0, 5000.1, 1000)

    def test_unaffordable_setup_yields_zero_not_an_error(self) -> None:
        result = size_by_risk(ES, 5000, 4900, 100)  # 100 pt stop, $100 budget
        assert result.quantity == 0
        assert not result.is_tradable
        assert "below one tradable unit" in result.rationale

    def test_reward_risk_ratio(self) -> None:
        assert reward_risk_ratio(100, 95, 115) == pytest.approx(3.0)


class TestRiskManager:
    def _fresh(self, **kwargs: object) -> tuple[RiskManager, Portfolio]:
        # Exposure is left generous here so these tests isolate the limit they
        # each target; the cap gets its own dedicated tests below. One ES
        # contract is ~$250k notional, so a tight cap would reject everything
        # for reasons unrelated to the property under test.
        params: dict[str, object] = {
            "account_equity": 100_000,
            "max_risk_per_trade_pct": 0.5,
            "daily_loss_limit_pct": 2.0,
            "max_positions": 3,
            "max_consecutive_losses": 4,
            "max_gross_exposure_pct": 2000.0,
        }
        params.update(kwargs)
        settings = RiskSettings(**params)  # type: ignore[arg-type]
        return RiskManager(settings), Portfolio(starting_cash=100_000)

    def test_approves_a_sane_trade(self) -> None:
        risk, pf = self._fresh()
        decision = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=pf, timestamp=TS,
        )
        assert decision.approved
        assert decision.quantity == 1  # $500 budget / $500 per contract

    def test_kill_switch_blocks_everything(self) -> None:
        risk, pf = self._fresh()
        risk.engage_kill_switch("manual halt")
        decision = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=pf, timestamp=TS,
        )
        assert not decision.approved
        assert "kill switch" in decision.explain()

    def test_kill_switch_release_requires_confirmation(self) -> None:
        risk, _ = self._fresh()
        risk.engage_kill_switch()
        with pytest.raises(ValueError, match="RELEASE"):
            risk.release_kill_switch("yes")
        risk.release_kill_switch("RELEASE")
        assert not risk.kill_switch

    def test_stop_on_the_wrong_side_is_refused(self) -> None:
        risk, pf = self._fresh()
        decision = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=5010,
            portfolio=pf, timestamp=TS,
        )
        assert not decision.approved
        assert "at or above entry" in decision.reasons[0]

    def test_neutral_direction_is_refused(self) -> None:
        risk, pf = self._fresh()
        assert not risk.evaluate(
            instrument=ES, direction=Direction.NEUTRAL, entry=5000, stop=4990,
            portfolio=pf, timestamp=TS,
        ).approved

    def test_daily_loss_limit_halts_trading(self) -> None:
        risk, pf = self._fresh()
        # Realise a loss beyond the 2% ($2,000) daily limit.
        pf.daily_realised[TS.tz_convert("America/New_York").normalize()] = -2500.0
        decision = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=pf, timestamp=TS,
        )
        assert not decision.approved
        assert "daily loss limit" in decision.reasons[0]

    def test_consecutive_losses_block_then_reset_next_day(self) -> None:
        risk, pf = self._fresh()
        for _ in range(4):
            risk.on_trade_closed(-100.0)
        blocked = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=pf, timestamp=TS,
        )
        assert not blocked.approved

        # The streak is a daily circuit breaker, not a permanent shutdown.
        next_day = TS + pd.Timedelta(days=1)
        assert risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=pf, timestamp=next_day,
        ).approved

    def test_a_win_resets_the_loss_streak(self) -> None:
        risk, _ = self._fresh()
        risk.on_trade_closed(-100.0)
        risk.on_trade_closed(-100.0)
        risk.on_trade_closed(+50.0)
        assert risk.state.consecutive_losses == 0

    def test_requested_risk_above_the_cap_is_refused(self) -> None:
        risk, pf = self._fresh()
        decision = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=pf, timestamp=TS, risk_pct=5.0,
        )
        assert not decision.approved
        assert "exceeds per-trade cap" in decision.reasons[0]

    def test_risk_is_trimmed_to_the_remaining_daily_budget(self) -> None:
        risk, pf = self._fresh()
        # $1,800 of the $2,000 daily budget already lost; only $200 remains.
        pf.daily_realised[TS.tz_convert("America/New_York").normalize()] = -1800.0
        decision = risk.evaluate(
            instrument=SPY, direction=Direction.BULLISH, entry=500, stop=495,
            portfolio=pf, timestamp=TS,
        )
        assert decision.approved
        assert decision.sizing is not None
        assert decision.sizing.risk_cash <= 200.0
        assert any("trimmed" in r for r in decision.reasons)

    def test_single_unit_exceeding_the_cap_says_so_explicitly(self) -> None:
        """One ES contract is ~$250k notional; on a small account with a tight
        cap it can never fit, and the message must say that rather than
        reporting an empty book as 'cap reached'."""
        risk, pf = self._fresh(max_gross_exposure_pct=200.0)
        decision = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=pf, timestamp=TS,
        )
        assert not decision.approved
        assert "alone exceeds" in decision.reasons[0]
        assert "untradable at this account size" in decision.reasons[0]

    def test_exposure_cap_trims_size_when_partially_affordable(self) -> None:
        risk, pf = self._fresh(max_gross_exposure_pct=100.0)
        # SPY at $500: cap is $100k = 200 shares, but risk would allow more.
        decision = risk.evaluate(
            instrument=SPY, direction=Direction.BULLISH, entry=500, stop=499,
            portfolio=pf, timestamp=TS,
        )
        assert decision.approved
        assert decision.quantity <= 200
        assert any("gross exposure cap" in r for r in decision.reasons)

    def test_settings_reject_incoherent_limits(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            RiskSettings(max_risk_per_trade_pct=5.0, daily_loss_limit_pct=2.0)


class TestPortfolioAccounting:
    def test_long_round_trip_pnl(self) -> None:
        pf = Portfolio(starting_cash=100_000)
        pf.apply_fill(ES, Side.BUY, 2, 5000, 0.0, TS)
        realised = pf.apply_fill(ES, Side.SELL, 2, 5010, 0.0, TS)
        assert realised == pytest.approx(1000.0)  # 10 pts * $50 * 2
        assert pf.position(ES).is_flat

    def test_short_round_trip_pnl(self) -> None:
        pf = Portfolio(starting_cash=100_000)
        pf.apply_fill(ES, Side.SELL, 1, 5000, 0.0, TS)
        realised = pf.apply_fill(ES, Side.BUY, 1, 4990, 0.0, TS)
        assert realised == pytest.approx(500.0)

    def test_adding_blends_the_average_price(self) -> None:
        pf = Portfolio(starting_cash=100_000)
        pf.apply_fill(ES, Side.BUY, 1, 5000, 0.0, TS)
        pf.apply_fill(ES, Side.BUY, 1, 5010, 0.0, TS)
        assert pf.position(ES).average_price == pytest.approx(5005.0)

    def test_partial_close_leaves_basis_untouched(self) -> None:
        pf = Portfolio(starting_cash=100_000)
        pf.apply_fill(ES, Side.BUY, 3, 5000, 0.0, TS)
        pf.apply_fill(ES, Side.SELL, 1, 5020, 0.0, TS)
        position = pf.position(ES)
        assert position.quantity == 2
        assert position.average_price == pytest.approx(5000.0)

    def test_reversal_through_flat_reopens_at_the_fill_price(self) -> None:
        pf = Portfolio(starting_cash=100_000)
        pf.apply_fill(ES, Side.BUY, 1, 5000, 0.0, TS)
        pf.apply_fill(ES, Side.SELL, 3, 5010, 0.0, TS)
        position = pf.position(ES)
        assert position.quantity == -2
        assert position.average_price == pytest.approx(5010.0)

    def test_commission_reduces_cash(self) -> None:
        pf = Portfolio(starting_cash=100_000)
        pf.apply_fill(ES, Side.BUY, 1, 5000, 2.50, TS)
        assert pf.cash == pytest.approx(100_000 - 2.50)


class TestFillModel:
    def test_order_cannot_fill_on_the_bar_that_created_it(self) -> None:
        venue = SimulatedVenue(FillModel(slippage_ticks=0, commission_per_unit=0))
        entry_bar = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(instrument=ES, side=Side.BUY, quantity=1)
        venue.submit(order, pd.Timestamp(entry_bar.timestamp))
        assert venue.process_bar(entry_bar) == []

    def test_market_order_fills_at_the_next_bar_open(self) -> None:
        venue = SimulatedVenue(FillModel(slippage_ticks=0, commission_per_unit=0))
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(instrument=ES, side=Side.BUY, quantity=1)
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        fills = venue.process_bar(bar(5007, 5020, 5000, 5015, minutes=15))
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(5007.0)

    def test_slippage_always_works_against_the_order(self) -> None:
        venue = SimulatedVenue(FillModel(slippage_ticks=2, commission_per_unit=0))
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        buy = Order(instrument=ES, side=Side.BUY, quantity=1)
        sell = Order(instrument=ES, side=Side.SELL, quantity=1)
        venue.submit(buy, pd.Timestamp(first.timestamp))
        venue.submit(sell, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        fills = {f.side: f.price for f in venue.process_bar(bar(5007, 5020, 5000, 5015, 15))}
        assert fills[Side.BUY] > 5007.0   # buyer pays up
        assert fills[Side.SELL] < 5007.0  # seller receives less

    def test_limit_order_does_not_fill_if_price_never_reaches_it(self) -> None:
        venue = SimulatedVenue(FillModel(slippage_ticks=0, commission_per_unit=0))
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(
            instrument=ES, side=Side.BUY, quantity=1,
            order_type=OrderType.LIMIT, limit_price=4900,
        )
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        assert venue.process_bar(bar(5007, 5020, 4950, 5015, 15)) == []

    def test_stop_that_gaps_through_fills_worse_than_its_trigger(self) -> None:
        venue = SimulatedVenue(FillModel(slippage_ticks=0, commission_per_unit=0))
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(
            instrument=ES, side=Side.SELL, quantity=1,
            order_type=OrderType.STOP, stop_price=4990,
        )
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        # Next bar gaps far below the stop: the fill is the open, not the stop.
        fills = venue.process_bar(bar(4950, 4955, 4940, 4945, 15))
        assert fills[0].price == pytest.approx(4950.0)
        assert fills[0].price < 4990

    def test_fill_price_never_escapes_the_bar_range(self) -> None:
        venue = SimulatedVenue(FillModel(slippage_ticks=100, commission_per_unit=0))
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(instrument=ES, side=Side.BUY, quantity=1)
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        next_bar = bar(5007, 5020, 5000, 5015, 15)
        fill = venue.process_bar(next_bar)[0]
        assert next_bar.low <= fill.price <= next_bar.high

    def test_commission_is_charged_per_unit(self) -> None:
        venue = SimulatedVenue(FillModel(slippage_ticks=0, commission_per_unit=1.50))
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(instrument=ES, side=Side.BUY, quantity=3)
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        assert venue.process_bar(bar(5007, 5020, 5000, 5015, 15))[0].commission == 4.50


class TestRouter:
    def test_router_refuses_to_wrap_a_live_venue_outside_live_mode(self) -> None:
        venue = SimulatedVenue()
        venue.is_live = True  # type: ignore[misc]
        with pytest.raises(PermissionError, match="Refusing"):
            OrderRouter(venue=venue, risk=RiskManager(), mode=TradingMode.PAPER)

    def test_kill_switch_stops_orders_at_the_router(self) -> None:
        risk = RiskManager()
        router = OrderRouter(venue=SimulatedVenue(), risk=risk)
        risk.engage_kill_switch("test")
        order = router.submit(Order(instrument=ES, side=Side.BUY, quantity=1), TS)
        assert not order.is_open
        assert router.routed_count == 0
        assert router.rejected

    def test_emergency_flatten_engages_kill_switch_and_cancels(self) -> None:
        risk = RiskManager()
        router = OrderRouter(venue=SimulatedVenue(), risk=risk)
        router.submit(Order(instrument=ES, side=Side.BUY, quantity=1), TS)
        cancelled = emergency_flatten(router, "circuit breaker")
        assert risk.kill_switch
        assert len(cancelled) == 1
        assert router.venue.working_orders() == []
