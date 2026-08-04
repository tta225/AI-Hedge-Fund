"""The risk-budget guarantee.

Before the entry-tolerance mechanism, sizing used the signal's *intended* entry
while fills happened at the next bar's open plus slippage. When the open gapped
away, realised risk exceeded the budget — measured at roughly 1.3-1.4x on
synthetic data.

The guarantee now has two halves, and both are tested here:

1. **sizing** assumes the worst case within the tolerance, so the budget is an
   upper bound rather than an estimate;
2. **the venue** expires any entry that would fill beyond that tolerance.

The end-to-end property is the last test: no completed trade may lose more than
the per-trade budget, plus commission.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from axiom.backtest.engine import Backtester
from axiom.core.config import RiskSettings
from axiom.core.series import OHLCVSeries
from axiom.core.types import ES, SPY, Bar, Direction, OrderStatus, Side
from axiom.execution.base import Order
from axiom.execution.simulated import FillModel, SimulatedVenue
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.risk.sizing import size_by_risk
from axiom.strategy.ict_strategies import LiquidityRaidReversal, SilverBulletStrategy

TS = pd.Timestamp("2025-03-03 14:30", tz="UTC")


def bar(o: float, h: float, low: float, c: float, minutes: int = 0) -> Bar:
    return Bar(
        timestamp=datetime(2025, 3, 3, 14, 30, tzinfo=UTC) + pd.Timedelta(minutes=minutes),
        open=o, high=h, low=low, close=c, volume=100.0,
    )


class TestWorstCaseSizing:
    def test_tolerance_reduces_size(self) -> None:
        without = size_by_risk(ES, 5000, 4990, 5000)
        with_tolerance = size_by_risk(ES, 5000, 4990, 5000, entry_tolerance=5.0)
        assert with_tolerance.quantity < without.quantity

    def test_stop_distance_includes_the_tolerance(self) -> None:
        result = size_by_risk(ES, 5000, 4990, 5000, entry_tolerance=5.0)
        assert result.stop_distance == pytest.approx(15.0)

    def test_worst_case_fill_stays_inside_budget(self) -> None:
        """The arithmetic the whole mechanism rests on."""
        budget = 5000.0
        tolerance = 5.0
        result = size_by_risk(ES, 5000, 4990, budget, entry_tolerance=tolerance)

        # Fill at the worst permitted price: 5 points adverse for a long.
        worst_fill = 5000 + tolerance
        realised_risk = abs(worst_fill - 4990) * result.quantity * ES.point_value
        assert realised_risk <= budget

    def test_zero_tolerance_matches_the_old_behaviour(self) -> None:
        assert (
            size_by_risk(ES, 5000, 4990, 5000, entry_tolerance=0.0).quantity
            == size_by_risk(ES, 5000, 4990, 5000).quantity
        )

    def test_negative_tolerance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="entry_tolerance"):
            size_by_risk(ES, 5000, 4990, 5000, entry_tolerance=-1.0)

    def test_risk_manager_threads_the_tolerance_through(self) -> None:
        risk = RiskManager(
            RiskSettings(account_equity=100_000, max_gross_exposure_pct=2000.0)
        )
        portfolio = Portfolio(starting_cash=100_000)
        # SPY, not ES: a 100-point stop on ES is $5,000 per contract, which
        # exceeds the whole per-trade budget, so both branches would come back
        # untradable and the test would pass without testing anything.
        tight = risk.evaluate(
            instrument=SPY, direction=Direction.BULLISH, entry=500, stop=490,
            portfolio=portfolio, timestamp=TS, entry_tolerance=0.0,
        )
        loose = risk.evaluate(
            instrument=SPY, direction=Direction.BULLISH, entry=500, stop=490,
            portfolio=portfolio, timestamp=TS, entry_tolerance=10.0,
        )
        assert loose.approved and tight.approved
        assert loose.sizing is not None and tight.sizing is not None
        assert loose.sizing.stop_distance > tight.sizing.stop_distance
        assert loose.quantity < tight.quantity


class TestVenueDeviationGuard:
    def _venue(self) -> SimulatedVenue:
        return SimulatedVenue(FillModel(slippage_ticks=0, commission_per_unit=0))

    def test_fill_within_tolerance_is_accepted(self) -> None:
        venue = self._venue()
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(
            instrument=ES, side=Side.BUY, quantity=1,
            reference_price=5005.0, max_entry_deviation=10.0,
        )
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        # Opens 2 points adverse — inside the 10-point tolerance.
        fills = venue.process_bar(bar(5007, 5020, 5000, 5015, 15))
        assert len(fills) == 1
        assert order.status is OrderStatus.FILLED

    def test_fill_beyond_tolerance_expires_the_order(self) -> None:
        venue = self._venue()
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(
            instrument=ES, side=Side.BUY, quantity=1,
            reference_price=5005.0, max_entry_deviation=2.0,
        )
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        # Gaps 45 points adverse — far beyond the 2-point tolerance.
        assert venue.process_bar(bar(5050, 5060, 5045, 5055, 15)) == []
        assert order.status is OrderStatus.EXPIRED
        assert "beyond the" in order.reject_reason

    def test_expired_order_leaves_the_working_book(self) -> None:
        """Otherwise it is retried on every subsequent bar, forever."""
        venue = self._venue()
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(
            instrument=ES, side=Side.BUY, quantity=1,
            reference_price=5005.0, max_entry_deviation=2.0,
        )
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        venue.process_bar(bar(5050, 5060, 5045, 5055, 15))
        assert venue.working_orders() == []

    def test_favourable_gap_still_fills(self) -> None:
        """Tolerance bounds adverse movement only; a better price is welcome."""
        venue = self._venue()
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(
            instrument=ES, side=Side.BUY, quantity=1,
            reference_price=5005.0, max_entry_deviation=2.0,
        )
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        fills = venue.process_bar(bar(4950, 4960, 4945, 4955, 15))
        assert len(fills) == 1

    def test_short_side_tolerance_is_mirrored(self) -> None:
        venue = self._venue()
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        order = Order(
            instrument=ES, side=Side.SELL, quantity=1,
            reference_price=5005.0, max_entry_deviation=2.0,
        )
        venue.submit(order, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        # For a short, adverse means *lower*.
        assert venue.process_bar(bar(4950, 4960, 4945, 4955, 15)) == []
        assert order.status is OrderStatus.EXPIRED

    def test_protective_exits_are_never_blocked(self) -> None:
        """A stop must fill however bad the price.

        Refusing an exit because the fill is unfavourable converts a bounded
        loss into an unbounded one, so bracket legs set no deviation cap.
        """
        venue = self._venue()
        first = bar(5000, 5010, 4990, 5005, minutes=0)
        from axiom.core.types import OrderType

        stop = Order(
            instrument=ES, side=Side.SELL, quantity=1,
            order_type=OrderType.STOP, stop_price=4990,
        )
        assert stop.max_entry_deviation is None
        venue.submit(stop, pd.Timestamp(first.timestamp))
        venue.process_bar(first)
        fills = venue.process_bar(bar(4900, 4910, 4890, 4895, 15))
        assert len(fills) == 1
        assert fills[0].price < 4990  # filled far worse than the trigger


class TestEndToEndBudget:
    """The property that matters: no trade loses more than it was allowed to."""

    EQUITY = 250_000.0
    RISK_PCT = 0.5

    def _settings(self) -> RiskSettings:
        return RiskSettings(
            account_equity=self.EQUITY,
            max_risk_per_trade_pct=self.RISK_PCT,
            max_gross_exposure_pct=2000.0,
        )

    @pytest.mark.parametrize(
        "strategy_cls", [SilverBulletStrategy, LiquidityRaidReversal]
    )
    def test_no_trade_exceeds_the_per_trade_budget(
        self, synthetic_series: OHLCVSeries, strategy_cls: type
    ) -> None:
        budget = self.EQUITY * self.RISK_PCT / 100.0
        result = Backtester(
            strategy_cls(),
            risk_settings=self._settings(),
            entry_tolerance_atr=0.35,
        ).run(synthetic_series, warmup=100)

        assert result.trades, "fixture must produce trades for this to mean anything"
        for trade in result.trades:
            # Commission is a cost, not risk; it is charged whatever happens.
            loss = -(trade.pnl + trade.commission)
            assert loss <= budget * 1.001, (
                f"trade lost ${loss:,.2f} against a ${budget:,.2f} budget "
                f"({loss / budget:.2f}x) — the risk guarantee is broken"
            )

    def test_r_multiples_are_bounded_below_by_minus_one(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """R is measured against the budgeted worst case, so a stop-out is -1R."""
        result = Backtester(
            SilverBulletStrategy(),
            risk_settings=self._settings(),
            entry_tolerance_atr=0.35,
        ).run(synthetic_series, warmup=100)

        r_values = [t.r_multiple for t in result.trades if t.r_multiple is not None]
        assert r_values
        assert min(r_values) >= -1.05

    def test_disabling_both_allowances_reintroduces_the_overrun(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Demonstrates the bug these mechanisms fix, so a regression is visible.

        With both allowances off, sizing uses the intended entry-to-stop
        distance while fills gap past it on *both* legs — and trades lose
        materially more than the budget allowed.
        """
        budget = self.EQUITY * self.RISK_PCT / 100.0
        result = Backtester(
            SilverBulletStrategy(),
            risk_settings=self._settings(),
            entry_tolerance_atr=0.0,
            stop_gap_atr=0.0,
        ).run(synthetic_series, warmup=100)

        assert result.trades
        worst = max(-(t.pnl + t.commission) for t in result.trades)
        assert worst > budget, (
            "expected an unbounded run to breach the budget; if this fails the "
            "fixture no longer exercises gapped fills and the guard tests above "
            "prove less than they appear to"
        )

    def test_stop_gap_allowance_is_what_bounds_the_exit_leg(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The entry guard alone is not sufficient.

        Bounding entry slippage still leaves the stop free to gap through its
        trigger while the position is already open. Only budgeting for that in
        size brings the worst case back inside the limit.
        """
        budget = self.EQUITY * self.RISK_PCT / 100.0
        entry_only = Backtester(
            SilverBulletStrategy(), risk_settings=self._settings(),
            entry_tolerance_atr=0.35, stop_gap_atr=0.0,
        ).run(synthetic_series, warmup=100)
        both = Backtester(
            SilverBulletStrategy(), risk_settings=self._settings(),
            entry_tolerance_atr=0.35, stop_gap_atr=0.75,
        ).run(synthetic_series, warmup=100)

        worst_entry_only = max(-(t.pnl + t.commission) for t in entry_only.trades)
        worst_both = max(-(t.pnl + t.commission) for t in both.trades)
        assert worst_both < worst_entry_only
        assert worst_both <= budget * 1.001

    def test_report_surfaces_any_residual_overrun(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Tail gaps beyond the allowance are reported, not hidden."""
        result = Backtester(
            SilverBulletStrategy(), risk_settings=self._settings(),
            entry_tolerance_atr=0.0, stop_gap_atr=0.0,
        ).run(synthetic_series, warmup=100)

        assert "worst_loss_vs_budget_x" in result.report.metrics
        assert result.report.metrics["worst_loss_vs_budget_x"] > 1.0
        assert any("stop-gap risk" in n for n in result.report.notes)
