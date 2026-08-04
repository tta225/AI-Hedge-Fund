"""Session clock behaviour and end-to-end backtest integrity."""

from __future__ import annotations

import pandas as pd
import pytest

from axiom.backtest.engine import Backtester
from axiom.backtest.metrics import Trade, compute_metrics, max_drawdown, sharpe_ratio
from axiom.core.config import ExecutionSettings, RiskSettings
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.sessions import (
    LONDON_KILLZONE,
    NY_AM_KILLZONE,
    SILVER_BULLET_AM,
    active_killzone,
    active_windows,
    in_silver_bullet,
    session_label,
    to_ny,
    trading_day,
    window_mask,
)
from axiom.core.types import ES, Side
from axiom.strategy.ict_strategies import LiquidityRaidReversal, SilverBulletStrategy


def ts(text: str) -> pd.Timestamp:
    """A New York wall-clock time, expressed in UTC."""
    return pd.Timestamp(text, tz="America/New_York").tz_convert("UTC")


class TestSessionClock:
    def test_ny_am_killzone_membership(self) -> None:
        assert active_killzone(ts("2025-03-03 08:30")) is NY_AM_KILLZONE
        assert active_killzone(ts("2025-03-03 06:00")) is None

    def test_london_killzone_membership(self) -> None:
        assert active_killzone(ts("2025-03-03 03:30")) is LONDON_KILLZONE

    def test_silver_bullet_window(self) -> None:
        assert in_silver_bullet(ts("2025-03-03 10:30")) is SILVER_BULLET_AM
        assert in_silver_bullet(ts("2025-03-03 11:30")) is None

    def test_windows_are_half_open(self) -> None:
        # NY AM runs [07:00, 10:00); 10:00 belongs to the next window.
        assert NY_AM_KILLZONE in active_windows(ts("2025-03-03 07:00"))
        assert NY_AM_KILLZONE not in active_windows(ts("2025-03-03 10:00"))

    def test_asia_window_crosses_midnight(self) -> None:
        from axiom.core.sessions import ASIA_KILLZONE

        assert ASIA_KILLZONE in active_windows(ts("2025-03-03 21:00"))
        assert ASIA_KILLZONE in active_windows(ts("2025-03-03 23:59"))
        assert ASIA_KILLZONE not in active_windows(ts("2025-03-03 01:00"))

    def test_killzones_survive_the_dst_transition(self) -> None:
        """The decisive session test.

        US DST began 2025-03-09. Both dates are 08:30 *New York* time, which is
        13:30 UTC before the switch and 12:30 UTC after. If the clock were
        anchored to UTC rather than converted per-timestamp, one of these would
        fall outside the killzone.
        """
        before = ts("2025-03-07 08:30")
        after = ts("2025-03-14 08:30")
        assert before.hour == 13 and after.hour == 12  # different UTC hours
        assert active_killzone(before) is NY_AM_KILLZONE
        assert active_killzone(after) is NY_AM_KILLZONE

    def test_window_mask_matches_scalar_membership(self) -> None:
        index = pd.date_range("2025-03-03", periods=96, freq="15min", tz="UTC")
        mask = window_mask(index, NY_AM_KILLZONE)
        for i, timestamp in enumerate(index):
            assert mask[i] == (NY_AM_KILLZONE in active_windows(timestamp))

    def test_trading_day_is_anchored_to_new_york_midnight(self) -> None:
        # 23:00 ET on the 3rd and 01:00 ET on the 4th are different days.
        index = pd.DatetimeIndex([ts("2025-03-03 23:00"), ts("2025-03-04 01:00")])
        days = trading_day(index)
        assert days[0] != days[1]

    def test_to_ny_converts_correctly(self) -> None:
        index = pd.DatetimeIndex([pd.Timestamp("2025-07-01 16:00", tz="UTC")])
        assert to_ny(index)[0].hour == 12  # EDT is UTC-4

    def test_session_label_falls_back_when_off_session(self) -> None:
        assert session_label(ts("2025-03-03 06:00")) in {"off_session", "asia", "london"}


class TestMetrics:
    def test_max_drawdown(self) -> None:
        equity = pd.Series(
            [100.0, 120.0, 90.0, 110.0],
            index=pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
        )
        cash, pct = max_drawdown(equity)
        assert cash == pytest.approx(30.0)
        assert pct == pytest.approx(25.0)

    def test_sharpe_is_nan_for_a_short_sample(self) -> None:
        equity = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
        )
        # Three daily points cannot support an annualised ratio.
        assert pd.isna(sharpe_ratio(equity))

    def test_profit_factor_is_nan_without_a_loss(self) -> None:
        """Reporting a huge finite number for an all-wins sample flatters it."""
        trades = [
            Trade(ES, Side.BUY, 1, 5000, 5010, pd.Timestamp("2025-01-01", tz="UTC"),
                  pd.Timestamp("2025-01-02", tz="UTC"), pnl=500.0, commission=1.0)
        ]
        equity = pd.Series(
            [100_000.0, 100_500.0],
            index=pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC"),
        )
        metrics = compute_metrics(trades, equity, 100_000.0)
        assert pd.isna(metrics["profit_factor"])
        assert metrics["win_rate_pct"] == 100.0

    def test_r_multiple_uses_planned_risk(self) -> None:
        trade = Trade(
            ES, Side.BUY, 1, 5000, 5020, pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"), pnl=1000.0, commission=0.0,
            risk_points=10.0,  # $500 risk on one ES contract
        )
        assert trade.r_multiple == pytest.approx(2.0)

    def test_r_multiple_is_none_without_stated_risk(self) -> None:
        trade = Trade(
            ES, Side.BUY, 1, 5000, 5020, pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"), pnl=1000.0, commission=0.0,
        )
        assert trade.r_multiple is None


class TestBacktestIntegration:
    def _run(self, series: OHLCVSeries, strategy=None):  # type: ignore[no-untyped-def]
        return Backtester(
            strategy or SilverBulletStrategy(),
            risk_settings=RiskSettings(
                account_equity=250_000, max_gross_exposure_pct=2000.0
            ),
        ).run(series, warmup=100)

    def test_runs_and_produces_a_report(self, synthetic_series: OHLCVSeries) -> None:
        result = self._run(synthetic_series)
        assert result.report.metrics["starting_equity"] == 250_000
        assert isinstance(result.report.render(), str)

    def test_synthetic_results_are_flagged_as_non_evidence(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The guarantee the whole provenance layer exists to provide."""
        result = self._run(synthetic_series)
        assert not result.report.is_evidence
        assert result.report.warning is not None
        assert "SYNTHETIC DATA" in result.report.render()

    def test_every_trade_paid_commission(self, synthetic_series: OHLCVSeries) -> None:
        result = self._run(synthetic_series)
        if result.trades:
            assert result.portfolio.total_commission > 0
            assert all(t.commission > 0 for t in result.trades)

    def test_costs_are_never_silently_zero(self, synthetic_series: OHLCVSeries) -> None:
        settings = ExecutionSettings()
        assert settings.commission_per_unit > 0
        assert settings.slippage_ticks > 0

    def test_equity_curve_covers_every_bar(self, synthetic_series: OHLCVSeries) -> None:
        result = self._run(synthetic_series)
        assert len(result.portfolio.equity_curve) == len(synthetic_series)

    def test_no_position_is_left_open_unaccounted(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        result = self._run(synthetic_series)
        # Realised P&L plus unrealised must reconcile to the equity change.
        expected = result.portfolio.equity - 250_000
        actual = result.portfolio.realised_pnl + result.portfolio.unrealised_pnl
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_trades_close_only_via_bracket_legs(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        result = self._run(synthetic_series)
        for trade in result.trades:
            assert trade.exit_reason in {"stop_loss", "take_profit"}

    def test_exit_always_follows_entry(self, synthetic_series: OHLCVSeries) -> None:
        result = self._run(synthetic_series)
        for trade in result.trades:
            assert trade.exit_time > trade.entry_time

    def test_funnel_explains_signal_attrition(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        result = self._run(synthetic_series, LiquidityRaidReversal())
        text = result.funnel()
        assert "signals generated" in text
        assert "trades completed" in text

    def test_refuses_a_series_shorter_than_the_warmup(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        with pytest.raises(ValueError, match="not enough"):
            Backtester(SilverBulletStrategy()).run(
                synthetic_series.head(40), warmup=100
            )

    def test_kill_switch_prevents_all_trading(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from axiom.core.types import Direction
        from axiom.portfolio.positions import Portfolio
        from axiom.risk.manager import RiskManager

        # The fixture must actually generate signals, or this proves nothing.
        assert self._run(synthetic_series).signals

        risk = RiskManager(RiskSettings(account_equity=250_000))
        risk.engage_kill_switch("test halt")
        decision = risk.evaluate(
            instrument=ES, direction=Direction.BULLISH, entry=5000, stop=4990,
            portfolio=Portfolio(starting_cash=250_000),
            timestamp=synthetic_series.index[-1],
        )
        assert not decision.approved
        assert "kill switch" in decision.reasons[0]

    def test_provenance_propagates_into_the_report(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        result = self._run(synthetic_series)
        assert result.report.provenance.source == "synthetic"
        assert not result.report.provenance.is_evidential

    def test_mock_data_is_also_non_evidential(self) -> None:
        assert not Provenance.mock("fixture").is_evidential
