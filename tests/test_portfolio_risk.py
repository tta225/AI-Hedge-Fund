"""Tests for portfolio-level risk: correlation, vol targeting, drawdown, tails."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from axiom.alpha.panel import Panel
from axiom.core.provenance import Provenance
from axiom.portfolio.risk import (
    CALENDAR_DAYS,
    TRADING_DAYS,
    InsufficientHistoryError,
    PortfolioRiskManager,
    correlation_matrix,
    covariance_matrix,
    diversification_ratio,
    drawdown_scalar,
    effective_bets,
    expected_shortfall,
    marginal_risk_contributions,
    periods_per_year,
    portfolio_volatility,
    volatility_target_scalar,
)


def _panel(
    correlation: float = 0.0,
    *,
    n: int = 400,
    freq_seconds: int = 3600,
    volatilities: tuple[float, ...] = (0.01, 0.01),
    seed: int = 0,
) -> Panel:
    """A two-asset panel with a known correlation, built from a Cholesky draw."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    other = correlation * base + np.sqrt(max(1 - correlation**2, 0.0)) * rng.normal(size=n)
    columns = [base * volatilities[0], other * volatilities[1]]

    start = datetime(2025, 1, 1, tzinfo=UTC)
    index = pd.DatetimeIndex(
        [start + i * timedelta(seconds=freq_seconds) for i in range(n)]
    )
    closes = np.column_stack([100.0 * np.exp(np.cumsum(c)) for c in columns])
    return Panel(
        symbols=("A", "B"),
        index=index,
        closes=closes,
        volumes=np.ones_like(closes),
        provenance=Provenance.real("test"),
    )


class TestAnnualisation:
    def test_hourly_bars_infer_hourly_periods(self) -> None:
        """The bug this exists to prevent: 252 on hourly bars understates 5x."""
        assert periods_per_year(_panel(freq_seconds=3600)) == pytest.approx(
            CALENDAR_DAYS * 24
        )

    def test_daily_bars_use_trading_days(self) -> None:
        assert periods_per_year(_panel(freq_seconds=86400)) == pytest.approx(
            TRADING_DAYS
        )

    def test_minute_bars_scale_further(self) -> None:
        assert periods_per_year(_panel(freq_seconds=60)) > periods_per_year(
            _panel(freq_seconds=3600)
        )

    def test_a_single_gap_does_not_move_the_estimate(self) -> None:
        """Median, not mean — one venue outage must not rescale all risk."""
        panel = _panel(freq_seconds=3600, n=200)
        index = panel.index.to_list()
        index[100] = index[100] + timedelta(hours=48)
        gapped = Panel(
            symbols=panel.symbols,
            index=pd.DatetimeIndex(sorted(index)),
            closes=panel.closes,
            volumes=panel.volumes,
            provenance=panel.provenance,
        )
        assert periods_per_year(gapped) == pytest.approx(periods_per_year(panel))

    def test_realistic_crypto_volatility(self) -> None:
        """A sanity anchor: 1% hourly moves annualise to a large number."""
        panel = _panel(freq_seconds=3600, volatilities=(0.01, 0.01), n=500)
        covariance = covariance_matrix(panel, len(panel) - 1, lookback=400)
        vol = float(np.sqrt(covariance[0, 0]))
        assert 0.5 < vol < 1.5


class TestCovariance:
    def test_is_causal(self) -> None:
        """A covariance that peeks makes every portfolio look diversifiable."""
        panel = _panel(n=400)
        early = covariance_matrix(panel, 200, lookback=100)
        late = covariance_matrix(panel, 380, lookback=100)
        assert not np.allclose(early, late)

    def test_refuses_a_short_sample(self) -> None:
        panel = _panel(n=400)
        with pytest.raises(InsufficientHistoryError, match="observations"):
            covariance_matrix(panel, 10, lookback=100)

    def test_annualisation_can_be_disabled(self) -> None:
        panel = _panel(n=400)
        raw = covariance_matrix(panel, 300, lookback=200, annualise=False)
        annual = covariance_matrix(panel, 300, lookback=200)
        assert (annual > raw).all()

    def test_shape_is_stable_for_one_instrument(self) -> None:
        panel = _panel(n=300)
        single = Panel(
            symbols=("A",),
            index=panel.index,
            closes=panel.closes[:, :1],
            volumes=panel.volumes[:, :1],
            provenance=panel.provenance,
        )
        assert covariance_matrix(single, 250, lookback=200).shape == (1, 1)


class TestCorrelation:
    def test_recovers_a_known_correlation(self) -> None:
        panel = _panel(correlation=0.8, n=2000)
        correlation = correlation_matrix(covariance_matrix(panel, 1999, lookback=1500))
        assert correlation[0, 1] == pytest.approx(0.8, abs=0.06)

    def test_diagonal_is_one(self) -> None:
        correlation = correlation_matrix(np.array([[4.0, 1.0], [1.0, 9.0]]))
        assert np.allclose(np.diag(correlation), 1.0)

    def test_bounded(self) -> None:
        panel = _panel(correlation=0.99, n=1000)
        correlation = correlation_matrix(covariance_matrix(panel, 999, lookback=800))
        assert (np.abs(correlation) <= 1.0).all()

    def test_a_dead_series_correlates_with_nothing(self) -> None:
        """A stalled feed carries no information about co-movement."""
        correlation = correlation_matrix(np.array([[4.0, 0.0], [0.0, 0.0]]))
        assert correlation[0, 1] == 0.0
        assert correlation[1, 1] == 1.0


class TestPortfolioVolatility:
    def test_correlated_positions_add_up(self) -> None:
        """The whole point: correlation is what makes three bets into one."""
        covariance = np.array([[0.04, 0.04], [0.04, 0.04]])  # correlation 1.0
        assert portfolio_volatility(np.array([0.5, 0.5]), covariance) == pytest.approx(
            0.2
        )

    def test_uncorrelated_positions_diversify(self) -> None:
        covariance = np.array([[0.04, 0.0], [0.0, 0.04]])
        combined = portfolio_volatility(np.array([0.5, 0.5]), covariance)
        assert combined == pytest.approx(np.sqrt(2) * 0.1)
        assert combined < 0.2

    def test_a_hedge_reduces_risk(self) -> None:
        """A short in a correlated name genuinely lowers portfolio risk."""
        covariance = np.array([[0.04, 0.036], [0.036, 0.04]])
        long_only = portfolio_volatility(np.array([0.5, 0.5]), covariance)
        hedged = portfolio_volatility(np.array([0.5, -0.5]), covariance)
        assert hedged < long_only

    def test_survives_a_non_psd_estimate(self) -> None:
        """Short samples can produce a slightly negative variance."""
        covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert portfolio_volatility(np.array([1.0, -1.0]), covariance) == 0.0

    def test_shape_mismatch_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="weights against"):
            portfolio_volatility(np.array([1.0]), np.eye(3))


class TestDiversification:
    def test_perfectly_correlated_book_is_one_bet(self) -> None:
        covariance = np.array([[0.04, 0.04], [0.04, 0.04]])
        weights = np.array([0.5, 0.5])
        assert diversification_ratio(weights, covariance) == pytest.approx(1.0)
        assert effective_bets(weights, covariance) == pytest.approx(1.0)

    def test_uncorrelated_book_holds_more_bets(self) -> None:
        covariance = np.eye(4) * 0.04
        weights = np.full(4, 0.25)
        assert effective_bets(weights, covariance) == pytest.approx(4.0)

    def test_an_empty_book_is_one_by_convention(self) -> None:
        assert diversification_ratio(np.zeros(2), np.eye(2)) == 1.0


class TestRiskContributions:
    def test_contributions_sum_to_one(self) -> None:
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        contributions = marginal_risk_contributions(np.array([0.6, 0.4]), covariance)
        assert contributions.sum() == pytest.approx(1.0)

    def test_notional_weight_hides_concentration(self) -> None:
        """A 20% notional position can easily be most of the risk."""
        covariance = np.array([[0.0004, 0.0], [0.0, 0.25]])
        contributions = marginal_risk_contributions(np.array([0.8, 0.2]), covariance)
        assert contributions[1] > 0.9

    def test_an_empty_book_contributes_nothing(self) -> None:
        assert marginal_risk_contributions(np.zeros(2), np.eye(2)).tolist() == [0.0, 0.0]


class TestVolatilityTargeting:
    def test_scales_up_in_a_quiet_regime(self) -> None:
        assert volatility_target_scalar(
            0.05, target=0.15, max_leverage=5.0
        ) == pytest.approx(3.0)

    def test_scales_down_in_a_violent_one(self) -> None:
        assert volatility_target_scalar(0.60, target=0.15) == pytest.approx(0.25)

    def test_leverage_is_capped(self) -> None:
        """Quiet stretches precede volatile ones; uncapped leverage is a trap."""
        assert volatility_target_scalar(0.001, target=0.15, max_leverage=2.0) == 2.0

    def test_an_unmeasurable_volatility_is_neutral_not_infinite(self) -> None:
        """'Cannot measure risk' must not read as 'there is no risk'."""
        assert volatility_target_scalar(0.0) == 1.0
        assert volatility_target_scalar(float("nan")) == 1.0

    def test_rejects_a_nonsense_target(self) -> None:
        with pytest.raises(ValueError, match="target must be positive"):
            volatility_target_scalar(0.2, target=0.0)


class TestDrawdownScalar:
    def test_full_size_at_the_high_water_mark(self) -> None:
        assert drawdown_scalar(100.0, 100.0) == 1.0

    def test_full_size_inside_the_start_band(self) -> None:
        assert drawdown_scalar(97.0, 100.0) == 1.0

    def test_scales_down_progressively(self) -> None:
        """A cliff means full size right up until none, with no time to react."""
        mild = drawdown_scalar(93.0, 100.0)
        worse = drawdown_scalar(89.0, 100.0)
        assert 1.0 > mild > worse > 0.25

    def test_floors_rather_than_stopping(self) -> None:
        """A hard stop is a halt, and a halt belongs in the guards."""
        assert drawdown_scalar(50.0, 100.0) == pytest.approx(0.25)
        assert drawdown_scalar(50.0, 100.0) > 0.0

    def test_is_monotonic(self) -> None:
        scalars = [drawdown_scalar(e, 100.0) for e in range(100, 70, -1)]
        assert scalars == sorted(scalars, reverse=True)

    def test_no_peak_gets_the_floor(self) -> None:
        assert drawdown_scalar(100.0, 0.0) == pytest.approx(0.25)

    def test_rejects_inverted_bands(self) -> None:
        with pytest.raises(ValueError, match="floor_pct"):
            drawdown_scalar(100.0, 100.0, start_pct=10.0, floor_pct=5.0)


class TestExpectedShortfall:
    def test_averages_the_tail_not_the_edge_of_it(self) -> None:
        """VaR reads the 5th percentile; ES reads everything past it."""
        returns = np.array([-0.10, -0.08, -0.05] + [0.01] * 97)
        # The 5% tail is five observations, so two gains dilute the three
        # losses. What matters is that it is materially non-zero.
        assert expected_shortfall(returns, 0.95) == pytest.approx(0.042)

    def test_a_small_tail_is_not_swallowed_by_the_quantile(self) -> None:
        """The bug this replaced: three catastrophic days reporting zero.

        With only three losses in a hundred observations, the interpolated 5th
        percentile sits at +0.01, so a `values <= cutoff` filter selects the
        entire sample and averages to a gain.
        """
        returns = np.array([-0.50, -0.40, -0.30] + [0.01] * 97)
        assert expected_shortfall(returns, 0.95) > 0.2

    def test_averages_exactly_the_worst_fraction(self) -> None:
        returns = np.concatenate([np.full(5, -0.20), np.full(95, 0.01)])
        # ceil(0.05 * 100) = 5, so the tail is precisely the five losses.
        assert expected_shortfall(returns, 0.95) == pytest.approx(0.20)

    def test_a_fatter_tail_scores_worse(self) -> None:
        mild = np.array([-0.05] * 5 + [0.01] * 95)
        fat = np.array([-0.50] * 5 + [0.01] * 95)
        assert expected_shortfall(fat) > expected_shortfall(mild)

    def test_reported_as_a_positive_loss(self) -> None:
        assert expected_shortfall(np.array([-0.1] * 10 + [0.01] * 90)) > 0

    def test_an_all_positive_sample_has_no_loss(self) -> None:
        assert expected_shortfall(np.full(100, 0.01)) == 0.0

    def test_empty_is_zero_not_an_error(self) -> None:
        assert expected_shortfall(np.array([])) == 0.0

    def test_rejects_a_nonsense_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            expected_shortfall(np.array([0.1]), confidence=1.5)


class TestPortfolioRiskManager:
    def test_reports_a_correlated_book_as_one_bet(self) -> None:
        panel = _panel(correlation=0.95, n=600)
        manager = PortfolioRiskManager(volatility_limit=0.30)
        report = manager.assess({"A": 0.5, "B": 0.5}, panel, 599, lookback=400)
        assert report.n_positions == 2
        assert report.effective_bets < 1.2
        assert "one bet wearing several names" in report.render()

    def test_reports_an_uncorrelated_book_as_diversified(self) -> None:
        panel = _panel(correlation=0.0, n=600)
        manager = PortfolioRiskManager(volatility_limit=0.30)
        report = manager.assess({"A": 0.5, "B": 0.5}, panel, 599, lookback=400)
        assert report.effective_bets > 1.5

    def test_a_breach_is_flagged(self) -> None:
        panel = _panel(correlation=0.95, n=600, volatilities=(0.02, 0.02))
        manager = PortfolioRiskManager(volatility_limit=0.05)
        report = manager.assess({"A": 1.0, "B": 1.0}, panel, 599, lookback=400)
        assert report.is_breached
        assert report.utilisation > 1.0

    def test_symbols_outside_the_panel_are_ignored(self) -> None:
        panel = _panel(n=400)
        manager = PortfolioRiskManager()
        report = manager.assess({"A": 0.5, "NOTINPANEL": 5.0}, panel, 399, lookback=300)
        assert report.n_positions == 1

    def test_scaling_never_exceeds_one(self) -> None:
        """This layer only reduces a size the per-trade budget already approved."""
        panel = _panel(correlation=0.0, n=600, volatilities=(0.0001, 0.0001))
        manager = PortfolioRiskManager(volatility_limit=1.0, volatility_target=1.0)
        scalar, _ = manager.scale_for(
            "A", 0.1, {}, panel, 599, equity=100.0, peak_equity=100.0, lookback=400
        )
        assert scalar <= 1.0

    def test_a_correlated_book_shrinks_the_next_position(self) -> None:
        panel = _panel(correlation=0.95, n=600, volatilities=(0.02, 0.02))
        manager = PortfolioRiskManager(volatility_limit=0.30)
        alone, _ = manager.scale_for(
            "A", 0.5, {}, panel, 599, equity=100.0, peak_equity=100.0, lookback=400
        )
        crowded, reason = manager.scale_for(
            "A", 0.5, {"B": 0.5}, panel, 599,
            equity=100.0, peak_equity=100.0, lookback=400,
        )
        assert crowded < alone
        assert "volatility" in reason

    def test_drawdown_shrinks_the_next_position(self) -> None:
        panel = _panel(n=600)
        manager = PortfolioRiskManager(volatility_limit=5.0, volatility_target=5.0)
        healthy, _ = manager.scale_for(
            "A", 0.1, {}, panel, 599, equity=100.0, peak_equity=100.0, lookback=400
        )
        bleeding, reason = manager.scale_for(
            "A", 0.1, {}, panel, 599, equity=88.0, peak_equity=100.0, lookback=400
        )
        assert bleeding < healthy
        assert "drawdown" in reason

    def test_concentration_is_capped(self) -> None:
        panel = _panel(correlation=0.0, n=600, volatilities=(0.05, 0.0001))
        manager = PortfolioRiskManager(
            volatility_limit=100.0, volatility_target=100.0, max_risk_contribution=0.5
        )
        _, reason = manager.scale_for(
            "A", 1.0, {"B": 1.0}, panel, 599,
            equity=100.0, peak_equity=100.0, lookback=400,
        )
        assert "concentration" in reason

    def test_no_covariance_estimate_says_so_rather_than_assuming_safety(self) -> None:
        panel = _panel(n=400)
        manager = PortfolioRiskManager()
        scalar, reason = manager.scale_for(
            "A", 0.5, {}, panel, 10, equity=100.0, peak_equity=100.0, lookback=100
        )
        assert "no covariance estimate" in reason
        assert scalar == 1.0

    def test_an_unmodelled_symbol_is_named(self) -> None:
        panel = _panel(n=600)
        manager = PortfolioRiskManager()
        _, reason = manager.scale_for(
            "UNKNOWN", 0.5, {}, panel, 599,
            equity=100.0, peak_equity=100.0, lookback=400,
        )
        assert "not in the risk panel" in reason

    def test_drawdown_still_applies_without_a_covariance_estimate(self) -> None:
        """Losing the correlation model must not also lose the drawdown brake."""
        panel = _panel(n=400)
        manager = PortfolioRiskManager()
        scalar, _ = manager.scale_for(
            "A", 0.5, {}, panel, 10, equity=85.0, peak_equity=100.0, lookback=100
        )
        assert scalar < 1.0

    def test_rejects_a_nonsense_limit(self) -> None:
        with pytest.raises(ValueError, match="volatility_limit"):
            PortfolioRiskManager(volatility_limit=0.0)

    def test_rejects_a_nonsense_contribution_cap(self) -> None:
        with pytest.raises(ValueError, match="max_risk_contribution"):
            PortfolioRiskManager(max_risk_contribution=1.5)
