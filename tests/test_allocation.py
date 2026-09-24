"""Capital allocation and cross-strategy netting.

The properties worth pinning are the ones a single-book risk manager cannot
see: that risk parity actually equalises risk *contribution* rather than
capital, that netting produces the book the desk holds rather than the sum of
what it asked for, and that two strategies which are secretly the same trade
are visible as such.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.portfolio.allocation import (
    Allocation,
    AllocationError,
    allocate,
    assess_cross_strategy,
    equal_weight,
    inverse_volatility,
    net_positions,
    risk_parity,
)


def _covariance(vols: list[float], correlation: float) -> np.ndarray:
    vol = np.array(vols)
    n = len(vol)
    corr = np.full((n, n), correlation)
    np.fill_diagonal(corr, 1.0)
    return np.outer(vol, vol) * corr


class TestEqualWeight:
    def test_divides_evenly(self) -> None:
        assert equal_weight(4).tolist() == [0.25] * 4

    def test_needs_a_strategy(self) -> None:
        with pytest.raises(AllocationError, match="at least one"):
            equal_weight(0)


class TestInverseVolatility:
    def test_the_quieter_strategy_gets_more(self) -> None:
        weights = inverse_volatility([0.10, 0.20])
        assert weights[0] == pytest.approx(2 / 3)
        assert weights[1] == pytest.approx(1 / 3)

    def test_equal_volatility_is_equal_weight(self) -> None:
        assert inverse_volatility([0.15, 0.15, 0.15]).tolist() == pytest.approx([1 / 3] * 3)

    def test_all_zero_volatility_is_a_measurement_failure(self) -> None:
        """Not a riskless portfolio — something failed to measure."""
        with pytest.raises(AllocationError, match="measurement failure"):
            inverse_volatility([0.0, 0.0])

    def test_negative_volatility_is_refused(self) -> None:
        with pytest.raises(AllocationError, match="cannot be negative"):
            inverse_volatility([0.1, -0.2])

    def test_a_zero_vol_strategy_is_dropped_not_infinite(self) -> None:
        weights = inverse_volatility([0.0, 0.1, 0.2])
        assert weights[0] == 0.0
        assert weights.sum() == pytest.approx(1.0)


class TestRiskParity:
    def test_equalises_risk_contribution_not_capital(self) -> None:
        """The distinction that justifies preferring this over inverse vol."""
        covariance = _covariance([0.10, 0.30], correlation=0.0)
        weights = risk_parity(covariance)
        variance = weights @ covariance @ weights
        contributions = weights * (covariance @ weights) / variance
        assert contributions[0] == pytest.approx(contributions[1], abs=1e-4)
        # Capital is emphatically not equal.
        assert weights[0] > 0.7

    def test_correlated_strategies_share_one_budget(self) -> None:
        """Two strategies that are the same trade must not each get a full
        risk budget — that is how a desk runs twice the risk it authorised."""
        independent = risk_parity(_covariance([0.2, 0.2, 0.2], correlation=0.0))
        # A and B are near-identical; C is independent.
        covariance = _covariance([0.2, 0.2, 0.2], correlation=0.0)
        covariance[0, 1] = covariance[1, 0] = 0.2 * 0.2 * 0.95
        correlated = risk_parity(covariance)
        assert independent.tolist() == pytest.approx([1 / 3] * 3, abs=1e-3)
        # The independent strategy earns more than its equal share.
        assert correlated[2] > correlated[0]
        assert correlated[2] > 1 / 3

    def test_single_strategy_takes_everything(self) -> None:
        assert risk_parity(np.array([[0.04]])).tolist() == [1.0]

    def test_weights_sum_to_one(self) -> None:
        weights = risk_parity(_covariance([0.1, 0.2, 0.15, 0.3], correlation=0.3))
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights > 0)

    def test_asymmetric_covariance_is_refused(self) -> None:
        bad = np.array([[0.04, 0.01], [0.02, 0.09]])
        with pytest.raises(AllocationError, match="symmetric"):
            risk_parity(bad)

    def test_non_square_is_refused(self) -> None:
        with pytest.raises(AllocationError, match="square"):
            risk_parity(np.zeros((2, 3)))

    def test_zero_variance_is_refused(self) -> None:
        with pytest.raises(AllocationError, match="not usable"):
            risk_parity(np.zeros((2, 2)))


class TestAllocate:
    def test_reports_diversification(self) -> None:
        covariance = _covariance([0.2, 0.2], correlation=0.0)
        result = allocate(["a", "b"], method="risk_parity", covariance=covariance)
        # Two uncorrelated equal-vol strategies: sqrt(2) improvement.
        assert result.diversification_ratio == pytest.approx(np.sqrt(2), abs=0.01)

    def test_identical_strategies_show_no_diversification(self) -> None:
        """1.0 means the desk runs one book while believing it runs several."""
        covariance = _covariance([0.2, 0.2], correlation=0.999)
        result = allocate(["a", "b"], method="risk_parity", covariance=covariance)
        assert result.diversification_ratio == pytest.approx(1.0, abs=0.01)
        assert "same trade" in result.render()

    def test_unknown_method_names_the_options(self) -> None:
        with pytest.raises(AllocationError, match="risk_parity"):
            allocate(["a"], method="magic")

    def test_risk_parity_requires_a_covariance(self) -> None:
        with pytest.raises(AllocationError, match="needs a covariance"):
            allocate(["a", "b"], method="risk_parity")

    def test_inverse_volatility_requires_volatilities(self) -> None:
        with pytest.raises(AllocationError, match="needs per-strategy"):
            allocate(["a", "b"], method="inverse_volatility")

    def test_no_strategies_is_refused(self) -> None:
        with pytest.raises(AllocationError, match="no strategies"):
            allocate([], method="equal")

    def test_render_lists_funded_strategies(self) -> None:
        result = allocate(["slow", "fast"], method="inverse_volatility",
                          volatilities=[0.1, 0.3])
        assert result.n_funded == 2
        assert "slow" in result.render()


class TestNetting:
    def test_two_strategies_wanting_the_same_name_hold_it_once(self) -> None:
        allocation = allocate(["a", "b"], method="equal")
        held = net_positions({"a": {"AAPL": 1.0}, "b": {"AAPL": 1.0}}, allocation)
        assert held["AAPL"] == pytest.approx(1.0)

    def test_opposing_strategies_net_to_flat(self) -> None:
        """Two orders to arrive at nothing, paying twice for the privilege —
        invisible from inside either strategy."""
        allocation = allocate(["a", "b"], method="equal")
        held = net_positions({"a": {"AAPL": 1.0}, "b": {"AAPL": -1.0}}, allocation)
        assert "AAPL" not in held

    def test_allocation_scales_each_strategy(self) -> None:
        allocation = allocate(["a", "b"], method="inverse_volatility",
                              volatilities=[0.1, 0.2])
        held = net_positions({"a": {"X": 1.0}, "b": {"Y": 1.0}}, allocation)
        assert held["X"] == pytest.approx(2 / 3)
        assert held["Y"] == pytest.approx(1 / 3)

    def test_an_unfunded_strategy_does_not_reach_the_market(self) -> None:
        allocation = Allocation(
            strategies=("a", "b"), weights=np.array([1.0, 0.0]), method="manual"
        )
        held = net_positions({"a": {"X": 1.0}, "b": {"Y": 1.0}}, allocation)
        assert "Y" not in held

    def test_a_strategy_with_no_allocation_is_refused(self) -> None:
        """Silently trading an unallocated strategy is how a desk runs a book
        nobody sized."""
        allocation = allocate(["a"], method="equal")
        with pytest.raises(AllocationError, match="hold no allocation"):
            net_positions({"a": {"X": 1.0}, "rogue": {"Y": 1.0}}, allocation)

    def test_works_without_an_allocation(self) -> None:
        held = net_positions({"a": {"X": 0.5}, "b": {"X": 0.5}})
        assert held["X"] == pytest.approx(1.0)


class TestCrossStrategyRisk:
    def test_internalisation_detects_strategies_trading_against_each_other(self) -> None:
        allocation = allocate(["a", "b"], method="equal")
        risk = assess_cross_strategy(
            {"a": {"AAPL": 1.0}, "b": {"AAPL": -1.0}}, allocation
        )
        assert risk.internalisation == pytest.approx(1.0)
        assert "trading against each other" in risk.render()

    def test_disjoint_books_internalise_nothing(self) -> None:
        allocation = allocate(["a", "b"], method="equal")
        risk = assess_cross_strategy({"a": {"X": 1.0}, "b": {"Y": 1.0}}, allocation)
        assert risk.internalisation == pytest.approx(0.0)
        assert risk.contested == []

    def test_contested_symbols_are_named(self) -> None:
        allocation = allocate(["a", "b"], method="equal")
        risk = assess_cross_strategy(
            {"a": {"AAPL": 1.0, "MSFT": 1.0}, "b": {"AAPL": 1.0}}, allocation
        )
        assert risk.contested == ["AAPL"]

    def test_gross_and_net_differ_for_a_long_short_desk(self) -> None:
        allocation = allocate(["a", "b"], method="equal")
        risk = assess_cross_strategy(
            {"a": {"X": 1.0}, "b": {"Y": -1.0}}, allocation
        )
        assert risk.gross_held == pytest.approx(1.0)
        assert risk.net_held == pytest.approx(0.0)
