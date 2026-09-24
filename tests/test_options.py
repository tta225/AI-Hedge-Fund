"""Options pricing and volatility strategies.

Black-Scholes is checked against identities rather than against remembered
numbers: put-call parity, the delta bounds, and the sign of theta hold for
every valid input, so an identity test catches a class of error a golden number
cannot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axiom.options.pricing import OptionType, black_scholes, implied_volatility
from axiom.options.strategies import (
    VolQuote,
    dispersion,
    realised_volatility,
    volatility_arbitrage,
)

CALL, PUT = OptionType.CALL, OptionType.PUT


class TestBlackScholesIdentities:
    @pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
    @pytest.mark.parametrize("vol", [0.1, 0.25, 0.6])
    def test_put_call_parity(self, spot: float, vol: float) -> None:
        """C - P = S·e^{-qT} - K·e^{-rT}, for every input."""
        strike, t, r, q = 100.0, 0.75, 0.04, 0.02
        call = black_scholes(spot, strike, t, r, vol, CALL, dividend_yield=q).price
        put = black_scholes(spot, strike, t, r, vol, PUT, dividend_yield=q).price
        expected = spot * math.exp(-q * t) - strike * math.exp(-r * t)
        assert call - put == pytest.approx(expected, abs=1e-9)

    def test_call_delta_is_bounded_by_zero_and_one(self) -> None:
        for spot in (50.0, 100.0, 200.0):
            delta = black_scholes(spot, 100.0, 1.0, 0.03, 0.2, CALL).delta
            assert 0.0 <= delta <= 1.0

    def test_put_delta_is_bounded_by_minus_one_and_zero(self) -> None:
        for spot in (50.0, 100.0, 200.0):
            delta = black_scholes(spot, 100.0, 1.0, 0.03, 0.2, PUT).delta
            assert -1.0 <= delta <= 0.0

    def test_gamma_and_vega_match_across_call_and_put(self) -> None:
        """Both are second-order in the same underlying distribution."""
        call = black_scholes(100.0, 100.0, 1.0, 0.03, 0.2, CALL)
        put = black_scholes(100.0, 100.0, 1.0, 0.03, 0.2, PUT)
        assert call.gamma == pytest.approx(put.gamma)
        assert call.vega == pytest.approx(put.vega)

    def test_price_rises_with_volatility(self) -> None:
        low = black_scholes(100.0, 100.0, 1.0, 0.03, 0.10, CALL).price
        high = black_scholes(100.0, 100.0, 1.0, 0.03, 0.40, CALL).price
        assert high > low

    def test_a_long_option_loses_value_with_time(self) -> None:
        assert black_scholes(100.0, 100.0, 0.5, 0.03, 0.2, CALL).theta < 0

    def test_vega_is_quoted_per_volatility_point(self) -> None:
        """Vega must be the move for a 1% vol change, not for a 100% one."""
        greeks = black_scholes(100.0, 100.0, 1.0, 0.03, 0.20, CALL)
        bumped = black_scholes(100.0, 100.0, 1.0, 0.03, 0.21, CALL)
        assert greeks.vega == pytest.approx(bumped.price - greeks.price, rel=0.02)

    def test_gamma_matches_a_numerical_delta_bump(self) -> None:
        base = black_scholes(100.0, 100.0, 1.0, 0.03, 0.2, CALL)
        up = black_scholes(100.01, 100.0, 1.0, 0.03, 0.2, CALL)
        assert base.gamma == pytest.approx((up.delta - base.delta) / 0.01, rel=0.01)


class TestExpiryHandling:
    def test_an_expiring_itm_call_has_delta_one(self) -> None:
        """The original returned 0 here — a book would think it was flat at
        exactly the moment its exposure was largest."""
        assert black_scholes(120.0, 100.0, 0.0, 0.03, 0.2, CALL).delta == 1.0

    def test_an_expiring_itm_put_has_delta_minus_one(self) -> None:
        assert black_scholes(80.0, 100.0, 0.0, 0.03, 0.2, PUT).delta == -1.0

    def test_an_expiring_otm_option_has_delta_zero(self) -> None:
        assert black_scholes(80.0, 100.0, 0.0, 0.03, 0.2, CALL).delta == 0.0

    def test_an_expired_option_is_worth_its_intrinsic(self) -> None:
        assert black_scholes(120.0, 100.0, 0.0, 0.03, 0.2, CALL).price == pytest.approx(20.0)

    def test_zero_volatility_is_treated_as_deterministic(self) -> None:
        assert black_scholes(120.0, 100.0, 1.0, 0.0, 0.0, CALL).price == pytest.approx(20.0)


class TestInvalidInputs:
    def test_a_non_positive_spot_raises_rather_than_returning_zeros(self) -> None:
        """Zeros would present a missing answer as a flat position."""
        with pytest.raises(ValueError, match="undefined"):
            black_scholes(0.0, 100.0, 1.0, 0.03, 0.2, CALL)

    def test_a_non_positive_strike_raises(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            black_scholes(100.0, -1.0, 1.0, 0.03, 0.2, CALL)

    def test_a_negative_volatility_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            black_scholes(100.0, 100.0, 1.0, 0.03, -0.2, CALL)


class TestImpliedVolatility:
    @pytest.mark.parametrize("vol", [0.08, 0.2, 0.45, 1.2])
    def test_it_inverts_the_pricer(self, vol: float) -> None:
        price = black_scholes(100.0, 105.0, 0.6, 0.03, vol, CALL).price
        assert implied_volatility(price, 100.0, 105.0, 0.6, 0.03, CALL) == pytest.approx(
            vol, abs=1e-4
        )

    def test_an_unattainable_price_returns_none_not_a_clamp(self) -> None:
        """A clamped number here is an invented volatility feeding vol arb."""
        assert implied_volatility(1e6, 100.0, 100.0, 1.0, 0.03, CALL) is None

    def test_a_zero_price_returns_none(self) -> None:
        assert implied_volatility(0.0, 100.0, 100.0, 1.0, 0.03, CALL) is None


class TestRealisedVolatility:
    def test_it_annualises(self) -> None:
        rng = np.random.default_rng(0)
        daily = 0.01
        closes = 100 * np.exp(np.cumsum(rng.normal(0, daily, 2000)))
        assert realised_volatility(closes) == pytest.approx(
            daily * math.sqrt(252), rel=0.1
        )

    def test_too_little_data_returns_none(self) -> None:
        """A vol from three observations is not a vol; it manufactures edge."""
        assert realised_volatility(np.array([100.0, 101.0, 99.0])) is None

    def test_non_positive_prices_return_none(self) -> None:
        assert realised_volatility(np.zeros(50)) is None


class TestVolatilityArbitrage:
    def test_rich_implied_vol_is_a_sell(self) -> None:
        signals = volatility_arbitrage(
            [VolQuote(100.0, CALL, implied_vol=0.35)], realised_vol=0.20
        )
        assert signals[0].direction == -1

    def test_cheap_implied_vol_is_a_buy(self) -> None:
        signals = volatility_arbitrage(
            [VolQuote(100.0, CALL, implied_vol=0.10)], realised_vol=0.25
        )
        assert signals[0].direction == 1

    def test_the_spread_is_subtracted_from_the_edge(self) -> None:
        """The original never crossed the spread, so a 6-point edge on an
        option quoted 5 points wide read as tradable."""
        signals = volatility_arbitrage(
            [VolQuote(100.0, CALL, implied_vol=0.26, vol_half_spread=0.05)],
            realised_vol=0.20,
        )
        assert signals[0].edge_vol == pytest.approx(0.06)
        assert signals[0].net_edge_vol == pytest.approx(0.01)
        assert signals[0].is_tradable

    def test_an_edge_smaller_than_the_spread_is_not_tradable(self) -> None:
        signals = volatility_arbitrage(
            [VolQuote(100.0, CALL, implied_vol=0.26, vol_half_spread=0.10)],
            realised_vol=0.20,
        )
        assert not signals[0].is_tradable

    def test_a_small_edge_is_ignored(self) -> None:
        assert volatility_arbitrage(
            [VolQuote(100.0, CALL, implied_vol=0.21)], realised_vol=0.20
        ) == []

    def test_open_interest_can_be_required(self) -> None:
        assert volatility_arbitrage(
            [VolQuote(100.0, CALL, implied_vol=0.40, open_interest=3)],
            realised_vol=0.20,
            min_open_interest=100,
        ) == []


class TestDispersion:
    def test_a_perfectly_correlated_basket_implies_correlation_one(self) -> None:
        """Every name at 20% vol, equal weights, index at 20% ⇒ ρ = 1."""
        view = dispersion(0.20, [0.20] * 4, [0.25] * 4)
        assert view.implied_correlation == pytest.approx(1.0, abs=1e-9)

    def test_an_uncorrelated_basket_implies_correlation_zero(self) -> None:
        vols, weights = [0.20] * 4, [0.25] * 4
        index_vol = math.sqrt(sum(w**2 * v**2 for w, v in zip(weights, vols, strict=True)))
        assert dispersion(index_vol, vols, weights).implied_correlation == pytest.approx(
            0.0, abs=1e-9
        )

    def test_the_original_formula_would_have_reported_one_here(self) -> None:
        """The uploaded version divided index variance by the *zero*-correlation
        basket variance and clamped at 1, so an uncorrelated basket reported
        ρ=1.0 and 'SELL_INDEX' — the clamp hid the error."""
        vols, weights = [0.20] * 4, [0.25] * 4
        index_vol = math.sqrt(sum(w**2 * v**2 for w, v in zip(weights, vols, strict=True)))
        original = min(index_vol**2 / (index_vol**2 + 1e-10), 1.0)
        view = dispersion(index_vol, vols, weights)
        assert original == pytest.approx(1.0, abs=1e-6)
        assert view.implied_correlation == pytest.approx(0.0, abs=1e-9)

    def test_high_correlation_triggers_the_dispersion_trade(self) -> None:
        assert dispersion(0.195, [0.20] * 4, [0.25] * 4).signal.startswith("SELL_INDEX")

    def test_an_impossible_correlation_is_surfaced_not_clamped(self) -> None:
        """Above 1 means the quotes disagree — a data problem, not a signal."""
        view = dispersion(0.50, [0.20] * 4, [0.25] * 4)
        assert view.signal == "NO_VIEW"
        assert any("exceeds 1.0" in w for w in view.warnings)

    def test_weights_that_do_not_sum_to_one_are_flagged(self) -> None:
        view = dispersion(0.20, [0.20] * 4, [0.10] * 4)
        assert any("sum to" in w for w in view.warnings)

    def test_mismatched_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same basket"):
            dispersion(0.2, [0.2, 0.2], [0.5])

    def test_an_empty_basket_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty basket"):
            dispersion(0.2, [], [])

    def test_a_degenerate_basket_reports_no_view(self) -> None:
        """One name at weight 1 identifies no correlation at all."""
        view = dispersion(0.20, [0.20], [1.0])
        assert view.signal == "NO_VIEW"
        assert any("degenerate" in w for w in view.warnings)
