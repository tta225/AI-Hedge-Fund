"""Options pricing, greeks, and the structure catalogue.

The put-call parity test is the load-bearing one: it checks the pricing formula
against an arbitrage identity rather than against a number someone typed in, so
it catches sign and discounting errors that a hardcoded expected value would
happily agree with.

`TestNoFabrication` pins the wall around expected return. The platform has no
options chain, so any statement about what a structure *earns* would be
fabricated from a model volatility. That refusal is a feature and is tested as
one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axiom.options import (
    CATALOGUE,
    ChainRequiredError,
    Direction,
    OptionType,
    VolatilityView,
    black_scholes,
    build,
    by_outlook,
    greeks,
    implied_volatility,
    realised_volatility,
    undefined_risk,
)

SPOT = 100.0
VOL = 0.25
RATE = 0.04
T = 0.5


class TestPricing:
    def test_put_call_parity(self) -> None:
        """C − P = S·e^(−qT) − K·e^(−rT). An identity, not a fitted value."""
        for strike in (80.0, 100.0, 125.0):
            call = black_scholes(
                spot=SPOT, strike=strike, time_to_expiry=T, volatility=VOL,
                rate=RATE, option_type=OptionType.CALL,
            )
            put = black_scholes(
                spot=SPOT, strike=strike, time_to_expiry=T, volatility=VOL,
                rate=RATE, option_type=OptionType.PUT,
            )
            expected = SPOT - strike * math.exp(-RATE * T)
            assert call - put == pytest.approx(expected, abs=1e-9)

    def test_price_is_intrinsic_at_expiry(self) -> None:
        for strike, option_type, expected in (
            (90.0, OptionType.CALL, 10.0),
            (110.0, OptionType.CALL, 0.0),
            (110.0, OptionType.PUT, 10.0),
            (90.0, OptionType.PUT, 0.0),
        ):
            assert black_scholes(
                spot=SPOT, strike=strike, time_to_expiry=0.0, volatility=VOL,
                option_type=option_type,
            ) == pytest.approx(expected)

    def test_price_rises_with_volatility(self) -> None:
        low = black_scholes(spot=SPOT, strike=SPOT, time_to_expiry=T, volatility=0.1)
        high = black_scholes(spot=SPOT, strike=SPOT, time_to_expiry=T, volatility=0.5)
        assert high > low

    def test_rejects_nonsense_inputs(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            black_scholes(spot=-1.0, strike=100.0, time_to_expiry=T, volatility=VOL)


class TestGreeks:
    def test_delta_bounds_and_sign(self) -> None:
        call = greeks(spot=SPOT, strike=SPOT, time_to_expiry=T, volatility=VOL,
                      rate=RATE, option_type=OptionType.CALL)
        put = greeks(spot=SPOT, strike=SPOT, time_to_expiry=T, volatility=VOL,
                     rate=RATE, option_type=OptionType.PUT)
        assert 0.0 < call.delta < 1.0
        assert -1.0 < put.delta < 0.0
        # Parity again: call delta − put delta = e^(−qT) = 1 with no dividend.
        assert call.delta - put.delta == pytest.approx(1.0, abs=1e-9)

    def test_delta_matches_a_numerical_derivative(self) -> None:
        """Guards against an algebra slip the parity test would not catch."""
        h = 1e-4
        up = black_scholes(spot=SPOT + h, strike=105.0, time_to_expiry=T, volatility=VOL, rate=RATE)
        down = black_scholes(spot=SPOT - h, strike=105.0, time_to_expiry=T, volatility=VOL, rate=RATE)
        analytic = greeks(spot=SPOT, strike=105.0, time_to_expiry=T,
                          volatility=VOL, rate=RATE).delta
        assert analytic == pytest.approx((up - down) / (2 * h), abs=1e-5)

    def test_gamma_and_vega_are_positive_for_both_types(self) -> None:
        for option_type in OptionType:
            g = greeks(spot=SPOT, strike=SPOT, time_to_expiry=T, volatility=VOL,
                       rate=RATE, option_type=option_type)
            assert g.gamma > 0
            assert g.vega > 0

    def test_theta_is_per_day_not_per_year(self) -> None:
        """A per-year theta is ~365× larger and silently wrong in every report."""
        g = greeks(spot=SPOT, strike=SPOT, time_to_expiry=T, volatility=VOL, rate=RATE)
        assert -1.0 < g.theta < 0.0

    def test_greeks_at_expiry_collapse(self) -> None:
        g = greeks(spot=SPOT, strike=90.0, time_to_expiry=0.0, volatility=VOL)
        assert g.delta == 1.0
        assert g.gamma == 0.0 and g.vega == 0.0 and g.theta == 0.0


class TestImpliedVolatility:
    def test_round_trips(self) -> None:
        for strike in (85.0, 100.0, 115.0):
            price = black_scholes(spot=SPOT, strike=strike, time_to_expiry=T,
                                  volatility=0.33, rate=RATE)
            recovered = implied_volatility(
                price=price, spot=SPOT, strike=strike, time_to_expiry=T, rate=RATE
            )
            assert recovered == pytest.approx(0.33, abs=1e-4)

    def test_converges_deep_out_of_the_money(self) -> None:
        """Where Newton-Raphson diverges — the reason bisection is used."""
        price = black_scholes(spot=SPOT, strike=400.0, time_to_expiry=T,
                              volatility=0.6, rate=RATE)
        recovered = implied_volatility(
            price=price, spot=SPOT, strike=400.0, time_to_expiry=T, rate=RATE
        )
        assert recovered == pytest.approx(0.6, abs=1e-3)

    def test_returns_nan_outside_arbitrage_bounds(self) -> None:
        below_intrinsic = implied_volatility(
            price=1.0, spot=SPOT, strike=50.0, time_to_expiry=T, rate=RATE
        )
        assert math.isnan(below_intrinsic)

    def test_realised_volatility(self) -> None:
        rng = np.random.default_rng(4)
        daily = 0.02
        closes = 100 * np.exp(np.cumsum(rng.normal(0, daily, 4000)))
        assert realised_volatility(closes) == pytest.approx(
            daily * math.sqrt(252), rel=0.1
        )


class TestCatalogue:
    def test_every_structure_has_legs_and_a_thesis(self) -> None:
        for key, spec in CATALOGUE.items():
            assert spec.key == key
            assert spec.legs, f"{key} has no legs"
            assert spec.thesis.strip(), f"{key} has no thesis"

    def test_undefined_risk_structures_are_flagged(self) -> None:
        """The field that stops a ranking putting a naked strangle on top."""
        flagged = {s.key for s in undefined_risk()}
        assert "short-call" in flagged
        assert "short-straddle" in flagged
        assert "short-strangle" in flagged
        assert "iron-condor" not in flagged

    def test_undefined_risk_structures_carry_a_caveat(self) -> None:
        for spec in undefined_risk():
            assert spec.caveat.strip(), f"{spec.key} has unbounded loss and no caveat"

    def test_filtering_by_outlook(self) -> None:
        bullish = by_outlook(Direction.BULLISH)
        assert {s.key for s in bullish} >= {"long-call", "bull-call-spread"}
        defined = by_outlook(Direction.NEUTRAL, defined_risk_only=True)
        assert all(s.defined_risk for s in defined)
        short_vol = by_outlook(volatility_view=VolatilityView.SHORT_VOL)
        assert "iron-condor" in {s.key for s in short_vol}


class TestPayoffs:
    def test_long_call_payoff_geometry(self) -> None:
        strategy = build("long-call", spot=SPOT, volatility=VOL)
        premium = strategy.net_premium()
        assert premium > 0, "a long call is a debit"
        # Far below the strike, the loss is exactly the premium paid.
        assert strategy.payoff([50.0])[0] == pytest.approx(-premium)
        # Far above it, profit grows one-for-one with the underlying.
        assert strategy.payoff([200.0])[0] == pytest.approx(
            (200.0 - SPOT) * 100 - premium
        )

    def test_iron_condor_is_a_credit_with_bounded_loss(self) -> None:
        strategy = build("iron-condor", spot=SPOT, volatility=VOL)
        assert strategy.net_premium() < 0, "an iron condor opens for a credit"
        max_loss = strategy.max_loss()
        assert math.isfinite(max_loss) and max_loss < 0
        # Width of one wing, less the credit received.
        credit = -strategy.net_premium()
        wing = (0.95 - 0.90) * SPOT * 100
        assert max_loss == pytest.approx(-(wing - credit), rel=0.02)

    def test_iron_condor_pays_the_full_credit_inside_the_short_strikes(self) -> None:
        strategy = build("iron-condor", spot=SPOT, volatility=VOL)
        assert strategy.payoff([SPOT])[0] == pytest.approx(-strategy.net_premium())

    def test_butterfly_peaks_at_the_body(self) -> None:
        strategy = build("long-call-butterfly", spot=SPOT, volatility=VOL)
        at_body = strategy.payoff([SPOT])[0]
        assert at_body > strategy.payoff([SPOT * 0.90])[0]
        assert at_body > strategy.payoff([SPOT * 1.10])[0]

    def test_straddle_has_two_breakevens(self) -> None:
        assert len(build("long-straddle", spot=SPOT, volatility=VOL).breakevens()) == 2

    def test_short_structures_report_unbounded_loss(self) -> None:
        assert math.isinf(build("short-call", spot=SPOT, volatility=VOL).max_loss())
        assert math.isinf(build("short-straddle", spot=SPOT, volatility=VOL).max_loss())

    def test_synthetic_long_tracks_the_underlying(self) -> None:
        """Put-call parity expressed as a payoff, not a price."""
        strategy = build("synthetic-long", spot=SPOT, volatility=VOL)
        gained = strategy.payoff([120.0])[0] - strategy.payoff([100.0])[0]
        assert gained == pytest.approx(20.0 * 100, rel=1e-6)

    def test_box_spread_payoff_is_flat(self) -> None:
        """A box pays the strike width regardless of the underlying."""
        strategy = build("box-spread", spot=SPOT, volatility=VOL)
        values = strategy.payoff([60.0, 100.0, 140.0])
        assert values.max() - values.min() == pytest.approx(0.0, abs=1e-6)

    def test_credit_structures_open_for_a_credit(self) -> None:
        for key, spec in CATALOGUE.items():
            if not spec.net_credit or any(leg.is_underlying for leg in spec.legs):
                continue
            if len({leg.expiry_years for leg in spec.legs}) > 1:
                continue  # calendars price against a term structure this lacks
            premium = build(key, spot=SPOT, volatility=VOL).net_premium()
            assert premium < 0, f"{key} is marked net_credit but prices as a debit"


class TestGreeksOfStructures:
    def test_short_premium_structures_are_short_vega_and_long_theta(self) -> None:
        for key in ("iron-condor", "short-strangle", "iron-butterfly"):
            g = build(key, spot=SPOT, volatility=VOL).greeks()
            assert g.vega < 0, f"{key} should be short vega"
            assert g.theta > 0, f"{key} should collect theta"

    def test_long_premium_structures_are_long_vega_and_short_theta(self) -> None:
        for key in ("long-straddle", "long-strangle", "long-call"):
            g = build(key, spot=SPOT, volatility=VOL).greeks()
            assert g.vega > 0, f"{key} should be long vega"
            assert g.theta < 0, f"{key} should pay theta"

    def test_neutral_structures_are_near_delta_flat(self) -> None:
        for key in ("long-straddle", "iron-condor"):
            g = build(key, spot=SPOT, volatility=VOL).greeks()
            assert abs(g.delta) < 20.0, f"{key} is not delta-neutral"


class TestNoFabrication:
    def test_expected_return_refuses_rather_than_inventing_one(self) -> None:
        """The wall this module is built around.

        Without a chain, the premium — the whole trade — would have to be
        invented from a model volatility. Refusing is the correct behaviour and
        this test exists so nobody quietly replaces it with a number.
        """
        strategy = build("iron-condor", spot=SPOT, volatility=VOL)
        with pytest.raises(ChainRequiredError, match="options chain"):
            strategy.expected_return()

    def test_unknown_structure_names_the_alternatives(self) -> None:
        with pytest.raises(KeyError, match="unknown structure"):
            build("no-such-structure", spot=SPOT, volatility=VOL)

    def test_summary_labels_the_volatility_as_a_model_input(self) -> None:
        text = build("iron-condor", spot=SPOT, volatility=VOL).summary()
        assert "MODEL" in text, "the summary must not present model vol as a quote"
