"""Black-Scholes-Merton pricing and greeks.

Scope and honesty
-----------------
This prices European options on a lognormal underlying with constant
volatility. That model is **wrong** in ways that matter for every strategy in
`axiom.options.catalogue`:

* Real implied volatility is not constant across strikes. The skew is the single
  largest source of error here, and it is precisely what most of the catalogued
  spreads are trading. A butterfly priced at flat vol is mispriced by roughly the
  curvature of the smile.
* American exercise is not modelled. For puts on a dividend-paying underlying
  the early-exercise premium is real.
* Dividends enter only as a continuous yield.

Use this for **scenario analysis and structure comparison** — what a spread pays
at expiry, how its delta behaves, what happens if vol moves. Do not use it to
claim an option is cheap or rich. That claim requires a market quote, and the
platform has no options feed. See `axiom.options.catalogue`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["Greeks", "OptionType", "black_scholes", "greeks", "implied_volatility"]


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """+1 for a call, −1 for a put. Lets one formula serve both."""
        return 1 if self is OptionType.CALL else -1


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order sensitivities.

    Units follow market convention, not calculus:

    * `theta` is **per calendar day**, not per year.
    * `vega` is per **1 volatility point** (0.01), not per unit of vol.
    * `rho` is per **1% rate move**.

    Reporting raw per-year derivatives is technically correct and operationally
    useless — a theta of −73 means nothing to anyone until it is −0.20/day.
    """

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def scaled(self, quantity: float, multiplier: float = 100.0) -> Greeks:
        """Position greeks for `quantity` contracts of `multiplier` shares each."""
        factor = quantity * multiplier
        return Greeks(
            price=self.price * factor,
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            theta=self.theta * factor,
            vega=self.vega * factor,
            rho=self.rho * factor,
        )

    def __add__(self, other: Greeks) -> Greeks:
        return Greeks(
            price=self.price + other.price,
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
        )


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(
    spot: float, strike: float, time: float, vol: float, rate: float, yield_: float
) -> tuple[float, float]:
    variance = vol * math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate - yield_ + 0.5 * vol * vol) * time) / variance
    return d1, d1 - variance


def black_scholes(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Price a European option. `time_to_expiry` is in years.

    At or past expiry, and at zero volatility, the price is intrinsic value —
    the limit the formula approaches but cannot evaluate (both divide by
    `vol * sqrt(t)`).
    """
    if spot <= 0 or strike <= 0:
        raise ValueError(f"spot and strike must be positive, got {spot}, {strike}")
    intrinsic = max(option_type.sign * (spot - strike), 0.0)
    if time_to_expiry <= 0 or volatility <= 0:
        return intrinsic

    d1, d2 = _d1_d2(spot, strike, time_to_expiry, volatility, rate, dividend_yield)
    discount = math.exp(-rate * time_to_expiry)
    carry = math.exp(-dividend_yield * time_to_expiry)
    sign = option_type.sign
    return sign * (
        spot * carry * _norm_cdf(sign * d1) - strike * discount * _norm_cdf(sign * d2)
    )


def greeks(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> Greeks:
    """Price and greeks, in market-convention units (see `Greeks`)."""
    price = black_scholes(
        spot=spot, strike=strike, time_to_expiry=time_to_expiry,
        volatility=volatility, rate=rate, dividend_yield=dividend_yield,
        option_type=option_type,
    )
    if time_to_expiry <= 0 or volatility <= 0:
        # At expiry delta is a step function and every other greek is zero.
        itm = option_type.sign * (spot - strike) > 0
        return Greeks(
            price=price,
            delta=float(option_type.sign) if itm else 0.0,
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
        )

    d1, d2 = _d1_d2(spot, strike, time_to_expiry, volatility, rate, dividend_yield)
    discount = math.exp(-rate * time_to_expiry)
    carry = math.exp(-dividend_yield * time_to_expiry)
    sign = option_type.sign
    pdf = _norm_pdf(d1)
    root_t = math.sqrt(time_to_expiry)

    delta = sign * carry * _norm_cdf(sign * d1)
    gamma = carry * pdf / (spot * volatility * root_t)
    vega = spot * carry * pdf * root_t
    theta_year = (
        -spot * carry * pdf * volatility / (2 * root_t)
        - sign * rate * strike * discount * _norm_cdf(sign * d2)
        + sign * dividend_yield * spot * carry * _norm_cdf(sign * d1)
    )
    rho = sign * strike * time_to_expiry * discount * _norm_cdf(sign * d2)

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta_year / 365.0,   # per calendar day
        vega=vega / 100.0,          # per 1 vol point
        rho=rho / 100.0,            # per 1% rate move
    )


def implied_volatility(
    *,
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """Invert Black-Scholes for volatility by bisection.

    Bisection rather than Newton-Raphson deliberately. Newton is faster but
    diverges for deep in- or out-of-the-money options, where vega approaches
    zero and the update step explodes. Those are exactly the strikes on the
    wings of a condor or butterfly, so the fast method fails precisely where
    this gets used. Bisection is monotone and cannot diverge.

    Returns NaN when the price is outside the no-arbitrage bounds, rather than
    an arbitrary number that would propagate into a greek.
    """
    intrinsic = max(option_type.sign * (spot - strike), 0.0)
    if price < intrinsic - tolerance or time_to_expiry <= 0:
        return float("nan")

    low, high = 1e-6, 5.0
    price_high = black_scholes(
        spot=spot, strike=strike, time_to_expiry=time_to_expiry, volatility=high,
        rate=rate, dividend_yield=dividend_yield, option_type=option_type,
    )
    if price > price_high:
        return float("nan")  # richer than 500% vol — not a real quote

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        value = black_scholes(
            spot=spot, strike=strike, time_to_expiry=time_to_expiry, volatility=mid,
            rate=rate, dividend_yield=dividend_yield, option_type=option_type,
        )
        if abs(value - price) < tolerance:
            return mid
        if value > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def realised_volatility(closes: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised close-to-close realised volatility.

    The only volatility input this platform can actually produce from its own
    data. Using it where implied volatility belongs — pricing a spread, judging
    whether a structure is cheap — silently assumes the volatility risk premium
    is zero, which is the one thing every options income strategy is betting it
    is not.
    """
    if len(closes) < 3:
        return float("nan")
    returns = np.diff(np.log(np.maximum(closes, 1e-12)))
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))
