"""Black-Scholes pricing and Greeks.

Ported from the uploaded architecture's ``BlackScholes.greeks`` with three
corrections. The original's put theta used ``norm.cdf(-d1 + sigma*sqrt(T))``,
which is ``N(-d2)`` written in a way that is easy to misread, and its call
theta used ``norm.cdf(d1 - sigma*sqrt(T))`` — correct, but the asymmetry
between the two lines is exactly where sign errors live. Both are written here
in terms of an explicit ``d2``.

The original also returned zeros for every Greek at ``T <= 0`` including delta,
which is wrong in the way that matters most: an expiring in-the-money call has
delta 1, not 0, and a book that believed otherwise would think it was flat at
the moment it was most exposed. Expiry is handled explicitly.

Rho is added because a rates-sensitive book cannot hedge what it does not
measure, and the original omitted it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

#: 1/sqrt(2π), the standard normal density's normalising constant.
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf``.

    Deliberately not ``scipy.stats.norm``. Black-Scholes needs exactly these two
    functions, and the error function is in the standard library to machine
    precision — taking a 30MB scientific-stack dependency for them would make
    the whole options module unavailable anywhere scipy is not installed.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class Greeks:
    """Sensitivities, in the conventional per-unit quoting of each.

    ``vega`` is per 1 volatility point (0.01), ``theta`` is per calendar day,
    and ``rho`` is per 1 rate point — the quoting a trader reads, not the raw
    partial derivatives, because raw units invite off-by-100 hedging errors.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def render(self) -> str:
        return (
            f"px {self.price:>9.4f}  Δ {self.delta:+.4f}  Γ {self.gamma:+.6f}  "
            f"ν {self.vega:+.4f}  Θ {self.theta:+.4f}  ρ {self.rho:+.4f}"
        )


def black_scholes(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    *,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Price and Greeks for a European option.

    Args:
        time_to_expiry: in years. Zero or negative is treated as expired.
        rate: continuously compounded risk-free rate.
        volatility: annualised, as a decimal (0.20 for 20%).
        dividend_yield: continuous yield. Non-zero matters for index options
            and for anything held across a dividend; the original omitted it,
            which biases delta on dividend-paying underlyings.

    Raises:
        ValueError: for a non-positive spot or strike, where the model's
            logarithm is undefined. Returning zeros there would present a
            missing answer as a flat position.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError(
            f"spot={spot} strike={strike}: Black-Scholes is undefined at or "
            "below zero; the log-moneyness does not exist"
        )
    if volatility < 0:
        raise ValueError(f"volatility={volatility} is negative")

    is_call = option_type is OptionType.CALL

    # At or past expiry the option is its intrinsic value, and delta is the
    # indicator of moneyness. Reporting delta 0 here would tell a book it was
    # flat at the moment its exposure is largest and most binary.
    if time_to_expiry <= 0 or volatility == 0:
        forward = spot * math.exp(-dividend_yield * max(time_to_expiry, 0.0))
        discounted = strike * math.exp(-rate * max(time_to_expiry, 0.0))
        intrinsic = max(forward - discounted, 0.0) if is_call else max(discounted - forward, 0.0)
        in_the_money = (spot > strike) if is_call else (spot < strike)
        return Greeks(
            price=intrinsic,
            delta=(1.0 if is_call else -1.0) if in_the_money else 0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
        )

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + volatility**2 / 2.0) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discount = math.exp(-rate * time_to_expiry)
    carry = math.exp(-dividend_yield * time_to_expiry)
    pdf_d1 = _norm_pdf(d1)

    gamma = carry * pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * carry * pdf_d1 * sqrt_t / 100.0

    if is_call:
        price = spot * carry * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        delta = carry * _norm_cdf(d1)
        theta = (
            -spot * carry * pdf_d1 * volatility / (2 * sqrt_t)
            - rate * strike * discount * _norm_cdf(d2)
            + dividend_yield * spot * carry * _norm_cdf(d1)
        ) / 365.0
        rho = strike * time_to_expiry * discount * _norm_cdf(d2) / 100.0
    else:
        price = strike * discount * _norm_cdf(-d2) - spot * carry * _norm_cdf(-d1)
        delta = -carry * _norm_cdf(-d1)
        theta = (
            -spot * carry * pdf_d1 * volatility / (2 * sqrt_t)
            + rate * strike * discount * _norm_cdf(-d2)
            - dividend_yield * spot * carry * _norm_cdf(-d1)
        ) / 365.0
        rho = -strike * time_to_expiry * discount * _norm_cdf(-d2) / 100.0

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType,
    *,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float | None:
    """Invert Black-Scholes for volatility by bisection.

    Bisection rather than Newton-Raphson: vega collapses toward zero for deep
    in- and out-of-the-money options, and a Newton step divided by a near-zero
    vega diverges precisely on the wings where quotes are least reliable.
    Bisection is slower and cannot diverge.

    Returns:
        The implied volatility, or ``None`` when the price is outside the
        no-arbitrage bounds the model can produce. ``None`` is deliberate — a
        clamped number here would be an invented volatility feeding a vol-arb
        signal, which is worse than an absent one.
    """
    if market_price <= 0 or time_to_expiry <= 0:
        return None

    low, high = 1e-6, 5.0
    price_low = black_scholes(
        spot, strike, time_to_expiry, rate, low, option_type,
        dividend_yield=dividend_yield,
    ).price
    price_high = black_scholes(
        spot, strike, time_to_expiry, rate, high, option_type,
        dividend_yield=dividend_yield,
    ).price
    if not price_low <= market_price <= price_high:
        return None

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        price = black_scholes(
            spot, strike, time_to_expiry, rate, mid, option_type,
            dividend_yield=dividend_yield,
        ).price
        if abs(price - market_price) < tolerance:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0
