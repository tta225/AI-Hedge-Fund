"""The options strategy catalogue — structures, payoffs, and what they bet on.

Read this first: what this can and cannot do
--------------------------------------------
This module defines option **structures**: their legs, their payoff at expiry,
their breakevens, their maximum profit and loss, and their aggregate greeks
under a supplied volatility.

It **cannot backtest them**, and nothing here should be read as if it could.
Backtesting an options strategy requires a historical **options chain** — every
strike, every expiry, bid and ask, with the implied volatility surface as it
stood at each point in time. The platform has no such feed: `axiom.data` returns
OHLCV bars on the underlying and nothing else. Pricing these structures from a
model volatility and calling the result a backtest would fabricate the single
most important number in the trade — the premium — and every strategy here is
fundamentally a bet about premium.

So the honest division of labour is:

* **Here**: structure definition, payoff geometry, greek profile, scenario
  analysis, and the risk facts that hold regardless of price (defined vs
  undefined risk, assignment exposure, directional and volatility bias).
* **Not here, and not possible without a chain feed**: expected return, win
  rate, Sharpe, or any statement that one structure has outperformed another.

`ChainRequiredError` is raised by anything that would need real quotes. That is a
deliberate wall, not a missing feature.

What "risk" means in the fields below
-------------------------------------
`max_loss` is `inf` for structures with genuinely unbounded loss (short naked
calls) and a finite number for defined-risk structures. An agent or UI that
sorts by expected credit without reading this field will rank a naked strangle
above an iron condor every time, on the strength of the one number that ignores
what can go wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from axiom.options.pricing import Greeks, OptionType, greeks

__all__ = [
    "CATALOGUE",
    "ChainRequiredError",
    "Direction",
    "Leg",
    "OptionStrategy",
    "StructureSpec",
    "VolatilityView",
    "build",
    "by_outlook",
    "catalogue_text",
]


class ChainRequiredError(RuntimeError):
    """Raised when an operation needs real option quotes the platform lacks."""


class Direction(str, Enum):
    """Directional bias of a structure."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    #: Profits from a large move in either direction.
    VOLATILE = "volatile"


class VolatilityView(str, Enum):
    """What the structure needs implied volatility to do."""

    LONG_VOL = "long-vol"
    SHORT_VOL = "short-vol"
    VOL_NEUTRAL = "vol-neutral"


@dataclass(frozen=True, slots=True)
class Leg:
    """One option or stock leg.

    `strike_offset` is expressed in **moneyness relative to spot** (1.0 = at the
    money, 1.05 = 5% out for a call) rather than an absolute strike, so a
    structure definition is instrument-independent. Concrete strikes are
    resolved against a real chain — which is where `ChainRequiredError` bites.
    """

    option_type: OptionType | None  # None = underlying
    quantity: float                 # positive long, negative short
    strike_offset: float = 1.0
    #: Expiry in years, so calendars/diagonals can differ leg to leg.
    expiry_years: float = 30.0 / 365.0

    @property
    def is_underlying(self) -> bool:
        return self.option_type is None

    def describe(self) -> str:
        side = "long" if self.quantity > 0 else "short"
        size = abs(self.quantity)
        count = "" if size == 1 else f"{size:g}× "
        if self.is_underlying:
            return f"{side} {count}underlying"
        return (
            f"{side} {count}{self.strike_offset:.2f}× {self.option_type.value}"  # type: ignore[union-attr]
            f" @ {self.expiry_years * 365:.0f}d"
        )


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """A catalogued structure: what it is and what it bets on."""

    key: str
    name: str
    legs: tuple[Leg, ...]
    direction: Direction
    volatility_view: VolatilityView
    #: The bet, in one sentence.
    thesis: str
    #: True when maximum loss is bounded by construction.
    defined_risk: bool
    #: True when the structure opens for a net credit.
    net_credit: bool
    #: Short options that can be assigned before expiry.
    assignment_risk: bool = False
    tags: tuple[str, ...] = ()
    caveat: str = ""

    @property
    def leg_count(self) -> int:
        return len(self.legs)

    def describe(self) -> str:
        lines = [
            f"{self.key}  —  {self.name}",
            f"  outlook   : {self.direction.value} / {self.volatility_view.value}",
            f"  thesis    : {self.thesis}",
            f"  structure : {'; '.join(leg.describe() for leg in self.legs)}",
            f"  risk      : {'defined' if self.defined_risk else 'UNDEFINED'}"
            f" · opens for a {'credit' if self.net_credit else 'debit'}"
            + (" · assignable" if self.assignment_risk else ""),
        ]
        if self.caveat:
            lines.append(f"  caveat    : {self.caveat}")
        return "\n".join(lines)


@dataclass(slots=True)
class OptionStrategy:
    """A structure resolved against a spot price and a volatility assumption.

    The volatility is an **assumption**, not a market quote. Every number this
    produces inherits that. See the module docstring.
    """

    spec: StructureSpec
    spot: float
    volatility: float
    rate: float = 0.04
    dividend_yield: float = 0.0
    multiplier: float = 100.0
    _strikes: tuple[float, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.volatility <= 0:
            raise ValueError(f"volatility must be positive, got {self.volatility}")
        self._strikes = tuple(
            self.spot * leg.strike_offset for leg in self.spec.legs
        )

    @property
    def strikes(self) -> tuple[float, ...]:
        return self._strikes

    def net_premium(self) -> float:
        """Model cost to open. Negative is a credit.

        A **model** cost. Without a chain this is the theoretical value at the
        assumed volatility, not a price anyone would fill.
        """
        total = 0.0
        for leg, strike in zip(self.spec.legs, self._strikes, strict=True):
            if leg.is_underlying:
                total += leg.quantity * self.spot
                continue
            assert leg.option_type is not None  # guarded by is_underlying above
            g = greeks(
                spot=self.spot, strike=strike, time_to_expiry=leg.expiry_years,
                volatility=self.volatility, rate=self.rate,
                dividend_yield=self.dividend_yield, option_type=leg.option_type,
            )
            total += leg.quantity * g.price
        return total * self.multiplier

    def greeks(self) -> Greeks:
        """Aggregate position greeks."""
        total = Greeks(price=0.0, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        for leg, strike in zip(self.spec.legs, self._strikes, strict=True):
            if leg.is_underlying:
                total = total + Greeks(
                    price=leg.quantity * self.spot * self.multiplier,
                    delta=leg.quantity * self.multiplier,
                    gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
                )
                continue
            assert leg.option_type is not None  # guarded by is_underlying above
            g = greeks(
                spot=self.spot, strike=strike, time_to_expiry=leg.expiry_years,
                volatility=self.volatility, rate=self.rate,
                dividend_yield=self.dividend_yield, option_type=leg.option_type,
            )
            total = total + g.scaled(leg.quantity, self.multiplier)
        return total

    def payoff(self, prices: Sequence[float] | np.ndarray) -> np.ndarray:
        """Profit and loss at expiry across `prices`, net of the model premium.

        Legs with different expiries are evaluated at *their own* intrinsic
        value at the terminal price, which is exact for a single-expiry
        structure and an **approximation for calendars and diagonals** — the
        far leg still has time value when the near one expires. `payoff` is
        therefore not the right tool for a calendar; its greeks are.
        """
        grid = np.asarray(prices, dtype=float)
        total = np.zeros_like(grid)
        for leg, strike in zip(self.spec.legs, self._strikes, strict=True):
            if leg.is_underlying:
                total += leg.quantity * (grid - self.spot)
                continue
            sign = leg.option_type.sign  # type: ignore[union-attr]
            total += leg.quantity * np.maximum(sign * (grid - strike), 0.0)
        return total * self.multiplier - self.net_premium()

    def breakevens(self, span: float = 0.6, points: int = 4001) -> list[float]:
        """Underlying prices where the expiry payoff crosses zero."""
        grid = np.linspace(self.spot * (1 - span), self.spot * (1 + span), points)
        values = self.payoff(grid)
        crossings: list[float] = []
        for k in range(len(values) - 1):
            a, b = values[k], values[k + 1]
            if a == 0.0:
                crossings.append(float(grid[k]))
            elif a * b < 0:
                # Linear interpolation between the bracketing grid points.
                crossings.append(
                    float(grid[k] + (grid[k + 1] - grid[k]) * abs(a) / (abs(a) + abs(b)))
                )
        return crossings

    def max_profit(self, span: float = 3.0, points: int = 6001) -> float:
        grid = np.linspace(max(self.spot * (1 - span), 1e-6), self.spot * (1 + span), points)
        best = float(np.max(self.payoff(grid)))
        # A long call's profit grows without bound; a wide grid only samples it.
        if self.payoff(np.array([grid[-1]]))[0] >= best - 1e-6 and not self.spec.defined_risk:
            return float("inf")
        return best

    def max_loss(self, span: float = 3.0, points: int = 6001) -> float:
        """Worst expiry outcome. `inf` when loss is genuinely unbounded."""
        if not self.spec.defined_risk:
            return float("inf")
        grid = np.linspace(max(self.spot * (1 - span), 1e-6), self.spot * (1 + span), points)
        return float(np.min(self.payoff(grid)))

    def summary(self) -> str:
        g = self.greeks()
        premium = self.net_premium()
        breaks = self.breakevens()
        max_p, max_l = self.max_profit(), self.max_loss()
        return "\n".join(
            [
                self.spec.describe(),
                "",
                f"  spot {self.spot:,.2f} · assumed vol {self.volatility:.1%} "
                f"(MODEL, not a market quote)",
                f"  strikes    : {', '.join(f'{k:,.2f}' for k in self._strikes)}",
                f"  premium    : {premium:,.2f} "
                f"({'credit' if premium < 0 else 'debit'})",
                "  breakevens : "
                + (", ".join(f"{b:,.2f}" for b in breaks) if breaks else "none in range"),
                f"  max profit : {'unbounded' if np.isinf(max_p) else f'{max_p:,.2f}'}",
                f"  max loss   : {'UNBOUNDED' if np.isinf(max_l) else f'{max_l:,.2f}'}",
                f"  delta {g.delta:+.2f}  gamma {g.gamma:+.4f}  "
                f"theta {g.theta:+.2f}/day  vega {g.vega:+.2f}/vol pt",
            ]
        )

    def expected_return(self) -> float:
        """Not available. Raises `ChainRequired`.

        Expected return depends on the premium actually received, which depends
        on the market's implied volatility surface. Computing it from a model
        vol would answer a question nobody asked: "what would this return if the
        market priced options exactly the way my model does" — under which
        assumption the volatility risk premium is zero and every income
        structure here has zero edge by construction.
        """
        raise ChainRequiredError(
            "expected return requires historical option quotes. The platform has "
            "no options chain feed — axiom.data serves OHLCV bars only. See "
            "axiom/options/catalogue.py."
        )


_M = 30.0 / 365.0   # one-month expiry
_Q = 90.0 / 365.0   # three-month expiry
_C = OptionType.CALL
_P = OptionType.PUT


def _spec(**kwargs: object) -> StructureSpec:
    return StructureSpec(**kwargs)  # type: ignore[arg-type]


#: Every catalogued structure, keyed by name.
CATALOGUE: dict[str, StructureSpec] = {
    # ── single leg ──────────────────────────────────────────────────────────
    "long-call": _spec(
        key="long-call", name="Long Call",
        legs=(Leg(_C, 1, 1.00, _M),),
        direction=Direction.BULLISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="Underlying rises more than the premium paid, before expiry.",
        defined_risk=True, net_credit=False,
        tags=("single-leg", "long-premium"),
    ),
    "long-put": _spec(
        key="long-put", name="Long Put",
        legs=(Leg(_P, 1, 1.00, _M),),
        direction=Direction.BEARISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="Underlying falls more than the premium paid, before expiry.",
        defined_risk=True, net_credit=False,
        tags=("single-leg", "long-premium", "hedge"),
    ),
    "short-call": _spec(
        key="short-call", name="Short (Naked) Call",
        legs=(Leg(_C, -1, 1.05, _M),),
        direction=Direction.BEARISH, volatility_view=VolatilityView.SHORT_VOL,
        thesis="Underlying stays below the strike; the premium is kept.",
        defined_risk=False, net_credit=True, assignment_risk=True,
        tags=("single-leg", "short-premium", "undefined-risk"),
        caveat=(
            "Loss is genuinely unbounded — the underlying can rise without "
            "limit. This is the highest-risk structure in the catalogue and is "
            "listed for completeness, not as a recommendation."
        ),
    ),
    "short-put": _spec(
        key="short-put", name="Short (Cash-Secured) Put",
        legs=(Leg(_P, -1, 0.95, _M),),
        direction=Direction.BULLISH, volatility_view=VolatilityView.SHORT_VOL,
        thesis="Underlying stays above the strike, or is acquired at a discount.",
        defined_risk=True, net_credit=True, assignment_risk=True,
        tags=("single-leg", "short-premium", "income"),
        caveat="Loss is bounded only because the underlying cannot go below zero.",
    ),
    # ── covered / protective ────────────────────────────────────────────────
    "covered-call": _spec(
        key="covered-call", name="Covered Call",
        legs=(Leg(None, 1), Leg(_C, -1, 1.05, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.SHORT_VOL,
        thesis="Collect premium on stock already held, capping the upside.",
        defined_risk=True, net_credit=True, assignment_risk=True,
        tags=("income", "overwrite"),
        caveat="Keeps the full downside of the stock; only the upside is sold.",
    ),
    "protective-put": _spec(
        key="protective-put", name="Protective Put",
        legs=(Leg(None, 1), Leg(_P, 1, 0.95, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="Insure a long position below the strike.",
        defined_risk=True, net_credit=False,
        tags=("hedge", "insurance"),
    ),
    "collar": _spec(
        key="collar", name="Collar",
        legs=(Leg(None, 1), Leg(_P, 1, 0.95, _M), Leg(_C, -1, 1.05, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.VOL_NEUTRAL,
        thesis="Fund downside insurance by selling the upside above a strike.",
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("hedge", "financed"),
    ),
    # ── vertical spreads ────────────────────────────────────────────────────
    "bull-call-spread": _spec(
        key="bull-call-spread", name="Bull Call Spread (call debit)",
        legs=(Leg(_C, 1, 1.00, _M), Leg(_C, -1, 1.05, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="A moderate rise, with cost reduced by capping the upside.",
        defined_risk=True, net_credit=False,
        tags=("vertical", "debit"),
    ),
    "bull-put-spread": _spec(
        key="bull-put-spread", name="Bull Put Spread (put credit)",
        legs=(Leg(_P, -1, 0.97, _M), Leg(_P, 1, 0.92, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.SHORT_VOL,
        thesis="Underlying stays above the short put; keep the credit.",
        defined_risk=True, net_credit=True, assignment_risk=True,
        tags=("vertical", "credit", "income"),
    ),
    "bear-put-spread": _spec(
        key="bear-put-spread", name="Bear Put Spread (put debit)",
        legs=(Leg(_P, 1, 1.00, _M), Leg(_P, -1, 0.95, _M)),
        direction=Direction.BEARISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="A moderate fall, with cost reduced by capping the downside.",
        defined_risk=True, net_credit=False,
        tags=("vertical", "debit"),
    ),
    "bear-call-spread": _spec(
        key="bear-call-spread", name="Bear Call Spread (call credit)",
        legs=(Leg(_C, -1, 1.03, _M), Leg(_C, 1, 1.08, _M)),
        direction=Direction.BEARISH, volatility_view=VolatilityView.SHORT_VOL,
        thesis="Underlying stays below the short call; keep the credit.",
        defined_risk=True, net_credit=True, assignment_risk=True,
        tags=("vertical", "credit", "income"),
    ),
    # ── straddles / strangles ───────────────────────────────────────────────
    "long-straddle": _spec(
        key="long-straddle", name="Long Straddle",
        legs=(Leg(_C, 1, 1.00, _M), Leg(_P, 1, 1.00, _M)),
        direction=Direction.VOLATILE, volatility_view=VolatilityView.LONG_VOL,
        thesis="A large move in either direction, bigger than the double premium.",
        defined_risk=True, net_credit=False,
        tags=("volatility", "long-premium", "event"),
    ),
    "short-straddle": _spec(
        key="short-straddle", name="Short Straddle",
        legs=(Leg(_C, -1, 1.00, _M), Leg(_P, -1, 1.00, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="The underlying stays pinned; both premiums decay.",
        defined_risk=False, net_credit=True, assignment_risk=True,
        tags=("volatility", "short-premium", "undefined-risk"),
        caveat="Unbounded loss on the upside. The classic 'picking up pennies' trade.",
    ),
    "long-strangle": _spec(
        key="long-strangle", name="Long Strangle",
        legs=(Leg(_C, 1, 1.05, _M), Leg(_P, 1, 0.95, _M)),
        direction=Direction.VOLATILE, volatility_view=VolatilityView.LONG_VOL,
        thesis="A large move; cheaper than a straddle but needs more of one.",
        defined_risk=True, net_credit=False,
        tags=("volatility", "long-premium", "event"),
    ),
    "short-strangle": _spec(
        key="short-strangle", name="Short Strangle",
        legs=(Leg(_C, -1, 1.05, _M), Leg(_P, -1, 0.95, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="The underlying stays inside a range; both wings decay.",
        defined_risk=False, net_credit=True, assignment_risk=True,
        tags=("volatility", "short-premium", "undefined-risk", "income"),
        caveat="Unbounded upside loss. Higher win rate, unbounded tail.",
    ),
    "long-guts": _spec(
        key="long-guts", name="Long Guts",
        legs=(Leg(_C, 1, 0.95, _M), Leg(_P, 1, 1.05, _M)),
        direction=Direction.VOLATILE, volatility_view=VolatilityView.LONG_VOL,
        thesis="Straddle-like exposure built from in-the-money options.",
        defined_risk=True, net_credit=False,
        tags=("volatility", "itm"),
    ),
    "strap": _spec(
        key="strap", name="Strap",
        legs=(Leg(_C, 2, 1.00, _M), Leg(_P, 1, 1.00, _M)),
        direction=Direction.VOLATILE, volatility_view=VolatilityView.LONG_VOL,
        thesis="A big move, weighted to the upside — two calls to one put.",
        defined_risk=True, net_credit=False,
        tags=("volatility", "skewed"),
    ),
    "strip": _spec(
        key="strip", name="Strip",
        legs=(Leg(_C, 1, 1.00, _M), Leg(_P, 2, 1.00, _M)),
        direction=Direction.VOLATILE, volatility_view=VolatilityView.LONG_VOL,
        thesis="A big move, weighted to the downside — two puts to one call.",
        defined_risk=True, net_credit=False,
        tags=("volatility", "skewed"),
    ),
    # ── butterflies ─────────────────────────────────────────────────────────
    "long-call-butterfly": _spec(
        key="long-call-butterfly", name="Long Call Butterfly",
        legs=(Leg(_C, 1, 0.95, _M), Leg(_C, -2, 1.00, _M), Leg(_C, 1, 1.05, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="The underlying finishes near the body strike.",
        defined_risk=True, net_credit=False,
        tags=("butterfly", "pin", "low-cost"),
    ),
    "long-put-butterfly": _spec(
        key="long-put-butterfly", name="Long Put Butterfly",
        legs=(Leg(_P, 1, 1.05, _M), Leg(_P, -2, 1.00, _M), Leg(_P, 1, 0.95, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="Same payoff as the call butterfly, built from puts.",
        defined_risk=True, net_credit=False,
        tags=("butterfly", "pin"),
    ),
    "iron-butterfly": _spec(
        key="iron-butterfly", name="Iron Butterfly",
        legs=(
            Leg(_P, 1, 0.95, _M), Leg(_P, -1, 1.00, _M),
            Leg(_C, -1, 1.00, _M), Leg(_C, 1, 1.05, _M),
        ),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="A pinned underlying, opened for a credit rather than a debit.",
        defined_risk=True, net_credit=True, assignment_risk=True,
        tags=("butterfly", "credit", "income"),
    ),
    "broken-wing-butterfly": _spec(
        key="broken-wing-butterfly", name="Broken Wing Butterfly",
        legs=(Leg(_C, 1, 1.00, _M), Leg(_C, -2, 1.05, _M), Leg(_C, 1, 1.13, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis=(
            "A butterfly with one wing pushed out, reducing the debit — often "
            "to a credit — in exchange for asymmetric risk."
        ),
        defined_risk=True, net_credit=False,
        tags=("butterfly", "asymmetric"),
        caveat=(
            "Whether this opens for a credit depends on the volatility SKEW, "
            "which flat Black-Scholes cannot represent — priced here at a "
            "single vol it comes out a small debit, and on a real chain with "
            "put skew it frequently does not. Treat `net_credit=False` as "
            "'the model cannot tell you', not as a fact about the market. "
            "Separately: the wide wing carries materially more loss than the "
            "narrow one, so 'no risk on one side' describes one side only."
        ),
    ),
    # ── condors ─────────────────────────────────────────────────────────────
    "iron-condor": _spec(
        key="iron-condor", name="Iron Condor",
        legs=(
            Leg(_P, 1, 0.90, _M), Leg(_P, -1, 0.95, _M),
            Leg(_C, -1, 1.05, _M), Leg(_C, 1, 1.10, _M),
        ),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="The underlying stays between the short strikes; keep the credit.",
        defined_risk=True, net_credit=True, assignment_risk=True,
        tags=("condor", "credit", "income", "range"),
        caveat=(
            "High win rate, small win, large loss. The distribution is precisely "
            "the shape that looks excellent over a short sample and is not."
        ),
    ),
    "long-call-condor": _spec(
        key="long-call-condor", name="Long Call Condor",
        legs=(
            Leg(_C, 1, 0.92, _M), Leg(_C, -1, 0.97, _M),
            Leg(_C, -1, 1.03, _M), Leg(_C, 1, 1.08, _M),
        ),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="A wider, flatter butterfly — a range rather than a point.",
        defined_risk=True, net_credit=False,
        tags=("condor", "debit", "range"),
    ),
    # ── ratio and backspreads ───────────────────────────────────────────────
    "call-ratio-spread": _spec(
        key="call-ratio-spread", name="Call Ratio Spread",
        legs=(Leg(_C, 1, 1.00, _M), Leg(_C, -2, 1.05, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="A modest rise to the short strike; financed by the extra short.",
        defined_risk=False, net_credit=False, assignment_risk=True,
        tags=("ratio", "undefined-risk"),
        caveat=(
            "Net short one call above the upper strike — unbounded upside loss. "
            "Whether it opens for a credit depends on strike spacing and skew; "
            "priced at a single flat vol it comes out a small debit, so "
            "`net_credit=False` here means 'the model cannot tell you'."
        ),
    ),
    "put-ratio-spread": _spec(
        key="put-ratio-spread", name="Put Ratio Spread",
        legs=(Leg(_P, 1, 1.00, _M), Leg(_P, -2, 0.95, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="A modest fall to the short strike; financed by the extra short.",
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("ratio",),
        caveat=(
            "Net short one put below the lower strike — large loss on a crash. "
            "Credit-versus-debit depends on put skew, which flat Black-Scholes "
            "cannot represent; on a real chain this is often a credit."
        ),
    ),
    "call-backspread": _spec(
        key="call-backspread", name="Call Backspread",
        legs=(Leg(_C, -1, 1.00, _M), Leg(_C, 2, 1.05, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="A large upside move; the short leg funds the extra longs.",
        defined_risk=True, net_credit=False,
        tags=("ratio", "backspread", "convex"),
    ),
    "put-backspread": _spec(
        key="put-backspread", name="Put Backspread",
        legs=(Leg(_P, -1, 1.00, _M), Leg(_P, 2, 0.95, _M)),
        direction=Direction.BEARISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="A large downside move; the short leg funds the extra longs.",
        defined_risk=True, net_credit=False,
        tags=("ratio", "backspread", "convex", "crash-hedge"),
    ),
    "jade-lizard": _spec(
        key="jade-lizard", name="Jade Lizard",
        legs=(Leg(_P, -1, 0.95, _M), Leg(_C, -1, 1.05, _M), Leg(_C, 1, 1.10, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis=(
            "A short put plus a call credit spread, sized so the total credit "
            "exceeds the call spread's width — no upside risk by construction."
        ),
        defined_risk=True, net_credit=True, assignment_risk=True,
        tags=("credit", "income", "asymmetric"),
        caveat="No risk *to the upside*. The downside is a naked short put.",
    ),
    # ── calendar / diagonal ─────────────────────────────────────────────────
    "calendar-spread": _spec(
        key="calendar-spread", name="Calendar Spread",
        legs=(Leg(_C, -1, 1.00, _M), Leg(_C, 1, 1.00, _Q)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.LONG_VOL,
        thesis=(
            "The near option decays faster than the far one while the underlying "
            "stays near the strike."
        ),
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("calendar", "time-decay", "term-structure"),
        caveat=(
            "`payoff()` is not valid for this structure — it treats the far leg "
            "as expiring with the near one, discarding the residual time value "
            "the trade exists to capture. Read the greeks instead."
        ),
    ),
    "diagonal-spread": _spec(
        key="diagonal-spread", name="Diagonal Spread",
        legs=(Leg(_C, -1, 1.05, _M), Leg(_C, 1, 1.00, _Q)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.LONG_VOL,
        thesis="A calendar with a directional tilt from the strike offset.",
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("calendar", "diagonal", "directional"),
        caveat="Same payoff caveat as the calendar spread.",
    ),
    "poor-mans-covered-call": _spec(
        key="poor-mans-covered-call", name="Poor Man's Covered Call",
        legs=(Leg(_C, 1, 0.80, 1.0), Leg(_C, -1, 1.05, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.VOL_NEUTRAL,
        thesis=(
            "A deep in-the-money long-dated call substitutes for the stock in a "
            "covered call, at a fraction of the capital."
        ),
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("diagonal", "capital-efficient", "income"),
    ),
    # ── synthetic ───────────────────────────────────────────────────────────
    "synthetic-long": _spec(
        key="synthetic-long", name="Synthetic Long Stock",
        legs=(Leg(_C, 1, 1.00, _M), Leg(_P, -1, 1.00, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.VOL_NEUTRAL,
        thesis="Replicate long stock from options; same payoff, different capital.",
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("synthetic", "parity"),
    ),
    "synthetic-short": _spec(
        key="synthetic-short", name="Synthetic Short Stock",
        legs=(Leg(_P, 1, 1.00, _M), Leg(_C, -1, 1.00, _M)),
        direction=Direction.BEARISH, volatility_view=VolatilityView.VOL_NEUTRAL,
        thesis="Replicate short stock from options.",
        defined_risk=False, net_credit=False, assignment_risk=True,
        tags=("synthetic", "parity", "undefined-risk"),
        caveat=(
            "Carries short stock's unbounded upside loss, because that is "
            "exactly what it replicates. The synthetic is not safer than the "
            "position it stands in for."
        ),
    ),
    "risk-reversal": _spec(
        key="risk-reversal", name="Risk Reversal",
        legs=(Leg(_P, -1, 0.95, _M), Leg(_C, 1, 1.05, _M)),
        direction=Direction.BULLISH, volatility_view=VolatilityView.VOL_NEUTRAL,
        thesis="Fund upside calls by selling downside puts — a skew trade.",
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("synthetic", "skew", "financed"),
        caveat=(
            "A pure bet on the volatility SKEW, which flat-vol Black-Scholes "
            "cannot represent at all — the model price is close to meaningless "
            "for this structure specifically. In practice put skew usually "
            "makes it a credit; the model says debit because it prices both "
            "wings at one volatility. `net_credit=False` records the model's "
            "answer, not the market's."
        ),
    ),
    "box-spread": _spec(
        key="box-spread", name="Box Spread",
        legs=(
            Leg(_C, 1, 0.95, _M), Leg(_C, -1, 1.05, _M),
            Leg(_P, 1, 1.05, _M), Leg(_P, -1, 0.95, _M),
        ),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.VOL_NEUTRAL,
        thesis=(
            "A synthetic long and short combined: the payoff is the strike "
            "width regardless of the underlying. A financing instrument, not a "
            "directional trade."
        ),
        defined_risk=True, net_credit=False,
        tags=("arbitrage", "financing", "rates"),
        caveat=(
            "Used to borrow or lend at the implied rate. American-style boxes "
            "carry early-assignment risk that has produced catastrophic losses "
            "for traders who believed the payoff was risk-free."
        ),
    ),
    # ── ladders ─────────────────────────────────────────────────────────────
    "call-ladder": _spec(
        key="call-ladder", name="Call Ladder (bull call ladder)",
        legs=(Leg(_C, 1, 1.00, _M), Leg(_C, -1, 1.05, _M), Leg(_C, -1, 1.10, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="A bull call spread financed by an extra short call above it.",
        defined_risk=False, net_credit=False, assignment_risk=True,
        tags=("ladder", "undefined-risk"),
        caveat="Net short one call at the top — unbounded loss on a large rally.",
    ),
    "put-ladder": _spec(
        key="put-ladder", name="Put Ladder (bear put ladder)",
        legs=(Leg(_P, 1, 1.00, _M), Leg(_P, -1, 0.95, _M), Leg(_P, -1, 0.90, _M)),
        direction=Direction.NEUTRAL, volatility_view=VolatilityView.SHORT_VOL,
        thesis="A bear put spread financed by an extra short put below it.",
        defined_risk=True, net_credit=False, assignment_risk=True,
        tags=("ladder",),
        caveat="Net short one put at the bottom — large loss on a crash.",
    ),
}


# ── helpers ────────────────────────────────────────────────────────────────


def build(
    key: str,
    *,
    spot: float,
    volatility: float,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
    multiplier: float = 100.0,
) -> OptionStrategy:
    """Resolve a catalogued structure against a spot and an assumed volatility."""
    try:
        spec = CATALOGUE[key]
    except KeyError:
        raise KeyError(
            f"unknown structure {key!r}. Available: {', '.join(sorted(CATALOGUE))}"
        ) from None
    return OptionStrategy(
        spec=spec, spot=spot, volatility=volatility,
        rate=rate, dividend_yield=dividend_yield, multiplier=multiplier,
    )


def by_outlook(
    direction: Direction | None = None,
    volatility_view: VolatilityView | None = None,
    *,
    defined_risk_only: bool = False,
) -> list[StructureSpec]:
    """Structures matching a view. `defined_risk_only` filters out the tails."""
    out = list(CATALOGUE.values())
    if direction is not None:
        out = [s for s in out if s.direction is direction]
    if volatility_view is not None:
        out = [s for s in out if s.volatility_view is volatility_view]
    if defined_risk_only:
        out = [s for s in out if s.defined_risk]
    return sorted(out, key=lambda s: s.key)


def undefined_risk() -> list[StructureSpec]:
    """Every structure whose loss is not bounded by construction."""
    return sorted(
        (s for s in CATALOGUE.values() if not s.defined_risk), key=lambda s: s.key
    )


def catalogue_text() -> str:
    """Full listing, grouped by directional outlook."""
    blocks: list[str] = []
    for direction in Direction:
        specs = by_outlook(direction)
        if not specs:
            continue
        blocks.append(f"\n{direction.value.upper()}")
        blocks.append("─" * 70)
        for spec in specs:
            blocks.append(spec.describe())
            blocks.append("")
    return "\n".join(blocks)
