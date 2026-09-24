"""What a trade actually costs, as a function of how big it is.

The flat basis-point charge used until now is a placeholder wearing a number.
It says a trade costs 10 bp whether it is a thousand dollars or a hundred
million, which is wrong in the one direction that matters: it makes a strategy
look identically tradable at every size, and therefore hides the only question
a desk ultimately has to answer — **how much money can this take?**

The model here has three components, in increasing order of how much they
depend on size:

**Commission.** Per unit of notional. Near zero for US equities now, and kept
because it is not zero everywhere.

**Spread.** Half the bid-ask, paid on entry and again on exit. Independent of
size until the trade exceeds what is quoted, which for the sizes here it does
not. This is what the flat model was approximating.

**Market impact.** The part that scales. Trading is a request for liquidity,
and the price moves against a large request. The empirically supported form is
the **square-root law** — impact grows with the square root of participation
rather than linearly:

    impact ≈ η · σ · √(Q / ADV)

where ``Q`` is the order's notional, ``ADV`` is average daily dollar volume,
``σ`` is daily volatility, and ``η`` is a coefficient near 1. Square-root
rather than linear is not a refinement: at 10% participation the two differ by
a factor of three, and the linear form makes any strategy look untradeable at
size while the square-root form gives a usable capacity estimate.

The consequence worth stating plainly: **cost is a property of the strategy
*and* the capital**, not of the strategy alone. The same book that clears its
costs at ten million may be underwater at five hundred.
:func:`~axiom.research.panel_lab.capacity_curve` exists to make that visible
rather than leaving it implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

#: Half-spread in basis points for a liquid US large cap. A penny on a $100
#: stock is 1 bp of spread, so half is 0.5 bp. Deliberately not optimistic:
#: quoted spread is not the spread a market order pays into a moving book.
DEFAULT_HALF_SPREAD_BPS = 1.0
#: Commission per unit of notional, in basis points. Zero-commission retail
#: brokers are genuinely near zero; institutional is a fraction of a cent per
#: share. Kept as a knob rather than assumed away.
DEFAULT_COMMISSION_BPS = 0.2
#: Square-root impact coefficient. Published estimates for the temporary
#: component cluster around 0.3–0.6; 1.0 is deliberately conservative, because
#: a research bar that flatters execution is worse than one that does not.
DEFAULT_IMPACT_COEFFICIENT = 1.0
#: Participation above which the square-root law is extrapolating beyond where
#: it was fitted. Trades that large are worked over days, not sent as one
#: order, and pretending otherwise produces a number nobody should believe.
PARTICIPATION_WARN = 0.10


@runtime_checkable
class CostModel(Protocol):
    """Anything that can price a set of weight changes."""

    def charge(
        self,
        weight_deltas: np.ndarray,
        *,
        aum: float,
        adv_notional: np.ndarray,
        volatility: np.ndarray,
    ) -> float:
        """Total cost as a fraction of ``aum``."""
        ...


@dataclass(frozen=True, slots=True)
class FlatBpsCost:
    """A fixed charge per unit of notional traded.

    Kept so the new model can be compared against the old one on the same run,
    rather than a change in results being attributable to either the model or
    the strategy with no way to separate them.
    """

    bps: float = 10.0

    def charge(
        self,
        weight_deltas: np.ndarray,
        *,
        aum: float = 0.0,
        adv_notional: np.ndarray | None = None,
        volatility: np.ndarray | None = None,
    ) -> float:
        return float(np.abs(weight_deltas).sum() * self.bps / 10_000.0)

    def describe(self) -> str:
        return f"flat {self.bps:.1f} bp per unit traded (size-independent)"


@dataclass(frozen=True, slots=True)
class SpreadImpactCost:
    """Commission plus half-spread plus square-root market impact.

    Args:
        half_spread_bps: half the quoted bid-ask, in basis points.
        commission_bps: per unit of notional.
        impact_coefficient: ``η`` in the square-root law.
        min_adv_notional: floor on average daily dollar volume, so a name with
            a missing or zero volume history does not divide by zero and
            produce infinite cost. Set high enough that a genuinely illiquid
            name is still charged heavily.
    """

    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS
    commission_bps: float = DEFAULT_COMMISSION_BPS
    impact_coefficient: float = DEFAULT_IMPACT_COEFFICIENT
    min_adv_notional: float = 1e6

    def __post_init__(self) -> None:
        if self.half_spread_bps < 0 or self.commission_bps < 0:
            raise ValueError("spread and commission must be non-negative")
        if self.impact_coefficient < 0:
            raise ValueError("impact_coefficient must be non-negative")
        if self.min_adv_notional <= 0:
            raise ValueError("min_adv_notional must be positive")

    def charge(
        self,
        weight_deltas: np.ndarray,
        *,
        aum: float,
        adv_notional: np.ndarray,
        volatility: np.ndarray,
    ) -> float:
        """Cost of moving the book by ``weight_deltas``, as a fraction of AUM.

        Every component is expressed per unit of *weight* traded so the result
        is directly comparable to a portfolio return, which is what the
        backtester subtracts it from.
        """
        deltas = np.abs(np.asarray(weight_deltas, dtype=float))
        if deltas.sum() <= 0:
            return 0.0
        if aum <= 0:
            raise ValueError("aum must be positive to price impact")

        adv = np.maximum(
            np.nan_to_num(np.asarray(adv_notional, dtype=float), nan=0.0),
            self.min_adv_notional,
        )
        vol = np.nan_to_num(np.asarray(volatility, dtype=float), nan=0.0)

        # Spread and commission are proportional to what is traded.
        linear = deltas.sum() * (self.half_spread_bps + self.commission_bps) / 10_000.0

        # Impact is proportional to trade size *and* to the square root of the
        # fraction of a day's volume that size represents. Both factors matter:
        # doubling the trade doubles the shares that pay impact and increases
        # the impact each pays by √2, so total impact grows as the 3/2 power.
        notional = deltas * aum
        participation = np.where(adv > 0, notional / adv, 0.0)
        impact_per_unit = self.impact_coefficient * vol * np.sqrt(participation)
        impact = float((deltas * impact_per_unit).sum())

        return float(linear + impact)

    def participation(
        self, weight_deltas: np.ndarray, *, aum: float, adv_notional: np.ndarray
    ) -> np.ndarray:
        """Fraction of each name's daily volume this trade represents.

        Reported separately because a cost number alone cannot tell you whether
        the model is being extrapolated. Anything above
        :data:`PARTICIPATION_WARN` is outside the regime the square-root law
        was fitted in.
        """
        deltas = np.abs(np.asarray(weight_deltas, dtype=float))
        adv = np.maximum(
            np.nan_to_num(np.asarray(adv_notional, dtype=float), nan=0.0),
            self.min_adv_notional,
        )
        return np.asarray(deltas * aum / adv, dtype=float)

    def describe(self) -> str:
        return (
            f"{self.commission_bps:.1f} bp commission + "
            f"{self.half_spread_bps:.1f} bp half-spread + "
            f"{self.impact_coefficient:.2f}·σ·√participation impact"
        )


def average_dollar_volume(
    closes: np.ndarray, volumes: np.ndarray, index: int, lookback: int = 20
) -> np.ndarray:
    """Trailing average daily dollar volume per symbol, strictly before ``index``.

    Dollar volume rather than share volume, because a hundred thousand shares
    of a $5 stock and of a $500 stock are not remotely the same liquidity.
    Causal, like everything else that feeds a decision: using the volume of the
    day being traded is a mild lookahead that flatters exactly the large trades
    the impact model is meant to penalise.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    start = max(0, index - lookback)
    if index <= start:
        return np.zeros(closes.shape[1] if closes.ndim > 1 else 1)
    window_notional = closes[start:index] * volumes[start:index]
    # An all-NaN column is routine on a point-in-time panel: a name that has
    # not listed yet, or has already delisted, has no volume history at all.
    # numpy warns on the empty slice; the answer we want is zero, which the
    # caller's ``min_adv_notional`` floor then turns into a heavy charge.
    with np.errstate(invalid="ignore"):
        observed = np.count_nonzero(~np.isnan(window_notional), axis=0)
        totals = np.nansum(window_notional, axis=0)
    return np.asarray(
        np.where(observed > 0, totals / np.maximum(observed, 1), 0.0), dtype=float
    )


def trailing_volatility(
    closes: np.ndarray, index: int, lookback: int = 20
) -> np.ndarray:
    """Trailing per-bar return volatility per symbol, strictly before ``index``.

    Per-bar rather than annualised: impact is a cost incurred on one trade, and
    scaling it to a year would overstate it by a factor of sixteen.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    start = max(0, index - lookback - 1)
    prices = closes[start:index]
    if len(prices) < 3:
        return np.zeros(prices.shape[1] if prices.ndim > 1 else 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(np.where(prices > 0, prices, np.nan)), axis=0)
    return np.asarray(
        np.nan_to_num(np.nanstd(returns, axis=0, ddof=1), nan=0.0), dtype=float
    )
