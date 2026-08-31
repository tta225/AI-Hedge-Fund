"""Dividing capital between strategies, and netting what they ask for.

:mod:`axiom.portfolio.risk` sizes *one* book against its own covariance. That
is the right shape for one strategy and the wrong shape for a desk, which faces
three questions a single-book risk manager cannot express:

**How much capital does each strategy get?** Equal weight is a decision, not an
absence of one, and it is usually the wrong decision: two strategies at very
different volatilities equally weighted means the louder one is the portfolio.

**What does the desk actually hold?** Two strategies that both want AAPL send
two orders and pay two spreads for one position; one long and one short send
two orders to hold nothing. Netting at the desk level is not an optimisation,
it is the difference between the book you think you have and the book you have.

**Is the whole thing riskier than its parts?** Strategies that look independent
in isolation can be the same trade wearing different signals. Correlated
allocations do not diversify, and a desk that sizes each strategy against its
own risk budget can quietly run twice the risk it authorised.

The allocation methods here go from crude to less crude, and the ordering is
deliberate — each is a defensible default in a different state of knowledge:

:func:`equal_weight` when nothing is known. Stated explicitly so it is chosen
rather than defaulted into.

:func:`inverse_volatility` when per-strategy volatility is known but the
correlations are not trustworthy. This is the honest default for a desk with a
few strategies and a short history, because correlation estimates need far more
data than volatility estimates do.

:func:`risk_parity` when the covariance is worth believing. Equalises each
strategy's *contribution* to portfolio risk rather than its capital, which is
what "diversified" is normally meant to mean.

Deliberately absent: mean-variance optimisation. It requires expected returns,
which is precisely the input this project has spent six campaigns failing to
estimate, and it is famously willing to concentrate the entire book into
whichever estimate is most wrong. Using it here would be optimising against
noise with extra steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Smallest capital share a strategy can be given before it is dropped
#: entirely. A 0.3% allocation cannot be traded in round lots and produces
#: turnover without exposure.
MIN_ALLOCATION = 0.01
#: Iterations for the risk-parity solve. It converges in tens, not thousands.
RISK_PARITY_STEPS = 500
RISK_PARITY_TOLERANCE = 1e-8


class AllocationError(ValueError):
    """Capital could not be divided as asked."""


def _normalise(weights: np.ndarray, *, floor: float = MIN_ALLOCATION) -> np.ndarray:
    """Scale to sum to one, dropping shares too small to trade.

    Dropping and renormalising rather than leaving a dust allocation: a
    position too small to express in round lots still generates orders, pays
    spread, and shows up in reconciliation as a discrepancy.
    """
    weights = np.clip(np.nan_to_num(weights, nan=0.0), 0.0, None)
    total = weights.sum()
    if total <= 0:
        raise AllocationError(
            "every strategy scored zero or negative capital; there is no book to "
            "allocate rather than a book of nothing"
        )
    weights = weights / total
    weights[weights < floor] = 0.0
    total = weights.sum()
    if total <= 0:
        # Everything was below the floor, which means the floor is wrong for
        # this many strategies rather than that nothing should trade.
        raise AllocationError(
            f"all allocations fell below the {floor:.1%} floor; lower it or run "
            "fewer strategies"
        )
    return np.asarray(weights / total, dtype=float)


def equal_weight(n: int) -> np.ndarray:
    """The same capital to each. A decision, not the absence of one."""
    if n < 1:
        raise AllocationError("need at least one strategy")
    return np.full(n, 1.0 / n)


def inverse_volatility(volatilities: Sequence[float]) -> np.ndarray:
    """Capital inversely proportional to each strategy's own volatility.

    Equalises risk contribution *under the assumption that strategies are
    uncorrelated*, which is wrong but wrong in a stable, understandable way.
    The honest default when the correlation matrix is estimated from too little
    data to trust — and with a handful of strategies and a few years of
    history, it usually is.
    """
    vol = np.asarray(volatilities, dtype=float)
    if vol.size == 0:
        raise AllocationError("need at least one strategy")
    if np.any(vol < 0):
        raise AllocationError("volatility cannot be negative")
    with np.errstate(divide="ignore"):
        raw = np.where(vol > 0, 1.0 / vol, 0.0)
    if not np.any(raw > 0):
        raise AllocationError(
            "every strategy reported zero volatility; that is a measurement "
            "failure, not a riskless portfolio"
        )
    return _normalise(raw)


def risk_parity(covariance: np.ndarray, *, steps: int = RISK_PARITY_STEPS) -> np.ndarray:
    """Weights where every strategy contributes equally to portfolio risk.

    Solved by the standard multiplicative fixed-point iteration: repeatedly
    scale each weight by the ratio of its target risk contribution to its
    actual one. It converges for any positive-definite covariance and needs no
    optimiser dependency.

    Note what this equalises. Risk *contribution* accounts for correlation, so
    two strategies that are secretly the same trade share one strategy's worth
    of budget between them rather than getting one each — which is the entire
    reason to prefer this over inverse volatility when the covariance is
    believable.
    """
    covariance = np.asarray(covariance, dtype=float)
    n = covariance.shape[0]
    if covariance.shape != (n, n):
        raise AllocationError(f"covariance must be square, got {covariance.shape}")
    if n == 1:
        return np.array([1.0])
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise AllocationError("covariance matrix must be symmetric")

    weights = np.full(n, 1.0 / n)
    target = 1.0 / n
    for _ in range(steps):
        portfolio_variance = float(weights @ covariance @ weights)
        if portfolio_variance <= 0:
            raise AllocationError(
                "portfolio variance is zero or negative; the covariance matrix is "
                "not usable for risk parity"
            )
        # Marginal contribution of each strategy to total risk.
        contributions = weights * (covariance @ weights) / portfolio_variance
        gap = float(np.max(np.abs(contributions - target)))
        if gap < RISK_PARITY_TOLERANCE:
            break
        # Damped with a square root rather than taking the full step. The
        # undamped update `w *= target / contribution` overshoots and
        # oscillates with period two — on a two-strategy problem it flips
        # between the two extremes forever and returns whichever it happened to
        # land on, which for an even iteration count is the equal-weight
        # starting point. That failure is silent: it returns plausible weights
        # that are not risk parity at all.
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(contributions > 0, target / contributions, 1.0)
            weights = weights * np.sqrt(step)
        weights = np.clip(np.nan_to_num(weights, nan=0.0), 1e-12, None)
        weights = weights / weights.sum()
    return _normalise(weights)


@dataclass(frozen=True, slots=True)
class Allocation:
    """How capital was divided, and what that implies for the whole book."""

    strategies: tuple[str, ...]
    weights: np.ndarray
    method: str
    #: Volatility of the combined book, when a covariance was supplied.
    portfolio_volatility: float = float("nan")
    #: Sum of each strategy's standalone volatility times its weight. The
    #: comparison that says whether diversification happened at all.
    weighted_standalone_volatility: float = float("nan")

    @property
    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.strategies, (float(w) for w in self.weights), strict=True))

    @property
    def diversification_ratio(self) -> float:
        """Weighted standalone risk over realised portfolio risk.

        1.0 means the strategies are the same trade and the desk is running one
        strategy at full size while believing it runs several. Above 1.0 is the
        benefit that justifies operating more than one.
        """
        if not np.isfinite(self.portfolio_volatility) or self.portfolio_volatility <= 0:
            return float("nan")
        return self.weighted_standalone_volatility / self.portfolio_volatility

    @property
    def n_funded(self) -> int:
        return int(np.count_nonzero(self.weights))

    def render(self) -> str:
        lines = [f"Allocation ({self.method}) across {self.n_funded} funded strategies"]
        for name, weight in sorted(self.as_dict.items(), key=lambda kv: -kv[1]):
            marker = "" if weight > 0 else "   (unfunded)"
            lines.append(f"  {name:<28}{weight:>7.1%}{marker}")
        if np.isfinite(self.portfolio_volatility):
            lines.append(
                f"  portfolio vol {self.portfolio_volatility:.2%}, "
                f"diversification ratio {self.diversification_ratio:.2f}"
            )
            if self.diversification_ratio < 1.05:
                lines.append(
                    "  ! these strategies are close to the same trade — the desk is "
                    "running one book while believing it runs several"
                )
        return "\n".join(lines)


def allocate(
    strategies: Sequence[str],
    *,
    method: str = "inverse_volatility",
    volatilities: Sequence[float] | None = None,
    covariance: np.ndarray | None = None,
) -> Allocation:
    """Divide capital between strategies by the named method.

    Args:
        strategies: names, in the order the other arrays are indexed.
        method: ``equal``, ``inverse_volatility``, or ``risk_parity``.
        volatilities: per-strategy annualised volatility.
        covariance: strategy-return covariance, required for risk parity.
    """
    names = tuple(strategies)
    if not names:
        raise AllocationError("no strategies to allocate between")

    if method == "equal":
        weights = equal_weight(len(names))
    elif method == "inverse_volatility":
        if volatilities is None:
            raise AllocationError("inverse_volatility needs per-strategy volatilities")
        weights = inverse_volatility(volatilities)
    elif method == "risk_parity":
        if covariance is None:
            raise AllocationError("risk_parity needs a covariance matrix")
        weights = risk_parity(covariance)
    else:
        raise AllocationError(
            f"unknown method {method!r}; expected equal, inverse_volatility, "
            "or risk_parity"
        )

    portfolio_vol = float("nan")
    weighted_standalone = float("nan")
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        portfolio_vol = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
        standalone = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        weighted_standalone = float(weights @ standalone)
    elif volatilities is not None:
        standalone = np.asarray(volatilities, dtype=float)
        weighted_standalone = float(weights @ standalone)

    return Allocation(
        strategies=names,
        weights=weights,
        method=method,
        portfolio_volatility=portfolio_vol,
        weighted_standalone_volatility=weighted_standalone,
    )


def net_positions(
    requests: Mapping[str, Mapping[str, float]],
    allocation: Allocation | None = None,
) -> dict[str, float]:
    """Combine per-strategy target books into the one book the desk holds.

    ``requests`` maps a strategy name to its desired weights per symbol,
    expressed as fractions of *that strategy's own* capital. Scaling by the
    allocation converts them to fractions of desk capital, which is the only
    frame in which they can be added.

    Netting matters more than it sounds. Two strategies that both want AAPL
    send two orders and pay two spreads for a position they could have taken
    once; one long and one short send two orders to arrive at nothing, paying
    twice for the privilege. Neither is visible from inside either strategy.
    """
    shares = allocation.as_dict if allocation is not None else {}
    if allocation is not None:
        unknown = set(requests) - set(shares)
        if unknown:
            raise AllocationError(
                f"strategies {sorted(unknown)} submitted target books but hold no "
                f"allocation; the roster is {sorted(shares)}"
            )

    netted: dict[str, float] = {}
    for strategy, book in requests.items():
        share = shares.get(strategy, 1.0) if allocation is not None else 1.0
        if share <= 0:
            continue  # unfunded: its opinions do not reach the market
        for symbol, weight in book.items():
            netted[symbol] = netted.get(symbol, 0.0) + share * float(weight)
    # A netted-to-flat symbol is not a position; keeping it would send an order
    # for zero and leave a row nothing can reconcile.
    return {symbol: weight for symbol, weight in netted.items() if abs(weight) > 1e-12}


@dataclass(slots=True)
class CrossStrategyRisk:
    """Aggregate exposure across strategies, and what it hides.

    The number worth watching is :attr:`internalisation` — how much of the
    gross the strategies would have traded is cancelled by netting. A high
    figure means the desk is paying two spreads to hold one position, and it is
    invisible from inside any single strategy.
    """

    gross_requested: float = 0.0
    gross_held: float = 0.0
    net_held: float = 0.0
    overlaps: dict[str, int] = field(default_factory=dict)

    @property
    def internalisation(self) -> float:
        """Fraction of requested gross exposure cancelled by netting."""
        if self.gross_requested <= 0:
            return 0.0
        return max(0.0, 1.0 - self.gross_held / self.gross_requested)

    @property
    def contested(self) -> list[str]:
        """Symbols more than one strategy has an opinion about."""
        return sorted(s for s, n in self.overlaps.items() if n > 1)

    def render(self) -> str:
        lines = [
            f"Cross-strategy: gross requested {self.gross_requested:.2f}, "
            f"held {self.gross_held:.2f}, net {self.net_held:+.2f}",
            f"  internalisation {self.internalisation:.1%} across "
            f"{len(self.contested)} contested symbol(s)",
        ]
        if self.internalisation > 0.2:
            lines.append(
                "  ! more than a fifth of requested exposure cancels internally — "
                "these strategies are trading against each other"
            )
        return "\n".join(lines)


def assess_cross_strategy(
    requests: Mapping[str, Mapping[str, float]], allocation: Allocation | None = None
) -> CrossStrategyRisk:
    """Measure what netting the strategies together actually did."""
    shares = allocation.as_dict if allocation is not None else {}
    gross_requested = 0.0
    overlaps: dict[str, int] = {}
    for strategy, book in requests.items():
        share = shares.get(strategy, 1.0) if allocation is not None else 1.0
        if share <= 0:
            continue
        for symbol, weight in book.items():
            gross_requested += abs(share * float(weight))
            overlaps[symbol] = overlaps.get(symbol, 0) + 1

    held = net_positions(requests, allocation)
    return CrossStrategyRisk(
        gross_requested=gross_requested,
        gross_held=sum(abs(w) for w in held.values()),
        net_held=sum(held.values()),
        overlaps=overlaps,
    )
