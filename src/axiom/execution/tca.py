"""What the desk paid, against what it assumed it would pay.

:mod:`axiom.execution.costs` prices a trade with a square-root impact model
whose coefficient ``η`` was set to 1.0 because published estimates cluster
around 0.3–0.6 and a research bar that flatters execution is worse than one
that does not. That is a defensible guess. It is still a guess, and the desk
has been recording every fill it ever received, which means the guess is
testable against evidence it already owns.

That is what transaction cost analysis is for, and it closes the only loop in
this project that runs from live trading back into research. Everything else
points one way: research proposes, execution disposes. TCA is the return path.

**Implementation shortfall** is the measure, because it is the one that
captures the cost the strategy actually bore. The decision price is what the
signal saw when it decided; the fill price is what the desk got. The gap
contains the spread, the impact of the desk's own order, and any drift while
the order was working — and a strategy is charged for all three whether or not
the execution desk considers them its fault.

Signed against the trade, always. A buy filled above the decision price cost
money; a sell filled above it made money. Reporting unsigned slippage is how a
venue reports a 3 bp average on a book that is systematically bought high and
sold low.

Two things this deliberately does not do. It does not compare against VWAP —
a benchmark the desk's own trading moves, which flatters a large order by
construction. And it does not attribute cost to a cause; it measures the total
and reports what the model predicted, and the difference between them is the
thing worth arguing about.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from axiom.core.types import AssetClass, Instrument, Side
from axiom.execution.base import Fill
from axiom.store.db import Store

#: Below this many fills, a calibrated impact coefficient is a number computed
#: from noise. Reported, but flagged as not yet meaningful.
MIN_FILLS_FOR_CALIBRATION = 30
#: Cost beyond this many basis points on a single fill is treated as a data
#: problem — a stale decision price, a mismatched reference — rather than a
#: real execution. Included in the count, excluded from the calibration fit.
OUTLIER_BPS = 500.0


@dataclass(frozen=True, slots=True)
class Execution:
    """One fill, joined to the decision that caused it.

    ``decision_price`` is the price the signal was scored against, not the
    price when the order reached the venue. The difference between those two is
    latency cost, and it is a real cost that belongs to the strategy: a signal
    that only works if you are already there is not a tradable signal.
    """

    fill: Fill
    decision_price: float
    #: Average daily dollar volume for the symbol at decision time, if known.
    #: Required to compare against the impact model; without it the execution
    #: still contributes to realised cost, just not to calibration.
    adv_notional: float | None = None
    #: Per-bar return volatility at decision time.
    volatility: float | None = None

    def __post_init__(self) -> None:
        if self.decision_price <= 0:
            raise ValueError("decision_price must be positive")

    @property
    def signed_cost_bps(self) -> float:
        """Implementation shortfall in basis points, positive when it cost money."""
        direction = 1.0 if self.fill.side is Side.BUY else -1.0
        return direction * (self.fill.price - self.decision_price) / self.decision_price * 1e4

    @property
    def notional(self) -> float:
        return abs(self.fill.notional)

    @property
    def commission_bps(self) -> float:
        return self.fill.commission / self.notional * 1e4 if self.notional else 0.0

    @property
    def participation(self) -> float | None:
        """Fraction of a day's dollar volume this fill represented."""
        if not self.adv_notional or self.adv_notional <= 0:
            return None
        return self.notional / self.adv_notional

    @property
    def is_outlier(self) -> bool:
        return abs(self.signed_cost_bps) > OUTLIER_BPS


@dataclass(frozen=True, slots=True)
class TCAReport:
    """Realised execution cost, and what the model said it would be."""

    n_fills: int
    n_outliers: int
    total_notional: float
    #: Notional-weighted, because a 50 bp cost on a thousand dollars and on a
    #: million are not the same event and a plain mean says they are.
    weighted_cost_bps: float
    median_cost_bps: float
    commission_bps: float
    #: Realised cost minus what :class:`~axiom.execution.costs.SpreadImpactCost`
    #: predicted, notional-weighted. Positive means the model was optimistic.
    model_error_bps: float
    predicted_cost_bps: float
    #: The impact coefficient that would have made the model fit, and whether
    #: there were enough observations for it to mean anything.
    fitted_impact_coefficient: float
    calibration_n: int
    by_symbol: dict[str, float]
    by_strategy: dict[str, float]

    @property
    def calibration_is_meaningful(self) -> bool:
        return self.calibration_n >= MIN_FILLS_FOR_CALIBRATION

    @property
    def model_is_optimistic(self) -> bool:
        """Did the desk pay more than the research bar assumed?

        The direction that matters. A model that overstates cost produces
        conservative research; one that understates it produces backtests that
        cannot be traded, which is the failure this whole loop exists to catch.
        """
        return self.model_error_bps > 0

    def render(self) -> str:
        lines = [
            f"TCA over {self.n_fills:,} fills, ${self.total_notional:,.0f} traded",
            f"  realised cost      {self.weighted_cost_bps:+8.2f} bp  "
            f"(median {self.median_cost_bps:+.2f}, commission {self.commission_bps:.2f})",
            f"  model predicted    {self.predicted_cost_bps:+8.2f} bp",
            f"  model error        {self.model_error_bps:+8.2f} bp  "
            + ("(model OPTIMISTIC)" if self.model_is_optimistic else "(model conservative)"),
        ]
        if self.n_outliers:
            lines.append(
                f"  {self.n_outliers} fill(s) beyond ±{OUTLIER_BPS:.0f} bp excluded from the fit"
            )
        if math.isfinite(self.fitted_impact_coefficient):
            note = "" if self.calibration_is_meaningful else "  ! too few fills to trust"
            lines.append(
                f"  fitted eta         {self.fitted_impact_coefficient:8.3f}  "
                f"(n={self.calibration_n}){note}"
            )
        if self.by_symbol:
            worst = sorted(self.by_symbol.items(), key=lambda kv: -kv[1])[:5]
            lines.append("  worst symbols: " + ", ".join(f"{s} {c:+.1f}bp" for s, c in worst))
        if self.by_strategy:
            lines.append(
                "  by strategy:   "
                + ", ".join(
                    f"{s} {c:+.1f}bp" for s, c in sorted(self.by_strategy.items())
                )
            )
        return "\n".join(lines)


def _weighted(values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.dot(values, weights) / total)


def analyse(
    executions: Iterable[Execution],
    *,
    half_spread_bps: float = 1.0,
    commission_bps: float = 0.2,
    impact_coefficient: float = 1.0,
) -> TCAReport:
    """Compare realised cost against the model, and fit the coefficient.

    The fit is a single-parameter least-squares through the origin::

        realised_impact_bps ≈ η · σ · √participation · 1e4

    Through the origin because the square-root law has no intercept: a trade of
    zero size has no impact, and fitting a constant would let the regression
    absorb the spread into it and report an impact coefficient near zero.

    The spread and commission are subtracted first rather than fitted, because
    they are known quantities. Fitting them would spend degrees of freedom
    re-estimating something the broker already told us.
    """
    items = list(executions)
    if not items:
        raise ValueError(
            "no executions to analyse; TCA over an empty book is not a zero, it is "
            "an absence of evidence"
        )

    costs = np.array([e.signed_cost_bps for e in items])
    notionals = np.array([e.notional for e in items])
    commissions = np.array([e.commission_bps for e in items])
    outliers = np.array([e.is_outlier for e in items])

    # Prediction per fill under the configured model, in basis points.
    predicted = np.full(len(items), np.nan)
    for i, execution in enumerate(items):
        participation = execution.participation
        if participation is None or execution.volatility is None:
            continue
        impact = impact_coefficient * execution.volatility * math.sqrt(participation) * 1e4
        predicted[i] = half_spread_bps + commission_bps + impact

    comparable = np.isfinite(predicted) & ~outliers
    predicted_bps = (
        _weighted(predicted[comparable], notionals[comparable])
        if comparable.any()
        else float("nan")
    )
    realised_comparable = (
        _weighted(costs[comparable], notionals[comparable]) if comparable.any() else float("nan")
    )
    model_error = realised_comparable - predicted_bps

    # Fit η on the fills where every input is present. Residual cost is what is
    # left after the known, size-independent charges.
    fitted = float("nan")
    calibration_n = 0
    if comparable.any():
        basis = []
        residual = []
        for i, execution in enumerate(items):
            if not comparable[i]:
                continue
            participation = execution.participation
            assert participation is not None and execution.volatility is not None
            x = execution.volatility * math.sqrt(participation) * 1e4
            if x <= 0:
                continue
            basis.append(x)
            residual.append(costs[i] - half_spread_bps - commission_bps)
        if basis:
            x_arr = np.array(basis)
            y_arr = np.array(residual)
            # Least squares through the origin, notional-weighted.
            weights = notionals[comparable][: len(x_arr)]
            denominator = float(np.sum(weights * x_arr * x_arr))
            if denominator > 0:
                fitted = float(np.sum(weights * x_arr * y_arr) / denominator)
            calibration_n = len(x_arr)

    def _group(key_of: Callable[[Execution], str]) -> dict[str, float]:
        groups: dict[str, tuple[list[float], list[float]]] = {}
        for execution, cost, notional in zip(items, costs, notionals, strict=True):
            values, weights = groups.setdefault(key_of(execution), ([], []))
            values.append(float(cost))
            weights.append(float(notional))
        return {key: _weighted(values, weights) for key, (values, weights) in groups.items()}

    return TCAReport(
        n_fills=len(items),
        n_outliers=int(outliers.sum()),
        total_notional=float(notionals.sum()),
        weighted_cost_bps=_weighted(costs, notionals),
        median_cost_bps=float(np.median(costs)),
        commission_bps=_weighted(commissions, notionals),
        model_error_bps=model_error,
        predicted_cost_bps=predicted_bps,
        fitted_impact_coefficient=fitted,
        calibration_n=calibration_n,
        by_symbol=_group(lambda e: e.fill.instrument.symbol),
        by_strategy=_group(lambda e: e.fill.strategy or "unattributed"),
    )


@dataclass(frozen=True, slots=True)
class LoadedExecutions:
    """What could be analysed, and what had to be dropped.

    The second number is not bookkeeping. A TCA report over 40 of 900 fills is
    a report about 40 fills, and presenting it as the desk's execution quality
    is the most common way this analysis misleads. Both counts travel together
    so a caller cannot quote the first without the second.
    """

    executions: list[Execution]
    n_rows: int
    n_missing_decision: int

    @property
    def coverage(self) -> float:
        return len(self.executions) / self.n_rows if self.n_rows else 0.0


def executions_from_store(store: Store, limit: int = 5000) -> LoadedExecutions:
    """Read fills joined to the decision that caused them.

    A fill whose decision price is unknown — a bracket leg, a manual trade, or
    anything recorded before the schema carried the column — is **dropped**,
    not defaulted. Filling in the fill's own price would report zero cost for
    exactly the executions nobody was watching.
    """
    rows = store.fills_with_decisions(limit=limit)
    executions: list[Execution] = []
    missing = 0
    for row in rows:
        reference = row.get("reference_price")
        if reference is None or float(reference) <= 0:
            missing += 1
            continue
        raw_class = row.get("asset_class")
        instrument = Instrument(
            symbol=str(row["symbol"]),
            asset_class=AssetClass(raw_class) if raw_class else AssetClass.EQUITY,
            point_value=float(row.get("point_value") or 1.0),
        )
        fill = Fill(
            order_id=int(row["order_id"]),
            instrument=instrument,
            side=Side(str(row["side"])),
            quantity=float(row["quantity"]),
            price=float(row["price"]),
            commission=float(row["commission"]),
            timestamp=pd.Timestamp(int(row["at_us"]), unit="us"),
            slippage=float(row["slippage"]),
            strategy=str(row["strategy"] or row.get("order_strategy") or ""),
        )
        adv = row.get("decision_adv")
        volatility = row.get("decision_volatility")
        executions.append(
            Execution(
                fill=fill,
                decision_price=float(reference),
                adv_notional=float(adv) if adv is not None else None,
                volatility=float(volatility) if volatility is not None else None,
            )
        )
    return LoadedExecutions(
        executions=executions, n_rows=len(rows), n_missing_decision=missing
    )
