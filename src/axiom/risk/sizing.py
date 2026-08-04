"""Position sizing.

Everything is risk-first: size is derived from the distance to the stop, never
from a fixed lot count. If the stop is wide the position is small, so the cash
at risk is constant across setups. A strategy that cannot state its stop cannot
be sized, and :func:`size_by_risk` raises rather than guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from axiom.core.types import AssetClass, Instrument


@dataclass(frozen=True, slots=True)
class SizingResult:
    """The outcome of a sizing calculation, with its arithmetic exposed.

    ``rationale`` is intentionally human-readable: a risk decision that cannot
    be explained in a sentence should not be executed.
    """

    quantity: float
    risk_cash: float
    risk_per_unit: float
    stop_distance: float
    rationale: str
    #: Stable machine-readable reason, free of embedded numbers so that
    #: rejections aggregate. The rationale carries the specifics.
    code: str = "sized"

    @property
    def is_tradable(self) -> bool:
        return self.quantity > 0


def size_by_risk(
    instrument: Instrument,
    entry: float,
    stop: float,
    risk_cash: float,
    *,
    max_quantity: float | None = None,
    allow_fractional: bool | None = None,
    entry_tolerance: float = 0.0,
    exit_gap_allowance: float = 0.0,
) -> SizingResult:
    """Size a position so a stop-out loses **at most** ``risk_cash``.

    Args:
        entry_tolerance: how far, in price points, the actual fill may land
            *adverse* to ``entry``. Size is computed against the worst case
            ``|entry - stop| + entry_tolerance`` rather than the intended
            distance.

            This closes a gap that otherwise silently breaches the risk budget.
            A signal is generated on a bar's close, but the fill happens at the
            next bar's open plus slippage. When that open gaps away from the
            intended entry, the true distance to the stop is larger than the one
            the size was computed from, and the realised loss exceeds the budget
            — measured at roughly 1.3-1.4x on synthetic data before this fix.

            Sizing for the worst case makes the budget an upper bound instead of
            an estimate. The venue enforces the other half of the guarantee by
            refusing fills beyond the tolerance; see
            :attr:`~axiom.execution.base.Order.max_entry_deviation`.

        exit_gap_allowance: how far, in price points, the protective stop is
            expected to fill *beyond* its trigger. Budgeted for the same way.

            This covers the other half of the gap problem, and the halves are
            not symmetric. An entry that gaps can simply be refused — you are
            not in the position yet. A stop that gaps cannot: you already hold
            the position, and the market is where it is. No order type prevents
            it. The only defences are carrying less size, which is what this
            allowance does, or buying a hard floor with options.

            Gaps larger than this allowance remain genuine tail risk. They are
            not eliminated, and the backtest reports them via the
            ``worst_loss_vs_budget_x`` metric rather than hiding them.

        allow_fractional: whether partial units are permitted. Defaults to True
            for crypto and FX, False for futures and equities, where you cannot
            buy a third of a contract.

    Raises:
        ValueError: if the stop is on the wrong side of entry or coincides with
            it, which would imply infinite size.
    """
    if risk_cash <= 0:
        raise ValueError(f"risk_cash must be positive, got {risk_cash}")
    if entry_tolerance < 0:
        raise ValueError(f"entry_tolerance must be non-negative, got {entry_tolerance}")
    if exit_gap_allowance < 0:
        raise ValueError(
            f"exit_gap_allowance must be non-negative, got {exit_gap_allowance}"
        )

    intended_distance = abs(entry - stop)
    stop_distance = intended_distance + entry_tolerance + exit_gap_allowance
    if intended_distance <= 0:
        raise ValueError(
            f"{instrument.symbol}: stop ({stop}) coincides with entry ({entry}); "
            "risk per unit would be zero and size unbounded"
        )
    if intended_distance < instrument.tick_size:
        raise ValueError(
            f"{instrument.symbol}: stop distance {intended_distance} is narrower than "
            f"one tick ({instrument.tick_size}); this is not an executable stop"
        )

    if allow_fractional is None:
        allow_fractional = instrument.asset_class in (AssetClass.CRYPTO, AssetClass.FX)

    risk_per_unit = stop_distance * instrument.point_value
    raw = risk_cash / risk_per_unit
    quantity = raw if allow_fractional else math.floor(raw)

    if max_quantity is not None:
        quantity = min(quantity, max_quantity)

    if quantity <= 0:
        return SizingResult(
            quantity=0.0,
            risk_cash=0.0,
            risk_per_unit=risk_per_unit,
            stop_distance=stop_distance,
            rationale=(
                f"{instrument.symbol}: risking ${risk_cash:,.2f} over a "
                f"{stop_distance:.4f} point stop affords {raw:.3f} units — "
                "below one tradable unit, so no position is taken"
            ),
            code="below_one_unit",
        )

    actual = quantity * risk_per_unit
    padding = entry_tolerance + exit_gap_allowance
    tolerance_note = (
        f" (incl. {entry_tolerance:.4f} entry + {exit_gap_allowance:.4f} exit-gap"
        f" allowance)" if padding > 0 else ""
    )
    return SizingResult(
        quantity=float(quantity),
        risk_cash=actual,
        risk_per_unit=risk_per_unit,
        stop_distance=stop_distance,
        rationale=(
            f"{instrument.symbol}: worst-case stop {stop_distance:.4f} pts{tolerance_note} × "
            f"${instrument.point_value:,.2f}/pt = ${risk_per_unit:,.2f} risk per unit; "
            f"${risk_cash:,.2f} budget → {quantity:g} units "
            f"(${actual:,.2f} worst-case risk)"
        ),
    )


def size_by_equity_fraction(
    instrument: Instrument,
    entry: float,
    stop: float,
    equity: float,
    risk_pct: float,
    **kwargs: object,
) -> SizingResult:
    """Size from a percentage of account equity."""
    if not 0 < risk_pct <= 100:
        raise ValueError(f"risk_pct must be in (0, 100], got {risk_pct}")
    return size_by_risk(instrument, entry, stop, equity * risk_pct / 100.0, **kwargs)  # type: ignore[arg-type]


def reward_risk_ratio(entry: float, stop: float, target: float) -> float:
    """R-multiple of a setup. Below ~2 an ICT setup rarely justifies the entry."""
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("stop coincides with entry; reward:risk is undefined")
    return abs(target - entry) / risk
