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
) -> SizingResult:
    """Size a position so a stop-out loses approximately ``risk_cash``.

    Args:
        allow_fractional: whether partial units are permitted. Defaults to True
            for crypto and FX, False for futures and equities, where you cannot
            buy a third of a contract.

    Raises:
        ValueError: if the stop is on the wrong side of entry or coincides with
            it, which would imply infinite size.
    """
    if risk_cash <= 0:
        raise ValueError(f"risk_cash must be positive, got {risk_cash}")

    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        raise ValueError(
            f"{instrument.symbol}: stop ({stop}) coincides with entry ({entry}); "
            "risk per unit would be zero and size unbounded"
        )
    if stop_distance < instrument.tick_size:
        raise ValueError(
            f"{instrument.symbol}: stop distance {stop_distance} is narrower than one "
            f"tick ({instrument.tick_size}); this is not an executable stop"
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
        )

    actual = quantity * risk_per_unit
    return SizingResult(
        quantity=float(quantity),
        risk_cash=actual,
        risk_per_unit=risk_per_unit,
        stop_distance=stop_distance,
        rationale=(
            f"{instrument.symbol}: stop {stop_distance:.4f} pts × "
            f"${instrument.point_value:,.2f}/pt = ${risk_per_unit:,.2f} risk per unit; "
            f"${risk_cash:,.2f} budget → {quantity:g} units "
            f"(${actual:,.2f} actual risk)"
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
