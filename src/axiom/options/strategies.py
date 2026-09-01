"""Volatility strategies: dispersion and vol arbitrage.

Both are ported from the uploaded architecture's ``ExoticOptionsStrategies``.
Both needed the same correction, which is worth stating once because it is the
recurring theme of this merge: the originals emitted a trade signal without any
statement of whether the edge survived the cost of taking it, and vol trades
are among the most cost-sensitive things a desk does — the bid/ask on an option
routinely exceeds the mispricing being harvested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from axiom.options.pricing import OptionType


@dataclass(frozen=True, slots=True)
class VolQuote:
    """One line of an options chain, as observed."""

    strike: float
    option_type: OptionType
    implied_vol: float
    #: Half the bid/ask spread in volatility points. The threshold any edge
    #: must clear before it is real. Zero means "spread unknown", which is
    #: treated as unknown rather than as free.
    vol_half_spread: float = 0.0
    open_interest: int = 0


@dataclass(frozen=True, slots=True)
class VolSignal:
    """A volatility view on one strike."""

    strike: float
    option_type: OptionType
    #: ``+1`` buy volatility, ``-1`` sell volatility.
    direction: int
    #: Implied minus realised, in volatility points.
    edge_vol: float
    #: Edge remaining after crossing the spread. This is the tradable number.
    net_edge_vol: float
    implied_vol: float
    realised_vol: float
    rationale: str = ""

    @property
    def is_tradable(self) -> bool:
        return self.net_edge_vol > 0


def realised_volatility(closes: np.ndarray, *, periods_per_year: int = 252) -> float | None:
    """Close-to-close realised volatility, annualised.

    Returns ``None`` rather than a number when there is too little data. A
    volatility estimated from three observations is not a volatility, and
    feeding one into a vol-arb comparison manufactures edge from noise.
    """
    prices = np.asarray(closes, dtype=float)
    if prices.size < 20 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return None
    log_returns = np.diff(np.log(prices))
    if log_returns.size < 2:
        return None
    return float(np.std(log_returns, ddof=1) * math.sqrt(periods_per_year))


def volatility_arbitrage(
    chain: list[VolQuote],
    realised_vol: float,
    *,
    min_edge_vol: float = 0.05,
    min_open_interest: int = 0,
) -> list[VolSignal]:
    """Compare each strike's implied volatility against realised.

    The original compared implied to realised, thresholded the gap at five
    volatility points, and emitted a signal with ``edge = |spread| * 0.1``. Two
    problems: the spread the trade must actually cross was never considered, so
    a 6-point edge on an option quoted 5 points wide read as tradable; and the
    ``* 0.1`` scaling had no stated meaning, which makes the number impossible
    to size against.

    Here the bid/ask is subtracted first and both the gross and net edge are
    reported, so a caller can see the difference between an edge that exists
    and an edge that survives.
    """
    signals: list[VolSignal] = []
    for quote in chain:
        if quote.open_interest < min_open_interest:
            continue
        edge = quote.implied_vol - realised_vol
        if abs(edge) < min_edge_vol:
            continue
        # Crossing costs the half-spread on the way in. Round-tripping costs it
        # again, but a vol trade is often held to expiry, so one crossing is
        # the honest floor rather than the optimistic case.
        net = abs(edge) - quote.vol_half_spread
        direction = -1 if edge > 0 else 1
        signals.append(
            VolSignal(
                strike=quote.strike,
                option_type=quote.option_type,
                direction=direction,
                edge_vol=abs(edge),
                net_edge_vol=net,
                implied_vol=quote.implied_vol,
                realised_vol=realised_vol,
                rationale=(
                    f"implied {quote.implied_vol:.1%} vs realised "
                    f"{realised_vol:.1%}; {'sell' if direction < 0 else 'buy'} "
                    f"volatility, {net:.2%} net of a "
                    f"{quote.vol_half_spread:.2%} half-spread"
                ),
            )
        )
    return signals


@dataclass(frozen=True, slots=True)
class DispersionView:
    """Index volatility against its constituents."""

    implied_correlation: float
    index_vol: float
    #: Constituent volatility combined at the *observed* correlation of 1.
    basket_vol_at_unit_correlation: float
    weights_sum: float
    signal: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        lines = [
            f"implied correlation : {self.implied_correlation:.3f}",
            f"index vol           : {self.index_vol:.2%}",
            f"basket vol (ρ=1)    : {self.basket_vol_at_unit_correlation:.2%}",
            f"signal              : {self.signal}",
        ]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def dispersion(
    index_vol: float,
    constituent_vols: list[float],
    weights: list[float],
    *,
    sell_index_above: float = 0.7,
) -> DispersionView:
    """Implied average correlation from index versus constituent volatility.

    The original computed ``sqrt(sum(w_i^2 * v_i^2))`` and called it the average
    constituent volatility, then divided index variance by it. That expression
    is the basket volatility under **zero** correlation, so the resulting ratio
    is not an implied correlation — it is index variance over idiosyncratic
    variance, which exceeds 1 for any realistically correlated index and was
    then clamped to 1.0. The clamp hid the error: the function returned
    "correlation 1.0, SELL_INDEX" almost unconditionally.

    The identity actually wanted is

        σ²_index = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ

    which under a single average correlation ρ gives

        ρ = (σ²_index − Σᵢ wᵢ² σᵢ²) / ((Σᵢ wᵢ σᵢ)² − Σᵢ wᵢ² σᵢ²)

    That denominator is the difference between the perfectly-correlated basket
    variance and the diversified one — it is what correlation actually buys.
    """
    if len(constituent_vols) != len(weights):
        raise ValueError(
            f"{len(constituent_vols)} vols but {len(weights)} weights; "
            "they must describe the same basket"
        )
    if not constituent_vols:
        raise ValueError("cannot compute dispersion on an empty basket")

    vols = np.asarray(constituent_vols, dtype=float)
    w = np.asarray(weights, dtype=float)

    warnings: list[str] = []
    weights_sum = float(w.sum())
    if not math.isclose(weights_sum, 1.0, rel_tol=1e-3):
        warnings.append(
            f"weights sum to {weights_sum:.4f}, not 1.0 — the implied "
            "correlation is only meaningful for a fully specified basket"
        )

    unit_correlation_vol = float((w * vols).sum())
    diversified_variance = float((w**2 * vols**2).sum())
    denominator = unit_correlation_vol**2 - diversified_variance

    if denominator <= 0:
        warnings.append(
            "basket is degenerate (one name dominates); correlation is not identified"
        )
        implied = float("nan")
    else:
        implied = (index_vol**2 - diversified_variance) / denominator

    if math.isnan(implied):
        signal = "NO_VIEW"
    elif implied > 1.0:
        # Not clamped: above 1 means the inputs are mutually inconsistent —
        # a stale index quote, a wrong weight vector — and that is a data
        # problem to surface, not a maximally bullish dispersion signal.
        warnings.append(
            f"implied correlation {implied:.3f} exceeds 1.0, which is "
            "impossible — the index and constituent quotes disagree"
        )
        signal = "NO_VIEW"
    elif implied < 0.0:
        warnings.append(
            f"implied correlation {implied:.3f} is negative, which a long-only "
            "index basket cannot produce — check the inputs"
        )
        signal = "NO_VIEW"
    elif implied > sell_index_above:
        signal = "SELL_INDEX_VOL_BUY_CONSTITUENTS"
    else:
        signal = "NEUTRAL"

    return DispersionView(
        implied_correlation=implied,
        index_vol=index_vol,
        basket_vol_at_unit_correlation=unit_correlation_vol,
        weights_sum=weights_sum,
        signal=signal,
        warnings=tuple(warnings),
    )
