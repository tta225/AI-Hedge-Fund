"""Measured base rates for ICT features — the number that replaces the guess.

The confluence weights in :class:`~axiom.ict.engine.ICTConfig` are defensible
priors and nothing more. Nobody has ever measured, on this codebase, how often a
fair value gap is actually revisited or how often a swept pool actually
reverses. This module measures it.

The control is the whole point
------------------------------
"70% of fair value gaps get filled" sounds like an edge and is almost certainly
not one. Price is mean-reverting at short horizons and volatile at all of them:
*any* band of that height near the current price gets touched most of the time.
A rate without a control measures the market's general behaviour and attributes
it to the pattern.

So every measurement here carries a **mirror control** — the same geometry,
same height, same distance from price, reflected to the opposite side of the
confirming bar's close. If a gap is merely a band that price wanders back into,
the real rate and the mirror rate match. The interesting quantity is the
difference between them, reported as ``lift``.

Lift near zero is the expected result, and reporting it honestly is the job.

Statistics
----------
Rates carry **Wilson score intervals** rather than the normal approximation,
which is badly wrong for the small samples and near-1.0 rates this produces.
Lift carries an interval too: a lift of +4pp on 40 observations is noise, and
the interval is what says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import ICTConfig, ICTEngine
from axiom.ict.models import ICTState, LiquidityKind, StructureEventType

#: Default forward window, in bars, over which an outcome is judged. A feature
#: that is revisited 400 bars later is not the same claim as one revisited in
#: 20, and an unbounded horizon makes every rate approach 1.0.
DEFAULT_HORIZON = 48


def wilson_interval(hits: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because these samples are small and
    the rates sit near the boundaries, where the normal interval produces
    bounds below 0 or above 1 and badly understates uncertainty.
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = hits / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured rate, its control, and the uncertainty on both."""

    name: str
    hits: int
    trials: int
    #: Hits for the mirrored control geometry over the same trials.
    control_hits: int | None = None
    detail: str = ""

    @property
    def rate(self) -> float:
        return self.hits / self.trials if self.trials else float("nan")

    @property
    def control_rate(self) -> float | None:
        if self.control_hits is None or not self.trials:
            return None
        return self.control_hits / self.trials

    @property
    def lift(self) -> float | None:
        """Rate minus control rate, in percentage points of probability."""
        control = self.control_rate
        return None if control is None else self.rate - control

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.hits, self.trials)

    @property
    def lift_interval(self) -> tuple[float, float] | None:
        """Interval on the difference of two proportions measured on one sample.

        Treated as independent, which is *conservative in the wrong direction*
        here — the real and mirror measurements share the same price path, so
        they are positively correlated and the true interval is narrower. It is
        reported anyway because a lift that is not significant under the wider
        interval is definitely not significant, and this module's failure mode
        should be claiming too little rather than too much.
        """
        control = self.control_rate
        if control is None or not self.trials:
            return None
        p1, p2, n = self.rate, control, self.trials
        se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        diff = p1 - p2
        return (diff - 1.96 * se, diff + 1.96 * se)

    @property
    def is_significant(self) -> bool:
        """True when the lift interval excludes zero."""
        interval = self.lift_interval
        return interval is not None and (interval[0] > 0 or interval[1] < 0)

    def render(self) -> str:
        if not self.trials:
            return f"  {self.name:<34} no observations"
        lo, hi = self.interval
        line = (
            f"  {self.name:<34} {self.rate:6.1%}  "
            f"[{lo:.1%}–{hi:.1%}]  n={self.trials}"
        )
        control = self.control_rate
        if control is not None:
            lift = (self.lift or 0.0) * 100.0  # fraction → percentage points
            mark = "*" if self.is_significant else " "
            line += f"   control {control:5.1%}   lift {lift:+5.1f}pp{mark}"
        if self.detail:
            line += f"\n      {self.detail}"
        return line


@dataclass(slots=True)
class BaseRateReport:
    """Measured base rates for one series."""

    label: str
    provenance: Provenance
    horizon: int
    bars: int
    measurements: list[Measurement] = field(default_factory=list)

    @property
    def is_evidence(self) -> bool:
        """False for synthetic or mock input, however many transforms later."""
        return self.provenance.is_evidential

    def render(self) -> str:
        lines: list[str] = []
        if not self.is_evidence:
            lines.append(
                "!! NOT EVIDENCE — measured on non-real data. These numbers "
                "describe a generator, not a market."
            )
        lines.append(f"ICT base rates — {self.label}")
        lines.append(
            f"  {self.bars} bars · horizon {self.horizon} bars · "
            f"source {self.provenance.source} ({self.provenance.kind.value})"
        )
        lines.append("")
        lines.extend(m.render() for m in self.measurements)
        lines.append("")
        lines.append(
            "  control = same geometry mirrored to the other side of the "
            "confirming close.\n"
            "  lift    = rate - control. * marks a lift whose 95% interval "
            "excludes zero.\n"
            "  A lift near zero means the feature is not doing work the "
            "surrounding volatility\n  was not already doing."
        )
        return "\n".join(lines)


def _touched(
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    stop: int,
    bottom: float,
    top: float,
) -> bool:
    """True when any bar in ``[start, stop)`` traded inside ``[bottom, top]``."""
    if start >= stop:
        return False
    return bool(
        np.any((highs[start:stop] >= bottom) & (lows[start:stop] <= top))
    )


def measure_gap_revisits(
    series: OHLCVSeries, state: ICTState, horizon: int
) -> list[Measurement]:
    """How often a fair value gap is revisited, against the mirror control."""
    highs, lows, closes = series.highs, series.lows, series.closes
    n = len(series)

    ce_hits = ce_control = full_hits = trials = 0
    lags: list[int] = []

    for gap in state.fair_value_gaps:
        start = gap.confirmed_index + 1
        stop = min(start + horizon, n)
        if start >= stop or gap.height <= 0:
            continue
        trials += 1

        mid = gap.midpoint
        if _touched(highs, lows, start, stop, mid, mid):
            ce_hits += 1
            for i in range(start, stop):
                if lows[i] <= mid <= highs[i]:
                    lags.append(i - gap.confirmed_index)
                    break
        if _touched(highs, lows, start, stop, gap.bottom, gap.top) and _touched(
            highs, lows, start, stop, gap.entry_edge, gap.entry_edge
        ):
            full_hits += 1

        # Mirror: same height, same distance from the confirming close, other side.
        anchor = float(closes[gap.confirmed_index])
        mirror_mid = anchor - (mid - anchor)
        if _touched(highs, lows, start, stop, mirror_mid, mirror_mid):
            ce_control += 1

    median_lag = f"median {int(np.median(lags))} bars to CE" if lags else ""
    return [
        Measurement("FVG reaches 50% (CE)", ce_hits, trials, ce_control, median_lag),
        Measurement("FVG fully mitigated", full_hits, trials),
    ]


def measure_order_block_revisits(
    series: OHLCVSeries, state: ICTState, horizon: int
) -> list[Measurement]:
    """How often an order block is revisited, and whether it then holds."""
    highs, lows, closes = series.highs, series.lows, series.closes
    atr = series.atr()
    n = len(series)

    revisited = control = held = trials = held_trials = 0

    for block in state.order_blocks:
        start = block.confirmed_index + 1
        stop = min(start + horizon, n)
        if start >= stop or block.height <= 0:
            continue
        trials += 1

        touch_index: int | None = None
        for i in range(start, stop):
            if highs[i] >= block.bottom and lows[i] <= block.top:
                touch_index = i
                break

        if touch_index is not None:
            revisited += 1
            # "Held" = price moved one ATR in the block's direction before
            # closing through the far side of the candle. A block that is
            # revisited and then simply traded through is not support.
            held_trials += 1
            target_atr = float(atr[touch_index])
            sign = 1.0 if block.direction is Direction.BULLISH else -1.0
            invalid = block.candle_low if sign > 0 else block.candle_high
            for j in range(touch_index, min(touch_index + horizon, n)):
                if (closes[j] - invalid) * sign < 0:
                    break  # traded through first
                if (closes[j] - closes[touch_index]) * sign >= target_atr:
                    held += 1
                    break

        anchor = float(closes[block.confirmed_index])
        half = block.height / 2.0
        mirror_mid = anchor - (block.midpoint - anchor)
        if _touched(highs, lows, start, stop, mirror_mid - half, mirror_mid + half):
            control += 1

    return [
        Measurement("Order block revisited", revisited, trials, control),
        Measurement(
            "…then holds for 1 ATR",
            held,
            held_trials,
            detail="conditional on being revisited",
        ),
    ]


def _directional_outcome(
    closes: np.ndarray,
    atr: np.ndarray,
    anchor: int,
    start: int,
    stop: int,
    sign: float,
) -> bool:
    """True when price travels one ATR in ``sign``'s direction within the window."""
    base = float(closes[anchor])
    target = float(atr[anchor])
    return bool(np.any((closes[start:stop] - base) * sign >= target))


def measure_sweep_reversals(
    series: OHLCVSeries, state: ICTState, horizon: int
) -> list[Measurement]:
    """How often a liquidity sweep is followed by a reversal of one ATR.

    The control is the **same bar in the opposite direction**, not a randomly
    chosen bar. An earlier version used a random pivot and it failed the
    random-walk null test: sweeps are detected at moments of elevated
    volatility, so the ATR threshold at a sweep bar is systematically larger
    than at an average bar, and comparing the two measures that difference
    rather than the sweep.

    Anchoring the control to the same bar holds volatility, ATR and horizon
    fixed and isolates the only thing the methodology actually claims —
    **direction**.
    """
    closes = series.closes
    atr = series.atr()
    n = len(series)

    hits = control = trials = 0

    for sweep in state.sweeps:
        anchor = sweep.confirmed_index
        start = anchor + 1
        stop = min(start + horizon, n)
        if start >= stop:
            continue
        trials += 1

        # A buyside sweep collects stops above highs, so the reversal is down.
        sign = -1.0 if sweep.kind is LiquidityKind.BUYSIDE else 1.0
        if _directional_outcome(closes, atr, anchor, start, stop, sign):
            hits += 1
        if _directional_outcome(closes, atr, anchor, start, stop, -sign):
            control += 1

    return [
        Measurement(
            "Sweep reverses 1 ATR",
            hits,
            trials,
            control,
            detail="control = continuation instead, same bar and same threshold",
        )
    ]


def measure_structure_continuation(
    series: OHLCVSeries, state: ICTState, horizon: int
) -> list[Measurement]:
    """How often a structure break continues one ATR in its own direction.

    Control is the opposite direction from the same bar, for the reason given
    in :func:`measure_sweep_reversals`.
    """
    closes = series.closes
    atr = series.atr()
    n = len(series)

    out: list[Measurement] = []
    for event_type in (StructureEventType.BOS, StructureEventType.MSS):
        hits = control = trials = 0
        for event in state.structure_events:
            if event.event_type is not event_type:
                continue
            anchor = event.confirmed_index
            start = anchor + 1
            stop = min(start + horizon, n)
            if start >= stop:
                continue
            trials += 1
            sign = float(event.direction)
            if _directional_outcome(closes, atr, anchor, start, stop, sign):
                hits += 1
            if _directional_outcome(closes, atr, anchor, start, stop, -sign):
                control += 1
        out.append(
            Measurement(
                f"{event_type.value.upper()} continues 1 ATR",
                hits,
                trials,
                control,
                detail="control = the opposite direction, same bar",
            )
        )
    return out


def measure_base_rates(
    series: OHLCVSeries,
    *,
    config: ICTConfig | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> BaseRateReport:
    """Measure every ICT base rate on one series.

    Outcomes are always scanned forward from ``confirmed_index``, never from
    ``origin_index`` — measuring from the origin would credit the feature with
    moves that happened before it could be seen, which is the same lookahead
    error that produces fantasy equity curves.
    """
    state = ICTEngine(config or ICTConfig()).analyse(series)
    report = BaseRateReport(
        label=f"{series.instrument.symbol} {series.timeframe}",
        provenance=series.provenance,
        horizon=horizon,
        bars=len(series),
    )
    report.measurements.extend(measure_gap_revisits(series, state, horizon))
    report.measurements.extend(measure_order_block_revisits(series, state, horizon))
    report.measurements.extend(measure_sweep_reversals(series, state, horizon))
    report.measurements.extend(measure_structure_continuation(series, state, horizon))
    return report
