"""Base rates conditioned on the ICT session windows.

The question this finally answers
---------------------------------
`docs/REAL_DATA_FINDINGS.md` closed with one unambiguous statement of what was
missing:

    Killzones, judas swings, Power of Three, the Silver Bullet window and
    everything else anchored to the equities session are **not tested here at
    all**. Those session concepts are the load-bearing half of the methodology
    and they remain entirely unmeasured.

They were unmeasurable because the only real data available was crypto. BTC
trades 24/7 — there is no cash open, no 09:30 auction, no London close. A
"killzone" on BTC is a slice of the clock with nothing behind it.

With equities history, the question becomes answerable: **does conditioning an
ICT feature on a killzone change its base rate?**

Why unconditional measurement was never the whole test
------------------------------------------------------
`REAL_DATA_FINDINGS.md` also gave ICT its due:

    ICT's own claim is that these features work **in confluence and in
    context**. Measuring each feature in isolation and finding nothing does not
    refute the conditional claim.

This module tests the conditional claim directly, on the one conditioning
variable ICT weights most heavily. If killzone-conditioned lift is materially
above unconditional lift, the session concepts are doing work. If it is not,
that defence is spent.

The multiple-comparison trap, and the control that survives it
--------------------------------------------------------------
There are five killzones, three Silver Bullet windows and several sessions. Test
a feature in every one and something will look significant — that is arithmetic,
not discovery.

Two defences, both applied:

* **The control travels with the window.** Each window's rate is compared to the
  *same mirrored control measured inside that same window*, so anything the
  window does to volatility hits rate and control alike. A killzone that merely
  raises volatility moves both and shows no lift.
* **`sessions_tested` is reported**, so a reader can see how many windows were
  searched before the best one was quoted. A significant result found after
  eleven tests is not the same object as one found after one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.sessions import (
    KILLZONES,
    SESSIONS,
    SILVER_BULLETS,
    SessionWindow,
    window_mask,
)
from axiom.ict.engine import ICTConfig, ICTEngine
from axiom.ict.models import ICTState
from axiom.research.base_rates import (
    DEFAULT_HORIZON,
    Measurement,
    measure_gap_revisits,
    measure_order_block_revisits,
    measure_structure_continuation,
    measure_sweep_reversals,
)

__all__ = ["SessionRateReport", "SessionSlice", "measure_session_rates"]

#: Below this many observations a window's rate is reported but not compared —
#: a 3-of-4 hit rate is not evidence about anything.
MIN_TRIALS = 30


@dataclass(slots=True)
class SessionSlice:
    """Every measurement restricted to one session window."""

    window: SessionWindow
    bars: int
    measurements: list[Measurement] = field(default_factory=list)

    def by_name(self, name: str) -> Measurement | None:
        return next((m for m in self.measurements if m.name == name), None)

    @property
    def usable(self) -> list[Measurement]:
        return [m for m in self.measurements if m.trials >= MIN_TRIALS]


@dataclass(slots=True)
class SessionRateReport:
    """Unconditional rates, plus the same rates inside each session window."""

    label: str
    provenance: Provenance
    horizon: int
    bars: int
    baseline: list[Measurement] = field(default_factory=list)
    slices: list[SessionSlice] = field(default_factory=list)

    @property
    def is_evidence(self) -> bool:
        return self.provenance.is_evidential

    @property
    def sessions_tested(self) -> int:
        return len(self.slices)

    def conditional_gain(self, name: str) -> list[tuple[str, float]]:
        """How much each window's lift exceeds the unconditional lift.

        This is the number the whole module exists to produce. A positive value
        means the feature works *better* inside that window than it does in
        general — which is what "context matters" has to mean if it means
        anything measurable.
        """
        base = next((m for m in self.baseline if m.name == name), None)
        if base is None or base.lift is None:
            return []
        out: list[tuple[str, float]] = []
        for slice_ in self.slices:
            measurement = slice_.by_name(name)
            if measurement is None or measurement.trials < MIN_TRIALS:
                continue
            if measurement.lift is None:
                continue
            out.append((slice_.window.name, measurement.lift - base.lift))
        return sorted(out, key=lambda kv: -kv[1])

    def render(self) -> str:
        lines: list[str] = []
        if not self.is_evidence:
            lines += [
                "⚠️  NOT MARKET DATA — these rates describe generated bars.",
                "",
            ]
        lines.append(f"Session-conditional ICT base rates — {self.label}")
        lines.append(
            f"  {self.bars:,} bars · horizon {self.horizon} bars · "
            f"{self.sessions_tested} windows tested · source "
            f"{self.provenance.source} ({self.provenance.kind.value})"
        )
        lines.append("")

        for base in self.baseline:
            if base.lift is None or base.trials < MIN_TRIALS:
                continue
            lines.append(f"  {base.name}")
            lines.append(
                f"    unconditional          {base.rate:>6.1%}  "
                f"control {base.control_rate:>6.1%}  "
                f"lift {base.lift * 100:>+5.1f}pp   n={base.trials}"
            )
            gains = self.conditional_gain(base.name)
            if not gains:
                lines.append(
                    f"    no window reached {MIN_TRIALS} observations"
                )
                lines.append("")
                continue
            for window_name, gain in gains:
                slice_ = next(s for s in self.slices if s.window.name == window_name)
                measurement = slice_.by_name(base.name)
                assert measurement is not None
                star = "*" if measurement.is_significant else " "
                lift = measurement.lift
                if lift is None:  # filtered by conditional_gain; narrows the type
                    continue
                lines.append(
                    f"      {window_name:<18} {measurement.rate:>6.1%}  "
                    f"control {measurement.control_rate:>6.1%}  "
                    f"lift {lift * 100:>+5.1f}pp{star}  "
                    f"vs uncond. {gain * 100:>+5.1f}pp   n={measurement.trials}"
                )
            lines.append("")

        lines += [
            "  lift        = rate − control, both measured INSIDE that window, so a",
            "                window that merely raises volatility moves both and",
            "                shows no lift.",
            "  vs uncond.  = how much better the feature does in this window than",
            "                it does overall. This is the session claim, measured.",
            f"  * marks a lift whose 95% interval excludes zero. {self.sessions_tested} windows",
            "    were tested, so treat a single star with the suspicion it deserves.",
        ]
        return "\n".join(lines)


def _restrict(state: ICTState, keep: np.ndarray) -> ICTState:
    """A view of `state` holding only features confirmed inside `keep`.

    Filters on `confirmed_index`, never `origin_index`. A feature that formed
    outside a killzone but was only *confirmed* inside it belongs to the window
    — confirmation is when it became tradeable, and that is the only moment a
    session claim can attach to.
    """

    def inside(feature: object) -> bool:
        index = int(getattr(feature, "confirmed_index", -1))
        return 0 <= index < len(keep) and bool(keep[index])

    return ICTState(
        symbol=state.symbol,
        timeframe=state.timeframe,
        as_of=state.as_of,
        index=state.index,
        swings=[s for s in state.swings if inside(s)],
        structure_events=[e for e in state.structure_events if inside(e)],
        fair_value_gaps=[g for g in state.fair_value_gaps if inside(g)],
        order_blocks=[b for b in state.order_blocks if inside(b)],
        liquidity_pools=[p for p in state.liquidity_pools if inside(p)],
        sweeps=[s for s in state.sweeps if inside(s)],
        dealing_range=state.dealing_range,
        smt=state.smt,
        bias=state.bias,
        session=state.session,
    )


def _all_measurements(
    series: OHLCVSeries, state: ICTState, horizon: int
) -> list[Measurement]:
    return [
        *measure_gap_revisits(series, state, horizon),
        *measure_order_block_revisits(series, state, horizon),
        *measure_sweep_reversals(series, state, horizon),
        *measure_structure_continuation(series, state, horizon),
    ]


def measure_session_rates(
    series: OHLCVSeries,
    *,
    config: ICTConfig | None = None,
    horizon: int = DEFAULT_HORIZON,
    windows: tuple[SessionWindow, ...] | None = None,
) -> SessionRateReport:
    """Measure every ICT base rate unconditionally, then inside each window.

    The ICT engine runs **once**, on the full series. Windows then select which
    confirmed features count. Re-running the engine per window would be both
    slower and wrong: swing detection and structure need the surrounding bars,
    and a killzone-only series has none of them.
    """
    engine = ICTEngine(config or ICTConfig())
    state = engine.analyse(series)

    report = SessionRateReport(
        label=f"{series.instrument.symbol} {series.timeframe}",
        provenance=series.provenance,
        horizon=horizon,
        bars=len(series),
        baseline=_all_measurements(series, state, horizon),
    )

    index = pd.DatetimeIndex(series.index)

    # Several catalogued windows share identical clock bounds — `london` and
    # `london_kz` are both 02:00–05:00, as are `asia` and `asia_kz`. They are
    # one window under two names, and measuring both would report the same
    # result twice AND inflate `sessions_tested`, making the multiple-comparison
    # disclosure understate itself in the one direction that flatters a finding.
    seen: dict[tuple[object, object], SessionWindow] = {}
    for window in windows or (*KILLZONES, *SILVER_BULLETS, *SESSIONS):
        seen.setdefault((window.start, window.end), window)

    for window in seen.values():
        mask = window_mask(index, window)
        bars_inside = int(mask.sum())
        if bars_inside == 0:
            continue
        report.slices.append(
            SessionSlice(
                window=window,
                bars=bars_inside,
                measurements=_all_measurements(
                    series, _restrict(state, mask), horizon
                ),
            )
        )
    return report
