"""Blending many alpha sources into one view per instrument.

This is the uploaded architecture's ``UnifiedQuantOrchestrator``, rebuilt. The
original combined model outputs with fixed weights and then returned
``{"prediction": combined, "confidence": 0.7}`` — a literal. That number is the
one the risk manager would size against, and it was the same whether every
source agreed or they were evenly split, whether six models spoke or one.

Three changes.

*Confidence is measured.* It comes from three things the blend actually knows:
how much of the roster reported, how strongly they agree in sign, and how
confident the contributors were in themselves. Disagreement lowers it, which is
the behaviour you want — an ensemble whose members contradict each other has
found no edge, it has found noise, and the whole point of running several
sources is to be able to notice that.

*Weights are renormalised over contributors.* The original multiplied each
prediction by its static weight and then took the mean over however many
happened to report, so a run where only the 0.1-weighted source fired produced
a signal an order of magnitude too small — and one where only the 0.2-weighted
source fired produced one too large. Weights now sum to one over the sources
that actually spoke.

*Provenance survives the blend.* A synthetic contributor makes the blended
signal non-evidential. Without this, an ensemble is a laundering step.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from axiom.alpha.base import AlphaAgent, AlphaSignal
from axiom.alpha.panel import Panel
from axiom.core.provenance import Provenance, weakest_kind


@dataclass(frozen=True, slots=True)
class BlendedView:
    """The desk's combined opinion on one instrument."""

    symbol: str
    #: Weighted, renormalised consensus on ``[-1, +1]``.
    signal: float
    #: Measured belief on ``[0, 1]``. See the module docstring.
    confidence: float
    #: Fraction of contributors whose sign matches the consensus.
    agreement: float
    #: Fraction of the roster that reported on this instrument.
    coverage: float
    contributors: tuple[str, ...] = ()
    provenance: Provenance | None = None
    per_agent: dict[str, float] = field(default_factory=dict)

    @property
    def is_evidential(self) -> bool:
        return self.provenance is not None and self.provenance.is_evidential

    @property
    def conviction(self) -> float:
        """Signal scaled by belief — what a sizing model should consume."""
        return self.signal * self.confidence

    def render(self) -> str:
        return (
            f"{self.symbol:<10} {self.signal:+.3f}  conf {self.confidence:.2f}  "
            f"agree {self.agreement:.0%}  cover {self.coverage:.0%}  "
            f"[{', '.join(self.contributors)}]"
        )


class AlphaEnsemble:
    """Runs a roster of alpha agents and blends what they say.

    Args:
        agents: the roster.
        weights: per-``agent_id`` prior weight. Missing entries default to an
            equal share. Renormalised over whoever actually reports.
        min_coverage: fraction of the roster that must report on an instrument
            before a view is emitted. A consensus of one is not a consensus.
    """

    def __init__(
        self,
        agents: list[AlphaAgent],
        *,
        weights: dict[str, float] | None = None,
        min_coverage: float = 0.0,
    ) -> None:
        if not agents:
            raise ValueError("an ensemble needs at least one agent")
        ids = [a.agent_id for a in agents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate agent_id in roster: {sorted(ids)}")
        self.agents = list(agents)
        self.weights = {a.agent_id: float((weights or {}).get(a.agent_id, 1.0)) for a in agents}
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("agent weights must be non-negative")
        self.min_coverage = min_coverage

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        """Every raw signal from the roster, causality-checked."""
        signals: list[AlphaSignal] = []
        for agent in self.agents:
            signals.extend(agent.generate_checked(panel, index))
        return signals

    def blend(self, signals: list[AlphaSignal]) -> list[BlendedView]:
        """Combine raw signals into one view per instrument."""
        by_symbol: dict[str, list[AlphaSignal]] = defaultdict(list)
        for signal in signals:
            by_symbol[signal.symbol].append(signal)

        views: list[BlendedView] = []
        for symbol, group in sorted(by_symbol.items()):
            coverage = len(group) / len(self.agents)
            if coverage < self.min_coverage:
                continue

            # Precision-weighted mean: each contributor's prior weight times
            # its own confidence. Confidence enters here, as a weight, and
            # nowhere else in the signal — folding it in a second time (by
            # blending `signal * confidence` and then reporting
            # `signal * confidence` again as conviction) would square it and
            # shrink every consensus toward zero for no stated reason.
            weights = [self.weights[s.agent_id] * s.confidence for s in group]
            total_weight = sum(weights)
            if total_weight <= 0:
                # Every contributor disclaimed belief. Fall back to the prior
                # weights so the direction still reports, and let the ensemble
                # confidence below carry the fact that nobody was sure.
                weights = [self.weights[s.agent_id] for s in group]
                total_weight = sum(weights)
                if total_weight <= 0:
                    continue
            blended = sum(w * s.signal for w, s in zip(weights, group, strict=True))
            blended /= total_weight

            signed = [np.sign(s.signal) for s in group if s.signal != 0.0]
            if signed:
                majority = np.sign(sum(signed)) or 1.0
                agreement = float(np.mean([1.0 if s == majority else 0.0 for s in signed]))
            else:
                agreement = 0.0

            mean_confidence = float(np.mean([s.confidence for s in group]))
            # Agreement is squared so a split roster is punished hard: 50/50
            # disagreement keeps a quarter of the belief, not half. An ensemble
            # that cannot agree with itself has found noise.
            confidence = float(
                np.clip(mean_confidence * (agreement**2) * coverage, 0.0, 1.0)
            )

            views.append(
                BlendedView(
                    symbol=symbol,
                    signal=float(np.clip(blended, -1.0, 1.0)),
                    confidence=confidence,
                    agreement=agreement,
                    coverage=coverage,
                    contributors=tuple(sorted(s.agent_id for s in group)),
                    provenance=_blend_provenance(group),
                    per_agent={s.agent_id: round(s.signal, 4) for s in group},
                )
            )
        return views

    def run(self, panel: Panel, index: int) -> list[BlendedView]:
        return self.blend(self.generate(panel, index))


def _blend_provenance(signals: list[AlphaSignal]) -> Provenance | None:
    """Weakest provenance across contributors. A blend cannot launder data."""
    sources = [s.provenance for s in signals if s.provenance is not None]
    if not sources:
        return None
    return Provenance(
        source="ensemble",
        kind=weakest_kind(p.kind for p in sources),
        detail=f"{len(sources)} contributing signal(s)",
    )


def default_ensemble() -> AlphaEnsemble:
    """The standing roster: trend, relative value, and participation.

    Equal prior weights. The uploaded architecture shipped hand-tuned weights
    (``tft: 0.2, gnn: 0.15, sentiment: 0.1``) with nothing behind them; equal
    weights at least state honestly that no evidence has yet been gathered
    about which source deserves more, and the ensemble's measured confidence
    does the discriminating that static weights were being asked to do.
    """
    from axiom.alpha.agents import MomentumAgent, StatArbAgent, VolumePressureAgent

    return AlphaEnsemble([MomentumAgent(), StatArbAgent(), VolumePressureAgent()])
