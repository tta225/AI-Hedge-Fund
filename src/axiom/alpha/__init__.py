"""Cross-sectional alpha: a universe of instruments, a roster of signal sources.

Complements the single-instrument ICT strategy layer rather than replacing it.
Everything here is causal by construction (:class:`~axiom.alpha.panel.Panel`
only serves history strictly before the bar being scored) and carries
provenance through blending, so a synthetic input cannot acquire a track record
by passing through an ensemble.
"""

from axiom.alpha.agents import MomentumAgent, StatArbAgent, VolumePressureAgent
from axiom.alpha.base import AlphaAgent, AlphaSignal, LookaheadError
from axiom.alpha.ensemble import AlphaEnsemble, BlendedView, default_ensemble
from axiom.alpha.panel import Panel

__all__ = [
    "AlphaAgent",
    "AlphaEnsemble",
    "AlphaSignal",
    "BlendedView",
    "LookaheadError",
    "MomentumAgent",
    "Panel",
    "StatArbAgent",
    "VolumePressureAgent",
    "default_ensemble",
]
