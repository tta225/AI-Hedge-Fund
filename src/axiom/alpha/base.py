"""Cross-sectional alpha: many instruments, many signals, one book.

AXIOM's existing strategy layer is single-instrument and pattern-driven — one
:class:`~axiom.core.series.OHLCVSeries` in, ICT structure out. That is the
right shape for structure trading and the wrong shape for everything a
quantitative desk does across a universe: momentum ranks instruments against
each other, statistical arbitrage is defined only relative to a factor model
fitted across names, and dispersion is a statement about an index versus its
constituents.

This package adds that second shape without disturbing the first. An
:class:`AlphaAgent` reads a :class:`~axiom.alpha.panel.Panel` — a cross-section
of instruments over aligned time — and emits :class:`AlphaSignal` objects that
the ensemble blends and the risk manager sizes.

Three properties are enforced here rather than left to each implementation,
because each is a way a cross-sectional backtest silently becomes fiction:

*Causality.* A signal carries the bar index it was computed at. Fitting a
factor model on a window that includes the bar being scored is the single most
common lookahead leak in statistical arbitrage, and it is invisible in the
output — the residual just looks unusually informative.

*Provenance.* A signal inherits the provenance of the bars it was computed
from, so a signal derived from synthetic data cannot be laundered into a
performance claim by passing through an ensemble.

*Measured confidence.* :attr:`AlphaSignal.confidence` must be derived from
something. A constant is not a confidence, and an ensemble that averages
constants produces a number that looks like a probability and is not one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from axiom.core.provenance import Provenance

if TYPE_CHECKING:
    from axiom.alpha.panel import Panel


class LookaheadError(RuntimeError):
    """A signal was computed from bars at or after the bar it is scoring."""


@dataclass(frozen=True, slots=True)
class AlphaSignal:
    """One agent's view on one instrument at one bar.

    Deliberately *not* an order and not a position. It is a scaled opinion; the
    ensemble combines opinions and the risk manager decides what, if anything,
    to do about the result.
    """

    symbol: str
    #: Direction and conviction on ``[-1, +1]``. Sign is the view; magnitude is
    #: relative strength within this agent's own scale, not across agents.
    signal: float
    #: How much to believe the signal, on ``[0, 1]``. Must be measured — see
    #: the module docstring on why a constant here is a defect.
    confidence: float
    #: The agent's own estimate of the move, in return terms. May be ``None``
    #: when the agent ranks without forecasting a magnitude.
    expected_return: float | None
    #: Bars ahead over which the view is expected to play out.
    horizon_bars: int
    strategy: str
    agent_id: str
    #: Bar index within the panel at which this was computed. Everything the
    #: signal depends on must come from strictly before it.
    as_of_index: int
    #: Provenance of the underlying bars, propagated so it cannot be lost.
    provenance: Provenance | None = None
    features_used: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.signal <= 1.0:
            raise ValueError(
                f"{self.agent_id}/{self.symbol}: signal {self.signal} outside [-1, 1]. "
                "Unscaled signals make the ensemble's weights meaningless."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"{self.agent_id}/{self.symbol}: confidence {self.confidence} "
                "outside [0, 1]"
            )

    @property
    def is_evidential(self) -> bool:
        """False when this signal traces back to generated bars."""
        return self.provenance is not None and self.provenance.is_evidential

    @property
    def weighted(self) -> float:
        """Signal scaled by belief — the quantity the ensemble actually blends."""
        return self.signal * self.confidence


class AlphaAgent(ABC):
    """A source of cross-sectional views.

    Distinct from :class:`~axiom.agents.base.Agent`, which is an LLM-backed
    analyst seat that interprets measured facts. An ``AlphaAgent`` *produces*
    measured facts and never calls a model.
    """

    #: Stable identifier. Appears in every signal and in the audit record.
    agent_id: str
    #: Minimum bars of history before this agent will emit anything.
    min_history: int = 2

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        """Emit views for bar ``index``, using only bars strictly before it."""

    def generate_checked(self, panel: Panel, index: int) -> list[AlphaSignal]:
        """:meth:`generate`, with the causality contract enforced.

        Calling this rather than :meth:`generate` is what makes "no lookahead"
        a property of the framework instead of a claim in each agent's
        docstring. The backtest path always uses it.
        """
        if index < self.min_history:
            return []
        signals = self.generate(panel, index)
        for signal in signals:
            if signal.as_of_index != index:
                raise LookaheadError(
                    f"{self.agent_id} emitted a signal stamped bar "
                    f"{signal.as_of_index} while scoring bar {index}. A signal "
                    "must be stamped with the bar it was computed at."
                )
            if signal.symbol not in panel.symbols:
                raise ValueError(
                    f"{self.agent_id} emitted a signal for {signal.symbol}, "
                    "which is not in the panel"
                )
        return signals
