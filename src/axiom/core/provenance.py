"""Data provenance — the mechanism that makes "never invent numbers" enforceable.

Every :class:`~axiom.core.series.OHLCVSeries` carries a :class:`Provenance`
record naming where its bars came from. Provenance propagates through the ICT
engine, the backtester, and the metrics layer, so any performance report can
state *exactly* what data produced it.

The critical field is :attr:`Provenance.kind`. Synthetic and mock data are
first-class citizens for testing, but :func:`require_real_data` lets any
reporting or execution path refuse to treat them as evidence. A backtest run on
generated bars is a smoke test, never a track record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Self


class DataKind(str, Enum):
    """Where a series' numbers actually came from."""

    #: Observed market data from a real venue or vendor.
    REAL = "real"
    #: Deterministically generated. Valid for tests and demos; never evidence.
    SYNTHETIC = "synthetic"
    #: Hand-authored fixtures, typically to pin a known ICT pattern in tests.
    MOCK = "mock"
    #: Real data that has been resampled, adjusted, stitched, or cleaned.
    DERIVED = "derived"

    @property
    def is_evidential(self) -> bool:
        """True when numbers from this source may back a performance claim."""
        return self in {DataKind.REAL, DataKind.DERIVED}


class SyntheticDataError(RuntimeError):
    """Raised when non-evidential data reaches a path that requires real data."""


@dataclass(frozen=True, slots=True)
class Provenance:
    """An auditable record of a dataset's origin."""

    source: str
    kind: DataKind
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: Free-form provider detail: endpoint, dataset id, RNG seed, file path...
    detail: str = ""
    #: Chain of transformations applied since retrieval, oldest first.
    transforms: tuple[str, ...] = ()

    @classmethod
    def real(cls, source: str, detail: str = "") -> Self:
        return cls(source=source, kind=DataKind.REAL, detail=detail)

    @classmethod
    def synthetic(cls, source: str, seed: int | None = None) -> Self:
        return cls(
            source=source,
            kind=DataKind.SYNTHETIC,
            detail=f"seed={seed}" if seed is not None else "",
        )

    @classmethod
    def mock(cls, source: str, detail: str = "") -> Self:
        return cls(source=source, kind=DataKind.MOCK, detail=detail)

    @property
    def is_evidential(self) -> bool:
        return self.kind.is_evidential

    def derive(self, transform: str) -> Provenance:
        """Return provenance for data produced by applying ``transform``.

        Real data becomes DERIVED; synthetic/mock stay tainted, which is the
        whole point — you cannot launder generated bars into evidence by
        resampling them.
        """
        kind = DataKind.DERIVED if self.kind is DataKind.REAL else self.kind
        return replace(self, kind=kind, transforms=(*self.transforms, transform))

    def merge(self, other: Provenance) -> Provenance:
        """Combine two provenances, taking the weakest (least evidential) kind."""
        if self.kind.is_evidential and not other.kind.is_evidential:
            weakest = other.kind
        elif other.kind.is_evidential and not self.kind.is_evidential:
            weakest = self.kind
        else:
            weakest = self.kind if self.kind is not DataKind.REAL else other.kind
        return Provenance(
            source=f"{self.source}+{other.source}",
            kind=weakest,
            retrieved_at=min(self.retrieved_at, other.retrieved_at),
            detail="; ".join(d for d in (self.detail, other.detail) if d),
            transforms=(*self.transforms, *other.transforms),
        )

    @property
    def label(self) -> str:
        """Short human label, e.g. ``yfinance/real`` or ``synthetic/seed=7``."""
        suffix = self.detail or self.kind.value
        return f"{self.source}/{suffix}"

    def describe(self) -> str:
        parts = [f"source={self.source}", f"kind={self.kind.value}",
                 f"retrieved={self.retrieved_at.isoformat(timespec='seconds')}"]
        if self.detail:
            parts.append(f"detail={self.detail}")
        if self.transforms:
            parts.append("transforms=" + ">".join(self.transforms))
        return " ".join(parts)


def require_real_data(provenance: Provenance, context: str) -> None:
    """Guard a path that must not run on generated numbers.

    Raises:
        SyntheticDataError: if ``provenance`` is not evidential.
    """
    if not provenance.is_evidential:
        raise SyntheticDataError(
            f"{context} requires real market data, but the input is "
            f"{provenance.kind.value} ({provenance.describe()}). "
            "Results from generated data are not evidence of anything."
        )


def provenance_warning(provenance: Provenance) -> str | None:
    """Return a loud banner when results must not be read as a track record."""
    if provenance.is_evidential:
        return None
    return (
        f"⚠️  {provenance.kind.value.upper()} DATA — these numbers describe "
        f"generated bars ({provenance.label}), not market history. "
        "They are a correctness check, not performance."
    )
