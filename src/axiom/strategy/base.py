"""Strategy interface.

:class:`StrategyContext` is the only view a strategy gets of the world, and
every accessor on it filters by ``confirmed_index <= context.index``. A strategy
therefore cannot read a feature the market had not yet formed, even by mistake.
Reaching around the context into ``context.ict`` directly is the one thing
strategies must not do — it is the door to lookahead bias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from axiom.core.series import OHLCVSeries
from axiom.core.sessions import SessionWindow, active_windows, session_label
from axiom.core.types import Bar, Direction, Instrument, OrderType
from axiom.ict.imbalance import unmitigated_gaps
from axiom.ict.liquidity import unswept_pools
from axiom.ict.models import (
    DealingRange,
    FairValueGap,
    ICTState,
    LiquidityKind,
    LiquidityPool,
    LiquiditySweep,
    OrderBlock,
    StructureEvent,
)
from axiom.ict.orderblocks import unmitigated_blocks
from axiom.portfolio.positions import Portfolio, Position
from axiom.risk.sizing import reward_risk_ratio


@dataclass(frozen=True, slots=True)
class Signal:
    """A proposed trade. Sizing is not a strategy's job — risk decides that."""

    direction: Direction
    entry: float
    stop: float
    targets: tuple[float, ...] = ()
    #: 0-1 conviction, used to rank competing signals. Not a probability.
    confidence: float = 0.5
    rationale: str = ""
    order_type: OrderType = OrderType.MARKET
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.direction is Direction.NEUTRAL:
            raise ValueError("a Signal must be directional")
        if self.direction is Direction.BULLISH and self.stop >= self.entry:
            raise ValueError(f"long signal stop {self.stop} is not below entry {self.entry}")
        if self.direction is Direction.BEARISH and self.stop <= self.entry:
            raise ValueError(f"short signal stop {self.stop} is not above entry {self.entry}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def primary_target(self) -> float | None:
        return self.targets[0] if self.targets else None

    @property
    def reward_risk(self) -> float | None:
        target = self.primary_target
        return None if target is None else reward_risk_ratio(self.entry, self.stop, target)

    @property
    def risk_points(self) -> float:
        return abs(self.entry - self.stop)


@dataclass(slots=True)
class StrategyContext:
    """Everything a strategy may see at bar ``index``."""

    series: OHLCVSeries
    index: int
    ict: ICTState
    portfolio: Portfolio
    timestamp: pd.Timestamp

    @property
    def instrument(self) -> Instrument:
        return self.series.instrument

    @property
    def bar(self) -> Bar:
        return self.series.bar_at(self.index)

    @property
    def price(self) -> float:
        return float(self.series.closes[self.index])

    @property
    def position(self) -> Position:
        return self.portfolio.position(self.instrument)

    @property
    def is_flat(self) -> bool:
        return self.position.is_flat

    def atr(self, period: int = 14) -> float:
        return float(self.series.atr(period)[self.index])

    # --- ICT accessors, all index-filtered ---------------------------------

    @property
    def bias(self) -> Direction:
        events = self.structure_events()
        return events[-1].direction if events else Direction.NEUTRAL

    def structure_events(self, within: int | None = None) -> list[StructureEvent]:
        events = [e for e in self.ict.structure_events if e.is_known_at(self.index)]
        if within is not None:
            events = [e for e in events if self.index - e.confirmed_index <= within]
        return events

    def active_fvgs(self, direction: Direction | None = None) -> list[FairValueGap]:
        return unmitigated_gaps(self.ict.fair_value_gaps, self.index, direction)

    def active_order_blocks(self, direction: Direction | None = None) -> list[OrderBlock]:
        return unmitigated_blocks(self.ict.order_blocks, self.index, direction)

    def unswept_pools(self, kind: LiquidityKind | None = None) -> list[LiquidityPool]:
        return unswept_pools(self.ict.liquidity_pools, self.index, kind)

    def recent_sweeps(self, within: int = 20) -> list[LiquiditySweep]:
        return [
            s
            for s in self.ict.sweeps
            if s.is_known_at(self.index) and self.index - s.confirmed_index <= within
        ]

    @property
    def dealing_range(self) -> DealingRange | None:
        dr = self.ict.dealing_range
        return dr if dr is not None and dr.is_known_at(self.index) else None

    def draw_on_liquidity(self, direction: Direction) -> LiquidityPool | None:
        """Nearest unswept pool in ``direction`` — the natural target."""
        kind = LiquidityKind.BUYSIDE if direction is Direction.BULLISH else LiquidityKind.SELLSIDE
        price = self.price
        pools = [
            p
            for p in self.unswept_pools(kind)
            if (p.price > price) == (direction is Direction.BULLISH)
        ]
        return min(pools, key=lambda p: abs(p.price - price)) if pools else None

    # --- session helpers ----------------------------------------------------

    @property
    def session(self) -> str:
        return session_label(self.timestamp)

    def in_window(self, window: SessionWindow) -> bool:
        return window in active_windows(self.timestamp)

    def in_any_window(self, windows: tuple[SessionWindow, ...]) -> bool:
        active = set(active_windows(self.timestamp))
        return any(w in active for w in windows)


class Strategy(ABC):
    """Base class for signal generators."""

    name: str = "strategy"

    def __init__(self, **params: object) -> None:
        self.params = params

    @abstractmethod
    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Return a proposed entry for this bar, or None."""

    def should_exit(self, context: StrategyContext) -> str | None:
        """Optional discretionary exit, checked while a position is open.

        Return a reason string to flatten. Protective stops and targets are
        handled by bracket orders and do not go through here.
        """
        return None

    def describe(self) -> str:
        if not self.params:
            return self.name
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.name}({rendered})"


@dataclass(slots=True)
class StrategyResult:
    """A signal plus the bar it was generated on, for the audit trail."""

    signal: Signal
    index: int
    timestamp: pd.Timestamp
    strategy: str
    context_notes: list[str] = field(default_factory=list)
