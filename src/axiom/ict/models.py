"""Object model for ICT market-structure primitives.

Lookahead discipline
--------------------
Every object records two positions:

``origin_index``
    where the feature physically sits on the chart;
``confirmed_index``
    the first bar at which the feature could actually be *known*.

They differ constantly. A swing high with strength 3 sits at bar *i* but is only
confirmed at bar *i+3*. A fair value gap spans bars *i-1..i+1* but is only
visible once bar *i+1* closes. An order block is the candle *before* a
displacement leg, so it is confirmed only when that leg breaks structure —
often many bars later.

Backtests must filter on ``confirmed_index``, never ``origin_index``. Reading
the origin is how a strategy accidentally trades information it did not have,
and it is the single most common way an ICT backtest produces a fantasy equity
curve. :meth:`ICTFeature.is_known_at` is the guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from axiom.core.types import Direction


class SwingKind(str, Enum):
    HIGH = "high"
    LOW = "low"


class StructureEventType(str, Enum):
    #: Break of structure — continuation; price takes out the prior swing in
    #: the direction of the existing trend.
    BOS = "bos"
    #: Change of character — the first break against the prevailing trend.
    #: The earliest structural evidence that the delivery algorithm has flipped.
    CHOCH = "choch"
    #: Market structure shift — a CHoCH delivered with displacement, i.e. one
    #: that left an imbalance behind. The higher-conviction reversal signal.
    MSS = "mss"


class LiquidityKind(str, Enum):
    #: Resting buy stops above highs — what a bearish raid goes to collect.
    BUYSIDE = "buyside"
    #: Resting sell stops below lows.
    SELLSIDE = "sellside"

    @property
    def direction(self) -> Direction:
        """Direction price must travel to reach this liquidity."""
        return Direction.BULLISH if self is LiquidityKind.BUYSIDE else Direction.BEARISH


@dataclass(slots=True)
class ICTFeature:
    """Base for every detected feature."""

    origin_index: int
    confirmed_index: int
    timestamp: pd.Timestamp

    def is_known_at(self, index: int) -> bool:
        """True when this feature was already confirmed at bar ``index``."""
        return self.confirmed_index <= index

    @property
    def confirmation_lag(self) -> int:
        return self.confirmed_index - self.origin_index


@dataclass(slots=True)
class PriceZone(ICTFeature):
    """A feature occupying a price band rather than a single level."""

    top: float = 0.0
    bottom: float = 0.0

    def __post_init__(self) -> None:
        if self.bottom > self.top:
            self.top, self.bottom = self.bottom, self.top

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        """Consequent encroachment — the 50% level ICT treats as the true fill."""
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def overlaps(self, other: PriceZone) -> bool:
        return self.bottom <= other.top and other.bottom <= self.top

    def penetration(self, price: float) -> float:
        """How far into the zone ``price`` reached, as a fraction of height.

        0.0 = just touched the near edge, 1.0 = traded fully through. Sign is
        resolved by the subclass, which knows which edge price approaches from.
        """
        if self.height <= 0:
            return 1.0
        return min(max((self.top - price) / self.height, 0.0), 1.0)


@dataclass(slots=True)
class SwingPoint(ICTFeature):
    """A confirmed fractal pivot."""

    price: float = 0.0
    kind: SwingKind = SwingKind.HIGH
    strength: int = 2
    #: True once price has traded through this pivot's level.
    taken_index: int | None = None

    @property
    def is_high(self) -> bool:
        return self.kind is SwingKind.HIGH

    @property
    def liquidity_kind(self) -> LiquidityKind:
        return LiquidityKind.BUYSIDE if self.is_high else LiquidityKind.SELLSIDE


@dataclass(slots=True)
class StructureEvent(ICTFeature):
    """A break of a prior swing that carries structural meaning."""

    event_type: StructureEventType = StructureEventType.BOS
    direction: Direction = Direction.NEUTRAL
    #: The swing level that was broken.
    level: float = 0.0
    #: Close-through distance in ATR units — the displacement measure.
    displacement_atr: float = 0.0
    broken_swing_index: int | None = None

    @property
    def is_reversal(self) -> bool:
        return self.event_type in (StructureEventType.CHOCH, StructureEventType.MSS)


@dataclass(slots=True)
class FairValueGap(PriceZone):
    """A three-bar imbalance: bar 1's wick never meets bar 3's wick.

    Bullish FVG: ``high[i-1] < low[i+1]``. Bearish: ``low[i-1] > high[i+1]``.
    The unfilled band between them is inefficient delivery that price is
    expected to revisit and rebalance.
    """

    direction: Direction = Direction.NEUTRAL
    #: Largest fraction of the gap filled so far, in [0, 1].
    fill_ratio: float = 0.0
    #: First bar that traded through the 50% level (consequent encroachment).
    ce_touched_index: int | None = None
    #: First bar that fully closed the gap.
    mitigated_index: int | None = None
    #: Set once price closes decisively through the gap and then respects it
    #: from the other side — an inverse FVG, which flips polarity.
    inverted_index: int | None = None
    #: Displacement of the leg that created the gap, in ATR units.
    displacement_atr: float = 0.0

    @property
    def is_mitigated(self) -> bool:
        return self.mitigated_index is not None

    @property
    def is_inverted(self) -> bool:
        return self.inverted_index is not None

    @property
    def is_active(self) -> bool:
        """Still an unfilled imbalance available as a target or entry."""
        return not self.is_mitigated and not self.is_inverted

    @property
    def entry_edge(self) -> float:
        """The edge price returns to: the far side, relative to travel."""
        return self.top if self.direction is Direction.BULLISH else self.bottom


@dataclass(slots=True)
class OrderBlock(PriceZone):
    """The last opposing candle before a displacement leg that broke structure.

    A bullish order block is the final down-close candle before an up leg that
    displaced and broke a swing high. The zone spans that candle's low to high.
    """

    direction: Direction = Direction.NEUTRAL
    open_price: float = 0.0
    close_price: float = 0.0
    displacement_atr: float = 0.0
    #: True when the displacement leg left an FVG — a far stronger block.
    has_imbalance: bool = False
    #: First bar to trade back into the zone.
    mitigated_index: int | None = None
    #: Set when price closes through the block against its direction; the block
    #: then flips polarity and becomes a breaker.
    broken_index: int | None = None
    #: Index of the structure event this block is credited with causing.
    structure_event_index: int | None = None

    @property
    def is_breaker(self) -> bool:
        return self.broken_index is not None

    @property
    def is_active(self) -> bool:
        return self.broken_index is None

    @property
    def effective_direction(self) -> Direction:
        """A broken block trades in the opposite direction."""
        return self.direction.opposite if self.is_breaker else self.direction

    @property
    def entry_edge(self) -> float:
        """Proximal edge — the first price at which the block is engaged."""
        return self.top if self.direction is Direction.BULLISH else self.bottom

    @property
    def protective_edge(self) -> float:
        """Distal edge — beyond it the block has failed. Natural stop anchor."""
        return self.bottom if self.direction is Direction.BULLISH else self.top


@dataclass(slots=True)
class LiquidityPool(ICTFeature):
    """Resting stops: a swing extreme, or a cluster of near-equal extremes.

    Equal highs and lows are the highest-quality pools — obvious to every
    participant, therefore the most reliably targeted.
    """

    price: float = 0.0
    kind: LiquidityKind = LiquidityKind.BUYSIDE
    #: Bar positions of every extreme forming this pool.
    member_indices: tuple[int, ...] = ()
    #: Half-width of the price cluster, in ATR units.
    tolerance_atr: float = 0.0
    swept_index: int | None = None

    @property
    def size(self) -> int:
        return len(self.member_indices)

    @property
    def is_equal_cluster(self) -> bool:
        """Equal highs/lows — a stronger draw than a lone swing."""
        return self.size >= 2

    @property
    def is_swept(self) -> bool:
        return self.swept_index is not None

    @property
    def strength(self) -> float:
        """Relative draw-on-liquidity score. More touches, tighter cluster."""
        return self.size / (1.0 + self.tolerance_atr)


@dataclass(slots=True)
class LiquiditySweep(ICTFeature):
    """A raid: price trades beyond a pool, then rejects back inside.

    The rejection is what distinguishes a stop raid from a genuine breakout —
    liquidity was taken to fill orders, not to continue.
    """

    pool_price: float = 0.0
    kind: LiquidityKind = LiquidityKind.BUYSIDE
    #: How far beyond the pool price traded, in ATR units.
    penetration_atr: float = 0.0
    #: True when the bar closed back on the origin side of the pool.
    closed_back_inside: bool = False
    #: Set when a structure break confirms the reversal the raid implied.
    reversal_confirmed_index: int | None = None

    @property
    def reversal_direction(self) -> Direction:
        """Where price is expected to go after the raid — away from the pool."""
        return self.kind.direction.opposite


@dataclass(slots=True)
class DealingRange(ICTFeature):
    """The swing high/low pair defining current premium and discount.

    ICT's framing: only sell in premium, only buy in discount, measured against
    the range the algorithm is currently working within.
    """

    high: float = 0.0
    low: float = 0.0
    high_index: int = 0
    low_index: int = 0

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2.0

    def level(self, fraction: float) -> float:
        """Price at ``fraction`` of the range measured up from the low."""
        return self.low + self.height * fraction

    def position_of(self, price: float) -> float:
        """Where ``price`` sits in the range: 0.0 at the low, 1.0 at the high."""
        if self.height <= 0:
            return 0.5
        return (price - self.low) / self.height

    def is_premium(self, price: float) -> bool:
        return self.position_of(price) > 0.5

    def is_discount(self, price: float) -> bool:
        return self.position_of(price) < 0.5

    def optimal_trade_entry(self, direction: Direction) -> tuple[float, float]:
        """The OTE band — the 62%-79% retracement of the range.

        For longs this is measured down from the high; for shorts, up from the
        low. Returns ``(near_edge, far_edge)`` ordered low-to-high.
        """
        if direction is Direction.BULLISH:
            return self.level(1 - 0.79), self.level(1 - 0.62)
        return self.level(0.62), self.level(0.79)

    def in_ote(self, price: float, direction: Direction) -> bool:
        low, high = self.optimal_trade_entry(direction)
        return low <= price <= high


@dataclass(slots=True)
class SMTDivergence(ICTFeature):
    """Smart Money Technique divergence between correlated instruments.

    One market makes a higher high while its correlate fails to — the failure
    signals that the move is not supported by the broader algorithm and is
    likely a raid rather than expansion.
    """

    primary_symbol: str = ""
    correlate_symbol: str = ""
    direction: Direction = Direction.NEUTRAL
    primary_made_extreme: bool = False
    #: Price levels compared, for audit.
    primary_level: float = 0.0
    correlate_level: float = 0.0


@dataclass(slots=True)
class ICTState:
    """Complete ICT read of one instrument at one point in time."""

    symbol: str
    timeframe: str
    as_of: pd.Timestamp
    index: int

    swings: list[SwingPoint] = field(default_factory=list)
    structure_events: list[StructureEvent] = field(default_factory=list)
    fair_value_gaps: list[FairValueGap] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    liquidity_pools: list[LiquidityPool] = field(default_factory=list)
    sweeps: list[LiquiditySweep] = field(default_factory=list)
    dealing_range: DealingRange | None = None
    smt: list[SMTDivergence] = field(default_factory=list)

    #: Prevailing structural bias from the most recent structure events.
    bias: Direction = Direction.NEUTRAL
    #: Session/killzone label for ``as_of``.
    session: str = ""

    def known(self, index: int | None = None) -> ICTState:
        """A view containing only features confirmed at or before ``index``.

        This is the method backtests and live strategies must use.
        """
        i = self.index if index is None else index
        return ICTState(
            symbol=self.symbol,
            timeframe=self.timeframe,
            as_of=self.as_of,
            index=i,
            swings=[s for s in self.swings if s.is_known_at(i)],
            structure_events=[e for e in self.structure_events if e.is_known_at(i)],
            fair_value_gaps=[f for f in self.fair_value_gaps if f.is_known_at(i)],
            order_blocks=[o for o in self.order_blocks if o.is_known_at(i)],
            liquidity_pools=[p for p in self.liquidity_pools if p.is_known_at(i)],
            sweeps=[s for s in self.sweeps if s.is_known_at(i)],
            dealing_range=(
                self.dealing_range
                if self.dealing_range and self.dealing_range.is_known_at(i)
                else None
            ),
            smt=[d for d in self.smt if d.is_known_at(i)],
            bias=self.bias,
            session=self.session,
        )

    # The three accessors below report state **as of ``self.index``**, not the
    # final state at the end of the tape. The distinction matters: a gap filled
    # 200 bars later is still live today, and using the terminal ``is_active``
    # flag here would silently discard most live features and understate every
    # confluence score computed from them.

    def active_fvgs(self, direction: Direction | None = None) -> list[FairValueGap]:
        i = self.index
        gaps = [
            g
            for g in self.fair_value_gaps
            if g.is_known_at(i)
            and (g.mitigated_index is None or g.mitigated_index >= i)
            and (g.inverted_index is None or g.inverted_index > i)
        ]
        if direction is not None:
            gaps = [g for g in gaps if g.direction is direction]
        return gaps

    def active_order_blocks(self, direction: Direction | None = None) -> list[OrderBlock]:
        i = self.index
        blocks = [
            b
            for b in self.order_blocks
            if b.is_known_at(i)
            and (b.mitigated_index is None or b.mitigated_index >= i)
            and (b.broken_index is None or b.broken_index > i)
        ]
        if direction is not None:
            blocks = [b for b in blocks if b.effective_direction is direction]
        return blocks

    def unswept_pools(self, kind: LiquidityKind | None = None) -> list[LiquidityPool]:
        i = self.index
        pools = [
            p
            for p in self.liquidity_pools
            if p.is_known_at(i) and (p.swept_index is None or p.swept_index > i)
        ]
        if kind is not None:
            pools = [p for p in pools if p.kind is kind]
        return pools

    def nearest_draw(self, price: float, direction: Direction) -> LiquidityPool | None:
        """The nearest unswept pool in ``direction`` — the draw on liquidity."""
        kind = LiquidityKind.BUYSIDE if direction is Direction.BULLISH else LiquidityKind.SELLSIDE
        candidates = [
            p for p in self.unswept_pools(kind)
            if (p.price > price) == (direction is Direction.BULLISH)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p.price - price))

    @property
    def last_structure_event(self) -> StructureEvent | None:
        return self.structure_events[-1] if self.structure_events else None

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe} @ {self.as_of.isoformat()} "
            f"[{self.session}] bias={self.bias} · "
            f"{len(self.active_fvgs())} active FVG · "
            f"{len(self.active_order_blocks())} active OB · "
            f"{len(self.unswept_pools())} unswept pools · "
            f"{len(self.structure_events)} structure events"
        )
