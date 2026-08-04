"""ICT strategies.

Both implement the same underlying model — *liquidity is taken, structure
shifts, price retraces into the imbalance left behind, then delivers to the
opposite liquidity* — but gate it differently:

:class:`SilverBulletStrategy`
    time-gated. Only trades the three one-hour Silver Bullet windows, entering
    on the first FVG formed after a same-session displacement.

:class:`LiquidityRaidReversal`
    event-gated. Trades any confirmed sweep of a liquidity pool that is followed
    by a market structure shift, entering on the retracement into the order
    block or FVG that the shift left behind.

Both size their stop from structure, not from a fixed distance, and target
opposing liquidity rather than a fixed R multiple. A trade with no identifiable
draw on liquidity is skipped — without a target there is nothing to size
reward against.
"""

from __future__ import annotations

from axiom.core.sessions import LONDON_KILLZONE, NY_AM_KILLZONE, NY_PM_KILLZONE, SILVER_BULLETS
from axiom.core.types import Direction
from axiom.ict.engine import confluence_score
from axiom.ict.models import StructureEventType
from axiom.strategy.base import Signal, Strategy, StrategyContext

#: Minimum reward:risk. Below this an ICT setup is not worth the commission.
DEFAULT_MIN_RR = 2.0
#: Stop buffer beyond the protective level, in ATR units.
DEFAULT_STOP_BUFFER_ATR = 0.15


class SilverBulletStrategy(Strategy):
    """Trade the Silver Bullet windows.

    Conditions, all required:

    1. the bar is inside a Silver Bullet window;
    2. a market structure shift occurred within ``mss_lookback`` bars — the
       algorithm has declared a direction;
    3. an unmitigated FVG exists in that direction and price is currently
       trading into it;
    4. price sits on the correct side of the dealing range (discount for longs,
       premium for shorts);
    5. an unswept liquidity pool exists ahead to target, giving at least
       ``min_rr`` reward:risk.
    """

    name = "ict_silver_bullet"

    def __init__(
        self,
        *,
        min_rr: float = DEFAULT_MIN_RR,
        mss_lookback: int = 12,
        stop_buffer_atr: float = DEFAULT_STOP_BUFFER_ATR,
        require_discount: bool = True,
        min_confluence: float = 0.5,
    ) -> None:
        super().__init__(
            min_rr=min_rr,
            mss_lookback=mss_lookback,
            stop_buffer_atr=stop_buffer_atr,
            require_discount=require_discount,
            min_confluence=min_confluence,
        )
        self.min_rr = min_rr
        self.mss_lookback = mss_lookback
        self.stop_buffer_atr = stop_buffer_atr
        self.require_discount = require_discount
        self.min_confluence = min_confluence

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.in_any_window(SILVER_BULLETS):
            return None

        shifts = [
            e
            for e in context.structure_events(within=self.mss_lookback)
            if e.event_type in (StructureEventType.MSS, StructureEventType.BOS)
        ]
        if not shifts:
            return None
        direction = shifts[-1].direction
        if direction is Direction.NEUTRAL:
            return None

        price = context.price
        gaps = [g for g in context.active_fvgs(direction) if g.contains(price)]
        if not gaps:
            return None
        # The most recent gap is the one the current leg left behind.
        gap = max(gaps, key=lambda g: g.confirmed_index)

        if self.require_discount:
            dealing = context.dealing_range
            if dealing is not None:
                if direction is Direction.BULLISH and dealing.is_premium(price):
                    return None
                if direction is Direction.BEARISH and dealing.is_discount(price):
                    return None

        score = confluence_score(context.ict.known(context.index), price, direction)
        if score < self.min_confluence:
            return None

        buffer = self.stop_buffer_atr * context.atr()
        stop = (gap.bottom - buffer) if direction is Direction.BULLISH else (gap.top + buffer)

        target_pool = context.draw_on_liquidity(direction)
        if target_pool is None:
            return None
        target = target_pool.price

        signal = _build(
            context, direction, price, stop, target, score,
            rationale=(
                f"Silver Bullet {context.session}: {shifts[-1].event_type.value.upper()} "
                f"{direction} {context.index - shifts[-1].confirmed_index} bars ago; price "
                f"retraced into {direction} FVG {gap.bottom:.2f}-{gap.top:.2f}; "
                f"targeting {target_pool.kind.value} liquidity at {target:.2f}"
            ),
            tags=("silver_bullet", context.session, shifts[-1].event_type.value),
            min_rr=self.min_rr,
        )
        return signal


class LiquidityRaidReversal(Strategy):
    """Trade the reversal after a stop raid.

    Conditions, all required:

    1. a liquidity pool was swept within ``sweep_lookback`` bars and price
       closed back inside — a raid, not a breakout;
    2. structure then shifted in the reversal direction (CHoCH or MSS);
    3. price has retraced into the order block or FVG that shift left behind;
    4. opposing liquidity exists to target at ``min_rr`` or better.

    The stop sits beyond the raid's extreme: if price returns through the level
    it just swept, the raid interpretation was wrong.
    """

    name = "ict_liquidity_raid"

    def __init__(
        self,
        *,
        min_rr: float = DEFAULT_MIN_RR,
        sweep_lookback: int = 24,
        stop_buffer_atr: float = DEFAULT_STOP_BUFFER_ATR,
        killzones_only: bool = True,
        min_confluence: float = 0.45,
    ) -> None:
        super().__init__(
            min_rr=min_rr,
            sweep_lookback=sweep_lookback,
            stop_buffer_atr=stop_buffer_atr,
            killzones_only=killzones_only,
            min_confluence=min_confluence,
        )
        self.min_rr = min_rr
        self.sweep_lookback = sweep_lookback
        self.stop_buffer_atr = stop_buffer_atr
        self.killzones_only = killzones_only
        self.min_confluence = min_confluence

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if self.killzones_only and not context.in_any_window(
            (LONDON_KILLZONE, NY_AM_KILLZONE, NY_PM_KILLZONE)
        ):
            return None

        sweeps = [s for s in context.recent_sweeps(self.sweep_lookback) if s.closed_back_inside]
        if not sweeps:
            return None
        sweep = max(sweeps, key=lambda s: s.confirmed_index)
        direction = sweep.reversal_direction

        confirmed = [
            e
            for e in context.structure_events(within=self.sweep_lookback)
            if e.direction is direction
            and e.is_reversal
            and e.confirmed_index > sweep.confirmed_index
        ]
        if not confirmed:
            return None

        price = context.price
        zone = _retracement_zone(context, direction, price)
        if zone is None:
            return None
        zone_label, protective = zone

        score = confluence_score(context.ict.known(context.index), price, direction)
        if score < self.min_confluence:
            return None

        # Stop beyond whichever is further: the swept extreme or the zone's far
        # edge. The raid level is the real invalidation.
        buffer = self.stop_buffer_atr * context.atr()
        if direction is Direction.BULLISH:
            stop = min(protective, sweep.pool_price) - buffer
        else:
            stop = max(protective, sweep.pool_price) + buffer

        target_pool = context.draw_on_liquidity(direction)
        if target_pool is None:
            return None

        return _build(
            context, direction, price, stop, target_pool.price, score,
            rationale=(
                f"{sweep.kind.value} liquidity raided at {sweep.pool_price:.2f} "
                f"({sweep.penetration_atr:.2f} ATR, closed back inside), "
                f"{confirmed[-1].event_type.value.upper()} confirmed reversal; "
                f"retraced into {zone_label}; targeting {target_pool.price:.2f}"
            ),
            tags=("liquidity_raid", context.session, zone_label),
            min_rr=self.min_rr,
        )


def _retracement_zone(
    context: StrategyContext, direction: Direction, price: float
) -> tuple[str, float] | None:
    """Find the order block or FVG price is currently trading back into.

    Order blocks are preferred: they carry a defined protective edge, whereas an
    FVG's far side is only an inefficiency boundary.
    """
    blocks = [b for b in context.active_order_blocks(direction) if b.contains(price)]
    if blocks:
        block = max(blocks, key=lambda b: b.confirmed_index)
        return "order_block", block.protective_edge

    gaps = [g for g in context.active_fvgs(direction) if g.contains(price)]
    if gaps:
        gap = max(gaps, key=lambda g: g.confirmed_index)
        return "fvg", gap.bottom if direction is Direction.BULLISH else gap.top

    return None


def _build(
    context: StrategyContext,
    direction: Direction,
    entry: float,
    stop: float,
    target: float,
    confidence: float,
    *,
    rationale: str,
    tags: tuple[str, ...],
    min_rr: float,
) -> Signal | None:
    """Validate geometry and reward:risk, then construct the signal."""
    instrument = context.instrument
    entry = instrument.round_to_tick(entry)
    stop = instrument.round_to_tick(stop)
    target = instrument.round_to_tick(target)

    risk = abs(entry - stop)
    if risk < instrument.tick_size:
        return None
    if direction is Direction.BULLISH and not stop < entry < target:
        return None
    if direction is Direction.BEARISH and not target < entry < stop:
        return None
    if abs(target - entry) / risk < min_rr:
        return None

    return Signal(
        direction=direction,
        entry=entry,
        stop=stop,
        targets=(target,),
        confidence=min(max(confidence, 0.0), 1.0),
        rationale=rationale,
        tags=tags,
    )
