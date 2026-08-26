"""Trade the one finding the data actually supports.

``docs/REAL_DATA_FINDINGS.md`` measured liquidity sweeps against a same-bar
opposite-direction control across four independent series and found the lift
**negative every time** — −17.8, −6.1, −6.4, −7.1 percentage points. After a
sweep, price continued in the direction of the raid more often than it reversed
against it.

ICT's claim is the opposite: a sweep marks exhaustion, and the reversal is the
trade. :class:`~axiom.strategy.ict_strategies.LiquidityRaidReversal` implements
that claim. This module implements its negation, because that is what was
measured.

**A directional tilt is not a strategy, and this file is the demonstration of
that gap rather than a refutation of it.** The base rate is unconditional: it
says nothing about where to put a stop, what to target, or whether the edge
survives commission and slippage. A 7pp tilt over a fixed horizon is thin
enough that entry timing and costs can consume all of it. That question is
settled by :class:`~axiom.research.lab.StrategyLab` walk-forward, not by
argument — which is exactly why this class exists in a form the lab can search.

Every gate is a constructor parameter so the lab can sweep it and the deflated
Sharpe can hold the winner to the number of trials it took to find.
"""

from __future__ import annotations

from axiom.core.types import Direction
from axiom.ict.models import LiquiditySweep
from axiom.strategy.base import Signal, Strategy, StrategyContext

#: Minimum reward:risk. Lower than the ICT strategies' 2.0 because continuation
#: targets the *next* pool rather than the far side of a range — a nearer draw,
#: so demanding 2.0 would reject most of the population the finding describes.
DEFAULT_MIN_RR = 1.2
#: Stop buffer beyond the swept pool, in ATR units.
DEFAULT_STOP_BUFFER_ATR = 0.25
#: Fallback target distance when no pool lies ahead, in ATR units.
DEFAULT_TARGET_ATR = 2.0


class SweepContinuationStrategy(Strategy):
    """Enter *with* a liquidity raid rather than against it.

    A buyside sweep — price raiding above a pool of highs — is read as a
    breakout that will extend, so the trade is long. The stop sits back inside
    the pool: if price returns below the level it just took, the continuation
    thesis is dead and there is nothing left to be right about. That is the one
    part of the setup ICT and this strategy agree on, because it is structural
    rather than directional.

    Args:
        entry_window: bars after the sweep's confirmation during which an entry
            is still valid. The measured effect was over a forward horizon, not
            at a precise bar, so this is genuinely a free parameter.
        require_hold: when True, only take sweeps that did **not** close back
            inside the pool. Closing back inside is ICT's reversal signature;
            excluding it should isolate the continuation population — and if
            the finding is real, this filter should *improve* the result. If it
            does not, the finding is likely an artefact of pooled populations.
        min_penetration_atr: ignore marginal pokes past a pool. A one-tick
            excursion is a data artefact as often as it is a raid.
        max_penetration_atr: ignore violent excursions. Past some distance the
            move has already happened and the entry is chasing.
        require_bias_alignment: only trade sweeps whose continuation direction
            agrees with the prevailing structural bias.
        min_rr: reject setups whose target does not pay for the stop.
    """

    name = "sweep_continuation"

    def __init__(
        self,
        *,
        entry_window: int = 3,
        require_hold: bool = True,
        min_penetration_atr: float = 0.10,
        max_penetration_atr: float = 2.0,
        require_bias_alignment: bool = False,
        min_rr: float = DEFAULT_MIN_RR,
        stop_buffer_atr: float = DEFAULT_STOP_BUFFER_ATR,
        target_atr: float = DEFAULT_TARGET_ATR,
    ) -> None:
        super().__init__(
            entry_window=entry_window,
            require_hold=require_hold,
            min_penetration_atr=min_penetration_atr,
            max_penetration_atr=max_penetration_atr,
            require_bias_alignment=require_bias_alignment,
            min_rr=min_rr,
            stop_buffer_atr=stop_buffer_atr,
            target_atr=target_atr,
        )
        self.entry_window = entry_window
        self.require_hold = require_hold
        self.min_penetration_atr = min_penetration_atr
        self.max_penetration_atr = max_penetration_atr
        self.require_bias_alignment = require_bias_alignment
        self.min_rr = min_rr
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.is_flat:
            return None

        sweep = self._select(context)
        if sweep is None:
            return None

        # The finding's direction: with the raid, not against it. `LiquiditySweep`
        # exposes `reversal_direction` because the methodology assumes reversal;
        # continuation is its opposite, which is `kind.direction`.
        direction = sweep.kind.direction
        if direction is Direction.NEUTRAL:
            return None
        if self.require_bias_alignment and context.bias is not direction:
            return None

        atr = context.atr()
        if atr <= 0:
            return None

        entry = context.price
        buffer = self.stop_buffer_atr * atr

        # Stop goes back inside the swept pool. Structural, not a fixed
        # distance: the thesis is "the level broke and held", so the level
        # failing to hold is the invalidation.
        if direction is Direction.BULLISH:
            stop = sweep.pool_price - buffer
            if stop >= entry:
                return None
        else:
            stop = sweep.pool_price + buffer
            if stop <= entry:
                return None

        risk = abs(entry - stop)
        if risk <= 0:
            return None

        target = self._target(context, direction, entry, atr)
        reward = abs(target - entry)
        if reward / risk < self.min_rr:
            return None

        return Signal(
            direction=direction,
            entry=entry,
            stop=stop,
            targets=(target,),
            # Deliberately flat rather than a function of penetration depth.
            # Nothing in the measurement graded conviction by depth, and a
            # confidence curve invented here would be a number with no evidence
            # behind it in a codebase whose whole premise is not doing that.
            confidence=0.5,
            rationale=(
                f"Sweep continuation: {sweep.kind.value} pool at "
                f"{sweep.pool_price:.2f} raided "
                f"{sweep.penetration_atr:.2f} ATR "
                f"{'and held' if not sweep.closed_back_inside else 'then closed back inside'}; "
                f"trading with the raid per measured base rates, target {target:.2f}"
            ),
            tags=("sweep_continuation", sweep.kind.value),
        )

    def _select(self, context: StrategyContext) -> LiquiditySweep | None:
        """The most recent qualifying sweep, or None."""
        candidates = [
            sweep
            for sweep in context.recent_sweeps(within=self.entry_window)
            if self._qualifies(sweep)
        ]
        if not candidates:
            return None
        # Most recent wins: an older sweep's continuation has already had its
        # chance to play out, and entering late is what the max-penetration
        # gate exists to avoid.
        return max(candidates, key=lambda s: s.confirmed_index)

    def _qualifies(self, sweep: LiquiditySweep) -> bool:
        if self.require_hold and sweep.closed_back_inside:
            return False
        return self.min_penetration_atr <= sweep.penetration_atr <= self.max_penetration_atr

    def _target(
        self, context: StrategyContext, direction: Direction, entry: float, atr: float
    ) -> float:
        """Next pool ahead, else a fixed ATR extension.

        Unlike the ICT strategies, a missing draw is not disqualifying. Those
        target the opposing liquidity that the reversal thesis says price is
        seeking; continuation has no such claim, so an ATR extension is an
        honest fallback rather than a substitute for a thesis.
        """
        draw = context.draw_on_liquidity(direction)
        if draw is not None:
            return float(draw.price)
        extension = self.target_atr * atr
        return entry + extension if direction is Direction.BULLISH else entry - extension


class SweepReversalControl(Strategy):
    """The methodology's version, as a control. Same gates, opposite direction.

    This is not a strategy anyone should trade — it exists so the lab evaluates
    both directions under identical entry, stop, target and cost assumptions.
    Without it, a positive continuation result could just as easily be an
    artefact of the entry logic as evidence about direction, and there would be
    no way to tell which.
    """

    name = "sweep_reversal_control"

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._inner = SweepContinuationStrategy(**params)  # type: ignore[arg-type]

    def evaluate(self, context: StrategyContext) -> Signal | None:
        signal = self._inner.evaluate(context)
        if signal is None:
            return None

        # Flip the direction and rebuild the levels symmetrically about entry,
        # so the control differs from the treatment in direction alone.
        entry = signal.entry
        risk = abs(entry - signal.stop)
        reward = abs(signal.targets[0] - entry) if signal.targets else risk
        direction = signal.direction.opposite
        if direction is Direction.BULLISH:
            stop, target = entry - risk, entry + reward
        else:
            stop, target = entry + risk, entry - reward

        return Signal(
            direction=direction,
            entry=entry,
            stop=stop,
            targets=(target,),
            confidence=signal.confidence,
            rationale="Control: ICT reversal reading of the same sweep",
            tags=("sweep_reversal_control",),
        )
