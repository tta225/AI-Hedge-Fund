"""The ICT engine — one pass over a series producing a complete structural read.

Ordering matters and is not arbitrary:

1. **swings** — every other primitive is defined relative to pivots;
2. **structure** — pivots plus closes give BOS / CHoCH / MSS;
3. **imbalance** — FVGs, needed before order blocks so block quality can be
   graded on whether its leg left a gap;
4. **order blocks** — anchored to structure events;
5. **liquidity** — pools from pivots, then sweeps against those pools;
6. **sweep confirmation** — links raids to the structure breaks that validated
   them, which requires steps 2 and 5 to both be complete;
7. **dealing range** — the positional frame for premium/discount.

The engine is deliberately stateless: :meth:`ICTEngine.analyse` recomputes from
whatever bars it is handed. A backtest passes a growing slice, and because every
feature carries ``confirmed_index``, :meth:`ICTState.known` yields exactly the
view the market had at that bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from axiom.core.series import OHLCVSeries
from axiom.core.sessions import session_label
from axiom.core.types import Direction
from axiom.ict import (
    imbalance,
    liquidity,
    orderblocks,
    ranges,
    smt,
    structure,
)
from axiom.ict import (
    swings as swings_mod,
)
from axiom.ict.models import ICTState


@dataclass(frozen=True, slots=True)
class ICTConfig:
    """Tuning for the whole engine. One object to version and audit."""

    swing_strength: int = 2
    atr_period: int = 14
    #: Close-through, in ATR units, that promotes a CHoCH to an MSS.
    displacement_atr: float = structure.DEFAULT_DISPLACEMENT_ATR
    #: Minimum FVG height in ATR units.
    min_gap_atr: float = imbalance.DEFAULT_MIN_GAP_ATR
    #: Bars searched back from a structure break for the origin candle.
    order_block_lookback: int = orderblocks.DEFAULT_LOOKBACK
    #: Require the displacement leg to have left an FVG. Stricter, fewer blocks.
    require_order_block_imbalance: bool = False
    #: Use the origin candle's full range as the order-block zone instead of
    #: its body. The body is canonical; the range is a broader variant.
    order_block_use_candle_range: bool = False
    #: Cluster half-width, in ATR units, for equal highs/lows.
    equality_atr: float = liquidity.DEFAULT_EQUALITY_ATR
    #: Minimum excursion beyond a pool to register as a sweep.
    min_penetration_atr: float = liquidity.DEFAULT_MIN_PENETRATION_ATR
    #: Require the raid bar to close back inside the pool. True is ICT's
    #: definition — a sweep is a rejection, not a breakout — and it is the
    #: default because it is what every measurement in this repository was made
    #: against. It also means ``LiquiditySweep.closed_back_inside`` is
    #: universally True under the default config, so a strategy filtering on it
    #: empties its own population. Set False to admit breakout-type raids as
    #: well, which makes that field discriminating.
    sweep_require_close_back: bool = True
    #: Bars a sweep has to produce a confirming structure break.
    sweep_confirmation_bars: int = 20
    #: Lookback for the dealing range.
    dealing_range_lookback: int = 100

    def validate(self) -> None:
        if self.swing_strength < 1:
            raise ValueError("swing_strength must be >= 1")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.min_gap_atr < 0 or self.displacement_atr < 0:
            raise ValueError("ATR thresholds must be non-negative")


#: Stricter preset: fewer, higher-conviction features. A reasonable starting
#: point for live signal generation, where false positives cost money.
STRICT_CONFIG = ICTConfig(
    swing_strength=3,
    displacement_atr=0.5,
    min_gap_atr=0.15,
    require_order_block_imbalance=True,
    min_penetration_atr=0.1,
)


class ICTEngine:
    """Computes a full ICT read of a series."""

    def __init__(self, config: ICTConfig | None = None) -> None:
        self.config = config or ICTConfig()
        self.config.validate()

    def analyse(
        self,
        series: OHLCVSeries,
        *,
        correlate: OHLCVSeries | None = None,
        at: int | None = None,
    ) -> ICTState:
        """Analyse ``series``, reporting state as of bar ``at`` (default: last).

        Args:
            correlate: optional correlated instrument for SMT divergence.
            at: bar position to report at. Features confirmed after this bar are
                retained in the returned state but excluded by
                :meth:`ICTState.known`.
        """
        cfg = self.config
        index = len(series) - 1 if at is None else at
        if not 0 <= index < len(series):
            raise IndexError(f"bar {index} out of range for {len(series)} bars")

        pivots = swings_mod.find_swings(series, strength=cfg.swing_strength)

        events = structure.detect_structure(
            series,
            pivots,
            atr_period=cfg.atr_period,
            displacement_atr=cfg.displacement_atr,
        )

        gaps = imbalance.find_fair_value_gaps(
            series, atr_period=cfg.atr_period, min_gap_atr=cfg.min_gap_atr
        )

        blocks = orderblocks.find_order_blocks(
            series,
            events,
            gaps,
            lookback=cfg.order_block_lookback,
            atr_period=cfg.atr_period,
            require_imbalance=cfg.require_order_block_imbalance,
            use_candle_range=cfg.order_block_use_candle_range,
            swings=pivots,
        )

        pools = liquidity.find_liquidity_pools(
            series, pivots, atr_period=cfg.atr_period, equality_atr=cfg.equality_atr
        )
        sweeps = liquidity.find_sweeps(
            series,
            pools,
            atr_period=cfg.atr_period,
            min_penetration_atr=cfg.min_penetration_atr,
            require_close_back=cfg.sweep_require_close_back,
        )
        liquidity.confirm_sweep_reversals(
            sweeps, events, max_bars=cfg.sweep_confirmation_bars
        )

        dealing = ranges.build_dealing_range(
            series, pivots, index, lookback=cfg.dealing_range_lookback
        )

        divergences = []
        if correlate is not None:
            divergences = smt.find_smt_divergences(series, correlate, pivots)

        timestamp = series.index[index]
        return ICTState(
            symbol=series.instrument.symbol,
            timeframe=str(series.timeframe),
            as_of=timestamp,
            index=index,
            swings=pivots,
            structure_events=events,
            fair_value_gaps=gaps,
            order_blocks=blocks,
            liquidity_pools=pools,
            sweeps=sweeps,
            dealing_range=dealing,
            smt=divergences,
            bias=structure.current_bias(events, index),
            session=session_label(timestamp),
        )

    def analyse_known(
        self, series: OHLCVSeries, at: int, *, correlate: OHLCVSeries | None = None
    ) -> ICTState:
        """Analyse and immediately restrict to what was knowable at ``at``.

        This is the entry point strategies should use. It cannot leak future
        information even if a caller forgets to filter.
        """
        return self.analyse(series, correlate=correlate, at=at).known(at)


def confluence_score(state: ICTState, price: float, direction: Direction) -> float:
    """Heuristic 0-1 score for how much ICT confluence supports ``direction``.

    Deliberately simple and transparent — a weighted tally of independent
    confirmations rather than a fitted model. Its role is to *rank* candidate
    setups within a strategy, not to be a standalone edge. The weights are
    priors, not estimates, and should be replaced by measured hit rates once
    real data has been run through the backtester.
    """
    if direction is Direction.NEUTRAL:
        return 0.0

    score = 0.0
    weight_total = 0.0

    def add(condition: bool, weight: float) -> None:
        nonlocal score, weight_total
        weight_total += weight
        if condition:
            score += weight

    add(state.bias is direction, 0.25)

    if state.dealing_range is not None:
        positional = ranges.premium_discount_bias(state.dealing_range, price)
        add(positional is direction, 0.20)
        add(state.dealing_range.in_ote(price, direction), 0.10)
    else:
        weight_total += 0.30

    add(bool(state.active_order_blocks(direction)), 0.15)
    add(bool(state.active_fvgs(direction)), 0.15)

    recent_sweeps = [
        s for s in state.sweeps
        if s.reversal_direction is direction and state.index - s.confirmed_index <= 20
    ]
    add(bool(recent_sweeps), 0.15)

    return score / weight_total if weight_total else 0.0
