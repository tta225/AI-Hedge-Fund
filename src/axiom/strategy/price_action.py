"""Discretionary price-action patterns, made systematic.

What this module is doing
-------------------------
These are the setups discretionary traders actually take: pin bars, engulfing
candles, inside-bar breaks, Wyckoff springs, double tops, retests of a broken
level. Each is encoded as an explicit, testable rule.

**Encoding them changes them.** Every source on price action says the same
thing — a pin bar at a random level is not a trade, a pin bar at a weekly
resistance in the direction of the daily trend is. The context is the setup; the
candle is the trigger. A discretionary trader supplies that context from
experience and will not take the pattern without it.

So each strategy here pairs the pattern with a machine-checkable context filter
— a trend filter, a prior level, a swing point. That is the closest honest
translation available, and it is still a *weaker* rule than the discretionary
version. A measured result here is evidence about the encoded rule, **not**
about whether skilled discretionary traders can trade the pattern. Those are
different claims and conflating them is the main way this kind of backtest
misleads.

Relationship to the ICT engine
------------------------------
There is deliberate overlap. ICT's order blocks are supply/demand zones, its
turtle soup is a Wyckoff spring, its rejection blocks are pin bars. Where a
strategy here restates an ICT concept in the vocabulary of another tradition,
the docstring says so — because two names for one observation must not be
counted as two confirmations, in a confluence score or anywhere else.
"""

from __future__ import annotations

import numpy as np

from axiom.core.types import Direction
from axiom.strategy.base import Signal, Strategy, StrategyContext
from axiom.strategy.quant_strategies import _atr_stop, _build

__all__ = [
    "DoubleTopBottom",
    "EngulfingReversal",
    "InsideBarBreakout",
    "LevelRetest",
    "PinBarRejection",
    "RoundNumberReversion",
    "ThreeBarReversal",
    "WyckoffSpring",
]


def _body(open_: float, close: float) -> float:
    return abs(close - open_)


class PinBarRejection(Strategy):
    """Long wick rejecting a level, taken only with the trend.

    A pin bar is a bar whose wick is a large fraction of its range and whose
    body sits at the opposite end — price pushed one way and was rejected before
    the close.

    Identical in substance to ICT's **rejection block**, arrived at from the
    retail price-action tradition. If both fire, that is one observation.

    The trend filter is what makes this testable rather than arbitrary. Without
    it the rule fires on roughly every third bar in a volatile market.
    """

    name = "pin_bar"
    requires_ict = False

    def __init__(
        self,
        *,
        wick_ratio: float = 0.6,
        max_body_ratio: float = 0.3,
        trend_filter: int = 50,
        stop_buffer_atr: float = 0.2,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            wick_ratio=wick_ratio, max_body_ratio=max_body_ratio,
            trend_filter=trend_filter, stop_buffer_atr=stop_buffer_atr,
            target_atr=target_atr,
        )
        self.wick_ratio = wick_ratio
        self.max_body_ratio = max_body_ratio
        self.trend_filter = trend_filter
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.trend_filter + 2:
            return None
        series = context.series
        high = float(series.highs[i])
        low = float(series.lows[i])
        open_ = float(series.opens[i])
        close = float(series.closes[i])
        span = high - low
        if span <= 0:
            return None

        body = _body(open_, close)
        if body / span > self.max_body_ratio:
            return None
        lower_wick = min(open_, close) - low
        upper_wick = high - max(open_, close)
        trend = float(np.mean(series.closes[i - self.trend_filter + 1 : i + 1]))

        if lower_wick / span >= self.wick_ratio and close > trend:
            direction = Direction.BULLISH
            stop = low - self.stop_buffer_atr * max(context.atr(), 1e-9)
            wick, side = lower_wick, "lower"
        elif upper_wick / span >= self.wick_ratio and close < trend:
            direction = Direction.BEARISH
            stop = high + self.stop_buffer_atr * max(context.atr(), 1e-9)
            wick, side = upper_wick, "upper"
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        offset = self.target_atr * atr
        target = close + offset if direction is Direction.BULLISH else close - offset
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, wick / span),
            rationale=(
                f"pin bar: {wick / span:.0%} {side} wick, body {body / span:.0%} "
                f"of range, aligned with the {self.trend_filter}-bar trend"
            ),
            tags=("price-action", "pin-bar", "rejection", context.session),
            min_rr=1.5,
        )


class EngulfingReversal(Strategy):
    """A bar whose body fully covers the previous bar's, at a swing extreme.

    Engulfing is the most-cited two-bar reversal pattern. The **swing-extreme
    requirement** is what stops it being noise: engulfing bars occur constantly
    mid-range and mean nothing there. Requiring the pattern to form at a local
    extreme is the systematic stand-in for a discretionary trader's "at a level".
    """

    name = "engulfing"
    requires_ict = False

    def __init__(
        self,
        *,
        extreme_lookback: int = 20,
        min_body_ratio: float = 1.2,
        stop_buffer_atr: float = 0.2,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            extreme_lookback=extreme_lookback, min_body_ratio=min_body_ratio,
            stop_buffer_atr=stop_buffer_atr, target_atr=target_atr,
        )
        self.extreme_lookback = extreme_lookback
        self.min_body_ratio = min_body_ratio
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.extreme_lookback + 2:
            return None
        series = context.series
        o0, c0 = float(series.opens[i - 1]), float(series.closes[i - 1])
        o1, c1 = float(series.opens[i]), float(series.closes[i])
        prev_body, body = _body(o0, c0), _body(o1, c1)
        if prev_body <= 0 or body / prev_body < self.min_body_ratio:
            return None

        window = slice(i - self.extreme_lookback, i)
        at_low = float(series.lows[i]) <= float(np.min(series.lows[window]))
        at_high = float(series.highs[i]) >= float(np.max(series.highs[window]))

        bullish = c1 > o1 and c0 < o0 and c1 >= o0 and o1 <= c0 and at_low
        bearish = c1 < o1 and c0 > o0 and c1 <= o0 and o1 >= c0 and at_high
        if bullish:
            direction = Direction.BULLISH
            stop = float(series.lows[i]) - self.stop_buffer_atr * max(context.atr(), 1e-9)
        elif bearish:
            direction = Direction.BEARISH
            stop = float(series.highs[i]) + self.stop_buffer_atr * max(context.atr(), 1e-9)
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        offset = self.target_atr * atr
        target = c1 + offset if direction is Direction.BULLISH else c1 - offset
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.4 + (body / prev_body - 1) / 3),
            rationale=(
                f"{'bullish' if bullish else 'bearish'} engulfing "
                f"({body / prev_body:.1f}× the prior body) at a "
                f"{self.extreme_lookback}-bar extreme"
            ),
            tags=("price-action", "engulfing", "reversal", context.session),
            min_rr=1.5,
        )


class InsideBarBreakout(Strategy):
    """Break of a bar contained entirely within its predecessor.

    An inside bar is a one-bar consolidation — the market declining to extend in
    either direction. The break of the mother bar's range is the resolution.

    Same premise as `NarrowRangeBreakout` and `BollingerSqueeze`, expressed as a
    two-bar pattern. All three are contraction-then-expansion rules and their
    signals correlate heavily.
    """

    name = "inside_bar"
    requires_ict = False

    def __init__(
        self,
        *,
        trend_filter: int = 50,
        stop_buffer_atr: float = 0.15,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            trend_filter=trend_filter, stop_buffer_atr=stop_buffer_atr,
            target_atr=target_atr,
        )
        self.trend_filter = trend_filter
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.trend_filter + 3:
            return None
        series = context.series
        mother_high = float(series.highs[i - 2])
        mother_low = float(series.lows[i - 2])
        inside = (
            float(series.highs[i - 1]) <= mother_high
            and float(series.lows[i - 1]) >= mother_low
        )
        if not inside:
            return None

        price = context.price
        trend = float(np.mean(series.closes[i - self.trend_filter + 1 : i + 1]))
        atr = max(context.atr(), context.instrument.tick_size)

        if price > mother_high and price > trend:
            direction = Direction.BULLISH
            stop = mother_low - self.stop_buffer_atr * atr
        elif price < mother_low and price < trend:
            direction = Direction.BEARISH
            stop = mother_high + self.stop_buffer_atr * atr
        else:
            return None

        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        compression = (mother_high - mother_low) / atr
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 1.2 / max(compression, 0.5)),
            rationale=(
                f"inside bar inside a {compression:.1f} ATR mother bar, "
                f"broken {'up' if direction is Direction.BULLISH else 'down'}"
            ),
            tags=("price-action", "inside-bar", "breakout", context.session),
            min_rr=1.5,
        )


class WyckoffSpring(Strategy):
    """A brief break of a range low that reclaims — the Wyckoff spring.

    Wyckoff's spring (and its mirror, the upthrust) is a false break through a
    range boundary that immediately reclaims: supply is exhausted below the
    range, so the break fails and the move goes the other way.

    **This is the same object as ICT's turtle soup**, and closely related to a
    liquidity sweep. Three traditions, three names, one pattern. The archive
    lists all three so that a confluence score built from them cannot silently
    triple-count one observation — and `docs/REAL_DATA_FINDINGS.md` measured the
    ICT version and found the reversal reading *negative* on real bars.
    """

    name = "wyckoff_spring"
    requires_ict = False

    def __init__(
        self,
        *,
        range_lookback: int = 40,
        max_penetration_atr: float = 1.0,
        reclaim_bars: int = 3,
        stop_buffer_atr: float = 0.25,
        target_atr: float = 3.5,
    ) -> None:
        super().__init__(
            range_lookback=range_lookback,
            max_penetration_atr=max_penetration_atr,
            reclaim_bars=reclaim_bars, stop_buffer_atr=stop_buffer_atr,
            target_atr=target_atr,
        )
        self.range_lookback = range_lookback
        self.max_penetration_atr = max_penetration_atr
        self.reclaim_bars = reclaim_bars
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.range_lookback + self.reclaim_bars + 2:
            return None
        series = context.series
        window = slice(i - self.range_lookback - self.reclaim_bars, i - self.reclaim_bars)
        range_high = float(np.max(series.highs[window]))
        range_low = float(np.min(series.lows[window]))
        atr = max(context.atr(), context.instrument.tick_size)
        recent = slice(i - self.reclaim_bars, i + 1)
        price = context.price

        broke_low = float(np.min(series.lows[recent]))
        broke_high = float(np.max(series.highs[recent]))

        # Spring: dipped below the range low, closed back inside, shallow break.
        if (
            broke_low < range_low
            and price > range_low
            and (range_low - broke_low) <= self.max_penetration_atr * atr
        ):
            direction = Direction.BULLISH
            stop = broke_low - self.stop_buffer_atr * atr
            depth = (range_low - broke_low) / atr
            kind = "spring below"
            level = range_low
        # Upthrust: the mirror image.
        elif (
            broke_high > range_high
            and price < range_high
            and (broke_high - range_high) <= self.max_penetration_atr * atr
        ):
            direction = Direction.BEARISH
            stop = broke_high + self.stop_buffer_atr * atr
            depth = (broke_high - range_high) / atr
            kind = "upthrust above"
            level = range_high
        else:
            return None

        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            # A shallow false break is the cleaner signal — a deep one is
            # increasingly indistinguishable from a genuine breakout.
            confidence=min(1.0, 1.0 - depth / self.max_penetration_atr * 0.6),
            rationale=(
                f"Wyckoff {kind} {level:,.2f} by {depth:.2f} ATR, reclaimed "
                f"within {self.reclaim_bars} bars (= ICT turtle soup)"
            ),
            tags=("price-action", "wyckoff", "false-break", context.session),
            min_rr=1.5,
        )


class DoubleTopBottom(Strategy):
    """Two extremes at a similar level, then a break of the intervening swing.

    The classic reversal chart pattern. Encoded strictly: two extremes within
    `tolerance_atr` of one another, separated by at least `min_separation`
    bars, and the trade triggers only on the break of the neckline between
    them — not on the second touch. Discretionary traders anticipate; a
    systematic rule that anticipates is a rule that is guessing.
    """

    name = "double_top_bottom"
    requires_ict = False

    def __init__(
        self,
        *,
        lookback: int = 60,
        tolerance_atr: float = 0.5,
        min_separation: int = 8,
        stop_buffer_atr: float = 0.3,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            lookback=lookback, tolerance_atr=tolerance_atr,
            min_separation=min_separation, stop_buffer_atr=stop_buffer_atr,
            target_atr=target_atr,
        )
        self.lookback = lookback
        self.tolerance_atr = tolerance_atr
        self.min_separation = min_separation
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.lookback + 2:
            return None
        series = context.series
        window = slice(i - self.lookback, i)
        highs = series.highs[window]
        lows = series.lows[window]
        atr = max(context.atr(), context.instrument.tick_size)
        tolerance = self.tolerance_atr * atr
        price = context.price
        prev = float(series.closes[i - 1])

        # Double top: two comparable highs, neckline = lowest low between them.
        first_high = int(np.argmax(highs))
        others = [
            k for k in range(len(highs))
            if abs(k - first_high) >= self.min_separation
            and abs(float(highs[k]) - float(highs[first_high])) <= tolerance
        ]
        if others:
            second = max(others)
            left, right = sorted((first_high, second))
            neckline = float(np.min(lows[left : right + 1]))
            peak = max(float(highs[first_high]), float(highs[second]))
            if price < neckline and prev >= neckline:
                return _build(
                    context, Direction.BEARISH,
                    peak + self.stop_buffer_atr * atr,
                    price - self.target_atr * atr,
                    confidence=0.5,
                    rationale=(
                        f"double top at ~{peak:,.2f}, neckline {neckline:,.2f} broken"
                    ),
                    tags=("price-action", "double-top", "reversal", context.session),
                    min_rr=1.5,
                )

        first_low = int(np.argmin(lows))
        others = [
            k for k in range(len(lows))
            if abs(k - first_low) >= self.min_separation
            and abs(float(lows[k]) - float(lows[first_low])) <= tolerance
        ]
        if others:
            second = max(others)
            left, right = sorted((first_low, second))
            neckline = float(np.max(highs[left : right + 1]))
            trough = min(float(lows[first_low]), float(lows[second]))
            if price > neckline and prev <= neckline:
                return _build(
                    context, Direction.BULLISH,
                    trough - self.stop_buffer_atr * atr,
                    price + self.target_atr * atr,
                    confidence=0.5,
                    rationale=(
                        f"double bottom at ~{trough:,.2f}, neckline "
                        f"{neckline:,.2f} broken"
                    ),
                    tags=("price-action", "double-bottom", "reversal", context.session),
                    min_rr=1.5,
                )
        return None


class ThreeBarReversal(Strategy):
    """Three consecutive bars against the trend, then a reclaim.

    A short-horizon exhaustion rule, and the systematic core of what
    discretionary traders describe as "three pushes into a level". Requires the
    third bar to be reclaimed rather than merely present, so the strategy is not
    catching a falling knife on the way down.
    """

    name = "three_bar_reversal"
    requires_ict = False

    def __init__(
        self,
        *,
        trend_filter: int = 100,
        stop_buffer_atr: float = 0.25,
        target_atr: float = 2.5,
    ) -> None:
        super().__init__(
            trend_filter=trend_filter, stop_buffer_atr=stop_buffer_atr,
            target_atr=target_atr,
        )
        self.trend_filter = trend_filter
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.trend_filter + 5:
            return None
        series = context.series
        closes, opens = series.closes, series.opens
        atr = max(context.atr(), context.instrument.tick_size)
        trend = float(np.mean(closes[i - self.trend_filter + 1 : i + 1]))

        down = [float(closes[k]) < float(opens[k]) for k in (i - 3, i - 2, i - 1)]
        up = [float(closes[k]) > float(opens[k]) for k in (i - 3, i - 2, i - 1)]
        price = context.price

        if all(down) and price > float(series.highs[i - 1]) and price > trend:
            direction = Direction.BULLISH
            stop = float(np.min(series.lows[i - 3 : i])) - self.stop_buffer_atr * atr
            shape = "three down bars reclaimed"
        elif all(up) and price < float(series.lows[i - 1]) and price < trend:
            direction = Direction.BEARISH
            stop = float(np.max(series.highs[i - 3 : i])) + self.stop_buffer_atr * atr
            shape = "three up bars rejected"
        else:
            return None

        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=0.45,
            rationale=f"{shape}, trend-aligned",
            tags=("price-action", "exhaustion", "reversal", context.session),
            min_rr=1.2,
        )


class LevelRetest(Strategy):
    """Return to a broken swing level that now holds from the other side.

    The polarity flip: broken resistance becomes support. This is the
    price-action tradition's version of ICT's **breaker block**, and the two
    should be expected to fire together.

    Waits for the retest to *hold* — a close back in the break direction — so it
    does not enter into a level that is failing.
    """

    name = "level_retest"
    requires_ict = False

    def __init__(
        self,
        *,
        swing_lookback: int = 30,
        proximity_atr: float = 0.4,
        break_lookback: int = 20,
        stop_buffer_atr: float = 0.3,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            swing_lookback=swing_lookback, proximity_atr=proximity_atr,
            break_lookback=break_lookback, stop_buffer_atr=stop_buffer_atr,
            target_atr=target_atr,
        )
        self.swing_lookback = swing_lookback
        self.proximity_atr = proximity_atr
        self.break_lookback = break_lookback
        self.stop_buffer_atr = stop_buffer_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.swing_lookback + self.break_lookback + 2:
            return None
        series = context.series
        atr = max(context.atr(), context.instrument.tick_size)
        prior = slice(
            i - self.swing_lookback - self.break_lookback, i - self.break_lookback
        )
        level_high = float(np.max(series.highs[prior]))
        level_low = float(np.min(series.lows[prior]))
        since = slice(i - self.break_lookback, i + 1)
        price = context.price
        proximity = self.proximity_atr * atr

        broke_up = float(np.max(series.closes[since])) > level_high
        broke_down = float(np.min(series.closes[since])) < level_low

        if broke_up and abs(price - level_high) <= proximity and price > level_high:
            direction = Direction.BULLISH
            stop = level_high - self.stop_buffer_atr * atr
            level, kind = level_high, "broken resistance holding as support"
        elif broke_down and abs(price - level_low) <= proximity and price < level_low:
            direction = Direction.BEARISH
            stop = level_low + self.stop_buffer_atr * atr
            level, kind = level_low, "broken support holding as resistance"
        else:
            return None

        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=0.5,
            rationale=f"retest of {level:,.2f} — {kind} (= ICT breaker)",
            tags=("price-action", "retest", "polarity", context.session),
            min_rr=1.5,
        )


class RoundNumberReversion(Strategy):
    """Fade an approach to a psychologically round price level.

    Documented in the FX microstructure literature: limit orders cluster at
    round numbers, so price decelerates into them. The mechanism is order
    clustering, which is the same mechanism ICT describes as liquidity resting
    at obvious levels.

    The increment is derived from the instrument's own price scale rather than
    hardcoded, because "round" means 1.1000 on EURUSD, 5000 on ES and 100000 on
    BTC — a fixed constant would be meaningless across them.
    """

    name = "round_number"
    requires_ict = False

    def __init__(
        self,
        *,
        proximity_atr: float = 0.25,
        stop_atr: float = 1.5,
        target_atr: float = 2.0,
    ) -> None:
        super().__init__(
            proximity_atr=proximity_atr, stop_atr=stop_atr, target_atr=target_atr
        )
        self.proximity_atr = proximity_atr
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    @staticmethod
    def increment(price: float) -> float:
        """The round-number spacing appropriate to this price's magnitude."""
        if price <= 0:
            return 1.0
        magnitude = 10 ** np.floor(np.log10(abs(price)))
        return float(magnitude / 10.0)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < 30:
            return None
        price = context.price
        step = self.increment(price)
        if step <= 0:
            return None
        nearest = round(price / step) * step
        atr = max(context.atr(), context.instrument.tick_size)
        if abs(price - nearest) > self.proximity_atr * atr:
            return None

        approach = price - float(context.series.closes[i - 1])
        if approach == 0:
            return None
        # Fade the direction of approach: rising into the level → short.
        direction = Direction.BEARISH if approach > 0 else Direction.BULLISH
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=0.35,
            rationale=(
                f"approached the round level {nearest:,.2f} "
                f"({abs(price - nearest) / atr:.2f} ATR away) — order clustering"
            ),
            tags=("price-action", "round-number", "microstructure", context.session),
            min_rr=1.2,
        )
