"""Indicator-based strategy families in institutional and retail use.

Scope
-----
These are the rules that appear on real trading desks and in every charting
package: Supertrend, Parabolic SAR, ADX/DMI, Ichimoku, Bollinger, Stochastic,
CCI, Williams %R, OBV. They are here for two reasons.

**Coverage.** The archive is meant to be the candidate set an automated search
draws from. A search that has never seen ADX cannot tell you whether your novel
idea beats ADX.

**Control.** Most of these are widely published, which means any edge they had
has been arbitraged hard. They make excellent *nulls*: a new strategy that
cannot beat Supertrend net of costs has not demonstrated anything, and a
research harness that rates them highly is a harness with a problem.

Indicator correlation is not incidental
---------------------------------------
Supertrend, Parabolic SAR, Keltner and Donchian are all ATR-scaled trend
followers. They agree most of the time and their signals are **not** independent
confirmations of one another. `Family` groups them so a diversified candidate
set does not accidentally consist of six spellings of the same rule. Where a
strategy overlaps an ICT concept, the docstring says so — an ICT displacement
signal and a Supertrend flip firing together is one observation, not two.

Causality
---------
Every indicator reads `[: i + 1]` only. Recursive indicators (Supertrend, PSAR,
OBV, Wilder-smoothed ADX) are seeded at a bounded lookback and stepped forward,
which is both causal and O(window) per bar rather than O(i).
"""

from __future__ import annotations

import numpy as np

from axiom.core.types import Direction
from axiom.strategy.base import Signal, Strategy, StrategyContext
from axiom.strategy.quant_strategies import _atr_stop, _build

__all__ = [
    "ADXTrend",
    "BollingerSqueeze",
    "CCIReversal",
    "IchimokuCloud",
    "OBVDivergence",
    "ParabolicSAR",
    "StochasticReversal",
    "Supertrend",
    "WilliamsR",
]

#: Bars of history used to seed recursive indicators. Long enough for Wilder
#: smoothing to converge, short enough that a bar's cost does not grow with the
#: length of the series.
_SEED = 250


def _wilder(values: np.ndarray, period: int) -> float:
    """Wilder's smoothing — the recursion ADX, ATR and RSI are defined with.

    Not the same as an EMA of span `period`: Wilder uses α = 1/period, which is
    equivalent to an EMA of span `2*period − 1`. Substituting one for the other
    silently changes every threshold calibrated against the original.
    """
    if len(values) < period:
        return float("nan")
    out = float(np.mean(values[:period]))
    for v in values[period:]:
        out = (out * (period - 1) + float(v)) / period
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev = np.concatenate(([close[0]], close[:-1]))
    result: np.ndarray = np.maximum(
        high - low, np.maximum(np.abs(high - prev), np.abs(low - prev))
    )
    return result


class Supertrend(Strategy):
    """ATR-band trend follower that flips on a close through the band.

    Very widely used, and structurally an ATR channel with hysteresis: the band
    only ever ratchets in the direction of the trend, so it does not whipsaw on
    a single wide bar the way a raw channel does.

    **Overlaps `KeltnerBreakout` and `DonchianBreakout` substantially.** All
    three are ATR-scaled trend rules. Treat agreement between them as one
    observation.
    """

    name = "supertrend"
    requires_ict = False

    def __init__(
        self,
        *,
        period: int = 10,
        multiplier: float = 3.0,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
    ) -> None:
        super().__init__(
            period=period, multiplier=multiplier,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.period = period
        self.multiplier = multiplier
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _trend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> list[int]:
        """Step the recursion forward, returning the trend sign per bar."""
        tr = _true_range(high, low, close)
        atr = np.full(len(close), np.nan)
        if len(tr) <= self.period:
            return [1] * len(close)
        atr[self.period] = float(np.mean(tr[: self.period + 1]))
        for k in range(self.period + 1, len(close)):
            atr[k] = (atr[k - 1] * (self.period - 1) + tr[k]) / self.period

        mid = (high + low) / 2.0
        upper = mid + self.multiplier * atr
        lower = mid - self.multiplier * atr
        trend = [1] * len(close)
        final_upper, final_lower = upper.copy(), lower.copy()

        for k in range(self.period + 1, len(close)):
            # The band ratchets: it may only tighten toward price while the
            # trend holds. This is what stops a single wide bar flipping it.
            final_upper[k] = (
                min(upper[k], final_upper[k - 1])
                if close[k - 1] <= final_upper[k - 1]
                else upper[k]
            )
            final_lower[k] = (
                max(lower[k], final_lower[k - 1])
                if close[k - 1] >= final_lower[k - 1]
                else lower[k]
            )
            if close[k] > final_upper[k - 1]:
                trend[k] = 1
            elif close[k] < final_lower[k - 1]:
                trend[k] = -1
            else:
                trend[k] = trend[k - 1]
        return trend

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.period * 3:
            return None
        start = max(0, i - _SEED)
        series = context.series
        trend = self._trend(
            series.highs[start : i + 1],
            series.lows[start : i + 1],
            series.closes[start : i + 1],
        )
        if len(trend) < 2 or trend[-1] == trend[-2]:
            return None  # signal on the flip only, not on every bar in trend

        direction = Direction.BULLISH if trend[-1] > 0 else Direction.BEARISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=0.55,
            rationale=(
                f"Supertrend({self.period}, {self.multiplier}×ATR) flipped "
                f"{'bullish' if direction is Direction.BULLISH else 'bearish'}"
            ),
            tags=("trend", "supertrend", "atr-band", context.session),
            min_rr=1.5,
        )


class ParabolicSAR(Strategy):
    """Wilder's stop-and-reverse. Signals on the flip.

    The acceleration factor steps up on each new extreme, so the stop tightens
    as a trend extends. That makes it a trailing-stop mechanism first and an
    entry rule second — which is why it is usually paired with a trend-strength
    filter such as ADX.
    """

    name = "parabolic_sar"
    requires_ict = False

    def __init__(
        self,
        *,
        step: float = 0.02,
        max_step: float = 0.2,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
    ) -> None:
        super().__init__(
            step=step, max_step=max_step, stop_atr=stop_atr, target_atr=target_atr
        )
        self.step = step
        self.max_step = max_step
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _sar(self, high: np.ndarray, low: np.ndarray) -> list[int]:
        if len(high) < 3:
            return [1] * len(high)
        trend = [1] * len(high)
        sar = float(low[0])
        extreme = float(high[0])
        af = self.step
        rising = True

        for k in range(1, len(high)):
            sar = sar + af * (extreme - sar)
            if rising:
                sar = min(sar, float(low[k - 1]), float(low[max(0, k - 2)]))
                if low[k] < sar:
                    rising = False
                    sar, extreme, af = extreme, float(low[k]), self.step
                elif high[k] > extreme:
                    extreme = float(high[k])
                    af = min(af + self.step, self.max_step)
            else:
                sar = max(sar, float(high[k - 1]), float(high[max(0, k - 2)]))
                if high[k] > sar:
                    rising = True
                    sar, extreme, af = extreme, float(high[k]), self.step
                elif low[k] < extreme:
                    extreme = float(low[k])
                    af = min(af + self.step, self.max_step)
            trend[k] = 1 if rising else -1
        return trend

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < 30:
            return None
        start = max(0, i - _SEED)
        trend = self._sar(context.series.highs[start : i + 1], context.series.lows[start : i + 1])
        if len(trend) < 2 or trend[-1] == trend[-2]:
            return None

        direction = Direction.BULLISH if trend[-1] > 0 else Direction.BEARISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=0.45,
            rationale=f"Parabolic SAR flipped {'long' if trend[-1] > 0 else 'short'}",
            tags=("trend", "psar", context.session),
            min_rr=1.5,
        )


class ADXTrend(Strategy):
    """Directional movement with a trend-strength gate.

    ADX measures how *trending* the market is without saying which way; +DI/−DI
    say which way without saying how strongly. Neither is a signal alone, and
    the standard formulation is the conjunction: take the DI cross only when ADX
    confirms there is a trend to join.

    The gate is the useful part. It generalises: any trend rule paired with a
    strength filter trades far less and, in most published tests, better.
    """

    name = "adx_trend"
    requires_ict = False

    def __init__(
        self,
        *,
        period: int = 14,
        min_adx: float = 25.0,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
    ) -> None:
        super().__init__(
            period=period, min_adx=min_adx, stop_atr=stop_atr, target_atr=target_atr
        )
        self.period = period
        self.min_adx = min_adx
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _dmi(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> tuple[float, float, float]:
        """Return `(adx, plus_di, minus_di)`."""
        up = np.diff(high)
        down = -np.diff(low)
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = _true_range(high, low, close)[1:]

        atr = _wilder(tr, self.period)
        if not np.isfinite(atr) or atr <= 0:
            return float("nan"), 0.0, 0.0
        plus_di = 100.0 * _wilder(plus_dm, self.period) / atr
        minus_di = 100.0 * _wilder(minus_dm, self.period) / atr
        total = plus_di + minus_di
        if total <= 0:
            return float("nan"), plus_di, minus_di

        # ADX is Wilder smoothing of DX, so it needs a DX history, not one point.
        dx: list[float] = []
        for k in range(self.period, len(tr)):
            window = slice(max(0, k - self.period * 2), k + 1)
            a = _wilder(tr[window], self.period)
            if not np.isfinite(a) or a <= 0:
                continue
            p = 100.0 * _wilder(plus_dm[window], self.period) / a
            m = 100.0 * _wilder(minus_dm[window], self.period) / a
            if p + m > 0:
                dx.append(100.0 * abs(p - m) / (p + m))
        adx = _wilder(np.array(dx), self.period) if len(dx) >= self.period else float("nan")
        return adx, plus_di, minus_di

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.period * 6:
            return None
        start = max(0, i - _SEED)
        series = context.series
        adx, plus_di, minus_di = self._dmi(
            series.highs[start : i + 1], series.lows[start : i + 1], series.closes[start : i + 1]
        )
        if not np.isfinite(adx) or adx < self.min_adx:
            return None

        direction = Direction.BULLISH if plus_di > minus_di else Direction.BEARISH
        spread = abs(plus_di - minus_di)
        if spread < 5.0:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, (adx - self.min_adx) / 30.0 + 0.4),
            rationale=(
                f"ADX({self.period}) = {adx:.1f} ≥ {self.min_adx:.0f} with "
                f"+DI {plus_di:.1f} / −DI {minus_di:.1f}"
            ),
            tags=("trend", "adx", "dmi", context.session),
            min_rr=1.5,
        )


class IchimokuCloud(Strategy):
    """Price breaking through the Ichimoku cloud, with the cloud's own bias.

    The cloud (Senkou A/B) is plotted `displacement` bars **ahead**, which is
    the one thing about Ichimoku that routinely produces lookahead bugs: the
    cloud visible at bar *i* was computed from data at bar *i − displacement*,
    and reading the cloud *drawn at* bar *i* means reading the future. This
    implementation indexes the historical span deliberately, and the causality
    test covers it.
    """

    name = "ichimoku"
    requires_ict = False

    def __init__(
        self,
        *,
        conversion: int = 9,
        base: int = 26,
        span_b: int = 52,
        displacement: int = 26,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
    ) -> None:
        super().__init__(
            conversion=conversion, base=base, span_b=span_b,
            displacement=displacement, stop_atr=stop_atr, target_atr=target_atr,
        )
        self.conversion = conversion
        self.base = base
        self.span_b = span_b
        self.displacement = displacement
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    @staticmethod
    def _midpoint(high: np.ndarray, low: np.ndarray, end: int, period: int) -> float:
        window = slice(max(0, end - period + 1), end + 1)
        return (float(np.max(high[window])) + float(np.min(low[window]))) / 2.0

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.span_b + self.displacement + 2:
            return None
        series = context.series
        high, low = series.highs, series.lows

        # The cloud in force *now* was computed `displacement` bars ago.
        anchor = i - self.displacement
        span_a = (
            self._midpoint(high, low, anchor, self.conversion)
            + self._midpoint(high, low, anchor, self.base)
        ) / 2.0
        span_b = self._midpoint(high, low, anchor, self.span_b)
        cloud_top, cloud_bottom = max(span_a, span_b), min(span_a, span_b)

        conversion = self._midpoint(high, low, i, self.conversion)
        base = self._midpoint(high, low, i, self.base)
        prev_close = float(series.closes[i - 1])
        price = context.price

        # Break of the cloud on this bar, aligned with the tenkan/kijun cross.
        if price > cloud_top and prev_close <= cloud_top and conversion > base:
            direction = Direction.BULLISH
        elif price < cloud_bottom and prev_close >= cloud_bottom and conversion < base:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        thickness = (cloud_top - cloud_bottom) / atr

        return _build(
            context, direction, stop, target,
            # A thick cloud is stronger resistance, so breaking it means more.
            confidence=min(1.0, 0.4 + thickness / 8.0),
            rationale=(
                f"close broke {'above' if direction is Direction.BULLISH else 'below'} "
                f"the Ichimoku cloud ({cloud_bottom:,.2f}–{cloud_top:,.2f}, "
                f"{thickness:.1f} ATR thick) with tenkan/kijun aligned"
            ),
            tags=("trend", "ichimoku", "breakout", context.session),
            min_rr=1.5,
        )


class BollingerSqueeze(Strategy):
    """Breakout from a Bollinger band contraction.

    Distinct from `MeanReversionZScore`, which *fades* band touches. This trades
    the opposite premise — that a contraction in realised volatility precedes an
    expansion — and fires only when bandwidth is in the bottom decile of its own
    recent history. That percentile gate is what separates it from "price
    touched a band", which happens constantly.

    Structurally the same claim as ICT displacement out of consolidation.
    """

    name = "bollinger_squeeze"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 20,
        num_std: float = 2.0,
        squeeze_lookback: int = 120,
        squeeze_percentile: float = 20.0,
        stop_atr: float = 1.5,
        target_atr: float = 3.5,
    ) -> None:
        super().__init__(
            window=window, num_std=num_std, squeeze_lookback=squeeze_lookback,
            squeeze_percentile=squeeze_percentile,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.window = window
        self.num_std = num_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _bandwidth(self, closes: np.ndarray, end: int) -> float:
        window = closes[max(0, end - self.window + 1) : end + 1]
        mean = float(np.mean(window))
        if mean == 0:
            return float("nan")
        return 2.0 * self.num_std * float(np.std(window)) / mean

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + self.squeeze_lookback + 2:
            return None
        closes = context.series.closes
        history = np.array(
            [self._bandwidth(closes, k) for k in range(i - self.squeeze_lookback, i)]
        )
        history = history[np.isfinite(history)]
        if len(history) < 10:
            return None
        now = self._bandwidth(closes, i)
        threshold = float(np.percentile(history, self.squeeze_percentile))
        if not np.isfinite(now) or now > threshold:
            return None  # not compressed

        window = closes[i - self.window + 1 : i + 1]
        mean, std = float(np.mean(window)), float(np.std(window))
        if std <= 0:
            return None
        upper, lower = mean + self.num_std * std, mean - self.num_std * std
        price = context.price

        if price > upper:
            direction = Direction.BULLISH
        elif price < lower:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.4 + (threshold - now) / max(threshold, 1e-9)),
            rationale=(
                f"Bollinger bandwidth {now:.4f} in the bottom "
                f"{self.squeeze_percentile:.0f}% of {self.squeeze_lookback} bars, "
                f"broken {'up' if direction is Direction.BULLISH else 'down'}"
            ),
            tags=("breakout", "volatility", "bollinger", context.session),
            min_rr=1.5,
        )


class StochasticReversal(Strategy):
    """%K/%D cross out of an extreme, filtered by the longer trend.

    The trend filter is not optional decoration. Unfiltered stochastic
    oscillators famously stay pinned at an extreme through an entire trend,
    generating a losing signal on every bar of the best move in the sample.
    """

    name = "stochastic_reversal"
    requires_ict = False

    def __init__(
        self,
        *,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        trend_filter: int = 100,
        stop_atr: float = 2.0,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            k_period=k_period, d_period=d_period, oversold=oversold,
            overbought=overbought, trend_filter=trend_filter,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        self.trend_filter = trend_filter
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _k(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, end: int) -> float:
        window = slice(max(0, end - self.k_period + 1), end + 1)
        hi, lo = float(np.max(high[window])), float(np.min(low[window]))
        if hi == lo:
            return 50.0
        return 100.0 * (float(close[end]) - lo) / (hi - lo)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < max(self.trend_filter, self.k_period + self.d_period) + 2:
            return None
        series = context.series
        high, low, close = series.highs, series.lows, series.closes
        ks = [self._k(high, low, close, k) for k in range(i - self.d_period, i + 1)]
        k_now, k_prev = ks[-1], ks[-2]
        d_now = float(np.mean(ks[-self.d_period :]))
        d_prev = float(np.mean(ks[-self.d_period - 1 : -1]))

        trend = float(np.mean(close[i - self.trend_filter + 1 : i + 1]))
        price = context.price
        crossed_up = k_prev <= d_prev and k_now > d_now and k_prev < self.oversold
        crossed_down = k_prev >= d_prev and k_now < d_now and k_prev > self.overbought

        if crossed_up and price > trend:
            direction = Direction.BULLISH
        elif crossed_down and price < trend:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=0.45,
            rationale=(
                f"Stochastic %K crossed %D at {k_now:.1f} out of "
                f"{'oversold' if direction is Direction.BULLISH else 'overbought'}, "
                f"trend-aligned"
            ),
            tags=("mean-reversion", "stochastic", context.session),
            min_rr=1.2,
        )


class CCIReversal(Strategy):
    """Commodity Channel Index reverting from an extreme.

    CCI is a z-score of typical price against its own mean, scaled by 0.015 so
    that roughly 70–80% of readings fall inside ±100. It is included because it
    is a *different normalisation* of the same reversion idea as
    `MeanReversionZScore` — mean absolute deviation rather than standard
    deviation, which is materially less sensitive to a single outlier bar.
    """

    name = "cci_reversal"
    requires_ict = False

    def __init__(
        self,
        *,
        period: int = 20,
        threshold: float = 150.0,
        stop_atr: float = 2.0,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            period=period, threshold=threshold, stop_atr=stop_atr, target_atr=target_atr
        )
        self.period = period
        self.threshold = threshold
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, end: int) -> float:
        window = slice(max(0, end - self.period + 1), end + 1)
        typical = (high[window] + low[window] + close[window]) / 3.0
        mean = float(np.mean(typical))
        mad = float(np.mean(np.abs(typical - mean)))
        if mad == 0:
            return 0.0
        return (float(typical[-1]) - mean) / (0.015 * mad)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.period + 3:
            return None
        series = context.series
        now = self._cci(series.highs, series.lows, series.closes, i)
        prev = self._cci(series.highs, series.lows, series.closes, i - 1)

        # Reversion signal is the *cross back*, not the extreme itself. Waiting
        # avoids standing in front of a reading that is still extending.
        if prev <= -self.threshold < now:
            direction = Direction.BULLISH
        elif prev >= self.threshold > now:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.4 + abs(prev) / 400.0),
            rationale=f"CCI({self.period}) crossed back from {prev:+.0f} to {now:+.0f}",
            tags=("mean-reversion", "cci", context.session),
            min_rr=1.2,
        )


class WilliamsR(Strategy):
    """Williams %R recovering from an extreme, with a trend filter.

    Mathematically %R is `%K − 100` — the same statistic as the stochastic on a
    different scale. It is catalogued **because** of that, not in spite of it:
    it is the archive's built-in demonstration that two differently-named
    indicators can be one indicator, and a "confirmation" from both is a
    confirmation from neither.
    """

    name = "williams_r"
    requires_ict = False

    def __init__(
        self,
        *,
        period: int = 14,
        oversold: float = -85.0,
        overbought: float = -15.0,
        trend_filter: int = 100,
        stop_atr: float = 2.0,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            period=period, oversold=oversold, overbought=overbought,
            trend_filter=trend_filter, stop_atr=stop_atr, target_atr=target_atr,
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.trend_filter = trend_filter
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, end: int) -> float:
        window = slice(max(0, end - self.period + 1), end + 1)
        hi, lo = float(np.max(high[window])), float(np.min(low[window]))
        if hi == lo:
            return -50.0
        return -100.0 * (hi - float(close[end])) / (hi - lo)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < max(self.trend_filter, self.period) + 3:
            return None
        series = context.series
        now = self._r(series.highs, series.lows, series.closes, i)
        prev = self._r(series.highs, series.lows, series.closes, i - 1)
        trend = float(np.mean(series.closes[i - self.trend_filter + 1 : i + 1]))
        price = context.price

        if prev <= self.oversold < now and price > trend:
            direction = Direction.BULLISH
        elif prev >= self.overbought > now and price < trend:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=0.4,
            rationale=f"Williams %R({self.period}) recovered from {prev:.0f} to {now:.0f}",
            tags=("mean-reversion", "williams-r", context.session),
            min_rr=1.2,
        )


class OBVDivergence(Strategy):
    """On-balance volume diverging from price.

    OBV accumulates signed volume, so it rises when volume arrives on up-closes.
    The tradeable claim is **divergence**: price makes a new extreme while OBV
    does not, meaning the move is not supported by participation.

    Closest thing here to ICT's SMT divergence, which makes the same argument
    using a correlated instrument instead of volume. On instruments where volume
    is unreliable — FX, and any single-venue equity feed such as IEX — this
    measures the feed, not the market. `docs/DATA_SETUP.md` covers that trap.
    """

    name = "obv_divergence"
    requires_ict = False

    def __init__(
        self,
        *,
        lookback: int = 40,
        stop_atr: float = 2.0,
        target_atr: float = 3.5,
    ) -> None:
        super().__init__(lookback=lookback, stop_atr=stop_atr, target_atr=target_atr)
        self.lookback = lookback
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    @staticmethod
    def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        signs = np.sign(np.diff(close))
        return np.concatenate(([0.0], np.cumsum(signs * volume[1:])))

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.lookback * 2:
            return None
        series = context.series
        start = max(0, i - self.lookback * 2)
        closes = series.closes[start : i + 1]
        obv = self._obv(closes, series.volumes[start : i + 1])
        if float(np.sum(np.abs(obv))) == 0:
            return None  # no usable volume on this feed

        recent = slice(len(closes) - self.lookback, len(closes))
        prior = slice(0, len(closes) - self.lookback)
        price_high_now = float(np.max(closes[recent]))
        price_high_before = float(np.max(closes[prior]))
        price_low_now = float(np.min(closes[recent]))
        price_low_before = float(np.min(closes[prior]))
        obv_high_now = float(np.max(obv[recent]))
        obv_high_before = float(np.max(obv[prior]))
        obv_low_now = float(np.min(obv[recent]))
        obv_low_before = float(np.min(obv[prior]))

        bearish = price_high_now > price_high_before and obv_high_now <= obv_high_before
        bullish = price_low_now < price_low_before and obv_low_now >= obv_low_before
        if bullish:
            direction = Direction.BULLISH
        elif bearish:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=0.45,
            rationale=(
                f"price made a new {self.lookback}-bar "
                f"{'low' if bullish else 'high'} but OBV did not — "
                f"move unsupported by volume"
            ),
            tags=("divergence", "volume", "obv", context.session),
            min_rr=1.3,
        )
