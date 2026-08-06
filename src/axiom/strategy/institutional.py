"""Institutional and academic strategy families.

What these are
--------------
Each class is a reference implementation of a **documented** effect — something
with a literature, a name, and a mechanism someone has argued for in print. None
of them is presented as an edge. They exist so the research harness and the
agent pipeline have a genuinely diverse candidate set: a sweep across ten
different *families* says something about robustness, while a sweep across a
hundred parameterisations of one family says almost nothing except how good you
are at overfitting.

Causality
---------
Every indicator here is computed on `closes[: i + 1]` and never touches a bar
past `context.index`. This is not a stylistic preference — a rolling mean that
accidentally includes bar `i+1` produces a beautiful equity curve that cannot be
traded, and it is the single most common defect in published backtests. The
suite asserts it directly (`test_institutional.py::TestCausality`).

Sizing is not here
------------------
No strategy sets a position size. Strategies propose direction, entry and stop;
`RiskManager` decides size against the portfolio's budget. Volatility targeting
in `VolatilityTargetedMomentum` scales *conviction*, not quantity.

Families
--------
:class:`DonchianBreakout`      Turtle/managed-futures channel breakout.
:class:`DualMovingAverage`     Classic CTA trend following.
:class:`KeltnerBreakout`       ATR-channel expansion.
:class:`MACDTrend`             Moving-average convergence/divergence.
:class:`RSIReversal`           Connors-style short-horizon oversold reversal.
:class:`OrnsteinUhlenbeck`     OU half-life mean reversion — stat-arb's workhorse.
:class:`KalmanTrend`           Adaptive local-level filter; no fixed lookback.
:class:`VolatilityTargetedMomentum`  TSMOM with inverse-vol conviction scaling.
:class:`OpeningRangeBreakout`  Session-anchored intraday breakout.
:class:`VWAPReversion`         Reversion to session volume-weighted price.
:class:`TurnOfMonth`           Calendar seasonality around the month boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom.core.types import Direction
from axiom.strategy.base import Signal, Strategy, StrategyContext
from axiom.strategy.quant_strategies import _atr_stop, _build

__all__ = [
    "DonchianBreakout",
    "DualMovingAverage",
    "KalmanTrend",
    "KeltnerBreakout",
    "MACDTrend",
    "OpeningRangeBreakout",
    "OrnsteinUhlenbeck",
    "RSIReversal",
    "TurnOfMonth",
    "VWAPReversion",
    "VolatilityTargetedMomentum",
]


def _ema(values: np.ndarray, span: int) -> float:
    """EMA of `values`, returning only the final value.

    Seeded with the first observation rather than zero. A zero seed leaves the
    filter converging for the first few spans, which shows up as spurious signal
    at the start of every backtest.
    """
    if len(values) == 0:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    out = float(values[0])
    for v in values[1:]:
        out = alpha * float(v) + (1 - alpha) * out
    return out


class DonchianBreakout(Strategy):
    """Buy an N-bar high, sell an N-bar low. The Turtle rule.

    The oldest documented systematic trend rule still in institutional use, and
    the honest baseline for any breakout idea — including ICT's displacement,
    which makes a structurally similar claim from a different tradition.

    The channel is measured over the `lookback` bars **ending at the previous
    bar**, so the current bar's own high cannot define the level it is required
    to break. Including it makes the rule trivially self-satisfying.
    """

    name = "donchian_breakout"
    requires_ict = False

    def __init__(
        self,
        *,
        lookback: int = 55,
        exit_lookback: int = 20,
        stop_atr: float = 2.0,
        target_atr: float = 6.0,
    ) -> None:
        super().__init__(
            lookback=lookback, exit_lookback=exit_lookback,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.lookback = lookback
        self.exit_lookback = exit_lookback
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.lookback + 1:
            return None
        highs = context.series.highs
        lows = context.series.lows
        window = slice(i - self.lookback, i)  # excludes bar i — see docstring
        upper = float(np.max(highs[window]))
        lower = float(np.min(lows[window]))
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
        breadth = (upper - lower) / atr if atr else 0.0

        return _build(
            context, direction, stop, target,
            # A breakout from a *narrow* channel is the higher-conviction one:
            # less prior disagreement had to be resolved to produce it.
            confidence=min(1.0, 2.0 / max(breadth, 0.5)),
            rationale=(
                f"{self.lookback}-bar Donchian break "
                f"{'above' if direction is Direction.BULLISH else 'below'} "
                f"{upper if direction is Direction.BULLISH else lower:,.2f} "
                f"(channel {breadth:.1f} ATR wide)"
            ),
            tags=("trend", "breakout", "donchian", context.session),
            min_rr=1.5,
        )


class DualMovingAverage(Strategy):
    """Fast EMA over slow EMA. The canonical CTA trend filter.

    Signals only on the **crossover bar**, not on every bar the fast average
    happens to sit above the slow one. A rule that fires continuously while a
    condition holds is not a trend follower — it is a position that re-enters
    every bar, and its backtest reports trade statistics that have no
    relationship to how it would trade.
    """

    name = "dual_ma"
    requires_ict = False

    def __init__(
        self,
        *,
        fast: int = 20,
        slow: int = 100,
        stop_atr: float = 2.5,
        target_atr: float = 5.0,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
        super().__init__(fast=fast, slow=slow, stop_atr=stop_atr, target_atr=target_atr)
        self.fast = fast
        self.slow = slow
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.slow + 2:
            return None
        closes = context.series.closes
        now_fast = _ema(closes[max(0, i - self.slow * 3) : i + 1], self.fast)
        now_slow = _ema(closes[max(0, i - self.slow * 3) : i + 1], self.slow)
        prev_fast = _ema(closes[max(0, i - self.slow * 3) : i], self.fast)
        prev_slow = _ema(closes[max(0, i - self.slow * 3) : i], self.slow)

        crossed_up = prev_fast <= prev_slow and now_fast > now_slow
        crossed_down = prev_fast >= prev_slow and now_fast < now_slow
        if not (crossed_up or crossed_down):
            return None

        direction = Direction.BULLISH if crossed_up else Direction.BEARISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        separation = abs(now_fast - now_slow) / atr

        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.35 + separation),
            rationale=(
                f"EMA({self.fast}) crossed "
                f"{'above' if crossed_up else 'below'} EMA({self.slow}) "
                f"— separation {separation:.2f} ATR"
            ),
            tags=("trend", "crossover", context.session),
            min_rr=1.5,
        )


class KeltnerBreakout(Strategy):
    """Close outside an ATR channel around an EMA.

    Volatility-normalised by construction, so the same parameters mean the same
    thing on ES at 20 points of daily range and on BTC at 3%.
    """

    name = "keltner_breakout"
    requires_ict = False

    def __init__(
        self,
        *,
        span: int = 20,
        width_atr: float = 2.0,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
    ) -> None:
        super().__init__(
            span=span, width_atr=width_atr, stop_atr=stop_atr, target_atr=target_atr
        )
        self.span = span
        self.width_atr = width_atr
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.span * 2:
            return None
        closes = context.series.closes
        middle = _ema(closes[max(0, i - self.span * 4) : i + 1], self.span)
        atr = max(context.atr(), context.instrument.tick_size)
        upper = middle + self.width_atr * atr
        lower = middle - self.width_atr * atr
        price = context.price

        if price > upper:
            direction = Direction.BULLISH
        elif price < lower:
            direction = Direction.BEARISH
        else:
            return None

        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        extension = abs(price - middle) / atr

        return _build(
            context, direction, stop, target,
            confidence=min(1.0, extension / 4.0),
            rationale=(
                f"close {extension:.2f} ATR from EMA({self.span}), outside the "
                f"{self.width_atr}-ATR Keltner channel"
            ),
            tags=("trend", "breakout", "keltner", context.session),
            min_rr=1.5,
        )


class MACDTrend(Strategy):
    """MACD line crossing its signal line.

    Included because it is the trend rule most widely present in institutional
    dashboards, which makes it a useful control: if a novel idea cannot beat
    MACD net of costs, it is not novel in any way that matters.
    """

    name = "macd"
    requires_ict = False

    def __init__(
        self,
        *,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
    ) -> None:
        super().__init__(
            fast=fast, slow=slow, signal=signal,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _macd_history(self, closes: np.ndarray, end: int, count: int) -> np.ndarray:
        span = self.slow * 4
        return np.array(
            [
                _ema(closes[max(0, k - span) : k + 1], self.fast)
                - _ema(closes[max(0, k - span) : k + 1], self.slow)
                for k in range(end - count + 1, end + 1)
            ]
        )

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.slow + self.signal + 2:
            return None
        closes = context.series.closes
        history = self._macd_history(closes, i, self.signal * 3)
        if len(history) < self.signal + 2:
            return None

        macd_now, macd_prev = float(history[-1]), float(history[-2])
        signal_now = _ema(history, self.signal)
        signal_prev = _ema(history[:-1], self.signal)

        crossed_up = macd_prev <= signal_prev and macd_now > signal_now
        crossed_down = macd_prev >= signal_prev and macd_now < signal_now
        if not (crossed_up or crossed_down):
            return None

        direction = Direction.BULLISH if crossed_up else Direction.BEARISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.35 + abs(macd_now - signal_now) / atr),
            rationale=(
                f"MACD({self.fast},{self.slow}) crossed "
                f"{'above' if crossed_up else 'below'} its {self.signal}-signal"
            ),
            tags=("trend", "macd", context.session),
            min_rr=1.5,
        )


class RSIReversal(Strategy):
    """Short-horizon oversold/overbought reversal, Connors-style.

    The short RSI period (2 by default, not the conventional 14) is the point:
    the documented effect is a *very* short-horizon reversal, and a 14-period
    RSI smooths it away entirely.

    Gated by a long trend filter — buying oversold in a downtrend is the classic
    way this rule loses money, and Connors' own formulation includes the filter.
    """

    name = "rsi_reversal"
    requires_ict = False

    def __init__(
        self,
        *,
        period: int = 2,
        oversold: float = 10.0,
        overbought: float = 90.0,
        trend_filter: int = 200,
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

    @staticmethod
    def _rsi(closes: np.ndarray, period: int) -> float:
        deltas = np.diff(closes[-(period + 1) :])
        gains = float(np.sum(np.clip(deltas, 0, None)))
        losses = float(-np.sum(np.clip(deltas, None, 0)))
        if losses == 0:
            return 100.0 if gains > 0 else 50.0
        rs = (gains / period) / (losses / period)
        return 100.0 - 100.0 / (1.0 + rs)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < max(self.trend_filter, self.period) + 2:
            return None
        closes = context.series.closes
        rsi = self._rsi(closes[: i + 1], self.period)
        trend = float(np.mean(closes[i - self.trend_filter + 1 : i + 1]))
        price = context.price

        # Long only above the trend filter, short only below it.
        if rsi <= self.oversold and price > trend:
            direction = Direction.BULLISH
        elif rsi >= self.overbought and price < trend:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        extremity = (
            (self.oversold - rsi) if direction is Direction.BULLISH else (rsi - self.overbought)
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.4 + extremity / 20.0),
            rationale=(
                f"RSI({self.period}) = {rsi:.1f} "
                f"{'≤' if direction is Direction.BULLISH else '≥'} "
                f"{self.oversold if direction is Direction.BULLISH else self.overbought:.0f}, "
                f"with price {'above' if direction is Direction.BULLISH else 'below'} "
                f"the {self.trend_filter}-bar mean"
            ),
            tags=("mean-reversion", "rsi", context.session),
            min_rr=1.2,
        )


class OrnsteinUhlenbeck(Strategy):
    """Mean reversion with the half-life estimated, not assumed.

    Fits `Δp_t = λ(μ − p_{t−1}) + ε` by OLS over a rolling window and converts
    λ into a half-life. The strategy trades only when the process actually looks
    mean-reverting on the evidence:

    * `λ > 0` — a positive λ is what mean reversion *is*. A negative one is a
      trending series, and fading it is the exact wrong trade.
    * the half-life falls inside `[min_half_life, max_half_life]` — a half-life
      longer than the holding period is untradeable, and one of a bar or two is
      microstructure noise, not a signal.

    This is the discipline that separates statistical arbitrage from "the price
    went up a lot so it should come down".
    """

    name = "ou_reversion"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 120,
        entry_sigma: float = 2.0,
        min_half_life: float = 3.0,
        max_half_life: float = 60.0,
        stop_atr: float = 2.5,
    ) -> None:
        super().__init__(
            window=window, entry_sigma=entry_sigma,
            min_half_life=min_half_life, max_half_life=max_half_life,
            stop_atr=stop_atr,
        )
        self.window = window
        self.entry_sigma = entry_sigma
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.stop_atr = stop_atr

    @staticmethod
    def fit(prices: np.ndarray) -> tuple[float, float, float]:
        """Return `(lambda, mu, half_life)` from an OLS fit of the OU process.

        A non-positive λ yields `inf` half-life — the honest encoding of "this
        does not revert", rather than a number that would let a caller trade it.
        """
        lagged = prices[:-1]
        deltas = np.diff(prices)
        if len(lagged) < 3 or np.std(lagged) == 0:
            return 0.0, float(np.mean(prices)), float("inf")
        slope, intercept = np.polyfit(lagged, deltas, 1)
        lam = -float(slope)
        if lam <= 0:
            return lam, float(np.mean(prices)), float("inf")
        mu = float(intercept) / lam
        return lam, mu, float(np.log(2.0) / lam)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 2:
            return None
        prices = context.series.closes[i - self.window + 1 : i + 1]
        lam, mu, half_life = self.fit(prices)
        if not np.isfinite(half_life):
            return None
        if not self.min_half_life <= half_life <= self.max_half_life:
            return None

        sigma = float(np.std(prices))
        if sigma <= 0:
            return None
        price = context.price
        z = (price - mu) / sigma
        if abs(z) < self.entry_sigma:
            return None

        direction = Direction.BULLISH if z < 0 else Direction.BEARISH
        stop = _atr_stop(context, direction, self.stop_atr)
        # The target is the fitted mean — the level the model actually predicts,
        # not an ATR multiple picked to make the reward:risk look acceptable.
        target = mu

        return _build(
            context, direction, stop, target,
            confidence=min(1.0, abs(z) / 4.0),
            rationale=(
                f"OU z-score {z:+.2f}σ from fitted mean {mu:,.2f}; "
                f"half-life {half_life:.1f} bars (λ={lam:.4f})"
            ),
            tags=("mean-reversion", "stat-arb", "ou", context.session),
            min_rr=1.0,
        )


class KalmanTrend(Strategy):
    """Local-level Kalman filter; trades the filtered slope.

    Its appeal over a fixed moving average is that it has no lookback. The
    filter's gain adapts to the ratio of process to observation noise, so it
    tightens in a trend and loosens in noise, rather than applying one window to
    both. That makes it markedly less sensitive to the lookback parameter every
    moving-average rule has to have chosen for it — which is where a lot of
    quiet overfitting lives.
    """

    name = "kalman_trend"
    requires_ict = False

    def __init__(
        self,
        *,
        process_var: float = 1e-4,
        observation_var: float = 1e-2,
        window: int = 200,
        min_slope_atr: float = 0.05,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
    ) -> None:
        super().__init__(
            process_var=process_var, observation_var=observation_var,
            window=window, min_slope_atr=min_slope_atr,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.process_var = process_var
        self.observation_var = observation_var
        self.window = window
        self.min_slope_atr = min_slope_atr
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _filter(self, prices: np.ndarray) -> tuple[float, float]:
        """Return `(level, slope)` of a local-linear-trend filter.

        A proper two-state filter: the state is `[level, slope]` and the error
        covariance is a full 2×2, so the slope gets its own Kalman gain from the
        level/slope covariance term. Updating the slope by a scaled copy of the
        level's gain — the obvious shortcut — makes the slope update smaller
        than the threshold it is compared against by two orders of magnitude,
        and the strategy then never signals at all.
        """
        level = float(prices[0])
        slope = 0.0
        # P = [[p_ll, p_ls], [p_ls, p_ss]]
        p_ll, p_ls, p_ss = 1.0, 0.0, 1.0
        q_level = self.process_var
        q_slope = self.process_var
        r = self.observation_var

        for price in prices[1:]:
            # Predict: level_{t} = level_{t-1} + slope_{t-1}
            level += slope
            # F = [[1, 1], [0, 1]] → P = F P Fᵀ + Q
            p_ll = p_ll + 2.0 * p_ls + p_ss + q_level
            p_ls = p_ls + p_ss
            p_ss = p_ss + q_slope

            # Update against the level observation. H = [1, 0].
            innovation_var = p_ll + r
            k_level = p_ll / innovation_var
            k_slope = p_ls / innovation_var
            residual = float(price) - level

            level += k_level * residual
            slope += k_slope * residual

            # P = (I − K H) P
            new_p_ll = p_ll * (1.0 - k_level)
            new_p_ls = p_ls * (1.0 - k_level)
            p_ss = p_ss - k_slope * p_ls
            p_ll, p_ls = new_p_ll, new_p_ls

        return level, slope

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window:
            return None
        prices = context.series.closes[i - self.window + 1 : i + 1]
        _level, slope = self._filter(prices)
        atr = max(context.atr(), context.instrument.tick_size)
        normalised = slope / atr
        if abs(normalised) < self.min_slope_atr:
            return None

        direction = Direction.BULLISH if normalised > 0 else Direction.BEARISH
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, abs(normalised) / (self.min_slope_atr * 6)),
            rationale=f"Kalman filtered slope {normalised:+.3f} ATR/bar",
            tags=("trend", "kalman", "adaptive", context.session),
            min_rr=1.5,
        )


class VolatilityTargetedMomentum(Strategy):
    """Time-series momentum with inverse-volatility conviction scaling.

    Follows Moskowitz, Ooi & Pedersen (2012): the raw momentum signal is
    normalised by realised volatility so that the same signal means the same
    thing across instruments and across time.

    The scaling enters `confidence`, not quantity. `RiskManager` owns sizing,
    and a strategy that also sized would give the portfolio two independent
    opinions about how much to risk — which is how a "1% risk" system ends up
    risking four.
    """

    name = "vol_targeted_momentum"
    requires_ict = False

    def __init__(
        self,
        *,
        lookback: int = 120,
        vol_window: int = 60,
        target_vol: float = 0.01,
        min_signal: float = 0.4,
        stop_atr: float = 2.5,
        target_atr: float = 5.0,
    ) -> None:
        super().__init__(
            lookback=lookback, vol_window=vol_window, target_vol=target_vol,
            min_signal=min_signal, stop_atr=stop_atr, target_atr=target_atr,
        )
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.min_signal = min_signal
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < max(self.lookback, self.vol_window) + 2:
            return None
        closes = context.series.closes
        returns = np.diff(closes[i - self.vol_window : i + 1]) / closes[i - self.vol_window : i]
        realised_vol = float(np.std(returns))
        if realised_vol <= 0:
            return None

        total_return = float(closes[i] / closes[i - self.lookback] - 1.0)
        # Sharpe-like: cumulative return per unit of realised volatility,
        # annualisation-free because only the sign and magnitude are used.
        signal = total_return / (realised_vol * np.sqrt(self.lookback))
        if abs(signal) < self.min_signal:
            return None

        direction = Direction.BULLISH if signal > 0 else Direction.BEARISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        vol_scalar = self.target_vol / realised_vol

        return _build(
            context, direction, stop, target,
            # Conviction falls when realised vol runs above target — the same
            # instinct as vol targeting, expressed where it belongs.
            confidence=min(1.0, abs(signal) / 2.0 * min(vol_scalar, 2.0)),
            rationale=(
                f"{self.lookback}-bar return {total_return:+.2%} at "
                f"{realised_vol:.4f} realised vol → signal {signal:+.2f} "
                f"(vol scalar {vol_scalar:.2f}×)"
            ),
            tags=("momentum", "vol-targeted", "tsmom", context.session),
            min_rr=1.5,
        )


class VWAPReversion(Strategy):
    """Fade extension from the session volume-weighted average price.

    VWAP is the benchmark institutional execution is measured against, which is
    the mechanism: desks working a large order have a mandate to buy below it
    and sell above it, so extension from VWAP meets real, non-discretionary flow
    in the opposite direction.

    The anchor resets at each session boundary. A VWAP computed over a rolling
    window instead is a different statistic with none of that meaning.
    """

    name = "vwap_reversion"
    requires_ict = False

    def __init__(
        self,
        *,
        entry_sigma: float = 2.0,
        min_bars: int = 20,
        stop_atr: float = 2.0,
    ) -> None:
        super().__init__(
            entry_sigma=entry_sigma, min_bars=min_bars, stop_atr=stop_atr
        )
        self.entry_sigma = entry_sigma
        self.min_bars = min_bars
        self.stop_atr = stop_atr

    @staticmethod
    def _session_start(index: pd.DatetimeIndex, i: int, tz: str) -> int:
        """First bar of the trading day containing bar `i`, in the instrument's tz."""
        local = index.tz_convert(tz)
        today = local[i].date()
        start = i
        while start > 0 and local[start - 1].date() == today:
            start -= 1
        return start

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        series = context.series
        start = self._session_start(series.index, i, context.instrument.session_tz)
        if i - start < self.min_bars:
            return None

        window = slice(start, i + 1)
        typical = (series.highs[window] + series.lows[window] + series.closes[window]) / 3.0
        volume = series.volumes[window]
        total = float(np.sum(volume))
        if total <= 0:
            return None
        vwap = float(np.sum(typical * volume) / total)
        dispersion = float(np.std(series.closes[window]))
        if dispersion <= 0:
            return None

        price = context.price
        z = (price - vwap) / dispersion
        if abs(z) < self.entry_sigma:
            return None

        direction = Direction.BULLISH if z < 0 else Direction.BEARISH
        stop = _atr_stop(context, direction, self.stop_atr)
        return _build(
            context, direction, stop, vwap,
            confidence=min(1.0, abs(z) / 4.0),
            rationale=(
                f"{z:+.2f}σ from session VWAP {vwap:,.2f} "
                f"({i - start + 1} bars into the session)"
            ),
            tags=("mean-reversion", "vwap", "microstructure", context.session),
            min_rr=1.0,
        )


class OpeningRangeBreakout(Strategy):
    """Break of the range established in the first minutes of a session.

    The oldest institutional intraday rule there is, and the one that overlaps
    most directly with ICT: both hold that the session open establishes a
    reference the rest of the day trades around. ORB arrives at it from the
    order-flow tradition (overnight inventory clearing into the open), ICT from
    the delivery-algorithm framing. The overlap is worth knowing about — a
    Silver Bullet signal and an ORB signal firing together are not two
    independent confirmations.

    The range is frozen once `range_minutes` have elapsed and never updated, so
    a later bar cannot widen the level it is required to break.
    """

    name = "opening_range"
    requires_ict = False

    def __init__(
        self,
        *,
        range_minutes: int = 30,
        stop_atr: float = 1.5,
        target_atr: float = 3.0,
        max_bars_after: int = 40,
    ) -> None:
        super().__init__(
            range_minutes=range_minutes, stop_atr=stop_atr,
            target_atr=target_atr, max_bars_after=max_bars_after,
        )
        self.range_minutes = range_minutes
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.max_bars_after = max_bars_after

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        series = context.series
        tz = context.instrument.session_tz
        local = series.index.tz_convert(tz)
        today = local[i].date()

        start = i
        while start > 0 and local[start - 1].date() == today:
            start -= 1

        elapsed = (local[i] - local[start]).total_seconds() / 60.0
        if elapsed < self.range_minutes:
            return None  # the range is still forming
        if i - start > self.max_bars_after + self.range_minutes:
            return None  # too late in the session for this to be an ORB

        # Freeze the range at the last bar inside the opening window.
        end = start
        while (
            end + 1 <= i
            and (local[end + 1] - local[start]).total_seconds() / 60.0 <= self.range_minutes
        ):
            end += 1
        if end <= start:
            return None

        window = slice(start, end + 1)
        upper = float(np.max(series.highs[window]))
        lower = float(np.min(series.lows[window]))
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
        width = (upper - lower) / atr

        return _build(
            context, direction, stop, target,
            # A tight opening range breaking is the higher-conviction setup;
            # a wide one has already spent the day's move establishing itself.
            confidence=min(1.0, 1.5 / max(width, 0.5)),
            rationale=(
                f"{self.range_minutes}-minute opening range "
                f"{lower:,.2f}–{upper:,.2f} ({width:.1f} ATR wide) broken "
                f"{'above' if direction is Direction.BULLISH else 'below'}"
            ),
            tags=("breakout", "session", "orb", context.session),
            min_rr=1.5,
        )


class TurnOfMonth(Strategy):
    """Long across the month boundary — the turn-of-month effect.

    One of the most persistent documented calendar anomalies, usually attributed
    to pension and payroll flows concentrating around the month end.

    It is included **as a control, and should be read as one.** A calendar rule
    has no mechanism that reacts to price, so if it survives the same
    walk-forward and deflated-Sharpe gauntlet as everything else, that is
    evidence the gauntlet is too easy — not that the calendar predicts markets.
    """

    name = "turn_of_month"
    requires_ict = False

    def __init__(
        self,
        *,
        days_before: int = 1,
        days_after: int = 3,
        stop_atr: float = 2.0,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            days_before=days_before, days_after=days_after,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.days_before = days_before
        self.days_after = days_after
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        stamp = context.timestamp
        days_in_month = stamp.days_in_month
        day = stamp.day
        near_end = day > days_in_month - self.days_before
        near_start = day <= self.days_after
        if not (near_end or near_start):
            return None

        # Fire once per day, on the first bar of it. Without this the rule
        # re-enters on every intraday bar of the window.
        i = context.index
        local = context.series.index
        if i > 0 and local[i - 1].date() == stamp.date():
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, Direction.BULLISH, self.stop_atr)
        target = context.price + self.target_atr * atr
        return _build(
            context, Direction.BULLISH, stop, target,
            confidence=0.4,
            rationale=(
                f"turn-of-month window (day {day} of {days_in_month}); "
                f"calendar effect, no price mechanism — control strategy"
            ),
            tags=("seasonality", "calendar", "control", context.session),
            min_rr=1.2,
        )
