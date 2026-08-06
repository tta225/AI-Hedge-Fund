"""Statistical, microstructure and cross-sectional strategy families.

These are the rules that come from a *measurement* rather than a chart pattern:
a fitted Hurst exponent, a variance ratio, an order-flow imbalance, a
cointegrating relationship. What they share is that each one estimates a
property of the series and trades only when the estimate says the property is
present — which is the discipline that separates statistical arbitrage from
pattern matching.

The honest caveat about volume
------------------------------
`OrderFlowImbalance` and `AmihudIlliquidity` are computed from **bar** volume,
not from the order book. Real order-flow imbalance is built from message-level
book events; a bar-level proxy is a much cruder instrument and does not become
the real thing by being called one. On a single-venue feed such as IEX, bar
volume is a fraction of what actually traded and these will measure the feed
rather than the market. See `docs/DATA_SETUP.md`.
"""

from __future__ import annotations

import numpy as np

from axiom.core.types import Direction
from axiom.strategy.base import Signal, Strategy, StrategyContext
from axiom.strategy.quant_strategies import _atr_stop, _build

__all__ = [
    "AmihudIlliquidity",
    "AutocorrelationReversal",
    "FiftyTwoWeekHigh",
    "GapFade",
    "HurstAdaptive",
    "NarrowRangeBreakout",
    "OrderFlowImbalance",
    "PreviousDayLevels",
    "SkewnessReversal",
    "VarianceRatio",
]


def hurst_exponent(prices: np.ndarray, max_lag: int = 40) -> float:
    """Hurst exponent by rescaled variance of lagged differences.

    * `H > 0.5` — persistent. Trends continue; trade with them.
    * `H ≈ 0.5` — a random walk. Neither trend nor reversion rules apply.
    * `H < 0.5` — anti-persistent. Moves revert; fade them.

    Estimated from the slope of `log(std(lagged diff))` against `log(lag)`.
    Noisy on short windows — which is why `HurstAdaptive` requires the estimate
    to sit clearly away from 0.5 before it acts on it, rather than treating
    0.499 and 0.51 as opposite instructions.

    **What this cannot see.** The estimator measures persistence in the
    *stochastic* increments. A deterministic drift contributes nothing to the
    dispersion of lagged differences — `X[t+k] − X[t]` shifts by the same
    `drift × k` at every `t` — so a steady linear trend with light noise
    estimates near **0**, not near 1. That is the estimator behaving correctly,
    and it is a real limitation for a trading rule: a smoothly trending market
    can read as anti-persistent. `HurstAdaptive` is measuring whether *shocks*
    persist, which is a different question from whether price is going up.
    Pair it with a drift measure; do not treat it as a trend detector.

    Returns NaN for a series whose lagged differences have zero dispersion (a
    noiseless line), where the log-log regression is undefined.
    """
    if len(prices) < max_lag * 2:
        return float("nan")
    lags = np.arange(2, max_lag)
    deviations = [float(np.std(prices[lag:] - prices[:-lag])) for lag in lags]
    if any(d <= 0 for d in deviations):
        return float("nan")
    slope = np.polyfit(np.log(lags), np.log(deviations), 1)[0]
    return float(slope)


def variance_ratio(prices: np.ndarray, period: int = 5) -> float:
    """Lo & MacKinlay variance ratio.

    `Var(q-period return) / (q · Var(1-period return))`. Equals 1 for a random
    walk, exceeds 1 under positive autocorrelation (trending), falls below 1
    under mean reversion. A direct, testable statement about which family of
    rule the series currently supports.
    """
    returns = np.diff(np.log(np.maximum(prices, 1e-12)))
    if len(returns) < period * 4:
        return float("nan")
    var_one = float(np.var(returns, ddof=1))
    if var_one <= 0:
        return float("nan")
    aggregated = np.array(
        [float(np.sum(returns[k : k + period])) for k in range(len(returns) - period + 1)]
    )
    return float(np.var(aggregated, ddof=1) / (period * var_one))


class HurstAdaptive(Strategy):
    """Let the Hurst exponent choose between trending and fading.

    This is the archive's answer to the oldest question in systematic trading:
    *is this market trending or ranging right now?* Rather than assuming one,
    it estimates persistence on a rolling window and applies the rule the
    estimate supports — momentum when `H > 0.5 + band`, reversion when
    `H < 0.5 − band`, and **nothing at all** in between.

    The dead band is the whole design. Without it, an estimate hovering around
    0.5 flips the strategy's entire logic bar to bar on measurement noise, which
    is worse than either rule applied consistently.

    Conceptually the same move as `RegimeGatedStrategy`, reached with a single
    statistic instead of an HMM, and cheap enough to run inside a sweep.
    """

    name = "hurst_adaptive"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 200,
        band: float = 0.08,
        entry_z: float = 1.5,
        momentum_lookback: int = 50,
        stop_atr: float = 2.0,
        target_atr: float = 3.5,
    ) -> None:
        super().__init__(
            window=window, band=band, entry_z=entry_z,
            momentum_lookback=momentum_lookback,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.window = window
        self.band = band
        self.entry_z = entry_z
        self.momentum_lookback = momentum_lookback
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 2:
            return None
        closes = context.series.closes
        prices = closes[i - self.window + 1 : i + 1]
        hurst = hurst_exponent(prices)
        if not np.isfinite(hurst):
            return None

        mean = float(np.mean(prices))
        std = float(np.std(prices))
        if std <= 0:
            return None
        z = (context.price - mean) / std

        if hurst > 0.5 + self.band:
            # Persistent: trade with the trailing move.
            change = float(closes[i] - closes[i - self.momentum_lookback])
            if change == 0:
                return None
            direction = Direction.BULLISH if change > 0 else Direction.BEARISH
            mode = f"persistent (H={hurst:.2f}) — following"
        elif hurst < 0.5 - self.band:
            # Anti-persistent: fade the stretch, but only a real one.
            if abs(z) < self.entry_z:
                return None
            direction = Direction.BULLISH if z < 0 else Direction.BEARISH
            mode = f"anti-persistent (H={hurst:.2f}) — fading {z:+.2f}σ"
        else:
            return None  # random walk: no rule applies

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.35 + abs(hurst - 0.5) * 4),
            rationale=f"Hurst {mode}",
            tags=("adaptive", "hurst", "regime", context.session),
            min_rr=1.3,
        )


class VarianceRatio(Strategy):
    """Trade the direction the Lo–MacKinlay variance ratio supports.

    A close relative of `HurstAdaptive` arrived at from the econometrics side.
    Both are included on purpose: they measure the same underlying property with
    different estimators, so **disagreement between them is informative** and
    agreement is one observation, not two.
    """

    name = "variance_ratio"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 250,
        period: int = 5,
        trend_threshold: float = 1.15,
        revert_threshold: float = 0.85,
        entry_z: float = 1.5,
        stop_atr: float = 2.0,
        target_atr: float = 3.5,
    ) -> None:
        super().__init__(
            window=window, period=period, trend_threshold=trend_threshold,
            revert_threshold=revert_threshold, entry_z=entry_z,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.window = window
        self.period = period
        self.trend_threshold = trend_threshold
        self.revert_threshold = revert_threshold
        self.entry_z = entry_z
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 2:
            return None
        prices = context.series.closes[i - self.window + 1 : i + 1]
        vr = variance_ratio(prices, self.period)
        if not np.isfinite(vr):
            return None

        mean, std = float(np.mean(prices)), float(np.std(prices))
        if std <= 0:
            return None
        z = (context.price - mean) / std

        if vr >= self.trend_threshold:
            change = float(prices[-1] - prices[-self.period - 1])
            if change == 0:
                return None
            direction = Direction.BULLISH if change > 0 else Direction.BEARISH
            mode = "trending"
        elif vr <= self.revert_threshold:
            if abs(z) < self.entry_z:
                return None
            direction = Direction.BULLISH if z < 0 else Direction.BEARISH
            mode = "mean-reverting"
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
            confidence=min(1.0, 0.35 + abs(vr - 1.0)),
            rationale=f"variance ratio {vr:.2f} over {self.period} bars — {mode}",
            tags=("adaptive", "variance-ratio", "econometric", context.session),
            min_rr=1.3,
        )


class OrderFlowImbalance(Strategy):
    """Signed-volume pressure from bar internals.

    Each bar's volume is apportioned by where it closed within its own range —
    a close at the high assigns all volume to buyers, at the midpoint it splits
    evenly. Summed over a window this approximates net order flow.

    **It is a proxy.** Real OFI is computed from order-book message events, and
    this is a bar-level reconstruction of it. The literature's predictive
    results are for the real thing; do not transfer them here without measuring.
    Unusable on feeds with unreliable volume — FX, and any single-venue equity
    feed.
    """

    name = "order_flow_imbalance"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 20,
        threshold: float = 0.35,
        stop_atr: float = 1.5,
        target_atr: float = 2.5,
    ) -> None:
        super().__init__(
            window=window, threshold=threshold, stop_atr=stop_atr, target_atr=target_atr
        )
        self.window = window
        self.threshold = threshold
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 2:
            return None
        series = context.series
        window = slice(i - self.window + 1, i + 1)
        high, low = series.highs[window], series.lows[window]
        close, volume = series.closes[window], series.volumes[window]
        span = high - low
        if float(np.sum(volume)) <= 0 or np.any(span < 0):
            return None

        # Close location value in [-1, 1]: +1 closed at the high, -1 at the low.
        with np.errstate(divide="ignore", invalid="ignore"):
            clv = np.where(span > 0, ((close - low) - (high - close)) / span, 0.0)
        imbalance = float(np.sum(clv * volume) / np.sum(volume))
        if abs(imbalance) < self.threshold:
            return None

        direction = Direction.BULLISH if imbalance > 0 else Direction.BEARISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, abs(imbalance)),
            rationale=(
                f"volume-weighted close-location imbalance {imbalance:+.2f} over "
                f"{self.window} bars (bar-level proxy for order flow)"
            ),
            tags=("microstructure", "order-flow", "volume", context.session),
            min_rr=1.3,
        )


class NarrowRangeBreakout(Strategy):
    """NR7 — break of the narrowest range in N bars.

    Crabel's rule: the narrowest bar in seven marks a volatility contraction,
    and the break of its extreme initiates the expansion. The same premise as
    `BollingerSqueeze` measured on a single bar instead of a window, and the
    same premise ICT expresses as consolidation before displacement.
    """

    name = "narrow_range_breakout"
    requires_ict = False

    def __init__(
        self,
        *,
        lookback: int = 7,
        stop_atr: float = 1.0,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(lookback=lookback, stop_atr=stop_atr, target_atr=target_atr)
        self.lookback = lookback
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.lookback + 2:
            return None
        series = context.series
        ranges = series.highs[i - self.lookback : i] - series.lows[i - self.lookback : i]
        if len(ranges) < self.lookback or float(np.min(ranges)) <= 0:
            return None
        # The narrowest bar must be the most recent completed one.
        if int(np.argmin(ranges)) != len(ranges) - 1:
            return None

        setup_high = float(series.highs[i - 1])
        setup_low = float(series.lows[i - 1])
        price = context.price
        if price > setup_high:
            direction = Direction.BULLISH
        elif price < setup_low:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        compression = float(ranges[-1]) / atr
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 1.0 - compression),
            rationale=(
                f"NR{self.lookback}: narrowest bar in {self.lookback} "
                f"({compression:.2f} ATR) broken "
                f"{'up' if direction is Direction.BULLISH else 'down'}"
            ),
            tags=("breakout", "volatility", "nr7", context.session),
            min_rr=1.5,
        )


class GapFade(Strategy):
    """Fade an opening gap back toward the prior close.

    The documented effect is that ordinary gaps fill: the open overshoots on
    thin pre-session flow and reverts once real liquidity arrives. The rule only
    applies to *ordinary* gaps, so an upper bound rules out the ones that are
    repricing on news — those keep going, and fading them is how this strategy's
    worst losses happen.
    """

    name = "gap_fade"
    requires_ict = False

    def __init__(
        self,
        *,
        min_gap_atr: float = 0.5,
        max_gap_atr: float = 3.0,
        stop_atr: float = 2.0,
    ) -> None:
        super().__init__(
            min_gap_atr=min_gap_atr, max_gap_atr=max_gap_atr, stop_atr=stop_atr
        )
        self.min_gap_atr = min_gap_atr
        self.max_gap_atr = max_gap_atr
        self.stop_atr = stop_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < 30:
            return None
        series = context.series
        local = series.index.tz_convert(context.instrument.session_tz)
        # Only on the session's first bar; the gap is defined at the open.
        if i == 0 or local[i].date() == local[i - 1].date():
            return None

        prior_close = float(series.closes[i - 1])
        gap = context.price - prior_close
        atr = max(context.atr(), context.instrument.tick_size)
        size = abs(gap) / atr
        if not self.min_gap_atr <= size <= self.max_gap_atr:
            return None

        direction = Direction.BEARISH if gap > 0 else Direction.BULLISH
        stop = _atr_stop(context, direction, self.stop_atr)
        return _build(
            context, direction, stop, prior_close,
            confidence=min(1.0, 0.4 + (self.max_gap_atr - size) / self.max_gap_atr),
            rationale=(
                f"{'up' if gap > 0 else 'down'} gap of {size:.2f} ATR from prior "
                f"close {prior_close:,.2f} — fading toward fill"
            ),
            tags=("mean-reversion", "gap", "session", context.session),
            min_rr=1.0,
        )


class PreviousDayLevels(Strategy):
    """Break of the prior session's high or low.

    The most-watched intraday levels there are, and the direct quantitative
    analogue of ICT's buyside/sellside liquidity pools: both say resting orders
    cluster at the previous session's extremes. This one trades the *break*
    (continuation); `LiquidityRaidReversal` trades the *rejection*. Running both
    is a clean way to measure which reading the data supports — that is exactly
    the comparison `docs/REAL_DATA_FINDINGS.md` ran, and the break side won.
    """

    name = "previous_day_levels"
    requires_ict = False

    def __init__(
        self,
        *,
        buffer_atr: float = 0.1,
        stop_atr: float = 1.5,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            buffer_atr=buffer_atr, stop_atr=stop_atr, target_atr=target_atr
        )
        self.buffer_atr = buffer_atr
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < 30:
            return None
        series = context.series
        local = series.index.tz_convert(context.instrument.session_tz)
        today = local[i].date()

        start = i
        while start > 0 and local[start - 1].date() == today:
            start -= 1
        if start == 0:
            return None
        prior_day = local[start - 1].date()
        prior_start = start - 1
        while prior_start > 0 and local[prior_start - 1].date() == prior_day:
            prior_start -= 1

        window = slice(prior_start, start)
        prior_high = float(np.max(series.highs[window]))
        prior_low = float(np.min(series.lows[window]))
        atr = max(context.atr(), context.instrument.tick_size)
        buffer = self.buffer_atr * atr
        price = context.price
        prev = float(series.closes[i - 1])

        if price > prior_high + buffer and prev <= prior_high + buffer:
            direction = Direction.BULLISH
            level = prior_high
        elif price < prior_low - buffer and prev >= prior_low - buffer:
            direction = Direction.BEARISH
            level = prior_low
        else:
            return None

        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=0.5,
            rationale=(
                f"broke the prior session "
                f"{'high' if direction is Direction.BULLISH else 'low'} at "
                f"{level:,.2f} (continuation reading)"
            ),
            tags=("breakout", "session", "liquidity", context.session),
            min_rr=1.5,
        )


class FiftyTwoWeekHigh(Strategy):
    """Proximity to the trailing extreme, George & Hwang (2004).

    The finding is that nearness to the 52-week high predicts returns better
    than raw momentum does — the anchoring interpretation being that traders
    under-react to news that pushes a price to a new extreme.

    `window` is in **bars**, not weeks: 252 on daily bars is a year, but on 15m
    bars it is about four days. Setting it without regard to the timeframe is
    the most common way this rule gets misapplied.
    """

    name = "fifty_two_week_high"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 252,
        proximity: float = 0.98,
        stop_atr: float = 2.5,
        target_atr: float = 5.0,
    ) -> None:
        super().__init__(
            window=window, proximity=proximity, stop_atr=stop_atr, target_atr=target_atr
        )
        self.window = window
        self.proximity = proximity
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 2:
            return None
        series = context.series
        window = slice(i - self.window + 1, i + 1)
        high = float(np.max(series.highs[window]))
        low = float(np.min(series.lows[window]))
        if high <= low:
            return None
        price = context.price
        ratio = price / high
        prev_ratio = float(series.closes[i - 1]) / high

        # Cross into the proximity band, not every bar spent inside it.
        if not (ratio >= self.proximity and prev_ratio < self.proximity):
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, Direction.BULLISH, self.stop_atr)
        target = price + self.target_atr * atr
        return _build(
            context, Direction.BULLISH, stop, target,
            confidence=min(1.0, 0.4 + (ratio - self.proximity) * 20),
            rationale=(
                f"within {ratio:.1%} of the {self.window}-bar high {high:,.2f} "
                f"— anchoring/underreaction effect"
            ),
            tags=("momentum", "anchoring", "52-week", context.session),
            min_rr=1.5,
        )


class AutocorrelationReversal(Strategy):
    """Fade the last move when lag-1 autocorrelation is reliably negative.

    A direct measurement of short-horizon reversion: estimate the lag-1
    autocorrelation of returns and fade only when it is clearly negative. This
    is `MeanReversionZScore` with its premise tested rather than assumed — and
    the two disagreeing is a signal that the reversion is a level effect rather
    than a serial-dependence one.
    """

    name = "autocorr_reversal"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 120,
        max_autocorr: float = -0.05,
        entry_atr: float = 1.0,
        stop_atr: float = 2.0,
        target_atr: float = 2.0,
    ) -> None:
        super().__init__(
            window=window, max_autocorr=max_autocorr, entry_atr=entry_atr,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.window = window
        self.max_autocorr = max_autocorr
        self.entry_atr = entry_atr
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 3:
            return None
        closes = context.series.closes[i - self.window + 1 : i + 1]
        returns = np.diff(closes)
        if len(returns) < 20 or float(np.std(returns)) <= 0:
            return None
        rho = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        if not np.isfinite(rho) or rho > self.max_autocorr:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        last_move = float(closes[-1] - closes[-2])
        if abs(last_move) < self.entry_atr * atr:
            return None

        direction = Direction.BEARISH if last_move > 0 else Direction.BULLISH
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.35 + abs(rho) * 3),
            rationale=(
                f"lag-1 autocorrelation {rho:+.3f} over {self.window} bars; "
                f"fading a {last_move / atr:+.2f} ATR move"
            ),
            tags=("mean-reversion", "autocorrelation", "econometric", context.session),
            min_rr=1.0,
        )


class SkewnessReversal(Strategy):
    """Fade into negatively-skewed return distributions.

    Idiosyncratic skewness is a documented cross-sectional predictor: investors
    overpay for lottery-like positive skew, so positively-skewed assets
    underperform. Applied here as a time-series filter rather than
    cross-sectionally, which is a **weaker** version of the claim and is
    labelled as such — the original result is about the cross-section.
    """

    name = "skewness_reversal"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 120,
        threshold: float = 0.5,
        entry_z: float = 1.5,
        stop_atr: float = 2.0,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            window=window, threshold=threshold, entry_z=entry_z,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.window = window
        self.threshold = threshold
        self.entry_z = entry_z
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 2:
            return None
        closes = context.series.closes[i - self.window + 1 : i + 1]
        returns = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
        std = float(np.std(returns))
        if std <= 0:
            return None
        skew = float(np.mean(((returns - np.mean(returns)) / std) ** 3))
        if abs(skew) < self.threshold:
            return None

        mean_price = float(np.mean(closes))
        price_std = float(np.std(closes))
        if price_std <= 0:
            return None
        z = (context.price - mean_price) / price_std
        if abs(z) < self.entry_z:
            return None
        # Positive skew → expect underperformance; fade strength, not weakness.
        if skew > 0 and z <= 0:
            return None
        if skew < 0 and z >= 0:
            return None

        direction = Direction.BEARISH if skew > 0 else Direction.BULLISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(1.0, 0.35 + abs(skew) / 3.0),
            rationale=(
                f"return skew {skew:+.2f} over {self.window} bars at {z:+.2f}σ "
                f"(time-series form of a cross-sectional effect)"
            ),
            tags=("mean-reversion", "skewness", "factor", context.session),
            min_rr=1.2,
        )


class AmihudIlliquidity(Strategy):
    """Trade the illiquidity premium: |return| per unit of volume.

    Amihud's (2002) measure. A spike means price is moving on little volume,
    which is the signature of a move that is not supported by participation and
    tends to retrace. Bar-level, so it inherits every caveat in this module's
    docstring about volume quality.
    """

    name = "amihud_illiquidity"
    requires_ict = False

    def __init__(
        self,
        *,
        window: int = 60,
        spike_percentile: float = 90.0,
        stop_atr: float = 2.0,
        target_atr: float = 2.5,
    ) -> None:
        super().__init__(
            window=window, spike_percentile=spike_percentile,
            stop_atr=stop_atr, target_atr=target_atr,
        )
        self.window = window
        self.spike_percentile = spike_percentile
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.window + 3:
            return None
        series = context.series
        window = slice(i - self.window + 1, i + 1)
        closes = series.closes[window]
        volumes = series.volumes[window]
        if float(np.sum(volumes)) <= 0:
            return None
        returns = np.abs(np.diff(closes) / np.maximum(closes[:-1], 1e-12))
        illiquidity = returns / np.maximum(volumes[1:], 1e-9)
        finite = illiquidity[np.isfinite(illiquidity)]
        if len(finite) < 10:
            return None

        now = float(illiquidity[-1])
        threshold = float(np.percentile(finite, self.spike_percentile))
        if not np.isfinite(now) or now < threshold:
            return None

        move = float(closes[-1] - closes[-2])
        if move == 0:
            return None
        direction = Direction.BEARISH if move > 0 else Direction.BULLISH
        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=0.4,
            rationale=(
                f"Amihud illiquidity in the top "
                f"{100 - self.spike_percentile:.0f}% of {self.window} bars — "
                f"price moved on thin volume; fading"
            ),
            tags=("mean-reversion", "liquidity", "amihud", context.session),
            min_rr=1.2,
        )
