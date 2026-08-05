"""Classical quantitative strategies, and the regime gate that combines them.

Each strategy here is a *reference implementation of a documented anomaly*, not
a proven edge. They exist so the research harness has a diverse, honest
candidate set to search over — a sweep across five genuinely different families
tells you far more than a sweep across fifty parameterisations of one.

Families implemented:

:class:`TimeSeriesMomentum`
    Buy what has gone up. The most robust documented anomaly in the literature
    and the natural baseline any new idea has to beat.

:class:`MeanReversionZScore`
    Fade stretched deviations from a moving anchor. The direct opposite of
    momentum, and profitable in the regimes where momentum is not — which is
    exactly why regime gating matters.

:class:`VolatilityBreakout`
    Trade expansion out of a compressed range. Structurally close to ICT's
    displacement idea, arrived at from a different tradition.

:class:`OpeningRangeBreakout`
    Session-anchored breakout. Shares ICT's premise that the session open
    establishes a reference the rest of the day trades around.

:class:`RegimeGatedStrategy`
    Wraps any strategy and permits it only in nominated regimes. This is the
    ICT/quant merge: the ICT engine finds the *setup*, the HMM decides whether
    the current state is one where that setup has any business working.
"""

from __future__ import annotations

import numpy as np

from axiom.core.types import Direction
from axiom.quant.regime import RegimeLabel, RegimeSeries
from axiom.strategy.base import Signal, Strategy, StrategyContext


def _atr_stop(context: StrategyContext, direction: Direction, multiple: float) -> float:
    atr = max(context.atr(), context.instrument.tick_size)
    price = context.price
    offset = multiple * atr
    return price - offset if direction is Direction.BULLISH else price + offset


def _build(
    context: StrategyContext,
    direction: Direction,
    stop: float,
    target: float,
    confidence: float,
    rationale: str,
    tags: tuple[str, ...],
    min_rr: float,
) -> Signal | None:
    instrument = context.instrument
    entry = instrument.round_to_tick(context.price)
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


class TimeSeriesMomentum(Strategy):
    """Trade in the direction of trailing return over a lookback."""

    name = "ts_momentum"
    requires_ict = False

    def __init__(
        self,
        *,
        lookback: int = 96,
        stop_atr: float = 2.0,
        target_atr: float = 4.0,
        min_strength: float = 0.5,
    ) -> None:
        super().__init__(
            lookback=lookback, stop_atr=stop_atr,
            target_atr=target_atr, min_strength=min_strength,
        )
        self.lookback = lookback
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.min_strength = min_strength

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.lookback:
            return None
        closes = context.series.closes
        change = closes[i] - closes[i - self.lookback]
        atr = max(context.atr(), 1e-12)
        # Normalising by ATR makes the threshold comparable across instruments
        # and across volatility regimes, which a raw percentage is not.
        strength = change / (atr * np.sqrt(self.lookback))
        if abs(strength) < self.min_strength:
            return None

        direction = Direction.BULLISH if strength > 0 else Direction.BEARISH
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=min(abs(strength) / 3.0, 1.0),
            rationale=(
                f"{self.lookback}-bar momentum {strength:+.2f} ATR-normalised "
                f"({change:+.2f} pts)"
            ),
            tags=("momentum", context.session),
            min_rr=self.target_atr / self.stop_atr * 0.9,
        )


class MeanReversionZScore(Strategy):
    """Fade deviations beyond ``entry_z`` standard deviations of the mean."""

    name = "mean_reversion"
    requires_ict = False

    def __init__(
        self,
        *,
        lookback: int = 96,
        entry_z: float = 2.0,
        stop_z: float = 3.5,
        target_z: float = 0.25,
    ) -> None:
        super().__init__(
            lookback=lookback, entry_z=entry_z, stop_z=stop_z, target_z=target_z
        )
        self.lookback = lookback
        self.entry_z = entry_z
        self.stop_z = stop_z
        self.target_z = target_z

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.lookback:
            return None
        window = context.series.closes[i - self.lookback : i + 1]
        mean = float(window.mean())
        sigma = float(window.std(ddof=1))
        if sigma <= 0:
            return None
        z = (context.price - mean) / sigma
        if abs(z) < self.entry_z:
            return None

        # Stretched high -> fade short. The stop sits further out along the
        # same deviation axis, so it scales with the dispersion that produced
        # the signal rather than with an unrelated ATR.
        direction = Direction.BEARISH if z > 0 else Direction.BULLISH
        if direction is Direction.BEARISH:
            stop = mean + self.stop_z * sigma
            target = mean + self.target_z * sigma
        else:
            stop = mean - self.stop_z * sigma
            target = mean - self.target_z * sigma

        return _build(
            context, direction, stop, target,
            confidence=min(abs(z) / 4.0, 1.0),
            rationale=(
                f"price {z:+.2f}σ from its {self.lookback}-bar mean "
                f"({mean:.2f}); fading toward the mean"
            ),
            tags=("mean_reversion", context.session),
            min_rr=0.5,
        )


class VolatilityBreakout(Strategy):
    """Trade expansion out of a volatility contraction.

    Compression then expansion is the same phenomenon ICT calls displacement,
    reached from the volatility-clustering tradition instead of order flow.
    """

    name = "vol_breakout"
    requires_ict = False

    def __init__(
        self,
        *,
        squeeze_lookback: int = 96,
        squeeze_pct: float = 0.3,
        breakout_lookback: int = 24,
        stop_atr: float = 1.5,
        target_atr: float = 3.5,
    ) -> None:
        super().__init__(
            squeeze_lookback=squeeze_lookback, squeeze_pct=squeeze_pct,
            breakout_lookback=breakout_lookback, stop_atr=stop_atr,
            target_atr=target_atr,
        )
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pct = squeeze_pct
        self.breakout_lookback = breakout_lookback
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i < self.squeeze_lookback:
            return None

        atr_series = context.series.atr(14)
        window = atr_series[i - self.squeeze_lookback : i + 1]
        # Was volatility recently in the bottom decile-ish of its own range?
        threshold = float(np.quantile(window, self.squeeze_pct))
        if atr_series[i] > threshold * 1.5:
            return None

        highs = context.series.highs
        lows = context.series.lows
        recent_high = float(highs[i - self.breakout_lookback : i].max())
        recent_low = float(lows[i - self.breakout_lookback : i].min())
        price = context.price

        if price > recent_high:
            direction = Direction.BULLISH
        elif price < recent_low:
            direction = Direction.BEARISH
        else:
            return None

        atr = max(context.atr(), 1e-12)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = price + offset if direction is Direction.BULLISH else price - offset
        return _build(
            context, direction, stop, target,
            confidence=0.55,
            rationale=(
                f"ATR {atr_series[i]:.2f} in the bottom {self.squeeze_pct:.0%} of its "
                f"{self.squeeze_lookback}-bar range, breaking the "
                f"{self.breakout_lookback}-bar "
                f"{'high' if direction is Direction.BULLISH else 'low'}"
            ),
            tags=("vol_breakout", context.session),
            min_rr=self.target_atr / self.stop_atr * 0.9,
        )


class RegimeGatedStrategy(Strategy):
    """Permit an inner strategy only in nominated market regimes.

    **This is the ICT/quant merge.** The inner strategy — an ICT setup, a
    momentum rule, anything — decides *what* the trade is. The regime model
    decides whether the market is currently in a state where that kind of trade
    has any business working. Momentum in a choppy high-volatility regime and
    mean reversion in a strong trend are both reliable ways to lose money with a
    rule that is otherwise sound.

    The regime series must come from
    :meth:`~axiom.quant.regime.RegimeModel.fit_causal`. Passing a full-history
    fit here would gate every historical trade on knowledge of the future and
    make the backtest meaningless; the constructor rejects a series whose
    ``warmup_end`` is zero, which is the signature of a non-causal fit.
    """

    name = "regime_gated"

    def __init__(
        self,
        inner: Strategy,
        regimes: RegimeSeries,
        allowed: tuple[RegimeLabel, ...],
        *,
        min_confidence: float = 0.5,
        require_known: bool = True,
    ) -> None:
        if regimes.warmup_end == 0 and regimes.refit_points == (0,):
            raise ValueError(
                "regime series appears to come from fit_full_history(), which "
                "decodes using the entire sample. Gating a backtest on it leaks "
                "the future into every decision. Use fit_causal()."
            )
        super().__init__(
            inner=inner.name,
            allowed=tuple(r.value for r in allowed),
            min_confidence=min_confidence,
        )
        self.inner = inner
        self.regimes = regimes
        self.allowed = set(allowed)
        self.min_confidence = min_confidence
        self.require_known = require_known
        self.name = f"regime_gated({inner.name})"

    def evaluate(self, context: StrategyContext) -> Signal | None:
        i = context.index
        if i >= len(self.regimes):
            return None
        if not self.regimes.is_known_at(i):
            # No regime yet (warmup) — fail closed rather than trading blind.
            return None if self.require_known else self.inner.evaluate(context)

        label = self.regimes.label_at(i)
        if label not in self.allowed:
            return None
        if self.regimes.confidence_at(i) < self.min_confidence:
            return None

        signal = self.inner.evaluate(context)
        if signal is None:
            return None

        from dataclasses import replace

        return replace(
            signal,
            rationale=(
                f"[{label.value} @ {self.regimes.confidence_at(i):.0%}] "
                f"{signal.rationale}"
            ),
            tags=(*signal.tags, f"regime:{label.value}"),
        )

    def describe(self) -> str:
        allowed = ",".join(sorted(r.value for r in self.allowed))
        return f"{self.name}[allowed={allowed}, min_conf={self.min_confidence}]"
