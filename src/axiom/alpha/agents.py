"""Cross-sectional alpha agents.

Three sources, ported from the uploaded architecture and rebuilt on causal
foundations. Each notes what changed and why, because in every case the
original would have produced a backtest that could not be traded.
"""

from __future__ import annotations

import numpy as np

from axiom.alpha.base import AlphaAgent, AlphaSignal
from axiom.alpha.panel import Panel


def _clip(value: float, limit: float = 1.0) -> float:
    return float(np.clip(value, -limit, limit))


class MomentumAgent(AlphaAgent):
    """Volatility-scaled trend, ranked across the universe.

    Two changes from the original.

    *Confidence was ``min(0.5 + |momentum| * 5, 0.95)``* — a monotone function
    of the signal, which means it added no information: an ensemble weighting
    by it is just squaring the signal. Here confidence is the **stability** of
    the trend, measured as the fraction of sub-windows agreeing on its sign. A
    move that came entirely from one gap is not the same evidence as a steady
    drift of the same size, and the ensemble should be able to tell them apart.

    *Signals are ranked cross-sectionally.* Raw volatility-scaled momentum is
    unbounded and its distribution shifts with market volatility, so a fixed
    threshold means different things in different months. Ranking makes the
    signal scale-free and the book roughly dollar-neutral by construction.
    """

    agent_id = "momentum"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 21))
        self.horizon = int(self.config.get("horizon_bars", 21))
        self.min_history = self.lookback + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        prices = panel.history(index, self.lookback + 1)
        if len(prices) < self.lookback + 1:
            return []

        scores = np.full(panel.n_symbols, np.nan)
        stability = np.zeros(panel.n_symbols)
        raw_returns = np.full(panel.n_symbols, np.nan)

        for col in range(panel.n_symbols):
            column = prices[:, col]
            if not np.all(np.isfinite(column)) or column[0] <= 0 or column[-1] <= 0:
                continue
            total = column[-1] / column[0] - 1.0
            steps = np.diff(np.log(column))
            volatility = float(np.std(steps))
            if volatility <= 0:
                continue
            scores[col] = total / volatility
            raw_returns[col] = total
            # Sign agreement across quarters of the window: a trend that only
            # existed in one quarter is a jump wearing a trend's clothes.
            quarters = np.array_split(steps, 4)
            agree = sum(
                1 for q in quarters if q.size and np.sign(q.sum()) == np.sign(total)
            )
            stability[col] = agree / 4.0

        valid = np.isfinite(scores)
        if valid.sum() < 2:
            return []

        ranked = _cross_sectional_rank(scores, valid)
        return [
            AlphaSignal(
                symbol=panel.symbols[col],
                signal=_clip(ranked[col]),
                # Floor at 0.1: a stable trend is more believable than an
                # erratic one, but no momentum reading is ever certain.
                confidence=float(np.clip(0.1 + 0.7 * stability[col], 0.0, 1.0)),
                expected_return=float(raw_returns[col]),
                horizon_bars=self.horizon,
                strategy="momentum",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
                features_used=("close",),
                metadata={
                    "vol_scaled_momentum": round(float(scores[col]), 4),
                    "trend_stability": round(float(stability[col]), 3),
                },
            )
            for col in range(panel.n_symbols)
            if valid[col]
        ]


class StatArbAgent(AlphaAgent):
    """PCA statistical arbitrage: trade the residual, not the return.

    The original fitted the factor model on the *entire* return matrix and then
    z-scored the last residual against statistics computed from that same fit.
    Two leaks compound there: the scored bar is inside its own training set, and
    the standardising mean and variance already know about it. The residual
    that comes out looks unusually informative because it partly *is* the
    answer. This is the single most common way a stat-arb backtest produces a
    Sharpe that evaporates in production.

    Here the factor model is fitted on the window strictly *before* the scored
    bar, then that fitted model projects the scored bar. Standardisation uses
    the training residuals only. Nothing the signal touches includes the bar it
    is scoring.

    Confidence is the fraction of cross-sectional variance the factor model
    actually explains. When the factors explain little, the "residual" is mostly
    unmodelled noise and a large z-score means nothing — exactly when a constant
    confidence would have kept betting.
    """

    agent_id = "stat_arb"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 60))
        self.n_factors = int(self.config.get("n_factors", 3))
        self.entry_z = float(self.config.get("entry_z", 1.5))
        self.horizon = int(self.config.get("horizon_bars", 5))
        self.min_history = self.lookback + 3

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        returns = panel.returns(index, self.lookback)
        if len(returns) < max(self.n_factors * 3, 20) or panel.n_symbols < 3:
            return []
        if not np.all(np.isfinite(returns)):
            return []

        # Strictly before the scored bar: the model never sees `target`.
        train, target = returns[:-1], returns[-1]
        if len(train) < self.n_factors + 2:
            return []

        centre = train.mean(axis=0)
        centred = train - centre
        n_factors = min(self.n_factors, min(centred.shape) - 1)
        if n_factors < 1:
            return []

        # SVD rather than a PCA object: it keeps the fitted basis explicit, so
        # projecting a single out-of-sample row is unambiguous.
        _, singular, components = np.linalg.svd(centred, full_matrices=False)
        basis = components[:n_factors]

        train_residuals = centred - (centred @ basis.T) @ basis
        target_centred = target - centre
        target_residual = target_centred - (target_centred @ basis.T) @ basis

        spread = train_residuals.std(axis=0)
        total_variance = float((singular**2).sum())
        explained = (
            float((singular[:n_factors] ** 2).sum() / total_variance)
            if total_variance > 0
            else 0.0
        )

        signals: list[AlphaSignal] = []
        for col in range(panel.n_symbols):
            if spread[col] <= 0:
                continue
            z = float(target_residual[col] / spread[col])
            if abs(z) < self.entry_z:
                continue
            signals.append(
                AlphaSignal(
                    symbol=panel.symbols[col],
                    # Fade the residual: cheap relative to factors ⇒ buy.
                    signal=_clip(-np.sign(z) * min(abs(z) / 3.0, 1.0)),
                    confidence=float(np.clip(explained, 0.0, 1.0)),
                    expected_return=float(-target_residual[col]),
                    horizon_bars=self.horizon,
                    strategy="stat_arb",
                    agent_id=self.agent_id,
                    as_of_index=index,
                    provenance=panel.provenance,
                    features_used=("close",),
                    metadata={
                        "residual_z": round(z, 3),
                        "variance_explained": round(explained, 3),
                        "n_factors": n_factors,
                    },
                )
            )
        return signals


class VolumePressureAgent(AlphaAgent):
    """Relative-volume confirmation of short-term direction.

    The uploaded architecture computed order-book imbalance from a live book.
    AXIOM's data layer serves OHLCV bars and no venue in it publishes depth, so
    an order-book agent here would have nothing to read — it would be a class
    that never runs. This is the honest bar-data analogue: whether the recent
    move came on unusual participation.

    A book-depth agent belongs here the day a depth feed does. This one does not
    pretend to be one.
    """

    agent_id = "volume_pressure"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 20))
        self.horizon = int(self.config.get("horizon_bars", 3))
        self.min_history = self.lookback + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        prices = panel.history(index, self.lookback + 1)
        volumes = panel.volume_history(index, self.lookback)
        if len(prices) < 3 or len(volumes) < self.lookback:
            return []

        signals: list[AlphaSignal] = []
        for col in range(panel.n_symbols):
            column, volume = prices[:, col], volumes[:, col]
            if not np.all(np.isfinite(column)) or column[-2] <= 0:
                continue
            baseline = float(np.mean(volume[:-1]))
            if baseline <= 0:
                continue
            relative = float(volume[-1] / baseline)
            move = float(column[-1] / column[-2] - 1.0)
            if relative < 1.2 or move == 0.0:
                continue
            signals.append(
                AlphaSignal(
                    symbol=panel.symbols[col],
                    signal=_clip(np.sign(move) * min((relative - 1.0), 1.0)),
                    # Belief grows with how unusual the participation was, and
                    # saturates — 10× volume is not ten times the evidence 2× is.
                    confidence=float(np.clip((relative - 1.0) / 2.0, 0.0, 0.8)),
                    expected_return=None,
                    horizon_bars=self.horizon,
                    strategy="volume_pressure",
                    agent_id=self.agent_id,
                    as_of_index=index,
                    provenance=panel.provenance,
                    features_used=("close", "volume"),
                    metadata={
                        "relative_volume": round(relative, 3),
                        "bar_return": round(move, 5),
                    },
                )
            )
        return signals


def _cross_sectional_rank(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map valid scores onto ``[-1, +1]`` by rank, centred on the median.

    Rank rather than z-score: a single outlier drags a z-score distribution and
    would size the whole book off one instrument's move.
    """
    ranked = np.zeros_like(scores)
    order = np.argsort(scores[valid])
    positions = np.empty(order.size)
    positions[order] = np.arange(order.size)
    if order.size > 1:
        ranked[valid] = 2.0 * positions / (order.size - 1) - 1.0
    return ranked
