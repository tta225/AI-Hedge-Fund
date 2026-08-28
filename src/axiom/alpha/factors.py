"""Signals that are slow by construction.

The turnover study established the one result in this project that replicated
out of sample: on a weak cross-sectional signal, holding a name until it left a
*wider* band was worth roughly 0.45 Sharpe and about five times the capacity,
on a universe that had selected neither the signal nor the buffer. Trading less
was worth more than any refinement of direction.

That is a finding about mechanism, and it has an obvious implication that the
earlier campaigns did not act on. Every signal tested so far was fast — 21-bar
momentum, intraday seasonality, sweep continuation — and then made slower by
machinery bolted on afterwards. The alternative is to search families that are
slow *because of what they measure*, where a quarterly rebalance is the natural
frequency rather than an imposition. A 12-month momentum ranking simply does
not reshuffle much in five days; a 3-year reversal ranking barely moves in a
quarter.

Five families, all computable from closes and volumes alone:

:class:`LongHorizonMomentum` — the Jegadeesh-Titman 12-1, skipping the most
recent month.

:class:`LowVolatility` — the low-beta/low-volatility anomaly, ranked on
trailing volatility.

:class:`ResidualMomentum` — momentum measured on returns orthogonal to the
equal-weight market, which is the version with the higher reported information
ratio and, separately, the lower turnover.

:class:`LongTermReversal` — De Bondt-Thaler, over three to five years.

:class:`Illiquidity` — Amihud's measure: absolute return per dollar traded.

**What is deliberately absent: value and quality.** Both are the strongest
low-turnover families in the literature and both need fundamentals — book
value, earnings, accruals, gross profitability — which this platform does not
have a source for. Constructing a "value" proxy from price alone would produce
something that is mostly long-term reversal wearing a different name, and
reporting it as value would be a claim the data cannot support. The gap is
stated here rather than filled with a substitute; closing it needs a
fundamentals feed, not more code.

Every agent here ranks cross-sectionally, for the reason
:func:`~axiom.alpha.agents._cross_sectional_rank` gives: a raw factor score's
distribution shifts with market conditions, so a fixed threshold means
different things in different years.
"""

from __future__ import annotations

import numpy as np

from axiom.alpha.agents import _clip, _cross_sectional_rank
from axiom.alpha.base import AlphaAgent, AlphaSignal
from axiom.alpha.panel import Panel

#: A quarter of trading days. The natural holding period for every family here,
#: and the frequency at which the impact model measurably favours the desk.
QUARTER = 63
#: One trading month, the standard skip in a 12-1 momentum construction.
MONTH = 21


def _finite_columns(prices: np.ndarray) -> np.ndarray:
    """Columns with a complete, strictly positive price history in the window.

    Strict on purpose. A name with a gap is a name whose factor value would be
    computed over a different window than its peers', and a cross-sectional
    ranking that compares windows of different lengths is comparing different
    quantities.
    """
    finite = np.all(np.isfinite(prices), axis=0) & np.all(prices > 0, axis=0)
    return np.asarray(finite, dtype=bool)


def _log_returns(prices: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(np.where(prices > 0, prices, np.nan)), axis=0)


class LongHorizonMomentum(AlphaAgent):
    """Twelve-month return, skipping the most recent month.

    The skip is not a detail. Short-horizon returns reverse — a name up sharply
    last week tends to give some back — so a 12-month window that includes the
    most recent month mixes a momentum effect with a reversal effect of the
    opposite sign, and the combination is weaker than either. Skipping the last
    month is what separates them, and it is why the published construction has
    a gap in it.

    Turnover is low for a structural reason: a ranking over 252 bars moves by
    one bar's worth of information per day, so consecutive rankings are almost
    the same ranking.
    """

    agent_id = "long_horizon_momentum"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 252))
        self.skip = int(self.config.get("skip", MONTH))
        self.horizon = int(self.config.get("horizon_bars", QUARTER))
        if self.skip >= self.lookback:
            raise ValueError("skip must be shorter than lookback")
        self.min_history = self.lookback + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        prices = panel.history(index, self.lookback + 1)
        if len(prices) < self.lookback + 1:
            return []

        # The window ends `skip` bars before the present, so the formation
        # period and the most recent month do not overlap.
        formation = prices[: len(prices) - self.skip]
        if len(formation) < 3:
            return []

        valid = _finite_columns(formation)
        if valid.sum() < 2:
            return []

        with np.errstate(divide="ignore", invalid="ignore"):
            total = formation[-1] / formation[0] - 1.0
        scores = np.where(valid, total, np.nan)

        # Scaling by the volatility of the formation window makes the ranking
        # comparable across names of different risk, so the book is not
        # implicitly long whichever names happen to move most.
        volatility = np.full(panel.n_symbols, np.nan)
        steps = _log_returns(formation)
        if len(steps):
            volatility = np.nanstd(steps, axis=0)
        usable = valid & np.isfinite(scores) & np.isfinite(volatility) & (volatility > 0)
        if usable.sum() < 2:
            return []
        scaled = np.where(usable, scores / np.where(volatility > 0, volatility, np.nan), np.nan)

        ranked = _cross_sectional_rank(np.nan_to_num(scaled, nan=0.0), usable)
        return [
            AlphaSignal(
                symbol=panel.symbols[col],
                signal=_clip(ranked[col]),
                # Belief is how much of the twelve months agreed with the whole.
                # A 40% return earned entirely in one week is a different object
                # from the same return earned steadily, and the ensemble should
                # be able to tell them apart.
                confidence=_persistence(formation[:, col], total[col]),
                expected_return=float(scores[col]),
                horizon_bars=self.horizon,
                strategy="long_horizon_momentum",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
                features_used=("close",),
                metadata={"lookback": self.lookback, "skip": self.skip},
            )
            for col in np.flatnonzero(usable)
        ]


def _persistence(column: np.ndarray, total: float) -> float:
    """Fraction of sub-periods whose sign matched the whole period's.

    Floored at 0.1 rather than 0: a factor reading is never certain, and a
    confidence of exactly zero would remove the name from an ensemble entirely
    rather than down-weighting it.
    """
    steps = np.diff(np.log(column))
    if steps.size < 4 or not np.isfinite(total) or total == 0:
        return 0.1
    parts = np.array_split(steps, 4)
    agree = sum(1 for part in parts if part.size and np.sign(part.sum()) == np.sign(total))
    return float(np.clip(0.1 + 0.7 * (agree / 4.0), 0.0, 1.0))


class LowVolatility(AlphaAgent):
    """Long the calm names, short the wild ones.

    The low-volatility anomaly is the observation that realised return does not
    rise with volatility the way theory says it should, and over long samples
    the low-volatility half of a universe has delivered comparable returns at
    materially lower risk. The usual explanation is leverage aversion: investors
    who want more return and cannot borrow buy volatile names instead, bidding
    them up.

    Turnover is low because trailing volatility is a slow-moving quantity — a
    name does not usually move from the calmest decile to the wildest in a
    quarter.

    Note what this does **not** do: it does not neutralise beta. A book long low
    volatility and short high volatility is structurally short beta, which will
    look like alpha in a falling market and like incompetence in a rising one.
    That is a property of the raw factor, and it is left visible rather than
    hedged away, because hedging it here would hide it from the study that has
    to judge it.
    """

    agent_id = "low_volatility"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 252))
        self.horizon = int(self.config.get("horizon_bars", QUARTER))
        self.min_history = self.lookback + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        prices = panel.history(index, self.lookback + 1)
        if len(prices) < max(self.lookback // 2, 20):
            return []
        valid = _finite_columns(prices)
        steps = _log_returns(prices)
        if len(steps) < 10:
            return []
        volatility = np.nanstd(steps, axis=0)
        usable = valid & np.isfinite(volatility) & (volatility > 0)
        if usable.sum() < 2:
            return []

        # Negated so the calmest name ranks highest.
        ranked = _cross_sectional_rank(np.where(usable, -volatility, 0.0), usable)

        # Belief is the stability of the volatility estimate itself: a name
        # whose first-half and second-half volatilities agree is one whose
        # ranking is likely to survive into the holding period.
        half = len(steps) // 2
        first = np.nanstd(steps[:half], axis=0)
        second = np.nanstd(steps[half:], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.minimum(first, second) / np.maximum(first, second)

        return [
            AlphaSignal(
                symbol=panel.symbols[col],
                signal=_clip(ranked[col]),
                confidence=float(np.clip(np.nan_to_num(ratio[col], nan=0.1), 0.1, 0.9)),
                expected_return=None,
                horizon_bars=self.horizon,
                strategy="low_volatility",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
                features_used=("close",),
                metadata={"volatility": round(float(volatility[col]), 6)},
            )
            for col in np.flatnonzero(usable)
        ]


class ResidualMomentum(AlphaAgent):
    """Momentum in the part of a return the market does not explain.

    Plain momentum is contaminated by market exposure: in a year when equities
    rise, the winners are disproportionately the high-beta names, so a momentum
    book ends up long beta and calls the result alpha. When the market turns,
    that position unwinds violently — the well-documented momentum crash.

    Regressing each name's returns on the equal-weight universe return and
    ranking on the *residual* removes that. The reported effect is a higher
    information ratio and, separately and usefully here, lower turnover: the
    residual ranking is not reshuffled every time the market moves.

    Equal-weight rather than cap-weight because the panel carries no market
    capitalisations. The difference matters for the interpretation — this is
    orthogonal to the universe's own average, not to a published index — and
    saying so is better than implying a benchmark that is not there.
    """

    agent_id = "residual_momentum"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 252))
        self.skip = int(self.config.get("skip", MONTH))
        self.horizon = int(self.config.get("horizon_bars", QUARTER))
        self.min_history = self.lookback + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        prices = panel.history(index, self.lookback + 1)
        if len(prices) < self.lookback + 1:
            return []
        formation = prices[: len(prices) - self.skip]
        valid = _finite_columns(formation)
        if valid.sum() < 3:
            return []

        steps = _log_returns(formation)
        if len(steps) < 30:
            return []
        # The market is the equal-weight average of the names with usable
        # history, computed inside the same window as the regression.
        market = np.nanmean(steps[:, valid], axis=1)
        if not np.all(np.isfinite(market)) or np.std(market) <= 0:
            return []

        scores = np.full(panel.n_symbols, np.nan)
        fit_quality = np.zeros(panel.n_symbols)
        design = np.column_stack([np.ones_like(market), market])
        for col in np.flatnonzero(valid):
            y = steps[:, col]
            if not np.all(np.isfinite(y)):
                continue
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            alpha, beta = float(coefficients[0]), float(coefficients[1])

            # The intercept is fitted so that beta is not biased by the name's
            # own drift, and then deliberately **left in** the residual. This
            # is the one subtlety in the construction: an OLS fit with an
            # intercept forces its residuals to sum to exactly zero, so scoring
            # on `y - design @ coefficients` would rank pure numerical noise for
            # every name. Subtracting only the market component keeps the
            # idiosyncratic drift, which is the quantity the factor is about.
            residuals = y - beta * market
            # Dispersion around the drift, not around zero — the scale the
            # t-statistic below needs.
            spread = float(np.std(residuals - alpha))
            if spread <= 0:
                continue
            # Cumulative residual return over its own standard error: the
            # t-statistic of the idiosyncratic drift, comparable across names
            # of different idiosyncratic risk.
            scores[col] = float(residuals.sum()) / (spread * np.sqrt(len(residuals)))
            total_variance = float(np.var(y))
            fit_quality[col] = (
                1.0 - spread**2 / total_variance if total_variance > 0 else 0.0
            )

        usable = np.isfinite(scores)
        if usable.sum() < 2:
            return []
        ranked = _cross_sectional_rank(np.nan_to_num(scores, nan=0.0), usable)

        return [
            AlphaSignal(
                symbol=panel.symbols[col],
                signal=_clip(ranked[col]),
                # More belief where the market model explained less: a high R²
                # means most of the name's movement was market, so the residual
                # is a small difference between two large numbers.
                confidence=float(np.clip(0.9 - 0.7 * np.clip(fit_quality[col], 0.0, 1.0), 0.1, 0.9)),
                expected_return=None,
                horizon_bars=self.horizon,
                strategy="residual_momentum",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
                features_used=("close",),
                metadata={"r_squared": round(float(fit_quality[col]), 4)},
            )
            for col in np.flatnonzero(usable)
        ]


class LongTermReversal(AlphaAgent):
    """Buy the three-year losers, sell the three-year winners.

    De Bondt and Thaler's result, and the slowest thing here: a ranking over
    750 bars is almost static from quarter to quarter, which makes it the
    natural test of whether the turnover finding generalises. If low turnover is
    genuinely where the cost structure favours this desk, the family that
    trades least should show it most clearly.

    The honest caveat travels with the signal rather than the footnotes: this
    family is the one most damaged by survivorship. A three-year loser that
    subsequently went to zero is not in a universe built from names liquid
    today, so the long leg is drawn from a sample where the worst outcomes have
    been removed. Whatever this measures on such a universe is an upper bound,
    and a generous one.
    """

    agent_id = "long_term_reversal"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 756))
        #: Skip the most recent year, so the medium-term momentum effect is not
        #: subtracted from the long-term reversal effect.
        self.skip = int(self.config.get("skip", 252))
        self.horizon = int(self.config.get("horizon_bars", QUARTER))
        if self.skip >= self.lookback:
            raise ValueError("skip must be shorter than lookback")
        self.min_history = self.lookback + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        prices = panel.history(index, self.lookback + 1)
        if len(prices) < self.lookback + 1:
            return []
        formation = prices[: len(prices) - self.skip]
        if len(formation) < 60:
            return []
        valid = _finite_columns(formation)
        if valid.sum() < 2:
            return []

        with np.errstate(divide="ignore", invalid="ignore"):
            total = formation[-1] / formation[0] - 1.0
        scores = np.where(valid, total, np.nan)
        usable = valid & np.isfinite(scores)
        if usable.sum() < 2:
            return []

        # Negated: the biggest loser gets the strongest long.
        ranked = _cross_sectional_rank(np.where(usable, -np.nan_to_num(scores, nan=0.0), 0.0), usable)
        return [
            AlphaSignal(
                symbol=panel.symbols[col],
                signal=_clip(ranked[col]),
                # Flat and modest across names: nothing in a three-year return
                # distinguishes a more believable reversal from a less
                # believable one, and inventing a spread here would be the
                # unmeasured confidence the base class exists to forbid.
                confidence=0.3,
                expected_return=None,
                horizon_bars=self.horizon,
                strategy="long_term_reversal",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
                features_used=("close",),
                metadata={"formation_return": round(float(scores[col]), 4)},
            )
            for col in np.flatnonzero(usable)
        ]


class Illiquidity(AlphaAgent):
    """Amihud: absolute return per dollar traded.

    A name whose price moves a lot on little volume is illiquid, and illiquid
    names have historically carried a return premium as compensation for the
    cost of getting out of them.

    This factor has a property the others do not, and it is the reason it is
    included in a study about costs: **the premium and the cost are the same
    quantity**. A book long illiquidity is long the names the square-root impact
    model charges most to trade, so any measured edge is competing directly
    against its own execution. If it survives the impact model it means
    something; if it only works gross, that is not a finding, it is the
    definition of the factor.
    """

    agent_id = "illiquidity"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.lookback = int(self.config.get("lookback", 252))
        self.horizon = int(self.config.get("horizon_bars", QUARTER))
        self.min_history = self.lookback + 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        prices = panel.history(index, self.lookback + 1)
        volumes = panel.volume_history(index, self.lookback + 1)
        if len(prices) < 30 or len(volumes) != len(prices):
            return []

        valid = _finite_columns(prices)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.abs(np.diff(prices, axis=0) / prices[:-1])
            dollar_volume = prices[1:] * volumes[1:]
            ratio = np.where(dollar_volume > 0, returns / dollar_volume, np.nan)

        with np.errstate(invalid="ignore"):
            illiquidity = np.nanmean(ratio, axis=0)
        usable = valid & np.isfinite(illiquidity) & (illiquidity > 0)
        if usable.sum() < 2:
            return []

        # Log, because Amihud's measure spans orders of magnitude across a
        # universe and a rank taken on the raw value is decided entirely by the
        # thinnest one or two names.
        scores = np.where(usable, np.log(np.where(illiquidity > 0, illiquidity, np.nan)), np.nan)
        ranked = _cross_sectional_rank(np.nan_to_num(scores, nan=0.0), usable)

        # Fraction of the window with real volume: a name that did not trade on
        # half its days has an illiquidity estimate built from few observations.
        coverage = np.mean(dollar_volume > 0, axis=0)
        return [
            AlphaSignal(
                symbol=panel.symbols[col],
                signal=_clip(ranked[col]),
                confidence=float(np.clip(coverage[col] * 0.8, 0.1, 0.8)),
                expected_return=None,
                horizon_bars=self.horizon,
                strategy="illiquidity",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
                features_used=("close", "volume"),
                metadata={"log_amihud": round(float(scores[col]), 4)},
            )
            for col in np.flatnonzero(usable)
        ]


#: Every low-turnover family, for a study that wants the whole set. Ordered by
#: how slowly each moves, which is also roughly the order of how much each
#: depends on the universe being free of survivorship bias.
LOW_TURNOVER_FAMILIES: tuple[type[AlphaAgent], ...] = (
    LongHorizonMomentum,
    ResidualMomentum,
    LowVolatility,
    Illiquidity,
    LongTermReversal,
)
