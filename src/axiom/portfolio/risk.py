"""Portfolio risk: what five positions actually are, and how big they may be.

:class:`~axiom.risk.manager.RiskManager` sizes each trade in isolation against a
per-trade budget, and caps the count and gross exposure. That is correct as far
as it goes, and it contains a specific blind spot that this module exists to
close.

**Five positions in correlated instruments are one position at five times the
size.** A desk long BTC, ETH and SOL with a 0.5% risk budget on each believes it
has three independent 0.5% bets. It has one bet of roughly 1.5%, because those
three move together, and on the day it is wrong it is wrong on all three at
once. The position count says "diversified" and the covariance says otherwise.

So the control here is not on the *sum* of individual risks but on the risk of
the portfolio as an object:

    σ_p = √(wᵀ Σ w)

which collapses toward the sum as correlations approach 1, and falls well below
it when they do not. Capping σ_p is the honest version of what capping position
count was trying to approximate.

Three further scalars multiply into the same decision:

*Volatility targeting.* Fixed fractional sizing takes more risk in volatile
regimes, because the same percentage of equity buys a wider swing. Scaling
exposure by the ratio of target to realised volatility keeps the risk taken
roughly constant instead of the position size.

*Drawdown de-risking.* Betting the same size into a losing streak is how a
drawdown becomes terminal. Exposure is cut progressively as equity falls from
its high-water mark, well before the halt threshold in
:mod:`axiom.desk.guards` is reached.

*Expected shortfall.* Value at risk answers "how bad is the 5th percentile
day", which is the wrong question — it says nothing about the tail beyond it.
Expected shortfall is the mean of that tail, and it is the number that
describes what a bad day actually costs.

Everything here is computed from :class:`~axiom.alpha.panel.Panel`, whose
accessors take the bar being scored and return only history strictly before it.
Correlations estimated with hindsight would make every portfolio look
diversifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from axiom.alpha.panel import Panel

#: Trading days per year, the annualisation factor for *daily* bars.
TRADING_DAYS = 252
#: Calendar days per year, used for instruments that trade continuously.
CALENDAR_DAYS = 365
#: Minimum observations before a covariance estimate is used at all. Below
#: this the matrix is noise wearing the shape of a matrix.
MIN_OBSERVATIONS = 30
#: Default annualised portfolio volatility target.
DEFAULT_VOL_TARGET = 0.15
#: Exposure is never scaled above this, whatever the volatility ratio says.
#: Without a cap, a quiet stretch produces enormous leverage precisely before
#: volatility mean-reverts.
MAX_LEVERAGE = 2.0
#: Drawdown at which de-risking begins, and at which it reaches its floor.
DERISK_START_PCT = 5.0
DERISK_FLOOR_PCT = 15.0
#: Exposure multiplier at or beyond the floor. Not zero: a hard stop belongs in
#: the guards, where it is explicit and persisted, not hidden in a size curve.
DERISK_FLOOR_SCALAR = 0.25


class InsufficientHistoryError(ValueError):
    """Not enough observations to estimate the quantity requested."""


def periods_per_year(panel: Panel) -> float:
    """Annualisation factor inferred from the panel's own bar spacing.

    This must be derived rather than assumed. A hardcoded 252 applied to hourly
    bars understates volatility by ``√24`` — roughly a factor of five — and a
    risk system that reports 8% when the truth is 40% is worse than no risk
    system, because it is confidently wrong in the permissive direction.

    Inferred from the **median** gap, so a single missing bar or a venue outage
    does not move the estimate. Continuous instruments annualise on calendar
    days; anything at or above daily spacing uses trading days.
    """
    if len(panel.index) < 2:
        return TRADING_DAYS
    gaps = np.diff(panel.index.to_numpy()).astype("timedelta64[s]").astype(float)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return TRADING_DAYS
    seconds = float(np.median(gaps))
    day = 24 * 60 * 60
    if seconds >= day:
        # Daily or slower: count trading days, since weekends are not bars.
        return TRADING_DAYS * day / seconds
    # Intraday: the bars themselves say how many periods a year holds. Crypto
    # runs continuously, so calendar days are the honest denominator; an
    # equity intraday series simply has fewer bars and the same arithmetic
    # still yields its own correct count.
    return CALENDAR_DAYS * day / seconds


def covariance_matrix(
    panel: Panel,
    index: int,
    *,
    lookback: int = 120,
    annualise: bool = True,
    periods: float | None = None,
) -> np.ndarray:
    """Return covariance over the history strictly before ``index``.

    Args:
        panel: the aligned universe.
        index: the bar being scored. History is exclusive of it.
        lookback: observations to use.
        annualise: scale to annual units. Turn it off when combining with
            per-bar quantities.
        periods: annualisation factor. Defaults to :func:`periods_per_year`,
            inferred from the panel's bar spacing.

    Raises:
        InsufficientHistoryError: below :data:`MIN_OBSERVATIONS` usable rows.
    """
    returns = panel.returns(index, lookback)
    usable = returns[np.isfinite(returns).all(axis=1)] if returns.size else returns
    if len(usable) < MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"{len(usable)} usable observations before bar {index}; "
            f"{MIN_OBSERVATIONS} are needed before a covariance estimate means "
            "anything"
        )
    covariance = np.cov(usable, rowvar=False, ddof=1)
    # np.cov collapses to a scalar for a single column; keep the shape stable
    # so callers never branch on universe size.
    covariance = np.atleast_2d(covariance)
    if not annualise:
        return covariance
    factor = periods_per_year(panel) if periods is None else periods
    return np.asarray(covariance * factor, dtype=float)


def correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to correlations.

    A zero-variance column (a stalled feed, a halted instrument) would divide
    by zero. Those are given a correlation of zero with everything, which is
    the honest answer: a series that did not move carries no information about
    co-movement.
    """
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    deviations = np.sqrt(np.diag(covariance))
    safe = np.where(deviations > 0, deviations, 1.0)
    correlation = covariance / np.outer(safe, safe)
    dead = deviations <= 0
    if dead.any():
        correlation[dead, :] = 0.0
        correlation[:, dead] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return np.asarray(np.clip(correlation, -1.0, 1.0), dtype=float)


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    """``√(wᵀ Σ w)`` — the risk of the book as one object.

    Weights are signed portfolio fractions, so a short contributes negatively
    and can genuinely reduce total risk. Clipping at zero before the square
    root guards against a covariance estimate that is not quite positive
    semi-definite, which happens with short samples.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    if weights.shape[0] != covariance.shape[0]:
        raise ValueError(
            f"{weights.shape[0]} weights against a "
            f"{covariance.shape[0]}x{covariance.shape[1]} covariance matrix"
        )
    variance = float(weights @ covariance @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def diversification_ratio(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Sum of standalone risks divided by portfolio risk.

    1.0 means the positions are perfectly correlated and the book is one bet
    wearing several names. Higher means genuine diversification. This is the
    number that tells you whether a position *count* means anything.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    deviations = np.sqrt(np.diag(np.atleast_2d(covariance)))
    standalone = float(np.abs(weights) @ deviations)
    portfolio = portfolio_volatility(weights, covariance)
    if portfolio <= 0:
        return 1.0
    return standalone / portfolio


def effective_bets(weights: np.ndarray, covariance: np.ndarray) -> float:
    """How many independent positions the book actually holds.

    The squared diversification ratio. Five perfectly correlated positions give
    1.0; five uncorrelated equal-sized ones give 5.0. Reporting this next to
    the raw position count is the clearest way to show the gap between what the
    risk manager counts and what the market sees.
    """
    return diversification_ratio(weights, covariance) ** 2


def marginal_risk_contributions(
    weights: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Each position's share of total portfolio risk, summing to 1.

    ``wᵢ · (Σw)ᵢ / σ_p``. This is where a concentrated book gives itself away:
    a position that is 20% of the notional can easily be 60% of the risk, and
    the notional weight will not show it.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    portfolio = portfolio_volatility(weights, covariance)
    if portfolio <= 0:
        return np.zeros_like(weights)
    contributions = weights * (covariance @ weights) / (portfolio**2)
    return np.asarray(contributions, dtype=float)


def volatility_target_scalar(
    realised_volatility: float,
    *,
    target: float = DEFAULT_VOL_TARGET,
    max_leverage: float = MAX_LEVERAGE,
) -> float:
    """Scale exposure so realised volatility lands near ``target``.

    Capped at ``max_leverage``, and the cap is not decoration. Volatility
    mean-reverts, so the quietest stretch — where the uncapped ratio is largest
    — is systematically the moment just before it is wrong to be levered.

    A zero or non-finite volatility estimate returns 1.0 rather than infinity:
    "we cannot measure risk" must not read as "there is no risk".
    """
    if target <= 0:
        raise ValueError("volatility target must be positive")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    if not np.isfinite(realised_volatility) or realised_volatility <= 0:
        return 1.0
    return float(min(target / realised_volatility, max_leverage))


def drawdown_scalar(
    equity: float,
    peak_equity: float,
    *,
    start_pct: float = DERISK_START_PCT,
    floor_pct: float = DERISK_FLOOR_PCT,
    floor_scalar: float = DERISK_FLOOR_SCALAR,
) -> float:
    """Cut exposure as equity falls from its high-water mark.

    Linear between ``start_pct`` and ``floor_pct``, flat outside. Deliberately
    gradual rather than a cliff: a single threshold means the desk is at full
    size right up until it is at none, with no intermediate state in which a
    human might intervene.

    It never returns zero. Stopping entirely is a halt, and a halt belongs in
    :mod:`axiom.desk.guards` where it is explicit, recorded, and survives a
    restart — not buried in a sizing curve where nobody can see it happen.

    **The default bands overlap the guard, deliberately.** De-risking runs from
    5% to 15% while :func:`~axiom.desk.guards.check_drawdown` halts at 10%, so
    on a live desk only the 5–10% stretch is ever exercised and the rest is
    headroom for a caller that raises the halt threshold. The overlap is the
    safe direction: a de-risk band that stopped short of the halt would leave a
    window in which the desk was at full size with nothing between it and a
    stop.
    """
    if floor_pct <= start_pct:
        raise ValueError("floor_pct must exceed start_pct")
    if not 0.0 < floor_scalar <= 1.0:
        raise ValueError("floor_scalar must be in (0, 1]")
    if peak_equity <= 0:
        return floor_scalar

    drawdown = max(0.0, (peak_equity - equity) / peak_equity * 100.0)
    if drawdown <= start_pct:
        return 1.0
    if drawdown >= floor_pct:
        return floor_scalar
    progress = (drawdown - start_pct) / (floor_pct - start_pct)
    return float(1.0 - progress * (1.0 - floor_scalar))


def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Mean loss in the worst ``1 - confidence`` tail, as a positive number.

    Value at risk answers "how bad is the 5th percentile day", which says
    nothing about the tail beyond it — and the tail beyond it is what ends
    funds. Expected shortfall is the average of that tail, so a fatter tail
    produces a larger number where VaR would not move at all.

    Computed by averaging the worst ``ceil((1 - confidence) · n)`` observations
    directly, **not** by thresholding on a quantile. Thresholding is the
    obvious implementation and it is wrong on exactly the samples that matter:
    when the tail is small relative to the sample, the interpolated quantile
    lands in positive territory, ``values <= cutoff`` selects almost the whole
    distribution, and a portfolio with three catastrophic days reports a
    shortfall of zero.

    Returns 0.0 for an empty sample rather than raising, because a desk with no
    history has no measured tail — but callers must not read that as "no risk".
    Also returns 0.0 when even the worst observations are gains, which is the
    literal truth: there is no shortfall in that tail.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    values = np.asarray(returns, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    # Rounded before the ceiling: `(1.0 - 0.95) * 100` is 5.000000000000004 in
    # binary floating point, and an unguarded ceil turns that into six
    # observations. One extra element in a five-element tail moves the answer
    # by roughly 20%, which is not an acceptable error in a risk number.
    fraction = round((1.0 - confidence) * values.size, 9)
    count = max(1, int(np.ceil(fraction)))
    tail = np.sort(values)[:count]
    return float(max(-tail.mean(), 0.0))


@dataclass(frozen=True, slots=True)
class PortfolioRiskReport:
    """What the book looks like as one object rather than a list of trades."""

    volatility: float
    #: Cap the volatility was measured against.
    limit: float
    diversification: float
    effective_bets: float
    n_positions: int
    contributions: dict[str, float] = field(default_factory=dict)
    expected_shortfall: float = 0.0

    @property
    def utilisation(self) -> float:
        """Fraction of the risk budget in use. Above 1.0 is a breach."""
        return self.volatility / self.limit if self.limit > 0 else 0.0

    @property
    def is_breached(self) -> bool:
        return self.utilisation > 1.0

    @property
    def concentration(self) -> float:
        """Largest single risk contribution. 1.0 is a one-position book."""
        return max(self.contributions.values(), default=0.0)

    def render(self) -> str:
        lines = [
            f"Portfolio volatility : {self.volatility:.2%} annualised "
            f"(limit {self.limit:.2%}, {self.utilisation:.0%} used)"
            + ("  [BREACH]" if self.is_breached else ""),
            f"Positions            : {self.n_positions} holding "
            f"{self.effective_bets:.2f} effective bets "
            f"(diversification {self.diversification:.2f}x)",
        ]
        if self.expected_shortfall > 0:
            lines.append(f"Expected shortfall   : {self.expected_shortfall:.2%} (95%)")
        if self.contributions:
            lines.append("Risk contribution:")
            for symbol, share in sorted(
                self.contributions.items(), key=lambda kv: -abs(kv[1])
            ):
                lines.append(f"  {symbol:<12} {share:+.1%}")
        if self.n_positions > 1 and self.effective_bets < 1.5:
            lines.append(
                "  ! this book is one bet wearing several names — the position "
                "count is not measuring diversification"
            )
        return "\n".join(lines)


class PortfolioRiskManager:
    """Sizes a new position against the risk of the book it is joining.

    Composes with :class:`~axiom.risk.manager.RiskManager` rather than
    replacing it: that one decides whether a trade is permitted and what a
    single stop-out may cost, this one decides how much of it the portfolio can
    carry given what is already on.

    Args:
        volatility_limit: annualised portfolio volatility cap.
        volatility_target: target used by the vol-targeting scalar. Defaults to
            the limit, so a desk at target is at 100% utilisation.
        max_risk_contribution: largest share of portfolio risk one position may
            hold. Catches the case a volatility cap alone misses — a book that
            is inside its total limit but is really a single bet.
    """

    def __init__(
        self,
        *,
        volatility_limit: float = DEFAULT_VOL_TARGET,
        volatility_target: float | None = None,
        max_risk_contribution: float = 0.60,
        max_leverage: float = MAX_LEVERAGE,
    ) -> None:
        if volatility_limit <= 0:
            raise ValueError("volatility_limit must be positive")
        if not 0.0 < max_risk_contribution <= 1.0:
            raise ValueError("max_risk_contribution must be in (0, 1]")
        self.volatility_limit = volatility_limit
        self.volatility_target = volatility_target or volatility_limit
        self.max_risk_contribution = max_risk_contribution
        self.max_leverage = max_leverage

    def assess(
        self,
        weights: dict[str, float],
        panel: Panel,
        index: int,
        *,
        lookback: int = 120,
        returns: np.ndarray | None = None,
    ) -> PortfolioRiskReport:
        """Measure the current book.

        Args:
            weights: signed portfolio fractions per symbol. Symbols absent from
                the panel are ignored — they cannot be risk-modelled, which is
                itself worth knowing.
            panel: the aligned universe.
            index: bar being scored; history is exclusive of it.
            returns: optional realised portfolio returns for expected shortfall.
        """
        covariance = covariance_matrix(panel, index, lookback=lookback)
        vector = np.array([weights.get(symbol, 0.0) for symbol in panel.symbols])
        contributions = marginal_risk_contributions(vector, covariance)

        return PortfolioRiskReport(
            volatility=portfolio_volatility(vector, covariance),
            limit=self.volatility_limit,
            diversification=diversification_ratio(vector, covariance),
            effective_bets=effective_bets(vector, covariance),
            n_positions=int(np.count_nonzero(vector)),
            contributions={
                symbol: float(share)
                for symbol, share in zip(panel.symbols, contributions, strict=True)
                if abs(share) > 1e-9
            },
            expected_shortfall=(
                expected_shortfall(returns) if returns is not None else 0.0
            ),
        )

    def scale_for(
        self,
        symbol: str,
        proposed_weight: float,
        current_weights: dict[str, float],
        panel: Panel,
        index: int,
        *,
        equity: float,
        peak_equity: float,
        lookback: int = 120,
    ) -> tuple[float, str]:
        """Return a multiplier in [0, 1] for a proposed position, and why.

        The multiplier never exceeds 1: this layer only ever *reduces* a size
        the per-trade risk manager already approved. Allowing it to increase
        one would let a portfolio model override a hard per-trade budget, and
        the per-trade budget is the guarantee that a single bad fill cannot
        exceed a known loss.

        Volatility targeting can legitimately argue for more size in a quiet
        regime; that argument is capped here at 1.0 and belongs in the
        per-trade budget if it is wanted, where it is visible.
        """
        reasons: list[str] = []

        drawdown = drawdown_scalar(equity, peak_equity)
        if drawdown < 1.0:
            reasons.append(f"drawdown de-risk x{drawdown:.2f}")

        try:
            covariance = covariance_matrix(panel, index, lookback=lookback)
        except InsufficientHistoryError as exc:
            # No estimate means no correlation adjustment, and saying so is
            # better than silently applying 1.0 as though the book were known
            # to be diversified.
            return drawdown, "; ".join([*reasons, f"no covariance estimate ({exc})"])

        symbols = panel.symbols
        if symbol not in symbols:
            return drawdown, "; ".join(
                [*reasons, f"{symbol} is not in the risk panel — not modelled"]
            )

        current = np.array([current_weights.get(s, 0.0) for s in symbols])
        proposed = current.copy()
        proposed[symbols.index(symbol)] += proposed_weight

        realised = portfolio_volatility(proposed, covariance)
        vol_scalar = volatility_target_scalar(
            realised, target=self.volatility_target, max_leverage=self.max_leverage
        )
        if vol_scalar < 1.0:
            reasons.append(
                f"volatility target x{vol_scalar:.2f} "
                f"(book at {realised:.1%} vs {self.volatility_target:.1%})"
            )

        # Correlation cap: scale so the *resulting* portfolio volatility sits
        # inside the limit. Solving exactly would need a quadratic; scaling the
        # whole book linearly is conservative and, unlike an exact solution,
        # cannot be defeated by an estimation error in one covariance entry.
        correlation_scalar = 1.0
        if realised > self.volatility_limit > 0:
            correlation_scalar = self.volatility_limit / realised
            reasons.append(
                f"volatility limit x{correlation_scalar:.2f} "
                f"(would be {realised:.1%} vs {self.volatility_limit:.1%} cap)"
            )

        contribution_scalar = 1.0
        contributions = marginal_risk_contributions(proposed, covariance)
        share = abs(float(contributions[symbols.index(symbol)]))
        if share > self.max_risk_contribution:
            contribution_scalar = self.max_risk_contribution / share
            reasons.append(
                f"concentration x{contribution_scalar:.2f} "
                f"({share:.0%} of portfolio risk vs "
                f"{self.max_risk_contribution:.0%} cap)"
            )

        scalar = min(
            1.0, drawdown * min(vol_scalar, correlation_scalar, contribution_scalar)
        )
        if not reasons:
            reasons.append("no portfolio constraint binding")
        return float(max(scalar, 0.0)), "; ".join(reasons)
