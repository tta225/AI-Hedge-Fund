"""Backtest a cross-sectional book, honestly.

:class:`~axiom.research.lab.StrategyLab` evaluates one instrument at a time,
because :class:`~axiom.backtest.engine.Backtester` takes a single series. That
shape cannot express the strategies the published literature actually
replicates: momentum ranks names *against each other*, statistical arbitrage is
defined only relative to a factor model fitted across the cross-section, and a
long-short book's return is a property of the whole set of weights rather than
of any one position. Every agent in :mod:`axiom.alpha` had that shape and no
way to be measured.

This module closes that gap. It turns a roster of
:class:`~axiom.alpha.base.AlphaAgent` into portfolio weights, holds them until
the next rebalance, and marks the book bar by bar with costs charged on
turnover.

The four things that make it a measurement rather than a demonstration:

**Weights are formed from information strictly before the bar they are held
into.** The agent scores bar ``t`` using history up to ``t-1`` — enforced by
:meth:`~axiom.alpha.base.AlphaAgent.generate_checked` — and the resulting
weights earn the return from ``t`` to ``t+1``. Off by one bar in the permissive
direction and a momentum book earns the return it was ranking on, which looks
spectacular and is not real.

**Costs are charged on turnover, not on holdings.** A cross-sectional book
rebalances its whole notional; a strategy that reranks daily can turn over
200% a month, and at that rate cost is not a haircut on the result, it *is* the
result. Charging per-position or per-bar rather than per-unit-traded is how a
high-turnover backtest survives contact with a cost model it should not have.

**The book is dollar-neutral and gross-normalised by default.** Long-short with
matched gross exposure is what the anomaly literature measures. A long-only
version of the same ranking is mostly a bet on the market, and reporting its
Sharpe as evidence for the ranking confuses beta with alpha.

**Nothing is annualised by assumption.** The factor is inferred from the
panel's own bar spacing, the same way :mod:`axiom.portfolio.risk` does it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from axiom.alpha.base import AlphaAgent
from axiom.alpha.ensemble import AlphaEnsemble
from axiom.alpha.panel import Panel
from axiom.core.provenance import Provenance
from axiom.execution.costs import (
    CostModel,
    FlatBpsCost,
    SpreadImpactCost,
    average_dollar_volume,
    trailing_volatility,
)
from axiom.portfolio.risk import periods_per_year
from axiom.research.reference import PlausibilityCheck, assess_plausibility
from axiom.research.statistics import (
    PBOResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)
from axiom.research.turnover import (
    HysteresisBands,
    hysteresis_weights,
    partial_rebalance,
    suppress_small_trades,
)

#: Round-trip cost per unit of notional traded. 10 bp is deliberately
#: pessimistic for liquid US large caps, where commission is near zero and the
#: spread is a basis point or two — but a cross-sectional book pays the spread
#: on both legs, moves size, and this is a research bar rather than an
#: execution estimate. A result that only survives at 2 bp is not a result.
DEFAULT_COST_BPS = 10.0
#: Minimum names that must score before a bar is traded at all.
MIN_BREADTH = 5
#: Deflated-Sharpe probability above which a result counts as evidence.
SKILL_THRESHOLD = 0.95


class InsufficientBreadthError(ValueError):
    """Too few instruments scored to form a cross-sectional book."""


def rank_to_weights(
    scores: dict[str, float],
    symbols: Sequence[str],
    *,
    long_short: bool = True,
    top_fraction: float = 0.2,
    gross: float = 1.0,
) -> np.ndarray:
    """Turn scores into portfolio weights over ``symbols``.

    Args:
        scores: symbol → conviction. Missing symbols get no weight, which is
            the correct handling of "the agent had no view" — not zero
            conviction, but no position.
        symbols: the panel's column order. Weights come back in this order.
        long_short: buy the top fraction and sell the bottom. False is
            long-only, which is mostly a market bet.
        top_fraction: portion of the scored universe taken on each side.
        gross: total absolute exposure. 1.0 means fully invested.

    Returns:
        Weights summing to ``gross`` in absolute value, and to zero net when
        ``long_short`` — equal-weighted within each leg rather than
        conviction-weighted, because conviction magnitudes are not comparable
        across agents and weighting by them silently lets the noisiest agent
        take the largest positions.
    """
    if not 0.0 < top_fraction <= 0.5:
        raise ValueError(f"top_fraction must be in (0, 0.5], got {top_fraction}")
    if gross <= 0:
        raise ValueError("gross must be positive")

    weights = np.zeros(len(symbols), dtype=float)
    scored = [(s, v) for s, v in scores.items() if s in symbols and np.isfinite(v)]
    if len(scored) < MIN_BREADTH:
        return weights

    ordered = sorted(scored, key=lambda kv: kv[1], reverse=True)
    count = max(1, int(len(ordered) * top_fraction))
    index_of = {symbol: i for i, symbol in enumerate(symbols)}

    if not long_short:
        for symbol, _ in ordered[:count]:
            weights[index_of[symbol]] = gross / count
        return weights

    longs = ordered[:count]
    shorts = ordered[-count:]
    # Each leg carries half the gross, so the book is dollar-neutral and its
    # total absolute exposure is `gross` regardless of how many names qualify.
    for symbol, _ in longs:
        weights[index_of[symbol]] += gross / (2 * count)
    for symbol, _ in shorts:
        weights[index_of[symbol]] -= gross / (2 * count)
    return weights


@dataclass(slots=True)
class PanelBacktestResult:
    """What a cross-sectional book did."""

    returns: np.ndarray
    turnover: np.ndarray
    periods_per_year: float
    n_rebalances: int = 0
    #: Bars skipped because too few names scored.
    skipped: int = 0
    error: str = ""

    @property
    def is_viable(self) -> bool:
        return not self.error and self.returns.size > 20 and self.n_rebalances >= 5

    @property
    def sharpe(self) -> float:
        if self.returns.size < 2:
            return float("nan")
        deviation = float(self.returns.std(ddof=1))
        if deviation <= 0:
            return float("nan")
        return float(self.returns.mean() / deviation * np.sqrt(self.periods_per_year))

    @property
    def total_return_pct(self) -> float:
        return float((np.prod(1.0 + self.returns) - 1.0) * 100.0)

    @property
    def max_drawdown_pct(self) -> float:
        if self.returns.size == 0:
            return 0.0
        curve = np.cumprod(1.0 + self.returns)
        peak = np.maximum.accumulate(curve)
        return float(np.max((peak - curve) / peak) * 100.0)

    @property
    def annual_turnover(self) -> float:
        """Fraction of the book traded per year. The cost driver."""
        if self.turnover.size == 0:
            return 0.0
        return float(self.turnover.mean() * self.periods_per_year)


def backtest_panel(
    agents: Sequence[AlphaAgent] | AlphaEnsemble,
    panel: Panel,
    *,
    start: int,
    end: int | None = None,
    rebalance_every: int = 5,
    cost_bps: float = DEFAULT_COST_BPS,
    long_short: bool = True,
    top_fraction: float = 0.2,
    gross: float = 1.0,
    phase: int = 0,
    cost_model: CostModel | None = None,
    aum: float = 10_000_000.0,
    bands: HysteresisBands | None = None,
    rebalance_fraction: float = 1.0,
    min_trade: float = 0.0,
) -> PanelBacktestResult:
    """Run a cross-sectional book over ``panel`` from ``start``.

    Args:
        agents: a roster, or a prebuilt ensemble.
        panel: the aligned universe.
        start: first bar to trade. Everything before it is warm-up.
        end: last bar, exclusive. Defaults to the end of the panel.
        rebalance_every: bars between reranks. Weights are held in between,
            which is what makes turnover — and therefore cost — meaningful.
        cost_bps: round-trip cost per unit of notional traded.
        long_short: dollar-neutral by default.
        top_fraction: portion of the scored universe per leg.
        gross: total absolute exposure.
        phase: which bar of the rebalance cycle to trade on. Explicit rather
            than implied by ``start``, so a walk-forward fold cannot silently
            change the trading days, and so phase sensitivity can be measured
            deliberately — a strategy whose result depends on which day of the
            month it rebalances has found a sequence of coin flips.
    """
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1")
    if cost_bps < 0:
        raise ValueError("cost_bps must not be negative")

    ensemble = (
        agents if isinstance(agents, AlphaEnsemble) else AlphaEnsemble(list(agents))
    )
    model: CostModel = cost_model if cost_model is not None else FlatBpsCost(cost_bps)
    stop = len(panel) - 1 if end is None else min(end, len(panel) - 1)
    if start < 1 or start >= stop:
        raise ValueError(f"start {start} and end {stop} do not enclose any bars")

    factor = periods_per_year(panel)
    weights = np.zeros(panel.n_symbols)
    returns: list[float] = []
    turnovers: list[float] = []
    rebalances = 0
    skipped = 0

    for i in range(start, stop):
        # Phase is anchored to the panel's own bar index, not to this window's
        # start. Anchoring to `start` makes each walk-forward fold rebalance on
        # different days than a continuous run over the same bars, and with a
        # monthly rebalance over six years that is only ~58 decisions — few
        # enough that the phase choice moved a measured Sharpe from 0.07 to
        # 0.15. A fold boundary must not change which days the book trades.
        if (i - phase) % rebalance_every == 0:
            try:
                views = ensemble.blend(ensemble.generate(panel, i))
            except Exception as exc:
                return PanelBacktestResult(
                    returns=np.array([]), turnover=np.array([]),
                    periods_per_year=factor,
                    error=f"{type(exc).__name__} at bar {i}: {exc}",
                )
            scores = {v.symbol: v.conviction for v in views}
            if bands is None:
                target = rank_to_weights(
                    scores,
                    panel.symbols,
                    long_short=long_short,
                    top_fraction=top_fraction,
                    gross=gross,
                )
            else:
                target = hysteresis_weights(
                    scores,
                    panel.symbols,
                    weights,
                    bands=bands,
                    gross=gross,
                    long_short=long_short,
                )
            if rebalance_fraction < 1.0:
                target = partial_rebalance(weights, target, rebalance_fraction)
            if min_trade > 0.0:
                target = suppress_small_trades(weights, target, min_trade)

            if not np.any(target):
                skipped += 1
            else:
                rebalances += 1

            deltas = target - weights
            traded = float(np.abs(deltas).sum())
            # Priced before the weights move, so the charge reflects the trade
            # that was actually required rather than the book that resulted.
            cost = model.charge(
                deltas,
                aum=aum,
                adv_notional=average_dollar_volume(panel.closes, panel.volumes, i),
                volatility=trailing_volatility(panel.closes, i),
            )
            weights = target
        else:
            traded = 0.0
            cost = 0.0

        # The return from i to i+1, earned by weights formed from data before
        # i. One bar later and this would be the return the ranking was
        # computed on, which is the leak that makes momentum look magical.
        previous = panel.closes[i]
        nxt = panel.closes[i + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(previous > 0, nxt / previous - 1.0, 0.0)
        step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)

        gross_return = float(weights @ step)
        returns.append(gross_return - cost)
        turnovers.append(traded)

    return PanelBacktestResult(
        returns=np.array(returns),
        turnover=np.array(turnovers),
        periods_per_year=factor,
        n_rebalances=rebalances,
        skipped=skipped,
    )


def capacity_curve(
    agents: Sequence[AlphaAgent] | AlphaEnsemble,
    panel: Panel,
    *,
    aums: Sequence[float],
    start: int,
    cost_model: CostModel | None = None,
    **kwargs: Any,
) -> dict[float, PanelBacktestResult]:
    """Run the same book at several capital levels.

    The question a flat cost model cannot ask. Impact scales with the square
    root of participation, so a strategy's Sharpe is a *decreasing function of
    the money in it* — and the interesting number is not the Sharpe at any one
    size but the size at which it crosses zero.

    Uses :class:`~axiom.execution.costs.SpreadImpactCost` by default, because a
    capacity curve computed with a size-independent cost model is a flat line
    and answers nothing.
    """
    model = cost_model if cost_model is not None else SpreadImpactCost()
    return {
        float(aum): backtest_panel(
            agents, panel, start=start, cost_model=model, aum=float(aum), **kwargs
        )
        for aum in aums
    }


@dataclass(slots=True)
class PanelCandidate:
    """One cross-sectional configuration to evaluate."""

    name: str
    factory: Callable[[], Sequence[AlphaAgent] | AlphaEnsemble]
    params: dict[str, object] = field(default_factory=dict)
    #: Per-candidate overrides of the lab's execution assumptions. Rebalance
    #: frequency belongs to the *candidate* rather than the lab because it
    #: trades signal against turnover, and a search that cannot vary it is
    #: assuming away the main thing standing between a cross-sectional ranking
    #: and a tradable one.
    rebalance_every: int | None = None
    top_fraction: float | None = None

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        rendered = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}[{rendered}]"


@dataclass(slots=True)
class PanelCandidateResult:
    """Out-of-sample outcome for one cross-sectional candidate."""

    candidate: PanelCandidate
    result: PanelBacktestResult
    folds: int = 0

    @property
    def is_viable(self) -> bool:
        return self.result.is_viable and np.isfinite(self.result.sharpe)


@dataclass(slots=True)
class PanelSearchResult:
    """Everything a cross-sectional search produced, with the honesty statistics."""

    results: list[PanelCandidateResult]
    n_trials: int
    provenance: Provenance
    deflated: dict[str, float] = field(default_factory=dict)
    skill_threshold_sharpe: float = 0.0
    pbo: PBOResult | None = None
    plausibility: PlausibilityCheck | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def viable(self) -> list[PanelCandidateResult]:
        return [r for r in self.results if r.is_viable]

    @property
    def leaderboard(self) -> list[PanelCandidateResult]:
        # Sharpe breaks the tie. Every candidate scoring a deflated Sharpe of
        # exactly 0.00 is the common case in a failed search, and sorting on
        # that alone leaves the order at the mercy of insertion — which made an
        # earlier run report a -0.16 Sharpe candidate as "best" while a +0.48
        # sat lower in the table.
        return sorted(
            self.viable,
            key=lambda r: (
                self.deflated.get(r.candidate.label, 0.0),
                r.result.sharpe if np.isfinite(r.result.sharpe) else -np.inf,
            ),
            reverse=True,
        )

    @property
    def survivors(self) -> list[PanelCandidateResult]:
        return [
            r for r in self.leaderboard
            if self.deflated.get(r.candidate.label, 0.0) >= SKILL_THRESHOLD
        ]

    @property
    def verdict(self) -> str:
        if not self.provenance.is_evidential:
            return (
                f"NO CONCLUSION — the search ran on {self.provenance.kind.value} "
                "data."
            )
        if not self.viable:
            return f"NO RESULT — none of {self.n_trials} candidates traded enough."
        if not self.survivors:
            best = self.leaderboard[0]
            score = self.deflated.get(best.candidate.label, 0.0)
            return (
                f"NOTHING SURVIVED — best is {best.candidate.label} at a "
                f"deflated Sharpe probability of {score:.2f}, below "
                f"{SKILL_THRESHOLD:.2f}. Its raw Sharpe of "
                f"{best.result.sharpe:.2f} is not distinguishable from the "
                f"best of {self.n_trials} trials with no edge."
            )
        best = self.survivors[0]
        return (
            f"{len(self.survivors)} of {self.n_trials} cleared the threshold. "
            f"Best: {best.candidate.label} — deflated Sharpe probability "
            f"{self.deflated[best.candidate.label]:.2f}, raw Sharpe "
            f"{best.result.sharpe:.2f}, annual turnover "
            f"{best.result.annual_turnover:.1f}x."
        )

    def render(self, top: int = 15) -> str:
        lines: list[str] = []
        if not self.provenance.is_evidential:
            lines += [f"⚠️  {self.provenance.kind.value.upper()} DATA", ""]
        lines.append(f"Trials evaluated : {self.n_trials}")
        lines.append(f"Viable           : {len(self.viable)}")
        lines.append(
            f"Skill bar        : Sharpe > {self.skill_threshold_sharpe:.2f} "
            f"(expected best of {self.n_trials} trials with no edge)"
        )
        if self.pbo is not None:
            lines.append(f"Overfitting      : {self.pbo.verdict}")
        if self.plausibility is not None:
            lines.append(
                f"Vs literature    : {self.plausibility.percentile:.0%} percentile, "
                f"95% CI [{self.plausibility.ci_low:.2f}, "
                f"{self.plausibility.ci_high:.2f}]"
            )
        lines.append("")

        header = (
            f"{'candidate':<42}{'Sharpe':>8}{'DSR':>7}{'ret%':>9}"
            f"{'maxDD%':>8}{'turn/yr':>9}"
        )
        lines += [header, "-" * len(header)]
        for row in self.leaderboard[:top]:
            score = self.deflated.get(row.candidate.label, float("nan"))
            lines.append(
                f"{row.candidate.label[:41]:<42}"
                f"{row.result.sharpe:>8.2f}{score:>7.2f}"
                f"{row.result.total_return_pct:>9.2f}"
                f"{row.result.max_drawdown_pct:>8.2f}"
                f"{row.result.annual_turnover:>9.1f}"
            )
        lines += ["", self.verdict]
        if self.plausibility is not None and self.plausibility.is_suspicious:
            lines += ["", self.plausibility.verdict]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


class PanelLab:
    """Walk-forward search over cross-sectional candidates.

    Folds are anchored: each test window follows the data that preceded it, and
    only test-window returns are scored. Nothing is fitted on a fold and then
    reported on the same fold.
    """

    def __init__(
        self,
        *,
        n_folds: int = 4,
        warmup: int = 250,
        rebalance_every: int = 5,
        cost_bps: float = DEFAULT_COST_BPS,
        long_short: bool = True,
        top_fraction: float = 0.2,
        cost_model: CostModel | None = None,
        aum: float = 10_000_000.0,
        bands: HysteresisBands | None = None,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.n_folds = n_folds
        self.warmup = warmup
        self.rebalance_every = rebalance_every
        self.cost_bps = cost_bps
        self.long_short = long_short
        self.top_fraction = top_fraction
        # Defaults preserve the flat charge, so an existing caller's numbers do
        # not move silently when a better model becomes available.
        self.cost_model = cost_model
        self.aum = aum
        self.bands = bands

    def search(
        self, panel: Panel, candidates: Iterable[PanelCandidate]
    ) -> PanelSearchResult:
        candidates = list(candidates)
        if not candidates:
            raise ValueError("no candidates to search")

        usable = len(panel) - self.warmup - 1
        if usable < self.n_folds * 40:
            raise ValueError(
                f"{len(panel)} bars is too few for {self.n_folds} folds after a "
                f"{self.warmup}-bar warm-up"
            )
        size = usable // self.n_folds
        windows = [
            (self.warmup + fold * size, self.warmup + (fold + 1) * size)
            for fold in range(self.n_folds)
        ]

        results = [self._evaluate(panel, c, windows) for c in candidates]
        deflated = self._deflate(results, n_trials=len(candidates))
        threshold = self._threshold(results, n_trials=len(candidates))

        best = max(
            (r for r in results if r.is_viable),
            key=lambda r: deflated.get(r.candidate.label, 0.0),
            default=None,
        )
        plausibility = (
            assess_plausibility(best.result.sharpe, int(best.result.returns.size))
            if best is not None
            else None
        )

        return PanelSearchResult(
            results=results,
            n_trials=len(candidates),
            provenance=panel.provenance,
            deflated=deflated,
            skill_threshold_sharpe=threshold,
            pbo=self._pbo(results),
            plausibility=plausibility,
            notes=[
                f"Walk-forward: {self.n_folds} anchored folds, "
                f"{self.warmup}-bar warm-up, test windows only.",
                "Costs: "
                + (
                    self.cost_model.describe()  # type: ignore[attr-defined]
                    if self.cost_model is not None
                    else f"flat {self.cost_bps:.1f} bp per unit traded"
                )
                + f", rebalanced every {self.rebalance_every} bars"
                + (
                    f", hysteresis {self.bands.entry_fraction:.0%}/"
                    f"{self.bands.exit_fraction:.0%}"
                    if self.bands is not None
                    else ", no hysteresis"
                )
                + ".",
                "Book is dollar-neutral long-short, equal-weighted within each "
                "leg — conviction magnitudes are not comparable across agents.",
                f"All {len(candidates)} candidates counted as trials.",
            ],
        )

    def _evaluate(
        self,
        panel: Panel,
        candidate: PanelCandidate,
        windows: list[tuple[int, int]],
    ) -> PanelCandidateResult:
        pooled: list[np.ndarray] = []
        pooled_turnover: list[np.ndarray] = []
        rebalances = 0
        skipped = 0
        folds = 0

        for start, end in windows:
            outcome = backtest_panel(
                candidate.factory(),
                panel,
                start=start,
                end=end,
                rebalance_every=candidate.rebalance_every or self.rebalance_every,
                cost_bps=self.cost_bps,
                long_short=self.long_short,
                top_fraction=candidate.top_fraction or self.top_fraction,
                cost_model=self.cost_model,
                aum=self.aum,
                bands=self.bands,
            )
            if outcome.error:
                return PanelCandidateResult(
                    candidate=candidate,
                    result=PanelBacktestResult(
                        returns=np.array([]), turnover=np.array([]),
                        periods_per_year=outcome.periods_per_year,
                        error=outcome.error,
                    ),
                )
            pooled.append(outcome.returns)
            pooled_turnover.append(outcome.turnover)
            rebalances += outcome.n_rebalances
            skipped += outcome.skipped
            folds += 1

        return PanelCandidateResult(
            candidate=candidate,
            result=PanelBacktestResult(
                returns=np.concatenate(pooled) if pooled else np.array([]),
                turnover=(
                    np.concatenate(pooled_turnover) if pooled_turnover else np.array([])
                ),
                periods_per_year=periods_per_year(panel),
                n_rebalances=rebalances,
                skipped=skipped,
            ),
            folds=folds,
        )

    @staticmethod
    def _deflate(
        results: list[PanelCandidateResult], *, n_trials: int
    ) -> dict[str, float]:
        viable = [r for r in results if r.is_viable]
        if len(viable) < 2:
            return {r.candidate.label: float("nan") for r in results}
        variance = float(np.var([r.result.sharpe for r in viable], ddof=1))
        out: dict[str, float] = {}
        for row in results:
            if not row.is_viable or row.result.returns.size < 3:
                out[row.candidate.label] = 0.0
                continue
            out[row.candidate.label] = deflated_sharpe_ratio(
                row.result.sharpe,
                n_observations=int(row.result.returns.size),
                n_trials=n_trials,
                variance_of_sharpes=variance,
            )
        return out

    @staticmethod
    def _threshold(results: list[PanelCandidateResult], *, n_trials: int) -> float:
        viable = [r for r in results if r.is_viable]
        if len(viable) < 2:
            return 0.0
        variance = float(np.var([r.result.sharpe for r in viable], ddof=1))
        return expected_max_sharpe(n_trials, variance)

    @staticmethod
    def _pbo(results: list[PanelCandidateResult]) -> PBOResult | None:
        viable = [r for r in results if r.is_viable and r.result.returns.size > 50]
        if len(viable) < 2:
            return None
        length = min(r.result.returns.size for r in viable)
        matrix = np.column_stack([r.result.returns[:length] for r in viable])
        try:
            return probability_of_backtest_overfitting(matrix)
        except (ValueError, np.linalg.LinAlgError):
            return None
