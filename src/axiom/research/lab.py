"""Strategy search: sweep many candidates, then judge the winner honestly.

The naive version of this tool — run everything, sort by Sharpe, return the top
row — is an overfitting machine. :class:`StrategyLab` therefore does three
things a plain sweep does not:

1. **Walk-forward evaluation.** Every candidate is scored on data that follows
   the window used to select it, so the reported number is out-of-sample by
   construction rather than by promise.
2. **Trial accounting.** Every candidate evaluated is counted, and the count is
   fed into the deflated Sharpe. Testing more ideas makes the winner's bar
   higher, not lower.
3. **Selection diagnostics.** Probability of Backtest Overfitting measures
   whether picking the in-sample best finds strategies that generalise *at
   all* — a property of the procedure, independent of what it returned today.

The leaderboard therefore ranks on a **deflated** score, and
:attr:`SearchResult.verdict` will say plainly when nothing survived. A search
that returns "no candidate is distinguishable from noise" has done its job.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from axiom.backtest.engine import Backtester
from axiom.backtest.metrics import daily_returns
from axiom.core.config import ExecutionSettings, RiskSettings
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.research.reference import PlausibilityCheck, assess_plausibility
from axiom.research.statistics import (
    PBOResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    parameter_stability,
    probability_of_backtest_overfitting,
)
from axiom.strategy.base import Strategy

#: Deflated-Sharpe probability above which a result is treated as evidence.
SKILL_THRESHOLD = 0.95
#: Minimum trades before a candidate is considered at all.
MIN_TRADES = 20


@dataclass(slots=True)
class Candidate:
    """One strategy configuration to evaluate."""

    name: str
    factory: Callable[[], Strategy]
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        rendered = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}[{rendered}]"


@dataclass(slots=True)
class CandidateResult:
    """Out-of-sample outcome for one candidate."""

    candidate: Candidate
    trades: int
    sharpe: float
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    avg_r: float
    worst_loss_vs_budget_x: float
    #: Daily returns concatenated across walk-forward folds, for PBO and DSR.
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    folds: int = 0
    error: str = ""

    @property
    def is_viable(self) -> bool:
        return not self.error and self.trades >= MIN_TRADES and math.isfinite(self.sharpe)


@dataclass(slots=True)
class SearchResult:
    """Everything the search produced, plus the honesty statistics."""

    results: list[CandidateResult]
    n_trials: int
    provenance: Provenance
    deflated: dict[str, float] = field(default_factory=dict)
    #: Sharpe a candidate must beat to be distinguishable from the best of
    #: `n_trials` coin flips, given the observed dispersion across trials.
    skill_threshold_sharpe: float = 0.0
    pbo: PBOResult | None = None
    stability: dict[str, float] = field(default_factory=dict)
    #: Outside view on the leaderboard leader: where its Sharpe sits against
    #: the published literature, and how wide its own error bar is. The
    #: deflated Sharpe and PBO are both internal to the trial set and cannot
    #: notice that a number is simply implausible.
    plausibility: PlausibilityCheck | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def viable(self) -> list[CandidateResult]:
        return [r for r in self.results if r.is_viable]

    @property
    def leaderboard(self) -> list[CandidateResult]:
        """Viable candidates ranked by deflated Sharpe, best first."""
        return sorted(
            self.viable,
            key=lambda r: self.deflated.get(r.candidate.label, 0.0),
            reverse=True,
        )

    @property
    def survivors(self) -> list[CandidateResult]:
        """Candidates whose deflated Sharpe clears the skill threshold."""
        return [
            r for r in self.leaderboard
            if self.deflated.get(r.candidate.label, 0.0) >= SKILL_THRESHOLD
        ]

    @property
    def verdict(self) -> str:
        if not self.provenance.is_evidential:
            return (
                f"NO CONCLUSION — the search ran on {self.provenance.kind.value} "
                "data. These rankings describe generated bars, not markets."
            )
        if not self.viable:
            return (
                f"NO RESULT — none of {self.n_trials} candidates produced "
                f"{MIN_TRADES}+ trades. Nothing to rank."
            )
        if not self.survivors:
            best = self.leaderboard[0]
            score = self.deflated.get(best.candidate.label, 0.0)
            return (
                f"NOTHING SURVIVED — best candidate {best.candidate.label} has a "
                f"deflated Sharpe probability of {score:.2f}, below the "
                f"{SKILL_THRESHOLD:.2f} threshold. After correcting for "
                f"{self.n_trials} trials, its raw Sharpe of {best.sharpe:.2f} is "
                "not distinguishable from the best of that many coin flips. "
                "This is the expected outcome of most searches and is not a bug."
            )
        best = self.survivors[0]
        return (
            f"{len(self.survivors)} of {self.n_trials} candidates cleared the "
            f"threshold. Best: {best.candidate.label} — deflated Sharpe "
            f"probability {self.deflated[best.candidate.label]:.2f}, "
            f"raw Sharpe {best.sharpe:.2f} over {best.trades} trades."
        )

    def render(self, top: int = 12) -> str:
        lines: list[str] = []
        if not self.provenance.is_evidential:
            lines += [
                f"⚠️  {self.provenance.kind.value.upper()} DATA — rankings below "
                "are a correctness check, not research.",
                "",
            ]
        lines.append(f"Trials evaluated : {self.n_trials}")
        lines.append(f"Viable (≥{MIN_TRADES} trades): {len(self.viable)}")
        lines.append(
            f"Skill bar        : Sharpe > {self.skill_threshold_sharpe:.2f} "
            f"(expected best of {self.n_trials} trials with no edge)"
        )
        if self.pbo is not None:
            lines.append(f"Overfitting      : {self.pbo.verdict}")
        if self.plausibility is not None:
            lines.append(
                f"Vs literature    : {self.plausibility.percentile:.0%} percentile "
                f"of published anomalies, 95% CI "
                f"[{self.plausibility.ci_low:.2f}, {self.plausibility.ci_high:.2f}]"
            )
        lines.append("")

        header = (
            f"{'candidate':<44}{'trades':>7}{'Sharpe':>8}{'DSR':>7}"
            f"{'ret%':>8}{'maxDD%':>8}{'avgR':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for result in self.leaderboard[:top]:
            score = self.deflated.get(result.candidate.label, float("nan"))
            lines.append(
                f"{result.candidate.label[:43]:<44}"
                f"{result.trades:>7}"
                f"{result.sharpe:>8.2f}"
                f"{score:>7.2f}"
                f"{result.total_return_pct:>8.2f}"
                f"{result.max_drawdown_pct:>8.2f}"
                f"{result.avg_r:>7.2f}"
            )
        if self.stability:
            lines.append("")
            lines.append("Parameter stability (1 = smooth surface, 0 = spike field):")
            for family, value in sorted(self.stability.items()):
                lines.append(f"  {family:<40} {value:.2f}")
        lines += ["", self.verdict]
        if self.plausibility is not None and self.plausibility.is_suspicious:
            lines += ["", self.plausibility.verdict]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """One train/test division. Only ``test`` contributes to the score."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def test_length(self) -> int:
        return self.test_end - self.test_start


def walk_forward_splits(
    n_bars: int, *, n_folds: int = 4, train_fraction: float = 0.5, warmup: int = 100
) -> list[WalkForwardSplit]:
    """Anchored walk-forward: an expanding train window, a rolling test window.

    Test windows never overlap and always follow their training window, so
    concatenating the test results gives a genuinely out-of-sample track.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    usable = n_bars - warmup
    if usable < n_folds * 50:
        raise ValueError(
            f"{n_bars} bars is too few for {n_folds} folds after a "
            f"{warmup}-bar warmup"
        )

    initial_train = int(usable * train_fraction)
    test_size = (usable - initial_train) // n_folds
    if test_size < 20:
        raise ValueError("test windows would be shorter than 20 bars")

    splits: list[WalkForwardSplit] = []
    for fold in range(n_folds):
        train_end = warmup + initial_train + fold * test_size
        splits.append(
            WalkForwardSplit(
                train_start=0,
                train_end=train_end,
                test_start=train_end,
                test_end=min(train_end + test_size, n_bars),
            )
        )
    return splits


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """Cartesian product of parameter axes.

    Every combination is a hypothesis test. ``grid(a=[1,2,3], b=[1,2,3])`` is
    nine trials, and the deflated Sharpe will hold the winner to a nine-trial
    standard.
    """
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


class StrategyLab:
    """Evaluates many candidates walk-forward and ranks them honestly."""

    def __init__(
        self,
        *,
        risk_settings: RiskSettings | None = None,
        execution_settings: ExecutionSettings | None = None,
        n_folds: int = 4,
        warmup: int = 100,
        entry_tolerance_atr: float = 0.35,
        stop_gap_atr: float = 0.75,
    ) -> None:
        self.risk_settings = risk_settings or RiskSettings()
        self.execution_settings = execution_settings or ExecutionSettings()
        self.n_folds = n_folds
        self.warmup = warmup
        self.entry_tolerance_atr = entry_tolerance_atr
        self.stop_gap_atr = stop_gap_atr

    def search(
        self, series: OHLCVSeries, candidates: Iterable[Candidate]
    ) -> SearchResult:
        """Evaluate every candidate walk-forward and rank the survivors."""
        candidates = list(candidates)
        if not candidates:
            raise ValueError("no candidates to search")

        splits = walk_forward_splits(
            len(series), n_folds=self.n_folds, warmup=self.warmup
        )
        results = [self._evaluate(series, c, splits) for c in candidates]

        deflated = self._deflate(results, n_trials=len(candidates))
        threshold = self._threshold(results, n_trials=len(candidates))
        pbo = self._pbo(results)
        stability = self._stability(results, deflated)
        plausibility = self._plausibility(results, deflated)

        notes = [
            f"Walk-forward: {len(splits)} folds, expanding train window, "
            "non-overlapping out-of-sample test windows.",
            "The skill bar scales with the DISPERSION of results across trials. "
            "A candidate set containing structurally opposite strategies (momentum "
            "and mean reversion, say) has high dispersion by construction, which "
            "raises the bar and can drive every deflated score to zero. That is "
            "the formula working, not failing — but compare like with like if you "
            "want the bar to be informative.",
            "Reported metrics are from test windows only.",
            f"All {len(candidates)} candidates counted as trials in the "
            "deflated Sharpe — including those that failed.",
        ]
        return SearchResult(
            results=results,
            n_trials=len(candidates),
            provenance=series.provenance,
            deflated=deflated,
            skill_threshold_sharpe=threshold,
            pbo=pbo,
            stability=stability,
            plausibility=plausibility,
            notes=notes,
        )

    @staticmethod
    def _plausibility(
        results: list[CandidateResult], deflated: dict[str, float]
    ) -> PlausibilityCheck | None:
        """Outside view on whichever candidate the leaderboard would promote.

        Assessed on the leader rather than on every row because this is a
        prior about *the number you are about to believe*. Applied across a
        whole grid it would only restate the dispersion the deflated Sharpe
        already charges for.
        """
        viable = [r for r in results if r.is_viable and r.returns.size >= 3]
        if not viable:
            return None
        best = max(viable, key=lambda r: deflated.get(r.candidate.label, 0.0))
        return assess_plausibility(best.sharpe, int(best.returns.size))

    def _evaluate(
        self, series: OHLCVSeries, candidate: Candidate, splits: list[WalkForwardSplit]
    ) -> CandidateResult:
        """Run one candidate across all test windows and pool the results."""
        trades = 0
        returns: list[np.ndarray] = []
        total_return = 0.0
        worst_dd = 0.0
        worst_overrun = 0.0
        r_values: list[float] = []
        profit_factors: list[float] = []
        folds = 0

        for split in splits:
            window = series.islice(0, split.test_end)
            try:
                backtester = Backtester(
                    candidate.factory(),
                    risk_settings=self.risk_settings,
                    execution_settings=self.execution_settings,
                    entry_tolerance_atr=self.entry_tolerance_atr,
                    stop_gap_atr=self.stop_gap_atr,
                )
                # Trading starts where this fold's test window starts, so the
                # earlier bars inform the analytics without being traded.
                outcome = backtester.run(window, warmup=split.test_start)
            except (ValueError, RuntimeError) as exc:
                return CandidateResult(
                    candidate=candidate, trades=0, sharpe=float("nan"),
                    total_return_pct=0.0, max_drawdown_pct=0.0,
                    profit_factor=float("nan"), avg_r=float("nan"),
                    worst_loss_vs_budget_x=float("nan"), error=str(exc),
                )

            metrics = outcome.report.metrics
            trades += int(metrics.get("trades", 0))
            total_return += float(metrics.get("total_return_pct", 0.0))
            worst_dd = max(worst_dd, float(metrics.get("max_drawdown_pct", 0.0)))
            overrun = metrics.get("worst_loss_vs_budget_x")
            if overrun is not None and math.isfinite(overrun):
                worst_overrun = max(worst_overrun, float(overrun))
            if math.isfinite(metrics.get("avg_r", float("nan"))):
                r_values.append(float(metrics["avg_r"]))
            if math.isfinite(metrics.get("profit_factor", float("nan"))):
                profit_factors.append(float(metrics["profit_factor"]))

            equity = outcome.portfolio.equity_series()
            test_equity = equity.iloc[split.test_start : split.test_end]
            fold_returns, _ = daily_returns(test_equity)
            if len(fold_returns) > 1:
                returns.append(fold_returns.to_numpy())
            folds += 1

        pooled = np.concatenate(returns) if returns else np.array([])
        sharpe = _annualised_sharpe(pooled)
        return CandidateResult(
            candidate=candidate,
            trades=trades,
            sharpe=sharpe,
            total_return_pct=total_return,
            max_drawdown_pct=worst_dd,
            profit_factor=float(np.mean(profit_factors)) if profit_factors else float("nan"),
            avg_r=float(np.mean(r_values)) if r_values else float("nan"),
            worst_loss_vs_budget_x=worst_overrun,
            returns=pooled,
            folds=folds,
        )

    @staticmethod
    def _deflate(results: list[CandidateResult], *, n_trials: int) -> dict[str, float]:
        viable = [r for r in results if r.is_viable]
        if len(viable) < 2:
            return {r.candidate.label: float("nan") for r in results}

        sharpes = np.array([r.sharpe for r in viable])
        variance = float(np.var(sharpes, ddof=1))

        out: dict[str, float] = {}
        for result in results:
            if not result.is_viable or result.returns.size < 3:
                out[result.candidate.label] = 0.0
                continue
            out[result.candidate.label] = deflated_sharpe_ratio(
                result.sharpe,
                n_observations=result.returns.size,
                n_trials=n_trials,
                variance_of_sharpes=variance,
                skewness=float(_skew(result.returns)),
                kurtosis=float(_kurtosis(result.returns)),
            )
        return out

    @staticmethod
    def _threshold(results: list[CandidateResult], *, n_trials: int) -> float:
        viable = [r for r in results if r.is_viable]
        if len(viable) < 2:
            return 0.0
        sharpes = np.array([r.sharpe for r in viable])
        return expected_max_sharpe(n_trials, float(np.var(sharpes, ddof=1)))

    @staticmethod
    def _pbo(results: list[CandidateResult]) -> PBOResult | None:
        usable = [r for r in results if r.is_viable and r.returns.size > 0]
        if len(usable) < 2:
            return None
        length = min(r.returns.size for r in usable)
        if length < 16:
            return None
        matrix = np.column_stack([r.returns[-length:] for r in usable])
        return probability_of_backtest_overfitting(matrix)

    @staticmethod
    def _stability(
        results: list[CandidateResult], deflated: dict[str, float]
    ) -> dict[str, float]:
        """Stability of the score surface within each strategy family."""
        families: dict[str, dict[tuple[float, ...], float]] = {}
        for result in results:
            params = result.candidate.params
            numeric = tuple(
                float(v) for _, v in sorted(params.items())
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
            if not numeric:
                continue
            families.setdefault(result.candidate.name, {})[numeric] = (
                result.sharpe if math.isfinite(result.sharpe) else float("nan")
            )
        return {
            name: parameter_stability(scores)
            for name, scores in families.items()
            if len(scores) >= 3
        }


def _annualised_sharpe(daily: np.ndarray, periods_per_year: float = 252.0) -> float:
    if daily.size < 3:
        return float("nan")
    sigma = daily.std(ddof=1)
    if sigma == 0:
        return float("nan")
    return float(daily.mean() / sigma * math.sqrt(periods_per_year))


def _skew(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    centred = x - x.mean()
    sigma = centred.std(ddof=0)
    return float((centred**3).mean() / sigma**3) if sigma > 0 else 0.0


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 3.0
    centred = x - x.mean()
    sigma = centred.std(ddof=0)
    return float((centred**4).mean() / sigma**4) if sigma > 0 else 3.0
