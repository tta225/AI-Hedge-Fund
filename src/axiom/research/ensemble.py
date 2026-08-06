"""Rank live signals across the whole archive by out-of-sample evidence.

The question this answers
------------------------
"Given every strategy the platform knows, which trade right now has the best
likelihood of working?"

The wrong way to answer it, and why
-----------------------------------
Backtest all 49 strategies, pick the one with the highest return, trade its
signal. That is the single most reliable way to lose money in systematic
trading, and it fails for a reason that has nothing to do with implementation
quality: with 49 candidates, the best in-sample result is dominated by
**selection noise**. Run 49 coin-flipping strategies and the winner still looks
excellent. Its in-sample Sharpe is a measurement of how many things you tried,
not of what it will do next.

What this does instead
----------------------
1. **Walk-forward only.** Every strategy is scored on data it was not fitted or
   selected on. In-sample performance is never used for ranking.
2. **Deflated for multiplicity.** The score is penalised by the number of
   strategies tried and the dispersion of their results, following the deflated
   Sharpe construction already in `axiom.research.statistics`. A strategy has to
   beat what the *best of 49 random* strategies would have produced.
3. **Shrunk toward zero.** Scores are shrunk by sample size, so a strategy with
   six out-of-sample trades cannot outrank one with two hundred on the strength
   of a lucky streak. This is the single most important line in the module.
4. **Family-aware.** Ten trend rules agreeing is not ten pieces of evidence.
   Concordant signals within one family are collapsed; agreement *across*
   families counts for more.
5. **Refuses to rank when it cannot.** With too little out-of-sample history,
   `rank` returns an empty result with a stated reason rather than a
   confident-looking ordering of noise.

What a `RankedSignal` is and is not
-----------------------------------
`likelihood` is a **relative ordering score in [0, 1]**, not a probability. It
does not say "this trade wins 68% of the time". No honest construction from this
data could say that, and labelling it a probability would invite exactly the
misreading the rest of this platform is built to prevent. It says: *of the
signals live right now, ranked by out-of-sample evidence adjusted for how many
things were tried, this one ranks here.*

If every strategy's evidence is weak — which is the expected result on real
data, per `docs/REAL_DATA_FINDINGS.md` — every likelihood will be low and the
ranking will say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from axiom.backtest.engine import Backtester
from axiom.core.config import RiskSettings
from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import ICTConfig, ICTEngine
from axiom.portfolio.positions import Portfolio
from axiom.strategy.archive import ARCHIVE, Evidence, Family, StrategyEntry
from axiom.strategy.base import Signal, StrategyContext

__all__ = [
    "EnsembleRanking",
    "RankedSignal",
    "SignalEnsemble",
    "StrategyScore",
]

#: Below this many out-of-sample trades, a strategy's score is shrunk hard.
#: Chosen so a strategy needs roughly a season of activity before its record
#: counts for much. It is a judgement call, and it is deliberately harsh.
_SHRINKAGE_TRADES = 30.0

#: Minimum out-of-sample folds before any ranking is produced at all.
_MIN_FOLDS = 2


@dataclass(slots=True)
class StrategyScore:
    """One strategy's out-of-sample record."""

    key: str
    family: Family
    trades: int
    #: Mean out-of-sample R-multiple per trade. The core statistic.
    mean_r: float
    #: Standard deviation of out-of-sample R-multiples.
    std_r: float
    win_rate: float
    folds: int
    #: Signals generated out-of-sample, before risk sizing.
    signals: int = 0
    #: Why signals failed to become trades, if they did.
    rejections: dict[str, int] = field(default_factory=dict)
    #: `mean_r` shrunk toward zero by sample size — see module docstring.
    shrunk_r: float = 0.0
    #: `shrunk_r` after the multiple-testing penalty.
    deflated_r: float = 0.0
    #: Set when this strategy is excluded from ranking, with the reason.
    excluded: str = ""

    @property
    def is_usable(self) -> bool:
        return not self.excluded and self.trades > 0

    def describe(self) -> str:
        if self.excluded:
            return f"{self.key:<20} excluded — {self.excluded}"
        return (
            f"{self.key:<20} {self.trades:>4} trades  "
            f"raw {self.mean_r:+.3f}R  shrunk {self.shrunk_r:+.3f}R  "
            f"deflated {self.deflated_r:+.3f}R  win {self.win_rate:.0%}"
        )


@dataclass(slots=True)
class RankedSignal:
    """A live signal, with the evidence behind the strategy that produced it."""

    strategy_key: str
    family: Family
    signal: Signal
    score: StrategyScore
    #: Relative ordering score in [0, 1]. NOT a probability. See module docs.
    likelihood: float
    #: Other strategies signalling the same direction right now.
    concordant: tuple[str, ...] = ()
    #: Distinct families in agreement — worth more than a count of strategies.
    concordant_families: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def direction(self) -> Direction:
        return self.signal.direction

    def describe(self) -> str:
        lines = [
            f"{self.strategy_key} [{self.family.value}] — {self.signal.direction} "
            f"@ {self.signal.entry:,.2f}, stop {self.signal.stop:,.2f}",
            f"  likelihood score : {self.likelihood:.2f}  "
            f"(relative ordering, NOT a probability)",
            f"  evidence         : {self.score.trades} out-of-sample trades, "
            f"deflated {self.score.deflated_r:+.3f}R",
            f"  rationale        : {self.signal.rationale}",
        ]
        if self.concordant_families:
            lines.append(
                f"  agreement        : {len(self.concordant)} strategies across "
                f"{len(self.concordant_families)} families "
                f"({', '.join(self.concordant_families)})"
            )
        for note in self.notes:
            lines.append(f"  ⚠ {note}")
        return "\n".join(lines)


@dataclass(slots=True)
class EnsembleRanking:
    """The full result: ranked signals plus every score behind them."""

    signals: list[RankedSignal]
    scores: dict[str, StrategyScore]
    folds: int
    as_of: pd.Timestamp
    #: Populated when no ranking could honestly be produced.
    refusal: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> RankedSignal | None:
        return self.signals[0] if self.signals else None

    def render(self) -> str:
        if self.refusal:
            return f"No ranking produced.\n  {self.refusal}"
        if not self.signals:
            return (
                f"No strategy signalled at {self.as_of}.\n"
                f"  {len(self.scores)} strategies evaluated over {self.folds} "
                f"out-of-sample folds; none has a live setup."
            )
        blocks = [
            f"Ranked signals at {self.as_of} "
            f"({len(self.scores)} strategies, {self.folds} out-of-sample folds)",
            "",
        ]
        for rank, signal in enumerate(self.signals, start=1):
            blocks.append(f"{rank}. {signal.describe()}")
            blocks.append("")
        blocks += self.notes
        return "\n".join(blocks)

    def scoreboard(self) -> str:
        rows = sorted(
            self.scores.values(), key=lambda s: -s.deflated_r if s.is_usable else 1e9
        )
        return "\n".join(s.describe() for s in rows)


class SignalEnsemble:
    """Score the archive walk-forward, then rank whatever is live now.

    Args:
        entries: strategies to consider. Defaults to the whole archive minus
            controls — a calendar rule cannot generate a *tradeable* live signal
            worth ranking, and including the nulls inflates the dispersion the
            multiple-testing penalty is computed from.
        folds: walk-forward splits. More folds means more out-of-sample data and
            slower runs.
    """

    def __init__(
        self,
        entries: dict[str, StrategyEntry] | None = None,
        *,
        folds: int = 4,
        risk_settings: RiskSettings | None = None,
        ict_config: ICTConfig | None = None,
        warmup: int = 250,
        include_controls: bool = False,
    ) -> None:
        source = entries if entries is not None else ARCHIVE
        self.entries = {
            k: e
            for k, e in source.items()
            if include_controls or e.evidence is not Evidence.CONTROL
        }
        self.folds = folds
        self.risk_settings = risk_settings or RiskSettings(
            account_equity=250_000.0, max_gross_exposure_pct=2000.0
        )
        self.ict_config = ict_config or ICTConfig()
        self.warmup = warmup
        self.include_controls = include_controls

    # --- scoring ------------------------------------------------------------

    def _splits(self, length: int) -> list[tuple[int, int]]:
        """Expanding-window out-of-sample segments, oldest first.

        Expanding rather than rolling: each fold's test segment sits strictly
        after everything used before it, so no fold is ever scored on data an
        earlier fold could have seen.
        """
        usable = length - self.warmup
        if usable <= 0:
            return []
        size = usable // (self.folds + 1)
        if size < 50:
            return []
        return [
            (self.warmup + size * k, self.warmup + size * (k + 1))
            for k in range(1, self.folds + 1)
        ]

    def score(self, series: OHLCVSeries) -> dict[str, StrategyScore]:
        """Walk-forward score for every entry."""
        splits = self._splits(len(series))
        scores: dict[str, StrategyScore] = {}

        for key, entry in self.entries.items():
            if not splits:
                scores[key] = StrategyScore(
                    key=key, family=entry.family, trades=0, mean_r=float("nan"),
                    std_r=float("nan"), win_rate=float("nan"), folds=0,
                    excluded="not enough history for a walk-forward split",
                )
                continue

            r_multiples: list[float] = []
            completed = 0
            signal_count = 0
            rejections: dict[str, int] = {}
            for start, stop in splits:
                window = series.islice(0, stop)
                try:
                    result = Backtester(
                        entry.build(),
                        risk_settings=self.risk_settings,
                        ict_config=self.ict_config,
                    ).run(window, warmup=start)
                except Exception:
                    # A strategy that cannot run on this window is excluded, not
                    # scored as zero — zero would read as "tried and flat".
                    continue
                completed += 1
                signal_count += len(result.signals)
                for reason, count in result.risk_rejections.items():
                    rejections[reason] = rejections.get(reason, 0) + count
                for trade in result.trades:
                    risk = trade.risk_points * trade.quantity * trade.instrument.point_value
                    if risk > 0:
                        r_multiples.append(trade.pnl / risk)

            if completed < _MIN_FOLDS:
                scores[key] = StrategyScore(
                    key=key, family=entry.family, trades=0, mean_r=float("nan"),
                    std_r=float("nan"), win_rate=float("nan"), folds=completed,
                    signals=signal_count, rejections=rejections,
                    excluded=f"only {completed} usable folds",
                )
                continue

            arr = np.array(r_multiples)
            n = len(arr)
            if n == 0:
                # Signalled but never traded is a *different* failure from never
                # signalling, and conflating them hides the most common cause:
                # a risk budget too small to size one unit of this instrument.
                top_reason = (
                    max(rejections.items(), key=lambda kv: kv[1])[0]
                    if rejections else ""
                )
                why = (
                    f"{signal_count} signals, none sized "
                    f"({top_reason or 'no reason recorded'})"
                    if signal_count
                    else "no signals out-of-sample"
                )
                scores[key] = StrategyScore(
                    key=key, family=entry.family, trades=0, mean_r=float("nan"),
                    std_r=float("nan"), win_rate=float("nan"), folds=completed,
                    signals=signal_count, rejections=rejections, excluded=why,
                )
                continue
            mean_r = float(np.mean(arr)) if n else 0.0
            std_r = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
            win = float(np.mean(arr > 0)) if n else float("nan")

            # Shrink toward zero by sample size. With n = _SHRINKAGE_TRADES the
            # estimate is halved; with n = 3 it is all but erased. A strategy
            # must earn its score with trade count, not with one good run.
            shrunk = mean_r * (n / (n + _SHRINKAGE_TRADES)) if n else 0.0

            scores[key] = StrategyScore(
                key=key, family=entry.family, trades=n, mean_r=mean_r,
                std_r=std_r, win_rate=win, folds=completed, shrunk_r=shrunk,
                signals=signal_count, rejections=rejections,
            )

        self._deflate(scores)
        return scores

    @staticmethod
    def _deflate(scores: dict[str, StrategyScore]) -> None:
        """Penalise every score for the number of strategies tried.

        With `k` candidates, the maximum of `k` draws from a zero-mean
        distribution is roughly `sigma * sqrt(2 ln k)` above zero. Subtracting
        that expected best-of-noise is what stops the ranking from promoting the
        luckiest strategy in a field of nulls. It is the same construction as the
        deflated Sharpe ratio, applied to mean R.
        """
        usable = [s for s in scores.values() if s.is_usable]
        k = len(usable)
        if k < 2:
            for s in usable:
                s.deflated_r = s.shrunk_r
            return

        dispersion = float(np.std([s.shrunk_r for s in usable], ddof=1))
        expected_best_noise = dispersion * np.sqrt(2.0 * np.log(k)) if dispersion > 0 else 0.0
        for s in usable:
            s.deflated_r = s.shrunk_r - expected_best_noise

    # --- ranking ------------------------------------------------------------

    def rank(
        self,
        series: OHLCVSeries,
        *,
        scores: dict[str, StrategyScore] | None = None,
        top: int = 10,
    ) -> EnsembleRanking:
        """Score the archive, collect live signals, and rank them."""
        as_of = series.index[-1]
        if len(series) <= self.warmup + 100:
            return EnsembleRanking(
                signals=[], scores={}, folds=0, as_of=as_of,
                refusal=(
                    f"{len(series)} bars is not enough to score anything "
                    f"out-of-sample past a {self.warmup}-bar warmup. Ranking on "
                    f"in-sample results would be worse than not ranking."
                ),
            )

        scores = scores if scores is not None else self.score(series)
        usable = {k: s for k, s in scores.items() if s.is_usable}
        if not usable:
            return EnsembleRanking(
                signals=[], scores=scores, folds=self.folds, as_of=as_of,
                refusal=(
                    "no strategy produced a usable out-of-sample record. "
                    + self._diagnose(scores)
                ),
            )

        # One ICT analysis, shared. Re-running it per strategy would dominate
        # the cost of the whole ranking.
        ict_state = ICTEngine(self.ict_config).analyse(series)
        index = len(series) - 1
        context = StrategyContext(
            series=series, index=index, ict=ict_state,
            portfolio=Portfolio(starting_cash=self.risk_settings.account_equity),
            timestamp=as_of,
        )

        live: list[tuple[str, Signal, StrategyScore]] = []
        for key, score in usable.items():
            try:
                signal = self.entries[key].build().evaluate(context)
            except Exception:
                continue
            if signal is not None:
                live.append((key, signal, score))

        if not live:
            return EnsembleRanking(
                signals=[], scores=scores, folds=self.folds, as_of=as_of,
                notes=self._notes(usable),
            )

        ranked = self._assemble(live, top)
        return EnsembleRanking(
            signals=ranked, scores=scores, folds=self.folds, as_of=as_of,
            notes=self._notes(usable),
        )

    def _assemble(
        self,
        live: list[tuple[str, Signal, StrategyScore]],
        top: int,
    ) -> list[RankedSignal]:
        deflated = [s.deflated_r for _, _, s in live]
        lo, hi = min(deflated), max(deflated)
        span = hi - lo

        ranked: list[RankedSignal] = []
        for key, signal, score in live:
            same_direction = [
                other_key
                for other_key, other_signal, _ in live
                if other_key != key and other_signal.direction is signal.direction
            ]
            families = sorted(
                {
                    self.entries[other].family.value
                    for other in same_direction
                }
            )

            # Base score from deflated evidence, normalised across live signals.
            base = 0.5 if span <= 0 else (score.deflated_r - lo) / span

            # Cross-family agreement is the only agreement worth much: ten trend
            # rules concurring is one observation restated ten times.
            agreement = min(len(families), 4) / 8.0
            likelihood = float(np.clip(0.65 * base + 0.35 * agreement, 0.0, 1.0))

            notes: list[str] = []
            if score.deflated_r <= 0:
                notes.append(
                    "deflated out-of-sample edge is not positive — this signal "
                    "ranks relative to the others, on evidence that does not "
                    "establish an edge at all"
                )
            if score.trades < _SHRINKAGE_TRADES:
                notes.append(
                    f"only {score.trades} out-of-sample trades; the score is "
                    f"heavily shrunk and should be treated as provisional"
                )
            entry = self.entries[key]
            if entry.caveat:
                notes.append(entry.caveat)

            ranked.append(
                RankedSignal(
                    strategy_key=key, family=entry.family, signal=signal,
                    score=score, likelihood=likelihood,
                    concordant=tuple(sorted(same_direction)),
                    concordant_families=tuple(families),
                    notes=tuple(notes),
                )
            )

        ranked.sort(key=lambda r: (-r.likelihood, -r.score.deflated_r))
        return ranked[:top]

    @staticmethod
    def _diagnose(scores: dict[str, StrategyScore]) -> str:
        """Name the dominant reason nothing scored, so the refusal is actionable."""
        sized_out = [s for s in scores.values() if s.signals and not s.trades]
        if sized_out and len(sized_out) >= len(scores) / 2:
            reasons: dict[str, int] = {}
            for s in sized_out:
                for reason, count in s.rejections.items():
                    reasons[reason] = reasons.get(reason, 0) + count
            top = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "unknown"
            return (
                f"{len(sized_out)} strategies signalled but none was sized — "
                f"dominant reason '{top}'. If that is 'below_one_unit', the risk "
                f"budget is too small to buy one unit of this instrument: raise "
                f"account_equity or max_risk_per_trade_pct."
            )
        return "Most strategies produced no signals in the out-of-sample windows."

    @staticmethod
    def _notes(usable: dict[str, StrategyScore]) -> list[str]:
        positive = [s for s in usable.values() if s.deflated_r > 0]
        notes = [
            f"{len(usable)} strategies scored out-of-sample; "
            f"{len(positive)} have a positive deflated edge.",
            "`likelihood` is a relative ordering score, not a probability.",
        ]
        if not positive:
            notes.append(
                "No strategy clears the multiple-testing penalty. The ranking "
                "below orders signals against each other, and should not be "
                "read as evidence that any of them works."
            )
        return notes
