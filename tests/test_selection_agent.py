"""The Selection agent, and the ensemble pipeline that drives it.

Selection is the stage with the most room to mislead, because it is the only
one that *chooses*. Every other agent reasons about a strategy someone already
picked; this one picks it, out of 49 across 9 families — and picking the best
of 49 records is itself a search. The best of 49 coin flips looks like skill.

So the tests here are almost entirely about what the agent refuses to let a
number mean:

* a likelihood score is a **relative ordering**, not a probability
* "best of 49" is not "good", and the deflated figure is the one that knows it
* twelve strategies from one family agreeing is one opinion counted twelve times
* when the evidence names nothing, the honest output is nothing — not a
  full-looking five-stage report about an arbitrary default

The last of those is the one that would do real damage, because a complete
report reads as a considered answer whether or not anything considered it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from axiom.agents.base import AgentRole
from axiom.agents.pipeline import AgentPipeline, SelectionAgent
from axiom.core.config import AgentSettings
from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.research.ensemble import EnsembleRanking, RankedSignal, StrategyScore
from axiom.strategy.archive import Family
from axiom.strategy.base import Signal

AS_OF = pd.Timestamp("2026-03-02 15:00", tz="UTC")


def _settings() -> AgentSettings:
    """No API key — deterministic mode. Facts must be complete regardless."""
    return AgentSettings(ANTHROPIC_API_KEY="", AXIOM_AGENT_MODEL="claude-opus-5")


def _score(
    key: str = "donchian",
    *,
    trades: int = 150,
    mean_r: float = 0.8,
    shrunk_r: float = 0.6,
    deflated_r: float = 0.2,
    excluded: str = "",
    family: Family = Family.BREAKOUT,
) -> StrategyScore:
    return StrategyScore(
        key=key, family=family, trades=trades, mean_r=mean_r, std_r=1.0,
        win_rate=0.55, folds=4, signals=trades + 10,
        shrunk_r=shrunk_r, deflated_r=deflated_r, excluded=excluded,
    )


def _signal(direction: Direction = Direction.BULLISH) -> Signal:
    return Signal(
        direction=direction, entry=5000.0, stop=4950.0, targets=(5150.0,),
        rationale="test setup",
    )


def _ranked(
    score: StrategyScore | None = None,
    *,
    likelihood: float = 0.74,
    concordant: tuple[str, ...] = (),
    concordant_families: tuple[str, ...] = (),
) -> RankedSignal:
    resolved = score or _score()
    return RankedSignal(
        strategy_key=resolved.key, family=resolved.family, signal=_signal(),
        score=resolved, likelihood=likelihood,
        concordant=concordant, concordant_families=concordant_families,
    )


def _ranking(
    signals: list[RankedSignal] | None = None,
    scores: dict[str, StrategyScore] | None = None,
    *,
    refusal: str = "",
    notes: list[str] | None = None,
) -> EnsembleRanking:
    return EnsembleRanking(
        signals=signals if signals is not None else [],
        scores=scores if scores is not None else {},
        folds=4, as_of=AS_OF, refusal=refusal, notes=notes or [],
    )


def _run(ranking: EnsembleRanking, series: OHLCVSeries) -> Any:
    from axiom.ict.engine import ICTEngine

    return SelectionAgent(_settings()).run(
        ranking=ranking,
        series=series,
        state=ICTEngine().analyse(series),
        provenance=series.provenance,
    )


class TestFactsAreComplete:
    def test_it_reports_the_selection_without_an_llm(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        score = _score()
        report = _run(_ranking([_ranked(score)], {"donchian": score}), synthetic_series)
        assert report.role is AgentRole.SELECTION
        assert report.facts["selected"] == "donchian"
        assert report.facts["family"] == "breakout"
        assert not report.used_llm

    def test_all_three_r_figures_are_reported_together(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Raw, shrunk and deflated side by side.

        Showing only the deflated figure hides how much of the raw number was
        sample noise; showing only the raw one is the overstatement itself. The
        distance between them *is* the finding.
        """
        score = _score(mean_r=0.83, shrunk_r=0.70, deflated_r=0.20)
        report = _run(_ranking([_ranked(score)], {"donchian": score}), synthetic_series)
        assert report.facts["raw_mean_r"] == pytest.approx(0.83)
        assert report.facts["shrunk_r"] == pytest.approx(0.70)
        assert report.facts["deflated_r"] == pytest.approx(0.20)

    def test_the_scale_of_the_search_is_a_fact_not_a_footnote(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        scores = {f"s{i}": _score(key=f"s{i}") for i in range(12)}
        report = _run(_ranking([_ranked(scores["s0"])], scores), synthetic_series)
        assert report.facts["strategies_evaluated"] == 12
        assert report.facts["strategies_with_usable_records"] == 12
        assert report.facts["walk_forward_folds"] == 4


class TestLikelihoodIsNotAProbability:
    def test_the_payload_says_so_in_a_field(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A number between 0 and 1 next to a trade idea will be read as a
        probability unless something states otherwise in the same breath."""
        score = _score()
        report = _run(_ranking([_ranked(score)], {"donchian": score}), synthetic_series)
        assert report.facts["likelihood_is_a_probability"] is False

    def test_a_top_score_from_a_tiny_field_is_flagged_as_an_artefact(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """With two live signals the better one scores 1.0 by construction —
        that says nothing about the evidence behind it."""
        score = _score()
        ranking = _ranking(
            [_ranked(score, likelihood=1.0), _ranked(_score(key="other"))],
            {"donchian": score, "other": _score(key="other")},
        )
        report = _run(ranking, synthetic_series)
        assert any("artefact of the" in w for w in report.warnings)

    def test_a_top_score_from_a_wide_field_is_not_flagged(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The warning must stay meaningful. Attaching it to every ranking
        would make it the thing people skip."""
        scores = {f"s{i}": _score(key=f"s{i}") for i in range(8)}
        ranking = _ranking(
            [_ranked(scores[f"s{i}"], likelihood=1.0 - i / 10) for i in range(8)],
            scores,
        )
        report = _run(ranking, synthetic_series)
        assert not any("artefact of the" in w for w in report.warnings)


class TestSelectionPenalty:
    def test_choosing_from_many_is_flagged(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        scores = {f"s{i}": _score(key=f"s{i}") for i in range(20)}
        report = _run(_ranking([_ranked(scores["s0"])], scores), synthetic_series)
        assert any("best of 20" in w for w in report.warnings)

    def test_the_warning_names_which_figure_accounts_for_it(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """'This is inflated' without saying which number is not inflated
        leaves the reader with the inflated one."""
        scores = {f"s{i}": _score(key=f"s{i}") for i in range(20)}
        report = _run(_ranking([_ranked(scores["s0"])], scores), synthetic_series)
        penalty = next(w for w in report.warnings if "best of 20" in w)
        assert "deflated_r" in penalty and "raw_mean_r" in penalty

    def test_a_single_candidate_is_not_accused_of_being_a_search(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        score = _score()
        report = _run(_ranking([_ranked(score)], {"donchian": score}), synthetic_series)
        assert not any("best of" in w for w in report.warnings)

    def test_a_negative_deflated_edge_is_called_what_it_is(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Ranking first among things that do not work is still first."""
        score = _score(deflated_r=-0.05)
        report = _run(_ranking([_ranked(score)], {"donchian": score}), synthetic_series)
        assert any("not evidence of an edge" in w for w in report.warnings)

    def test_a_thin_sample_makes_the_ordering_arbitrary(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        score = _score(trades=8)
        report = _run(_ranking([_ranked(score)], {"donchian": score}), synthetic_series)
        assert any("close to arbitrary" in w for w in report.warnings)


class TestAgreementIsNotConfirmation:
    def test_agreement_within_one_family_is_flagged(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Twelve trend rules agreeing is one opinion measured twelve times."""
        score = _score()
        ranked = _ranked(
            score,
            concordant=("a", "b", "c", "d"),
            concordant_families=("trend",),
        )
        report = _run(_ranking([ranked], {"donchian": score}), synthetic_series)
        assert any("one opinion measured 4 times" in w for w in report.warnings)

    def test_agreement_across_families_is_not_flagged(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        score = _score()
        ranked = _ranked(
            score,
            concordant=("a", "b", "c"),
            concordant_families=("trend", "breakout", "mean_reversion"),
        )
        report = _run(_ranking([ranked], {"donchian": score}), synthetic_series)
        assert not any("one opinion" in w for w in report.warnings)

    def test_both_counts_are_reported_rather_than_summed(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Strategies and families answer different questions. A single
        'agreement: 5' hides which one it is."""
        score = _score()
        ranked = _ranked(
            score, concordant=("a", "b", "c", "d", "e"),
            concordant_families=("trend", "breakout"),
        )
        report = _run(_ranking([ranked], {"donchian": score}), synthetic_series)
        assert report.facts["concordant_strategies"] == 5
        assert report.facts["concordant_families"] == 2


class TestRefusal:
    def test_a_refused_ranking_selects_nothing(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        report = _run(_ranking(refusal="not enough bars"), synthetic_series)
        assert report.facts["selected"] is None
        assert report.facts["refusal"] == "not enough bars"

    def test_the_refusal_leads_the_warnings(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """Buried under other notes it reads as one caveat among several,
        rather than as the reason there is no answer."""
        ranking = _ranking(refusal="not enough bars", notes=["some note"])
        report = _run(ranking, synthetic_series)
        assert report.warnings[0].startswith("No strategy selected")

    def test_nothing_signalling_is_a_finding_not_an_error(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        scores = {"a": _score(key="a")}
        report = _run(_ranking([], scores), synthetic_series)
        assert report.facts["selected"] is None
        assert any("is a finding" in w for w in report.warnings)


class TestEnsemblePipeline:
    """End to end, against the real ensemble on generated bars."""

    def test_it_prefixes_a_selection_stage(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from axiom.strategy.archive import ARCHIVE

        entries = dict(list(ARCHIVE.items())[:4])
        result = AgentPipeline(_settings()).run_ensemble(
            synthetic_series, entries=entries, folds=2, warmup=100
        )
        assert result.reports[0].role is AgentRole.SELECTION

    def test_a_refusal_produces_one_report_not_six(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """The failure this matters for: a full five-stage report about an
        arbitrary default reads as a considered answer. An empty one cannot.
        """
        from axiom.strategy.archive import ARCHIVE

        entries = dict(list(ARCHIVE.items())[:3])
        # Deliberately too short to score anything out-of-sample.
        short = synthetic_series.islice(0, 120)
        result = AgentPipeline(_settings()).run_ensemble(
            short, entries=entries, folds=2, warmup=100
        )
        assert len(result.reports) == 1
        assert result.reports[0].facts["selected"] is None

    def test_a_refusal_asks_for_no_approval(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from axiom.strategy.archive import ARCHIVE

        short = synthetic_series.islice(0, 120)
        result = AgentPipeline(_settings()).run_ensemble(
            short, entries=dict(list(ARCHIVE.items())[:3]), folds=2, warmup=100
        )
        assert not result.approval_required
        assert "NOTHING TO APPROVE" in result.render()
        assert "HUMAN APPROVAL REQUIRED" not in result.render()

    def test_a_refusal_never_falls_back_to_a_default_strategy(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        from axiom.strategy.archive import ARCHIVE

        short = synthetic_series.islice(0, 120)
        result = AgentPipeline(_settings()).run_ensemble(
            short, entries=dict(list(ARCHIVE.items())[:3]), folds=2, warmup=100
        )
        assert result.by_role(AgentRole.BACKTEST) is None
        assert result.by_role(AgentRole.RISK) is None

    def test_the_ranking_is_carried_so_the_losers_stay_visible(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """A ranking that only ships its winner cannot be audited."""
        from axiom.strategy.archive import ARCHIVE

        result = AgentPipeline(_settings()).run_ensemble(
            synthetic_series,
            entries=dict(list(ARCHIVE.items())[:4]),
            folds=2,
            warmup=100,
        )
        assert result.ranking is not None
        assert len(result.ranking.scores) >= 1

    def test_the_plain_run_carries_no_ranking(
        self, synthetic_series: OHLCVSeries
    ) -> None:
        """`ranking` present means a selection happened. It must not appear
        when the caller named the strategy themselves."""
        from axiom.strategy.ict_strategies import SilverBulletStrategy

        result = AgentPipeline(_settings()).run(
            synthetic_series, SilverBulletStrategy(), warmup=100
        )
        assert result.ranking is None


@dataclass
class _StubResult:
    """Minimal stand-in for PipelineResult, for the serialiser."""

    reports: list[Any] = field(default_factory=list)
    approval_required: bool = True
    approval_summary: str = ""
    ranking: Any = None

    @property
    def all_warnings(self) -> list[str]:
        return []


class TestSerialisation:
    def test_the_scoreboard_reaches_the_wire(self) -> None:
        from axiom.web.serialise import agent_payload

        scores = {
            "winner": _score(key="winner", deflated_r=0.3),
            "loser": _score(key="loser", deflated_r=-0.4),
        }
        payload = agent_payload(_StubResult(ranking=_ranking([], scores)))
        keys = [row["key"] for row in payload["ranking"]["scoreboard"]]
        assert keys == ["winner", "loser"]  # best first

    def test_excluded_strategies_sort_last_rather_than_vanish(self) -> None:
        """An excluded strategy dropped from the board looks like one that was
        never tried, which understates the size of the search."""
        from axiom.web.serialise import agent_payload

        scores = {
            "ok": _score(key="ok", deflated_r=0.1),
            "dead": _score(key="dead", excluded="no trades"),
        }
        payload = agent_payload(_StubResult(ranking=_ranking([], scores)))
        board = payload["ranking"]["scoreboard"]
        assert [row["key"] for row in board] == ["ok", "dead"]
        assert board[-1]["excluded"] == "no trades"

    def test_a_result_without_a_ranking_omits_the_key(self) -> None:
        from axiom.web.serialise import agent_payload

        assert "ranking" not in agent_payload(_StubResult())

    def test_execution_is_still_refused_on_the_wire(self) -> None:
        from axiom.web.serialise import agent_payload

        assert agent_payload(_StubResult())["can_execute"] is False
