"""Portfolio-level backtesting and the signal ensemble.

The property that makes `PortfolioBacktester` worth having is tested first:
running N instruments through one shared risk budget must produce *fewer* trades
than running them independently. If it does not, the budget is not shared and
the class is an expensive loop.

For the ensemble, the property under test is that it does not rank on in-sample
results and does not present a lucky small sample as evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.backtest.engine import Backtester
from axiom.backtest.portfolio_engine import PortfolioBacktester, run_portfolio
from axiom.core.config import RiskSettings
from axiom.core.series import OHLCVSeries
from axiom.core.types import get_instrument
from axiom.data.base import BarRequest
from axiom.data.synthetic import SyntheticProvider
from axiom.research.ensemble import SignalEnsemble
from axiom.strategy.archive import ARCHIVE
from axiom.strategy.institutional import DonchianBreakout

RISK = RiskSettings(account_equity=5_000_000.0, max_gross_exposure_pct=5000.0)


def _series(symbol: str, seed: int, days: int = 120) -> OHLCVSeries:
    request = BarRequest.lookback(get_instrument(symbol), "1h", days)
    return SyntheticProvider(seed=seed, start_price=5200.0).fetch_bars(request)


@pytest.fixture(scope="module")
def book() -> dict[str, OHLCVSeries]:
    return {s: _series(s, 11 + i) for i, s in enumerate(("ES", "NQ", "SPY"))}


class TestPortfolioBacktester:
    def test_runs_and_attributes_per_instrument(self, book: dict[str, OHLCVSeries]) -> None:
        result = run_portfolio(
            DonchianBreakout(), book, risk_settings=RISK, warmup=300
        )
        assert set(result.attribution) == set(book)
        assert len(result.trades) == sum(
            len(a.trades) for a in result.attribution.values()
        )
        realised = sum(a.realised_pnl for a in result.attribution.values())
        assert realised == pytest.approx(sum(t.pnl for t in result.trades))

    def test_one_shared_budget_binds_across_instruments(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        """The property that makes this a portfolio backtest.

        With `max_positions=1`, three instruments compete for one slot. Run
        independently each would take its own trades; run together they cannot,
        and the difference is reported as `contested_slots`.
        """
        tight = RiskSettings(
            account_equity=5_000_000.0, max_gross_exposure_pct=5000.0, max_positions=1
        )
        shared = run_portfolio(DonchianBreakout(), book, risk_settings=tight, warmup=300)

        independent = 0
        for series in book.values():
            independent += len(
                Backtester(DonchianBreakout(), risk_settings=tight)
                .run(series, warmup=300)
                .trades
            )

        assert len(shared.trades) < independent, (
            "a shared one-position budget must take fewer trades than three "
            "independent runs — otherwise the budget is not shared"
        )
        assert shared.contested_slots > 0

    def test_provenance_is_tainted_by_any_synthetic_input(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        result = run_portfolio(DonchianBreakout(), book, risk_settings=RISK, warmup=300)
        assert not result.report.is_evidence
        assert result.report.warning

    def test_correlation_matrix_shape_and_diagonal(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        matrix = PortfolioBacktester.correlation_matrix(book)
        assert list(matrix.columns) == list(book)
        assert np.allclose(np.diag(matrix.to_numpy()), 1.0)

    def test_independent_seeds_are_uncorrelated(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        """Sanity check on the fixture as much as on the estimator."""
        matrix = PortfolioBacktester.correlation_matrix(book)
        off_diagonal = matrix.to_numpy()[~np.eye(len(book), dtype=bool)]
        assert np.abs(off_diagonal).max() < 0.3

    def test_timeline_is_the_union_of_all_instruments(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        timeline = PortfolioBacktester._merged_timeline(book)
        assert timeline.is_monotonic_increasing
        for series in book.values():
            assert series.index.isin(timeline).all()

    def test_per_symbol_strategies_are_accepted(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        strategies = {s: ARCHIVE["donchian"].build() for s in book}
        result = run_portfolio(strategies, book, risk_settings=RISK, warmup=300)
        assert set(result.attribution) == set(book)

    def test_missing_strategy_for_a_symbol_is_an_error(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        partial = {"ES": DonchianBreakout()}
        with pytest.raises(ValueError, match="no strategy supplied"):
            run_portfolio(partial, book, risk_settings=RISK, warmup=300)

    def test_notes_disclose_the_ordering_bias(
        self, book: dict[str, OHLCVSeries]
    ) -> None:
        """Same-timestamp ordering decides contested slots. It must be stated."""
        result = run_portfolio(DonchianBreakout(), book, risk_settings=RISK, warmup=300)
        assert any("fixed" in note and "order" in note for note in result.report.notes)

    def test_rejects_a_series_too_short_for_the_warmup(self) -> None:
        with pytest.raises(ValueError, match="not enough for a"):
            run_portfolio(
                DonchianBreakout(), {"ES": _series("ES", 3, days=2)},
                risk_settings=RISK, warmup=5000,
            )


class TestSignalEnsemble:
    @pytest.fixture(scope="class")
    @classmethod
    def subset(cls) -> dict[str, object]:
        keys = ("donchian", "momentum", "mean-reversion", "rsi-reversal", "macd")
        return {k: ARCHIVE[k] for k in keys}

    def test_refuses_to_rank_without_enough_history(self, subset: dict) -> None:
        """Ranking on in-sample results would be worse than not ranking."""
        short = _series("ES", 5, days=3)
        ranking = SignalEnsemble(subset, folds=3, warmup=200).rank(short)
        assert ranking.refusal
        assert not ranking.signals

    def test_scores_are_shrunk_toward_zero(self, subset: dict) -> None:
        """A small sample must not outrank a large one on a lucky streak."""
        series = _series("ES", 5, days=250)
        ensemble = SignalEnsemble(subset, folds=3, warmup=200, risk_settings=RISK)
        scores = ensemble.score(series)
        for score in scores.values():
            if score.is_usable and score.trades:
                # Shrinkage pulls toward zero, so magnitude cannot grow.
                assert abs(score.shrunk_r) <= abs(score.mean_r) + 1e-12
                # Deflation SUBTRACTS a penalty. A small positive edge can
                # become a larger-magnitude negative one — that is the penalty
                # working, so the comparison is signed, not absolute.
                assert score.deflated_r <= score.shrunk_r + 1e-9

    def test_deflation_penalises_every_candidate(self, subset: dict) -> None:
        series = _series("ES", 5, days=250)
        scores = SignalEnsemble(
            subset, folds=3, warmup=200, risk_settings=RISK
        ).score(series)
        usable = [s for s in scores.values() if s.is_usable]
        assert len(usable) >= 2
        assert all(s.deflated_r <= s.shrunk_r + 1e-9 for s in usable)

    def test_likelihood_is_bounded_and_never_called_a_probability(
        self, subset: dict
    ) -> None:
        series = _series("ES", 5, days=250)
        ranking = SignalEnsemble(
            subset, folds=3, warmup=200, risk_settings=RISK
        ).rank(series)
        for signal in ranking.signals:
            assert 0.0 <= signal.likelihood <= 1.0
            assert "NOT a probability" in signal.describe()

    def test_negative_edge_is_disclosed_on_every_ranked_signal(
        self, subset: dict
    ) -> None:
        """A ranking is an ordering, not a claim that the top entry works."""
        series = _series("ES", 5, days=250)
        ranking = SignalEnsemble(
            subset, folds=3, warmup=200, risk_settings=RISK
        ).rank(series)
        for signal in ranking.signals:
            if signal.score.deflated_r <= 0:
                assert any("not positive" in note for note in signal.notes)

    def test_controls_are_excluded_by_default(self) -> None:
        ensemble = SignalEnsemble()
        assert "day-of-week" not in ensemble.entries
        assert "sell-in-may" not in ensemble.entries
        assert "donchian" in ensemble.entries

    def test_controls_can_be_opted_in(self) -> None:
        assert "day-of-week" in SignalEnsemble(include_controls=True).entries

    def test_unsized_strategies_are_diagnosed_not_silently_dropped(self) -> None:
        """`below_one_unit` is the most common cause and must be named."""
        starved = RiskSettings(account_equity=2_000.0, max_gross_exposure_pct=100.0)
        series = _series("ES", 5, days=250)
        ranking = SignalEnsemble(
            {"donchian": ARCHIVE["donchian"]}, folds=3, warmup=200,
            risk_settings=starved,
        ).rank(series)
        assert ranking.refusal
        assert "below_one_unit" in ranking.refusal or "signalled" in ranking.refusal

    def test_scoreboard_renders_every_strategy(self, subset: dict) -> None:
        series = _series("ES", 5, days=250)
        ranking = SignalEnsemble(
            subset, folds=3, warmup=200, risk_settings=RISK
        ).rank(series)
        board = ranking.scoreboard()
        for key in subset:
            assert key in board
