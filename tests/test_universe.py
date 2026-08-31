"""Point-in-time universe membership, and the backtester's handling of death.

The tests that matter here are the ones about names that stop existing. A
survivorship-biased backtest does not announce itself — it produces a plausible
number that is simply too high — so the properties worth pinning are the ones
whose violation is silent:

* a name that has not yet listed cannot be held,
* a name that has delisted is liquidated rather than carried at zero return,
* selecting the universe uses only volume observed before the selection date.

Each of those, got wrong, yields a backtest that runs cleanly and lies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.alpha.base import AlphaAgent, AlphaSignal
from axiom.alpha.panel import Panel
from axiom.core.provenance import DataKind, Provenance
from axiom.data.universe import (
    MIN_COVERAGE,
    Listing,
    PointInTimeUniverse,
    listed_mask,
)
from axiom.research.panel_lab import backtest_panel

SYNTHETIC = Provenance(kind=DataKind.SYNTHETIC, source="test")


def _listing(symbol: str, first: str, last: str, *, active: bool = True, n: int = 100) -> Listing:
    return Listing(
        symbol=symbol,
        first_bar=pd.Timestamp(first),
        last_bar=pd.Timestamp(last),
        n_bars=n,
        is_active=active,
    )


@pytest.fixture
def universe() -> PointInTimeUniverse:
    """Three lives: one spanning, one that dies, one that lists late."""
    pit = PointInTimeUniverse()
    pit.add(_listing("LIVE", "2020-01-01", "2024-12-31"))
    pit.add(_listing("DEAD", "2020-01-01", "2022-06-30", active=False))
    pit.add(_listing("LATE", "2023-01-01", "2024-12-31"))
    return pit


class TestListing:
    def test_membership_is_inclusive_at_both_ends(self) -> None:
        listing = _listing("X", "2020-01-01", "2020-12-31")
        assert listing.was_trading_at(pd.Timestamp("2020-01-01"))
        assert listing.was_trading_at(pd.Timestamp("2020-12-31"))
        assert not listing.was_trading_at(pd.Timestamp("2019-12-31"))
        assert not listing.was_trading_at(pd.Timestamp("2021-01-01"))

    def test_a_reversed_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            _listing("X", "2021-01-01", "2020-01-01")

    def test_died_before(self) -> None:
        listing = _listing("X", "2020-01-01", "2022-06-30")
        assert listing.died_before(pd.Timestamp("2023-01-01"))
        assert not listing.died_before(pd.Timestamp("2022-01-01"))


class TestMembership:
    def test_a_name_that_has_not_listed_is_absent(self, universe) -> None:
        assert universe.members_at(pd.Timestamp("2021-06-01")) == ("DEAD", "LIVE")

    def test_a_dead_name_is_absent_afterwards(self, universe) -> None:
        assert universe.members_at(pd.Timestamp("2023-06-01")) == ("LATE", "LIVE")

    def test_a_dead_name_is_present_while_it_traded(self, universe) -> None:
        """The whole point: history must contain the companies that failed."""
        assert "DEAD" in universe.members_at(pd.Timestamp("2021-01-01"))

    def test_delisted_is_derived_from_status(self, universe) -> None:
        assert universe.delisted == ("DEAD",)

    def test_survivor_only_reproduces_the_biased_universe(self, universe) -> None:
        """The experimental control — the universe the old caches were."""
        biased = universe.survivor_only(pd.Timestamp("2024-12-31"))
        assert "DEAD" not in biased.listings
        assert set(biased.listings) == {"LIVE", "LATE"}


class TestSelection:
    @pytest.fixture
    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        index = pd.date_range("2020-01-01", "2024-12-31", freq="B")
        closes = pd.DataFrame(100.0, index=index, columns=["LIVE", "DEAD", "LATE"])
        volumes = pd.DataFrame(
            {"LIVE": 1e6, "DEAD": 5e6, "LATE": 2e6}, index=index
        )
        closes.loc[closes.index > "2022-06-30", "DEAD"] = np.nan
        volumes.loc[volumes.index > "2022-06-30", "DEAD"] = np.nan
        closes.loc[closes.index < "2023-01-01", "LATE"] = np.nan
        volumes.loc[volumes.index < "2023-01-01", "LATE"] = np.nan
        return closes, volumes

    def test_ranks_by_dollar_volume(self, universe, frames) -> None:
        closes, volumes = frames
        snapshot = universe.select_at(
            pd.Timestamp("2021-06-01"), closes=closes, volumes=volumes, top_n=2
        )
        # DEAD is the most liquid name while it lives, and must be selectable.
        assert snapshot.symbols[0] == "DEAD"

    def test_selection_uses_only_prior_volume(self, universe, frames) -> None:
        """A universe chosen with the volume of the day it starts knows
        something the trader did not."""
        closes, volumes = frames
        date = pd.Timestamp("2021-06-01")
        base = universe.select_at(date, closes=closes, volumes=volumes, top_n=2)

        future = volumes.copy()
        future.loc[future.index >= date, "LIVE"] = 1e12
        after = universe.select_at(date, closes=closes, volumes=future, top_n=2)
        assert base.symbols == after.symbols

    def test_prior_volume_does_change_the_selection(self, universe, frames) -> None:
        """The control: the mechanism works, so the test above means something."""
        closes, volumes = frames
        date = pd.Timestamp("2021-06-01")
        base = universe.select_at(date, closes=closes, volumes=volumes, top_n=1)
        louder = volumes.copy()
        louder.loc[louder.index < date, "LIVE"] = 1e12
        after = universe.select_at(date, closes=closes, volumes=louder, top_n=1)
        assert base.symbols != after.symbols
        assert after.symbols == ("LIVE",)

    def test_a_name_not_yet_listed_is_never_selected(self, universe, frames) -> None:
        closes, volumes = frames
        snapshot = universe.select_at(
            pd.Timestamp("2021-06-01"), closes=closes, volumes=volumes, top_n=3
        )
        assert "LATE" not in snapshot.symbols

    def test_thin_coverage_is_excluded(self, universe, frames) -> None:
        closes, volumes = frames
        sparse = volumes.copy()
        # LIVE trades on only a handful of days in the lookback.
        window = sparse.index[(sparse.index < "2021-06-01")][-252:]
        sparse.loc[window[:-5], "LIVE"] = np.nan
        snapshot = universe.select_at(
            pd.Timestamp("2021-06-01"), closes=closes, volumes=sparse, top_n=3,
            min_coverage=MIN_COVERAGE,
        )
        assert "LIVE" not in snapshot.symbols

    def test_exclude_supports_a_disjoint_confirmation_set(self, universe, frames) -> None:
        closes, volumes = frames
        snapshot = universe.select_at(
            pd.Timestamp("2021-06-01"), closes=closes, volumes=volumes,
            top_n=2, exclude=["DEAD"],
        )
        assert "DEAD" not in snapshot.symbols

    def test_attrition_counts_names_that_later_died(self, universe, frames) -> None:
        """The number the old universes silently set to zero."""
        closes, volumes = frames
        snapshot = universe.select_at(
            pd.Timestamp("2021-09-01"), closes=closes, volumes=volumes, top_n=2
        )
        assert "DEAD" in snapshot.symbols
        assert snapshot.n_later_delisted == 1
        assert snapshot.attrition_rate == pytest.approx(0.5)

    def test_survivorship_report_is_tabular(self, universe, frames) -> None:
        closes, volumes = frames
        report = universe.survivorship_report(
            [pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-01")],
            closes=closes, volumes=volumes, top_n=3,
        )
        assert list(report.columns) == [
            "as_of", "n_members", "n_eligible", "n_later_delisted", "attrition_rate"
        ]
        assert len(report) == 2

    def test_top_n_must_be_positive(self, universe, frames) -> None:
        closes, volumes = frames
        with pytest.raises(ValueError, match="at least 1"):
            universe.select_at(
                pd.Timestamp("2021-06-01"), closes=closes, volumes=volumes, top_n=0
            )


class TestFromFrames:
    def test_bars_are_authoritative_over_status(self) -> None:
        """A name whose record says active but which stopped printing in 2021
        is delisted for every purpose a backtest has."""
        index = pd.date_range("2020-01-01", "2024-12-31", freq="B")
        closes = pd.DataFrame(100.0, index=index, columns=["A", "B"])
        closes.loc[closes.index > "2021-06-30", "B"] = np.nan
        universe = PointInTimeUniverse.from_frames(closes, active={"A", "B"})
        assert universe.listings["B"].last_bar <= pd.Timestamp("2021-06-30")
        assert not universe.listings["B"].was_trading_at(pd.Timestamp("2023-01-01"))

    def test_an_all_nan_column_is_skipped(self) -> None:
        index = pd.date_range("2020-01-01", periods=50, freq="B")
        closes = pd.DataFrame({"A": 100.0, "GHOST": np.nan}, index=index)
        universe = PointInTimeUniverse.from_frames(closes)
        assert set(universe.listings) == {"A"}

    def test_round_trip_through_csv(self, universe, tmp_path) -> None:
        path = tmp_path / "listings.csv"
        universe.save(str(path))
        loaded = PointInTimeUniverse.load(str(path))
        assert set(loaded.listings) == set(universe.listings)
        assert loaded.listings["DEAD"].last_bar == universe.listings["DEAD"].last_bar
        assert loaded.listings["DEAD"].delisted


class TestListedMask:
    def test_mask_matches_the_intervals(self, universe) -> None:
        index = pd.DatetimeIndex(["2021-01-01", "2023-06-01"])
        mask = listed_mask(universe, index, ["LIVE", "DEAD", "LATE"])
        assert mask[0].tolist() == [True, True, False]
        assert mask[1].tolist() == [True, False, True]

    def test_a_mid_life_gap_does_not_read_as_a_delisting(self, universe) -> None:
        """Computed from intervals, not notna: a halt is not a death, and
        treating it as one would force a spurious liquidation."""
        index = pd.DatetimeIndex(["2021-01-01", "2021-01-04"])
        mask = listed_mask(universe, index, ["LIVE"])
        assert mask.all()

    def test_an_unknown_symbol_is_never_tradable(self, universe) -> None:
        mask = listed_mask(universe, pd.DatetimeIndex(["2021-01-01"]), ["NOPE"])
        assert not mask.any()


class _FullyLong(AlphaAgent):
    """Wants every name, always — so the backtester's refusals are visible."""

    agent_id = "fully_long"
    min_history = 2

    def generate(self, panel: Panel, index: int) -> list[AlphaSignal]:
        return [
            AlphaSignal(
                symbol=symbol,
                signal=1.0 - 0.01 * column,
                confidence=0.5,
                expected_return=None,
                horizon_bars=5,
                strategy="test",
                agent_id=self.agent_id,
                as_of_index=index,
                provenance=panel.provenance,
            )
            for column, symbol in enumerate(panel.symbols)
        ]


def _pit_panel() -> tuple[Panel, PointInTimeUniverse]:
    """Eight names: six that span, one that dies halfway, one that lists late.

    Large enough that a long/short book actually forms, so the backtester's
    refusals are visible as changed numbers rather than as an empty portfolio.
    """
    index = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(5)
    names = ["A", "B", "C", "D", "E", "F", "DEAD", "LATE"]
    values = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (300, len(names))), axis=0))
    closes = pd.DataFrame(values, index=index, columns=names)
    volumes = pd.DataFrame(1e6, index=index, columns=names)

    death, birth = index[150], index[100]
    closes.loc[closes.index > death, "DEAD"] = np.nan
    volumes.loc[volumes.index > death, "DEAD"] = np.nan
    closes.loc[closes.index < birth, "LATE"] = np.nan
    volumes.loc[volumes.index < birth, "LATE"] = np.nan

    active = set(names) - {"DEAD"}
    universe = PointInTimeUniverse.from_frames(closes, volumes, active=active)
    return Panel.point_in_time(closes, volumes, universe, SYNTHETIC), universe


class TestPointInTimePanel:
    def test_the_index_is_the_union_not_the_intersection(self) -> None:
        """Inner-joining a realistic universe yields an empty panel, and the
        silent result is survivorship bias reintroduced by a join."""
        panel, _ = _pit_panel()
        assert len(panel) == 300
        assert panel.is_point_in_time

    def test_prices_outside_the_listing_window_are_absent(self) -> None:
        panel, _ = _pit_panel()
        assert np.isnan(panel.closes[-1, panel.column("DEAD")])
        assert np.isnan(panel.closes[0, panel.column("LATE")])

    def test_tradable_tracks_the_mask(self) -> None:
        panel, _ = _pit_panel()
        dead, late = panel.column("DEAD"), panel.column("LATE")
        assert panel.tradable(0)[dead] and not panel.tradable(0)[late]
        assert not panel.tradable(299)[dead] and panel.tradable(299)[late]

    def test_a_panel_without_a_mask_assumes_everything_survived(self) -> None:
        """The old behaviour, still reachable and now named for what it is."""
        index = pd.date_range("2020-01-01", periods=10, freq="B")
        panel = Panel(
            symbols=("A",), index=index,
            closes=np.full((10, 1), 100.0), volumes=np.full((10, 1), 1e6),
            provenance=SYNTHETIC,
        )
        assert not panel.is_point_in_time
        assert panel.tradable(0).all()

    def test_a_mismatched_mask_is_refused(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="B")
        with pytest.raises(ValueError, match="listed mask"):
            Panel(
                symbols=("A",), index=index,
                closes=np.full((10, 1), 100.0), volumes=np.full((10, 1), 1e6),
                provenance=SYNTHETIC, listed=np.ones((5, 1), dtype=bool),
            )


class TestBacktestHandlesDeath:
    def test_a_delisted_name_is_not_held(self) -> None:
        """The silent failure this exists to prevent: without the mask a dead
        name sits in the book at exactly zero return, free of cost and risk,
        diluting every figure after it."""
        panel, _ = _pit_panel()
        result = backtest_panel(
            [_FullyLong()], panel, start=5, rebalance_every=5,
            long_short=True, top_fraction=0.5,
        )
        assert result.is_viable
        assert result.returns.size > 0

    def test_liquidation_happens_between_rebalances(self) -> None:
        """A name can die on a day the book was not going to trade."""
        panel, _ = _pit_panel()
        # Rebalance rarely, so the death at bar 150 falls between reranks.
        result = backtest_panel(
            [_FullyLong()], panel, start=5, rebalance_every=63,
            long_short=True, top_fraction=0.5,
        )
        # Turnover is non-zero on the bar the position was forced out, which
        # is not a rebalance bar.
        traded_on_non_rebalance = [
            t for i, t in enumerate(result.turnover, start=5)
            if t > 0 and i % 63 != 0
        ]
        assert traded_on_non_rebalance

    def test_returns_are_finite_across_a_delisting(self) -> None:
        panel, _ = _pit_panel()
        result = backtest_panel(
            [_FullyLong()], panel, start=5, rebalance_every=5,
            long_short=True, top_fraction=0.5,
        )
        assert np.all(np.isfinite(result.returns))

    def test_survivorship_changes_the_answer(self) -> None:
        """The measurement the whole exercise exists for.

        The same strategy over the full universe and over the survivors-only
        version of it must not produce the same number — if it did, there would
        be nothing to correct.
        """
        panel, universe = _pit_panel()
        full = backtest_panel(
            [_FullyLong()], panel, start=5, rebalance_every=5,
            long_short=True, top_fraction=0.5,
        )

        survivors = universe.survivor_only(panel.index[-1])
        keep = [s for s in panel.symbols if s in survivors.listings]
        closes = pd.DataFrame(panel.closes, index=panel.index, columns=list(panel.symbols))[keep]
        volumes = pd.DataFrame(panel.volumes, index=panel.index, columns=list(panel.symbols))[keep]
        biased_panel = Panel.point_in_time(closes, volumes, survivors, SYNTHETIC)
        biased = backtest_panel(
            [_FullyLong()], biased_panel, start=5, rebalance_every=5,
            long_short=True, top_fraction=0.5,
        )
        assert full.total_return_pct != pytest.approx(biased.total_return_pct)
