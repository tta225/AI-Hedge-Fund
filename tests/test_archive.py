"""The strategy archive: every entry runs, is causal, and is honestly labelled.

The causality test is the one that matters. A strategy that reads one bar into
the future produces a beautiful equity curve that cannot be traded, and nothing
else in the suite would catch it — the backtest completes, the metrics compute,
the report renders. `TestCausality` asserts it directly by mutating the future
and requiring the signal not to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.backtest.engine import Backtester
from axiom.core.config import RiskSettings
from axiom.core.series import OHLCVSeries
from axiom.core.types import get_instrument
from axiom.data.base import BarRequest
from axiom.data.synthetic import SyntheticProvider
from axiom.ict.engine import ICTEngine
from axiom.portfolio.positions import Portfolio
from axiom.strategy.archive import (
    ARCHIVE,
    Evidence,
    Family,
    build,
    by_family,
    diversified,
    families,
    get,
)
from axiom.strategy.base import StrategyContext

#: Generous equity — several catalogued strategies place wide ATR stops, and on
#: a $50/point instrument a small account rejects every one of them as
#: `below_one_unit`, which would test the risk manager rather than the strategy.
RISK = RiskSettings(account_equity=5_000_000.0, max_gross_exposure_pct=5000.0)


@pytest.fixture(scope="module")
def series() -> OHLCVSeries:
    request = BarRequest.lookback(get_instrument("ES"), "1h", 200)
    return SyntheticProvider(seed=11, start_price=5200.0).fetch_bars(request)


class TestRegistry:
    def test_every_entry_key_matches_its_dict_key(self) -> None:
        for key, entry in ARCHIVE.items():
            assert entry.key == key

    def test_every_entry_states_a_thesis_and_a_source(self) -> None:
        """An entry without a mechanism is a pattern someone liked the look of."""
        for entry in ARCHIVE.values():
            assert entry.thesis.strip(), f"{entry.key} has no thesis"
            assert entry.source.strip(), f"{entry.key} has no source"

    def test_grids_only_name_real_constructor_arguments(self) -> None:
        """A grid key that is not a parameter fails silently during a sweep."""
        import inspect

        for entry in ARCHIVE.values():
            signature = inspect.signature(entry.factory)
            for name in entry.grid:
                assert name in signature.parameters, (
                    f"{entry.key} grid names {name!r}, which is not a parameter"
                )

    def test_grid_values_construct(self) -> None:
        for entry in ARCHIVE.values():
            for name, values in entry.grid.items():
                for value in values:
                    entry.build(**{name: value})

    def test_nothing_claims_a_measured_edge_without_a_citation(self) -> None:
        """The most dangerous possible mislabel in this repository.

        `docs/REAL_DATA_FINDINGS.md` records that the first real measurement
        found no edge anywhere. Any entry claiming MEASURED_EDGE must cite where
        that was established.
        """
        for entry in ARCHIVE.values():
            if entry.evidence is Evidence.MEASURED_EDGE:
                assert entry.caveat, (
                    f"{entry.key} claims a measured edge with no citation"
                )

    def test_controls_are_labelled_as_controls(self) -> None:
        for entry in by_family(Family.SEASONALITY):
            assert entry.evidence is Evidence.CONTROL, (
                f"{entry.key} is a calendar rule and must be marked CONTROL"
            )

    def test_lookup_helpers(self) -> None:
        assert get("donchian").key == "donchian"
        with pytest.raises(KeyError, match="unknown strategy"):
            get("no-such-strategy")
        assert build("donchian").name == "donchian_breakout"

    def test_diversified_spans_families_and_excludes_controls(self) -> None:
        picked = diversified()
        assert len({e.family for e in picked}) == len(picked), "one per family"
        assert all(e.evidence is not Evidence.CONTROL for e in picked)

    def test_families_covers_every_entry(self) -> None:
        assert sum(len(v) for v in families().values()) == len(ARCHIVE)


class TestEveryStrategyRuns:
    def test_all_entries_backtest_without_error(self, series: OHLCVSeries) -> None:
        failures: list[str] = []
        for key, entry in sorted(ARCHIVE.items()):
            try:
                Backtester(entry.build(), risk_settings=RISK).run(series, warmup=600)
            except Exception as exc:
                failures.append(f"{key}: {type(exc).__name__}: {exc}")
        assert not failures, "\n".join(failures)

    def test_signals_respect_their_own_stop_direction(self, series: OHLCVSeries) -> None:
        """`Signal.__post_init__` enforces this; a violation raises, not returns."""
        for entry in ARCHIVE.values():
            result = Backtester(entry.build(), risk_settings=RISK).run(series, warmup=600)
            for signal in result.signals:
                if signal.direction.name == "BULLISH":
                    assert signal.stop < signal.entry
                else:
                    assert signal.stop > signal.entry


class TestCausality:
    """No strategy may read a bar it could not have seen.

    Method: evaluate at bar `i` on the full series, then evaluate again on a
    series whose bars *after* `i` have been replaced with garbage. A causal
    strategy returns exactly the same signal. Anything that changes is reading
    the future.
    """

    @staticmethod
    def _corrupt_future(series: OHLCVSeries, index: int) -> OHLCVSeries:
        frame = series.df.copy()
        tail = frame.index[index + 1 :]
        rng = np.random.default_rng(0)
        shock = pd.Series(rng.uniform(3.0, 6.0, len(tail)), index=tail)
        for column in ("open", "high", "low", "close"):
            frame.loc[tail, column] = frame.loc[tail, column] * shock
        # Keep the OHLC invariants intact so construction still validates.
        frame.loc[tail, "high"] = frame.loc[tail, ["open", "high", "low", "close"]].max(axis=1)
        frame.loc[tail, "low"] = frame.loc[tail, ["open", "high", "low", "close"]].min(axis=1)
        return OHLCVSeries(
            instrument=series.instrument,
            timeframe=series.timeframe,
            df=frame,
            provenance=series.provenance,
        )

    @pytest.mark.parametrize("key", sorted(ARCHIVE))
    def test_signal_is_unchanged_by_the_future(self, key: str, series: OHLCVSeries) -> None:
        index = len(series) - 200
        corrupted = self._corrupt_future(series, index)

        ict_full = ICTEngine().analyse(series)
        ict_corrupt = ICTEngine().analyse(corrupted)
        portfolio = Portfolio(starting_cash=RISK.account_equity)

        entry = ARCHIVE[key]
        original = entry.build().evaluate(
            StrategyContext(
                series=series, index=index, ict=ict_full.known(index),
                portfolio=portfolio, timestamp=series.index[index],
            )
        )
        replayed = entry.build().evaluate(
            StrategyContext(
                series=corrupted, index=index, ict=ict_corrupt.known(index),
                portfolio=portfolio, timestamp=corrupted.index[index],
            )
        )

        if original is None:
            assert replayed is None, f"{key} signalled only because of future bars"
            return
        assert replayed is not None, f"{key} lost its signal when the future changed"
        assert original.direction is replayed.direction, f"{key} direction changed"
        assert original.entry == pytest.approx(replayed.entry), f"{key} entry changed"
        assert original.stop == pytest.approx(replayed.stop), f"{key} stop changed"


class TestStatisticalEstimators:
    def test_hurst_detects_a_trend_and_a_random_walk(self) -> None:
        from axiom.strategy.statistical import hurst_exponent

        rng = np.random.default_rng(7)
        walk = np.cumsum(rng.normal(0, 1, 3000)) + 100
        assert hurst_exponent(walk) == pytest.approx(0.5, abs=0.12)

        # Persistence means *shocks* persist, so the test process must have
        # positively autocorrelated increments. A deterministic drift is
        # invisible to this estimator by construction — see the docstring, and
        # `test_hurst_cannot_see_deterministic_drift` below.
        noise = rng.normal(0, 1, 3000)
        increments = np.empty(3000)
        increments[0] = noise[0]
        for i in range(1, 3000):
            increments[i] = 0.9 * increments[i - 1] + noise[i]
        persistent = np.cumsum(increments) + 100
        assert hurst_exponent(persistent) > 0.7

        anti = np.cumsum(noise * np.where(np.arange(3000) % 2 == 0, 1, -1)) + 100
        assert hurst_exponent(anti) < 0.5

    def test_hurst_cannot_see_deterministic_drift(self) -> None:
        """A documented blind spot, pinned so it cannot be forgotten.

        A steadily trending series with light noise estimates near zero, not
        near one: the drift shifts every lagged difference equally and adds no
        dispersion. Anyone reading `HurstAdaptive` as a trend detector is
        misreading it, and this test is where that is written down.
        """
        from axiom.strategy.statistical import hurst_exponent

        rng = np.random.default_rng(7)
        drifting = np.arange(3000, dtype=float) * 0.5 + rng.normal(0, 2, 3000) + 100
        assert hurst_exponent(drifting) < 0.3

    def test_hurst_refuses_a_degenerate_series(self) -> None:
        from axiom.strategy.statistical import hurst_exponent

        line = np.arange(3000, dtype=float) * 0.5 + 100
        assert np.isnan(hurst_exponent(line)), (
            "a noiseless line has no defined Hurst exponent; NaN is the honest "
            "answer, and a number here would be invented"
        )

    def test_variance_ratio_is_one_for_a_random_walk(self) -> None:
        from axiom.strategy.statistical import variance_ratio

        rng = np.random.default_rng(11)
        walk = np.exp(np.cumsum(rng.normal(0, 0.01, 5000))) * 100
        assert variance_ratio(walk, 5) == pytest.approx(1.0, abs=0.25)

    def test_ou_fit_recovers_a_known_half_life(self) -> None:
        """The estimator is the whole strategy; if it is wrong, so is the rule."""
        from axiom.strategy.institutional import OrnsteinUhlenbeck

        rng = np.random.default_rng(3)
        lam, mu, n = 0.05, 100.0, 20000
        prices = np.empty(n)
        prices[0] = mu
        for i in range(1, n):
            prices[i] = prices[i - 1] + lam * (mu - prices[i - 1]) + rng.normal(0, 0.5)

        fitted_lam, fitted_mu, half_life = OrnsteinUhlenbeck.fit(prices)
        assert fitted_lam == pytest.approx(lam, rel=0.3)
        assert fitted_mu == pytest.approx(mu, abs=1.0)
        assert half_life == pytest.approx(np.log(2) / lam, rel=0.3)

    def test_a_trending_series_reports_no_reversion(self) -> None:
        from axiom.strategy.institutional import OrnsteinUhlenbeck

        trending = np.arange(500, dtype=float) + 100.0
        _lam, _mu, half_life = OrnsteinUhlenbeck.fit(trending)
        assert not np.isfinite(half_life), "a trending series must not report a half-life"


class TestKalman:
    def test_slope_tracks_a_real_trend(self) -> None:
        """Regression test for a two-orders-of-magnitude slope bug.

        The first implementation updated the slope with the level's gain scaled
        by `process_var / level_var`, which made the slope so small it never
        cleared `min_slope_atr` and the strategy silently never traded.
        """
        from axiom.strategy.institutional import KalmanTrend

        model = KalmanTrend()
        rising = np.arange(400, dtype=float) * 0.5 + 5000.0
        _level, slope = model._filter(rising)
        assert slope == pytest.approx(0.5, rel=0.35)

        falling = 5000.0 - np.arange(400, dtype=float) * 0.5
        _level, slope = model._filter(falling)
        assert slope == pytest.approx(-0.5, rel=0.35)

    def test_strategy_actually_signals(self, series: OHLCVSeries) -> None:
        from axiom.strategy.institutional import KalmanTrend

        result = Backtester(KalmanTrend(), risk_settings=RISK).run(series, warmup=600)
        assert result.signals, "kalman produced no signals at all — the bug is back"
