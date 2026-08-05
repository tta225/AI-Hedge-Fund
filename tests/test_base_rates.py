"""Base-rate measurement.

The tests that matter here are the ones about *honesty*: a measurement module
whose bug direction is "reports an edge that is not there" is worse than no
module. So the properties pinned below are mostly about refusing to overclaim —
forward-only scanning, controls that actually track the real rate on data with
no structure, and intervals that widen rather than narrow on small samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import AssetClass, Instrument
from axiom.research.base_rates import (
    Measurement,
    measure_base_rates,
    wilson_interval,
)


def _random_walk(n: int = 1500, seed: int = 3) -> OHLCVSeries:
    """A driftless random walk — no structure for any feature to find.

    This is the null case. Every lift measured on it should be indistinguishable
    from zero; if one is not, the measurement is manufacturing signal.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, n).cumsum() + 500.0
    spread = np.abs(rng.normal(0.0, 0.4, n)) + 0.05
    frame = pd.DataFrame(
        {
            "open": steps,
            "high": steps + spread,
            "low": steps - spread,
            "close": steps + rng.normal(0.0, 0.2, n),
            "volume": rng.integers(100, 1000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return OHLCVSeries.from_dataframe(
        frame,
        Instrument("TEST", AssetClass.EQUITY),
        "15m",
        Provenance.synthetic("random-walk"),
    )


class TestWilsonInterval:
    def test_contains_the_point_estimate(self) -> None:
        lo, hi = wilson_interval(30, 100)
        assert lo < 0.30 < hi

    def test_stays_inside_zero_one_at_the_boundary(self) -> None:
        """The normal approximation returns bounds outside [0,1] here."""
        lo, hi = wilson_interval(10, 10)
        assert lo >= 0.0
        assert hi <= 1.0

    def test_narrows_as_the_sample_grows(self) -> None:
        small = wilson_interval(5, 10)
        large = wilson_interval(500, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_no_observations_spans_everything(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestMeasurement:
    def test_lift_is_rate_minus_control(self) -> None:
        m = Measurement("x", hits=70, trials=100, control_hits=60)
        assert m.rate == pytest.approx(0.70)
        assert m.control_rate == pytest.approx(0.60)
        assert m.lift == pytest.approx(0.10)

    def test_no_control_means_no_lift(self) -> None:
        m = Measurement("x", hits=70, trials=100)
        assert m.lift is None
        assert m.lift_interval is None
        assert not m.is_significant

    def test_a_small_sample_is_not_significant(self) -> None:
        """+20pp on 10 observations is noise, and must not be starred."""
        m = Measurement("x", hits=7, trials=10, control_hits=5)
        assert m.lift == pytest.approx(0.20)
        assert not m.is_significant

    def test_a_large_consistent_gap_is_significant(self) -> None:
        m = Measurement("x", hits=800, trials=1000, control_hits=500)
        assert m.is_significant

    def test_lift_renders_in_percentage_points(self) -> None:
        """A lift of 0.10 is 10 percentage points, not 0.1 of one."""
        m = Measurement("x", hits=70, trials=100, control_hits=60)
        assert "+10.0pp" in m.render()

    def test_zero_trials_does_not_divide_by_zero(self) -> None:
        assert "no observations" in Measurement("x", 0, 0).render()


class TestOnARandomWalk:
    """The null case: structureless data must not produce a measured edge."""

    def test_produces_measurements(self) -> None:
        report = measure_base_rates(_random_walk(), horizon=32)
        assert report.measurements
        assert all(m.trials >= 0 for m in report.measurements)

    def test_no_lift_is_significant(self) -> None:
        report = measure_base_rates(_random_walk(), horizon=32)
        significant = [m.name for m in report.measurements if m.is_significant]
        assert not significant, f"manufactured signal on noise: {significant}"

    def test_synthetic_input_is_not_evidence(self) -> None:
        report = measure_base_rates(_random_walk(), horizon=32)
        assert not report.is_evidence
        assert "NOT EVIDENCE" in report.render()

    def test_rates_are_probabilities(self) -> None:
        report = measure_base_rates(_random_walk(), horizon=32)
        for m in report.measurements:
            if m.trials:
                assert 0.0 <= m.rate <= 1.0
                assert m.hits <= m.trials


class TestNoLookahead:
    def test_horizon_bounds_the_outcome_window(self) -> None:
        """A longer horizon can only find more, never fewer, outcomes.

        The forward scan runs from confirmed_index, so widening the window is
        monotone. A violation means an outcome is being read from somewhere
        other than the future of the confirmation bar.
        """
        series = _random_walk()
        short = {m.name: m for m in measure_base_rates(series, horizon=8).measurements}
        long = {m.name: m for m in measure_base_rates(series, horizon=64).measurements}
        for name, s in short.items():
            other = long[name]
            if s.trials and other.trials and s.trials == other.trials:
                assert other.hits >= s.hits, name
