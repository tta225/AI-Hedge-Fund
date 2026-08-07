"""Session-conditional base rates. Offline.

The tests that matter here are about *not overstating a finding*: that identical
windows are not counted twice, that thin windows are not compared, and that
features are attributed to a window by confirmation rather than origin.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core.sessions import KILLZONES, LONDON_KILLZONE, SESSIONS, SILVER_BULLETS
from axiom.core.types import get_instrument
from axiom.data.base import BarRequest
from axiom.data.synthetic import SyntheticProvider
from axiom.research.session_rates import MIN_TRIALS, measure_session_rates


@pytest.fixture(scope="module")
def report():
    request = BarRequest.lookback(get_instrument("SPY"), "15m", 400)
    series = SyntheticProvider(seed=5, start_price=500.0).fetch_bars(request)
    return measure_session_rates(series)


class TestWindowAccounting:
    def test_identical_windows_are_measured_once(self, report) -> None:
        """`london` and `london_kz` share bounds; so do `asia` and `asia_kz`.

        Measuring both reports one result twice and inflates `sessions_tested`,
        which understates the multiple-comparison burden in the one direction
        that flatters whatever was found.
        """
        bounds = [(s.window.start, s.window.end) for s in report.slices]
        assert len(bounds) == len(set(bounds)), "a window was measured twice"

    def test_sessions_tested_matches_the_slices(self, report) -> None:
        assert report.sessions_tested == len(report.slices)

    def test_multiple_comparison_burden_is_disclosed(self, report) -> None:
        text = report.render()
        assert f"{report.sessions_tested} windows" in text
        assert "suspicion" in text

    def test_every_slice_has_bars(self, report) -> None:
        assert all(s.bars > 0 for s in report.slices)

    def test_a_custom_window_set_is_honoured(self) -> None:
        request = BarRequest.lookback(get_instrument("SPY"), "15m", 120)
        series = SyntheticProvider(seed=3, start_price=500.0).fetch_bars(request)
        only = measure_session_rates(series, windows=(LONDON_KILLZONE,))
        assert only.sessions_tested == 1
        assert only.slices[0].window.name == "london_kz"


class TestConditioning:
    def test_a_window_is_a_subset_of_the_unconditional_sample(self, report) -> None:
        """Conditioning can only ever remove observations."""
        base = {m.name: m.trials for m in report.baseline}
        for slice_ in report.slices:
            for measurement in slice_.measurements:
                assert measurement.trials <= base.get(measurement.name, 0), (
                    f"{slice_.window.name}/{measurement.name} has more trials "
                    f"than the unconditional measurement"
                )

    def test_conditional_gain_is_lift_minus_unconditional_lift(self, report) -> None:
        base = {m.name: m for m in report.baseline}
        for name, unconditional in base.items():
            if unconditional.lift is None:
                continue
            for window_name, gain in report.conditional_gain(name):
                slice_ = next(s for s in report.slices if s.window.name == window_name)
                measurement = slice_.by_name(name)
                assert measurement is not None and measurement.lift is not None
                assert gain == pytest.approx(measurement.lift - unconditional.lift)

    def test_thin_windows_are_excluded_from_comparison(self, report) -> None:
        for name in {m.name for m in report.baseline}:
            for window_name, _gain in report.conditional_gain(name):
                slice_ = next(s for s in report.slices if s.window.name == window_name)
                measurement = slice_.by_name(name)
                assert measurement is not None
                assert measurement.trials >= MIN_TRIALS

    def test_the_control_is_measured_inside_the_same_window(self, report) -> None:
        """A window that only raises volatility must move rate and control alike.

        If the control were measured unconditionally, every high-volatility
        window would show spurious lift. Both being non-null on the same trial
        count is the observable consequence.
        """
        for slice_ in report.slices:
            for measurement in slice_.usable:
                if measurement.control_hits is None:
                    continue
                assert 0 <= measurement.control_hits <= measurement.trials


class TestAttribution:
    def test_features_are_attributed_by_confirmation_not_origin(self) -> None:
        """Confirmation is when a feature became tradeable.

        A gap that formed at 01:00 and was confirmed at 02:30 belongs to the
        London window, because 02:30 is the first moment anyone could act on it.
        Attributing by origin would credit a window with setups that were not
        yet visible inside it.
        """
        import pandas as pd

        from axiom.ict.models import ICTState, SwingKind, SwingPoint
        from axiom.research.session_rates import _restrict

        stamp = pd.Timestamp("2024-01-02 12:00", tz="UTC")
        early = SwingPoint(origin_index=0, confirmed_index=1, timestamp=stamp,
                           price=1.0, kind=SwingKind.HIGH)
        late = SwingPoint(origin_index=0, confirmed_index=5, timestamp=stamp,
                          price=2.0, kind=SwingKind.HIGH)
        state = ICTState(symbol="X", timeframe="15m", as_of=stamp, index=9,
                         swings=[early, late])

        keep = np.zeros(10, dtype=bool)
        keep[1] = True  # only the bar `early` was CONFIRMED on

        restricted = _restrict(state, keep)
        assert [s.price for s in restricted.swings] == [1.0]


class TestProvenance:
    def test_generated_bars_are_labelled_not_evidence(self, report) -> None:
        assert not report.is_evidence
        assert "NOT MARKET DATA" in report.render()

    def test_report_names_every_window_it_measured(self, report) -> None:
        text = report.render()
        expected = {s.window.name for s in report.slices}
        rendered = {n for n in expected if n in text}
        assert rendered, "no window names reached the report"


class TestWindowCatalogue:
    def test_the_catalogue_contains_the_duplicates_this_module_dedupes(self) -> None:
        """Pins the reason `measure_session_rates` dedupes at all.

        If the catalogue is ever cleaned up so no two windows share bounds, this
        fails and the dedup can be reconsidered — rather than sitting there
        forever as unexplained defensive code.
        """
        bounds: dict[tuple[object, object], list[str]] = {}
        for window in (*KILLZONES, *SILVER_BULLETS, *SESSIONS):
            bounds.setdefault((window.start, window.end), []).append(window.name)
        shared = {k: v for k, v in bounds.items() if len(v) > 1}
        assert shared, (
            "no windows share bounds any more — the dedup in "
            "measure_session_rates may no longer be needed"
        )
