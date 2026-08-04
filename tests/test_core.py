"""Core domain invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from tests.conftest import make_series

from axiom.core.provenance import (
    DataKind,
    Provenance,
    SyntheticDataError,
    provenance_warning,
    require_real_data,
)
from axiom.core.series import DataIntegrityError, OHLCVSeries
from axiom.core.timeframe import Timeframe
from axiom.core.types import ES, EURUSD, Bar, Direction, Instrument, Side


class TestInstrument:
    def test_tick_value_derives_from_tick_size_and_point_value(self) -> None:
        assert ES.tick_value == pytest.approx(12.50)

    def test_round_to_tick_snaps_to_grid(self) -> None:
        assert ES.round_to_tick(5000.13) == 5000.25
        assert ES.round_to_tick(5000.10) == 5000.0

    def test_round_to_tick_has_no_float_dust(self) -> None:
        # 0.1 + 0.2 territory: the result must be exactly representable.
        assert EURUSD.round_to_tick(1.234567) == 1.23457

    def test_cash_pnl_respects_side_and_point_value(self) -> None:
        # Long 2 ES from 5000 to 5010 = 10 pts * $50 * 2.
        assert ES.cash_pnl(5000, 5010, 2, Side.BUY) == pytest.approx(1000.0)
        assert ES.cash_pnl(5000, 5010, 2, Side.SELL) == pytest.approx(-1000.0)

    def test_rejects_nonpositive_tick_size(self) -> None:
        with pytest.raises(ValueError, match="tick_size"):
            Instrument("BAD", ES.asset_class, tick_size=0.0)


class TestBar:
    def test_requires_timezone_aware_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            Bar(datetime(2025, 1, 1), 1, 2, 0.5, 1.5)

    def test_rejects_inconsistent_ohlc(self) -> None:
        with pytest.raises(ValueError, match="inconsistent bar"):
            Bar(datetime(2025, 1, 1, tzinfo=UTC), open=5, high=1, low=0, close=2)

    def test_direction_and_midpoint(self) -> None:
        bar = Bar(datetime(2025, 1, 1, tzinfo=UTC), 100, 110, 90, 105)
        assert bar.direction is Direction.BULLISH
        assert bar.midpoint == 100.0
        assert bar.upper_wick == 5.0
        assert bar.lower_wick == 10.0


class TestTimeframe:
    @pytest.mark.parametrize(
        ("text", "seconds"), [("1m", 60), ("15m", 900), ("4h", 14400), ("1d", 86400)]
    )
    def test_parse(self, text: str, seconds: int) -> None:
        assert Timeframe.parse(text).seconds == seconds

    def test_ordering_is_by_duration(self) -> None:
        assert Timeframe.parse("1m") < Timeframe.parse("1h") < Timeframe.parse("1d")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="unparseable"):
            Timeframe.parse("banana")

    def test_pandas_freq_uses_min_not_m(self) -> None:
        # "15m" would mean months to pandas; the mapping must yield minutes.
        assert Timeframe.parse("15m").pandas_freq == "15min"


class TestOHLCVSeries:
    def test_rejects_missing_columns(self) -> None:
        frame = pd.DataFrame(
            {"open": [1.0], "high": [2.0]},
            index=pd.DatetimeIndex([datetime(2025, 1, 1, tzinfo=UTC)]),
        )
        with pytest.raises(DataIntegrityError, match="missing required column"):
            OHLCVSeries.from_dataframe(frame, ES, "1d", Provenance.mock("t"))

    def test_rejects_duplicate_timestamps(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        frame = pd.DataFrame(
            {"open": [1.0, 1.0], "high": [2.0, 2.0], "low": [0.5, 0.5],
             "close": [1.5, 1.5], "volume": [1.0, 1.0]},
            index=pd.DatetimeIndex([ts, ts]),
        )
        with pytest.raises(DataIntegrityError, match="duplicate timestamp"):
            OHLCVSeries.from_dataframe(frame, ES, "1d", Provenance.mock("t"))

    def test_rejects_impossible_bars(self) -> None:
        frame = pd.DataFrame(
            {"open": [5.0], "high": [1.0], "low": [0.0], "close": [2.0], "volume": [1.0]},
            index=pd.DatetimeIndex([datetime(2025, 1, 1, tzinfo=UTC)]),
        )
        with pytest.raises(DataIntegrityError, match="violate low"):
            OHLCVSeries.from_dataframe(frame, ES, "1d", Provenance.mock("t"))

    def test_naive_index_is_localised_to_utc(self) -> None:
        series = make_series([(1.0, 2.0, 0.5, 1.5)] * 3)
        assert str(series.index.tz) == "UTC"

    def test_resample_upward_aggregates_correctly(self) -> None:
        series = make_series([
            (100.0, 105.0, 99.0, 104.0),
            (104.0, 108.0, 103.0, 107.0),
            (107.0, 110.0, 106.0, 109.0),
            (109.0, 112.0, 108.0, 111.0),
        ], timeframe="15m")
        hourly = series.resample("1h")
        assert len(hourly) == 1
        bar = hourly.bar_at(0)
        assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 112.0, 99.0, 111.0)

    def test_resample_downward_is_refused(self) -> None:
        series = make_series([(1.0, 2.0, 0.5, 1.5)] * 4, timeframe="1h")
        with pytest.raises(ValueError, match="invent bars"):
            series.resample("1m")

    def test_atr_is_positive_and_matches_length(self) -> None:
        series = make_series([(100 + i, 102 + i, 99 + i, 101 + i) for i in range(30)])
        atr = series.atr(14)
        assert len(atr) == len(series)
        assert (atr > 0).all()


class TestProvenance:
    def test_synthetic_is_not_evidential(self) -> None:
        assert not Provenance.synthetic("gen", seed=1).is_evidential
        assert Provenance.real("vendor").is_evidential

    def test_deriving_real_data_stays_evidential(self) -> None:
        derived = Provenance.real("vendor").derive("resample")
        assert derived.kind is DataKind.DERIVED
        assert derived.is_evidential

    def test_synthetic_cannot_be_laundered_by_transformation(self) -> None:
        # This is the property that makes the whole guarantee hold.
        tainted = Provenance.synthetic("gen", seed=1).derive("resample").derive("clean")
        assert tainted.kind is DataKind.SYNTHETIC
        assert not tainted.is_evidential

    def test_merge_takes_the_weakest_kind(self) -> None:
        merged = Provenance.real("vendor").merge(Provenance.synthetic("gen"))
        assert not merged.is_evidential

    def test_require_real_data_raises_on_synthetic(self) -> None:
        with pytest.raises(SyntheticDataError, match="not evidence"):
            require_real_data(Provenance.synthetic("gen"), "live trading")

    def test_require_real_data_passes_for_real(self) -> None:
        require_real_data(Provenance.real("vendor"), "live trading")

    def test_warning_present_only_for_generated_data(self) -> None:
        assert provenance_warning(Provenance.synthetic("gen")) is not None
        assert provenance_warning(Provenance.real("vendor")) is None
