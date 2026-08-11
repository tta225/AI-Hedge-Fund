"""Shared fixtures.

Handcrafted bars are used wherever a test pins specific ICT geometry. Generated
data is fine for smoke tests but useless for asserting that a particular
three-bar pattern is detected, because you cannot control what the generator
produces.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.timeframe import Timeframe
from axiom.core.types import ES, Instrument
from axiom.data.base import BarRequest
from axiom.data.synthetic import SyntheticProvider


def make_series(
    rows: list[tuple[float, float, float, float]],
    *,
    instrument: Instrument = ES,
    timeframe: str = "15m",
    start: datetime | None = None,
    volume: float = 1000.0,
) -> OHLCVSeries:
    """Build a series from ``(open, high, low, close)`` tuples.

    Timestamps start at a fixed UTC instant so session-dependent tests are
    deterministic across machines and dates.
    """
    origin = start or datetime(2025, 3, 3, 13, 0, tzinfo=UTC)  # 08:00 ET, NY AM
    step = timedelta(seconds=Timeframe.parse(timeframe).seconds)
    index = pd.DatetimeIndex([origin + i * step for i in range(len(rows))])
    frame = pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [volume] * len(rows),
        },
        index=index,
    )
    return OHLCVSeries.from_dataframe(
        frame, instrument, timeframe, Provenance.mock("test_fixture")
    )


@pytest.fixture
def synthetic_series() -> OHLCVSeries:
    """A reproducible multi-week synthetic tape."""
    end = datetime(2025, 6, 30, tzinfo=UTC)
    return SyntheticProvider(seed=11, start_price=5200.0).fetch_bars(
        BarRequest(ES, Timeframe.parse("15m"), end - timedelta(days=60), end)
    )


@pytest.fixture(scope="module")
def synthetic_series_module() -> OHLCVSeries:
    """The same tape as :func:`synthetic_series`, built once per module.

    Identical bars, module scope. A test that runs the full agent pipeline pays
    for an ICT pass plus a backtest, and doing that per test function turns one
    file into the slowest part of the suite.
    """
    end = datetime(2025, 6, 30, tzinfo=UTC)
    return SyntheticProvider(seed=11, start_price=5200.0).fetch_bars(
        BarRequest(ES, Timeframe.parse("15m"), end - timedelta(days=60), end)
    )


@pytest.fixture
def uptrend_series() -> OHLCVSeries:
    """A clean uptrend: higher highs and higher lows, with a retracement.

    Deliberately shaped so a swing low, a swing high, a break of structure, and
    a bullish fair value gap all exist at known positions.
    """
    rows = [
        (100.0, 101.0, 99.5, 100.5),
        (100.5, 101.5, 100.0, 101.0),
        (101.0, 101.5, 99.0, 99.5),    # 2: swing low at 99.0
        (99.5, 102.0, 99.4, 101.8),
        (102.0, 106.0, 101.9, 105.5),  # 4: displacement up
        (105.6, 107.0, 105.0, 106.5),  # 5: FVG bull (low 105.0 > high 101.5 @3? no)
        (106.5, 108.0, 106.0, 107.5),  # 6: swing high region
        (107.5, 108.2, 104.0, 104.5),  # 7: retrace
        (104.5, 105.0, 103.0, 103.5),  # 8: swing low
        (103.5, 106.0, 103.4, 105.8),
        (105.8, 110.0, 105.7, 109.5),  # 10: BOS above 108.2
        (109.5, 111.0, 109.0, 110.5),
        (110.5, 111.5, 110.0, 111.0),
    ]
    return make_series(rows)
