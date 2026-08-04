"""Deterministic synthetic market generator.

Purpose: exercise the full pipeline offline and give tests a reproducible tape.
It is **not** a market simulator and its output is never evidence — every series
it produces is stamped ``DataKind.SYNTHETIC``, which
:func:`~axiom.core.provenance.require_real_data` rejects wherever numbers are
reported as performance.

The generator deliberately produces structure ICT analytics can find — trending
legs with displacement, retracements, and session-shaped volatility — because a
pure random walk yields degenerate test cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom.core.provenance import Provenance
from axiom.core.sessions import to_ny
from axiom.core.timeframe import Timeframe
from axiom.core.types import Instrument
from axiom.data.base import BarRequest, BaseProvider

#: Relative volatility by New York hour — quiet Asia, active London/NY killzones.
_HOUR_VOL_PROFILE = np.array([
    0.6, 0.5, 1.4, 1.5, 1.3, 1.1, 0.9, 1.5, 1.6, 1.7,  # 00-09
    1.5, 1.1, 0.7, 1.2, 1.4, 1.3, 0.8, 0.6, 0.6, 0.7,  # 10-19
    0.8, 0.7, 0.6, 0.6,                                  # 20-23
])


class SyntheticProvider(BaseProvider):
    """Generates regime-switching OHLCV bars from a seeded RNG."""

    name = "synthetic"

    def __init__(
        self,
        seed: int = 7,
        start_price: float = 5000.0,
        annual_vol: float = 0.18,
        trend_persistence: float = 0.985,
        trend_strength: float = 0.25,
    ) -> None:
        """
        Args:
            trend_persistence: AR(1) coefficient of the drift process. Higher
                values give longer, cleaner legs.
            trend_strength: stationary standard deviation of the drift, as a
                multiple of per-bar volatility. Values much above ~0.4 make the
                drift dominate the noise and the tape trends without pause,
                which is not a useful test fixture.
        """
        if not 0.0 <= trend_persistence < 1.0:
            raise ValueError("trend_persistence must be in [0, 1)")
        self.seed = seed
        self.start_price = start_price
        self.annual_vol = annual_vol
        self.trend_persistence = trend_persistence
        self.trend_strength = trend_strength

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance.synthetic(self.name, seed=self.seed)

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        return self.generate(
            request.instrument, request.timeframe, request.start, request.end
        )

    def generate(
        self,
        instrument: Instrument,
        timeframe: str | Timeframe,
        start: pd.Timestamp | object,
        end: pd.Timestamp | object,
    ) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        index = pd.date_range(start=start, end=end, freq=tf.pandas_freq, tz="UTC")
        if len(index) < 2:
            raise ValueError("synthetic range too short to generate bars")
        return self._simulate(instrument, tf, index)

    def _simulate(
        self, instrument: Instrument, tf: Timeframe, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = len(index)

        # Per-bar volatility, scaled from annual and shaped by session.
        bars_per_year = (365.0 * 24 * 3600) / tf.seconds
        base_sigma = self.annual_vol / np.sqrt(bars_per_year)
        hours = to_ny(index).hour.to_numpy()
        sigma = base_sigma * _HOUR_VOL_PROFILE[hours]

        # Persistent drift produces trending legs and clean retracements rather
        # than the structureless wandering of an iid walk. The innovation scale
        # is solved from the AR(1) stationary variance so that
        # std(drift) == trend_strength * base_sigma regardless of persistence —
        # otherwise raising persistence silently explodes the trend.
        phi = self.trend_persistence
        innovation_sd = self.trend_strength * base_sigma * np.sqrt(1.0 - phi**2)
        shocks = rng.standard_normal(n)
        innovation = rng.standard_normal(n) * innovation_sd
        drift = np.empty(n)
        drift[0] = innovation[0]
        for i in range(1, n):
            drift[i] = phi * drift[i - 1] + innovation[i]

        log_returns = drift + sigma * shocks
        closes = self.start_price * np.exp(np.cumsum(log_returns))
        opens = np.concatenate(([self.start_price], closes[:-1]))

        # Wick extents scale with the bar's own realised move, so displacement
        # bars get small wicks and indecision bars get large ones.
        body = np.abs(closes - opens)
        wick_scale = sigma * closes * rng.uniform(0.3, 1.6, size=n)
        upper = np.maximum(opens, closes) + wick_scale * rng.uniform(0.1, 1.0, size=n)
        lower = np.minimum(opens, closes) - wick_scale * rng.uniform(0.1, 1.0, size=n)
        highs = np.maximum.reduce([opens, closes, upper])
        lows = np.minimum.reduce([opens, closes, lower])

        volume = np.round(
            rng.lognormal(mean=8.0, sigma=0.45, size=n)
            * _HOUR_VOL_PROFILE[hours]
            * (1.0 + body / np.maximum(closes * base_sigma, 1e-12) * 0.15)
        )

        tick = instrument.tick_size

        def snap(values: np.ndarray) -> np.ndarray:
            return np.asarray(np.round(values / tick) * tick)

        frame = pd.DataFrame(
            {
                "open": snap(opens),
                "high": snap(highs),
                "low": snap(lows),
                "close": snap(closes),
                "volume": volume,
            },
            index=index,
        )
        # Snapping can invert H/L against O/C by one tick; re-clamp.
        frame["high"] = frame[["open", "high", "close"]].max(axis=1)
        frame["low"] = frame[["open", "low", "close"]].min(axis=1)
        return frame
