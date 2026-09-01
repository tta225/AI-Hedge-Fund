"""Time-of-day seasonality, and the machinery to test whether it is real.

The claim
---------
"Overnight Seasonality in Bitcoin" (Quantpedia; surveyed in
``paperswithbacktest/awesome-systematic-trading``) reports a Sharpe of **0.892**
for a rule with no discretion in it at all: go long BTC at 22:00 UTC, hold two
hours, flatten. That is the highest number in a survey of 61 published
systematic strategies, and it is the only entry in that survey testable with
data this repository already holds.

This module implements the *idea* from the paper's description — a long-only
position opened at a fixed UTC hour and closed after a fixed holding period.
The published QuantConnect implementation was not used and is not a dependency.

Two honest deviations from the paper, both of which cost the strategy
something rather than flattering it:

*A stop exists.* The paper's rule has none; it holds for a fixed window and
exits. But :class:`~axiom.strategy.base.Signal` requires a stop because the
risk manager sizes from it, and a strategy that cannot be sized cannot be
traded. The stop is deliberately wide (``stop_atr``, default 3.0) so that it
binds rarely and the measurement is of the seasonal effect rather than of stop
placement — but when it does bind it is a real loss the paper never took.

*Costs are charged.* The backtester applies commission and slippage to both
legs. A two-hour hold repeated daily is roughly 365 round trips a year, so
costs are not a rounding error here; they are the main thing standing between
the claimed Sharpe and a realised one.

:class:`SeasonalityControl` is the null. It takes the same rule at a *shifted*
hour, which has no seasonal claim attached to it whatsoever. If the control
performs like the treatment, what has been measured is crypto's drift and the
cost model, not an hour of the day.
"""

from __future__ import annotations

import pandas as pd

from axiom.core.types import Direction
from axiom.strategy.base import Signal, Strategy, StrategyContext

#: The hour the paper nominates, in UTC.
PUBLISHED_ENTRY_HOUR = 22
#: The holding period the paper nominates, in hours.
PUBLISHED_HOLD_HOURS = 2
#: Wide by design: this strategy is a test of an hour, not of a stop.
DEFAULT_STOP_ATR = 3.0


class TimeOfDaySeasonality(Strategy):
    """Enter at a fixed UTC hour, hold for a fixed number of hours, exit.

    There is no signal here in the usual sense — no confirmation, no structure,
    no filter. That is the point. A rule this mechanical either works or it
    does not, and there is nowhere for a researcher's judgement to leak in and
    manufacture a result.

    Args:
        entry_hour: UTC hour at which to open. The paper says 22.
        hold_hours: hours to hold before flattening. The paper says 2.
        direction: side to take. Long is the published claim; the parameter
            exists so the lab can test the short side as a further control.
        stop_atr: protective stop distance in ATR units. Wide on purpose.
        target_atr: profit target in ATR units. Also wide — the exit that is
            supposed to matter is the clock.
    """

    name = "time_of_day_seasonality"
    #: Pure calendar arithmetic. Skipping ICT analysis makes a parameter sweep
    #: dramatically cheaper and changes nothing, because nothing here reads it.
    requires_ict = False

    def __init__(
        self,
        *,
        entry_hour: int = PUBLISHED_ENTRY_HOUR,
        hold_hours: int = PUBLISHED_HOLD_HOURS,
        direction: Direction = Direction.BULLISH,
        stop_atr: float = DEFAULT_STOP_ATR,
        target_atr: float = DEFAULT_STOP_ATR,
    ) -> None:
        if not 0 <= entry_hour <= 23:
            raise ValueError(f"entry_hour must be a UTC hour in [0, 23], got {entry_hour}")
        if hold_hours < 1:
            raise ValueError(f"hold_hours must be at least 1, got {hold_hours}")
        if direction is Direction.NEUTRAL:
            raise ValueError("direction must be BULLISH or BEARISH")
        super().__init__(
            entry_hour=entry_hour,
            hold_hours=hold_hours,
            direction=direction,
            stop_atr=stop_atr,
            target_atr=target_atr,
        )
        self.entry_hour = entry_hour
        self.hold_hours = hold_hours
        self.direction = direction
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if not context.is_flat:
            return None
        if _utc_hour(context.timestamp) != self.entry_hour:
            return None

        atr = context.atr()
        if atr <= 0:
            return None

        entry = context.price
        sign = 1.0 if self.direction is Direction.BULLISH else -1.0
        stop = entry - sign * self.stop_atr * atr
        target = entry + sign * self.target_atr * atr

        return Signal(
            direction=self.direction,
            entry=entry,
            stop=stop,
            targets=(target,),
            # Flat, and flat for the same reason as everywhere else in this
            # codebase: nothing in the published claim grades conviction, so a
            # confidence curve here would be a number with no evidence behind
            # it.
            confidence=0.5,
            rationale=(
                f"Seasonal entry at {self.entry_hour:02d}:00 UTC, "
                f"holding {self.hold_hours}h"
            ),
            tags=("seasonality", f"hour_{self.entry_hour:02d}"),
        )

    def should_exit(self, context: StrategyContext) -> str | None:
        """Flatten once the holding period has elapsed.

        The clock is the thesis. Stop and target are risk plumbing that the
        paper's rule does not have, and if either fires first it does so
        despite this method rather than through it.
        """
        opened_at = context.position.opened_at
        if opened_at is None:
            return None
        elapsed = context.timestamp - opened_at
        if elapsed >= pd.Timedelta(hours=self.hold_hours):
            return f"held {self.hold_hours}h from {self.entry_hour:02d}:00 UTC"
        return None


class SeasonalityControl(TimeOfDaySeasonality):
    """The same rule at an hour with no claim attached. The null.

    Shifting the entry hour holds everything else fixed — same instrument, same
    holding period, same stop geometry, same costs, same number of round trips
    per year. The only thing that varies is the one thing the paper says
    matters. If this performs like the treatment, the result is about crypto's
    drift and the cost model rather than about 22:00 UTC.
    """

    name = "seasonality_control"

    def __init__(
        self, *, entry_hour: int = PUBLISHED_ENTRY_HOUR, shift_hours: int = 12, **params: object
    ) -> None:
        super().__init__(entry_hour=(entry_hour + shift_hours) % 24, **params)  # type: ignore[arg-type]
        self.params["shift_hours"] = shift_hours
        # Record the hour actually traded, not the one that was asked for, so
        # a leaderboard label never misstates what ran.
        self.params["entry_hour"] = self.entry_hour


def _utc_hour(timestamp: pd.Timestamp) -> int:
    """Hour of ``timestamp`` in UTC, whatever timezone it arrived in.

    A naive timestamp is assumed to already be UTC — that is this repository's
    convention everywhere — but a tz-aware one is converted rather than read
    off, because reading the local hour off an aware timestamp is how a
    seasonality study silently measures the wrong hour.
    """
    if timestamp.tzinfo is None:
        return int(timestamp.hour)
    return int(timestamp.tz_convert("UTC").hour)
