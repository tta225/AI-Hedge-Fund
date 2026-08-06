"""Calendar and seasonality effects.

Read this before using anything in this module
----------------------------------------------
A calendar rule has **no mechanism that reacts to price**. It cannot know
whether the market is trending, whether liquidity has evaporated, or whether the
thing it is anchored to still exists. Every effect here was found by slicing
historical returns by date, and the number of ways to slice a calendar is very
large — days of the week, days of the month, months, weeks around expiries,
holidays, and every combination. Some of these effects will be data mining.

They are catalogued for two honest reasons:

1. **As controls.** They are the sharpest available test of whether a research
   harness is too permissive. If `sell-in-may` clears the same walk-forward and
   deflated-Sharpe bar as a mechanism-based strategy, the bar is too low. That
   diagnostic is worth more than any of these are as strategies.

2. **As overlays.** Several are better used to *modulate* another strategy's
   conviction than to trade alone — `SeasonalWindow` exposes `is_active` for
   exactly that.

Every entry in the archive from this module carries `Evidence.CONTROL`. Nothing
here should be traded standalone on the strength of the label alone.

Decay
-----
Published calendar anomalies decay. The turn-of-month and January effects were
both substantially weaker after publication than before, which is what you would
expect if they were ever real and tradeable. Any use of these must be measured
on **recent** data, not on the full sample the original paper used.
"""

from __future__ import annotations

import pandas as pd

from axiom.core.types import Direction
from axiom.strategy.base import Signal, Strategy, StrategyContext
from axiom.strategy.quant_strategies import _atr_stop, _build

__all__ = [
    "DayOfWeek",
    "FOMCDrift",
    "JanuaryEffect",
    "OptionExpiryWeek",
    "SantaClausRally",
    "SeasonalWindow",
    "SellInMay",
]


class SeasonalWindow(Strategy):
    """Base for calendar rules: fires once per day inside an active window.

    Subclasses implement `is_active`. Keeping the plumbing in one place means
    the once-per-day guard and the ATR bracket are identical across every
    calendar effect, so a comparison between them measures the calendar rule and
    not two different implementations of "fire a trade".
    """

    name = "seasonal_window"
    requires_ict = False
    #: Set False for a rule whose documented direction is short.
    long_only: bool = True

    def __init__(
        self,
        *,
        stop_atr: float = 2.0,
        target_atr: float = 3.0,
        confidence: float = 0.35,
    ) -> None:
        super().__init__(
            stop_atr=stop_atr, target_atr=target_atr, confidence=confidence
        )
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.confidence = confidence

    def is_active(self, stamp: pd.Timestamp) -> tuple[bool, Direction, str]:
        """Return `(active, direction, why)` for this timestamp."""
        raise NotImplementedError

    def _first_bar_of_day(self, context: StrategyContext) -> bool:
        i = context.index
        if i == 0:
            return True
        index = context.series.index
        return bool(index[i - 1].date() != index[i].date())

    def evaluate(self, context: StrategyContext) -> Signal | None:
        # Once per day. Without this every intraday bar inside the window
        # re-enters, and the trade count reports the timeframe, not the effect.
        if not self._first_bar_of_day(context):
            return None

        active, direction, why = self.is_active(context.timestamp)
        if not active or direction is Direction.NEUTRAL:
            return None

        atr = max(context.atr(), context.instrument.tick_size)
        stop = _atr_stop(context, direction, self.stop_atr)
        offset = self.target_atr * atr
        target = (
            context.price + offset if direction is Direction.BULLISH
            else context.price - offset
        )
        return _build(
            context, direction, stop, target,
            confidence=self.confidence,
            rationale=f"{why} — calendar effect, no price mechanism (control)",
            tags=("seasonality", "calendar", "control", context.session),
            min_rr=1.2,
        )


class JanuaryEffect(SeasonalWindow):
    """Long through the first trading days of January.

    Small caps outperforming large caps in January, usually attributed to
    tax-loss selling in December reversing. **The size premium is the effect** —
    applied to a large-cap index future this is at best a diluted version of it,
    and that mismatch is the most likely reason it fails here.
    """

    name = "january_effect"

    def __init__(self, *, days: int = 10, **kwargs: float) -> None:
        super().__init__(**kwargs)
        self.days = days
        self.params["days"] = days

    def is_active(self, stamp: pd.Timestamp) -> tuple[bool, Direction, str]:
        active = stamp.month == 1 and stamp.day <= self.days
        return active, Direction.BULLISH, f"January effect (day {stamp.day})"


class SantaClausRally(SeasonalWindow):
    """Long the last trading days of December into the first of January.

    Documented by Yale Hirsch (1972). Roughly seven trading days, approximated
    here on calendar dates because the exchange holiday calendar is not modelled.
    That approximation is a real source of error in any measured result.
    """

    name = "santa_claus_rally"

    def is_active(self, stamp: pd.Timestamp) -> tuple[bool, Direction, str]:
        late_december = stamp.month == 12 and stamp.day >= 24
        early_january = stamp.month == 1 and stamp.day <= 3
        return (
            late_december or early_january,
            Direction.BULLISH,
            f"Santa Claus rally window ({stamp:%b %d})",
        )


class SellInMay(SeasonalWindow):
    """The Halloween indicator: long November–April, flat May–October.

    Among the most-tested calendar effects there is, and among the most likely
    to be a survivor of multiple testing across countries and decades. Traded
    here as long-only in the strong half rather than short in the weak half,
    which is what the original formulation actually claims.
    """

    name = "sell_in_may"

    def is_active(self, stamp: pd.Timestamp) -> tuple[bool, Direction, str]:
        strong = stamp.month in (11, 12, 1, 2, 3, 4)
        # Enter once at the start of the strong period, not every day of it.
        entry = strong and stamp.month == 11 and stamp.day <= 5
        return entry, Direction.BULLISH, f"Halloween/sell-in-May window opens ({stamp:%b %d})"


class DayOfWeek(SeasonalWindow):
    """Long or short a nominated weekday.

    The weakest claim in the module and included deliberately as the archive's
    purest null: there are only five weekdays and two directions, so testing all
    of them guarantees some combination looks good in any finite sample. If a
    search ranks this highly, the search is broken.
    """

    name = "day_of_week"

    def __init__(self, *, weekday: int = 0, direction: int = 1, **kwargs: float) -> None:
        super().__init__(**kwargs)
        if not 0 <= weekday <= 4:
            raise ValueError(f"weekday must be 0 (Mon) to 4 (Fri), got {weekday}")
        self.weekday = weekday
        self.direction = Direction.BULLISH if direction > 0 else Direction.BEARISH
        self.params.update(weekday=weekday, direction=direction)

    def is_active(self, stamp: pd.Timestamp) -> tuple[bool, Direction, str]:
        names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
        return (
            stamp.weekday() == self.weekday,
            self.direction,
            f"{names[self.weekday]} effect",
        )


class OptionExpiryWeek(SeasonalWindow):
    """Long the week of monthly option expiry.

    Mechanism, unusually for this module, is plausible: dealer gamma hedging
    into a large open-interest expiry pins and supports the underlying. Monthly
    expiry is the third Friday, so the week containing it is days 15–21.
    """

    name = "option_expiry_week"

    def is_active(self, stamp: pd.Timestamp) -> tuple[bool, Direction, str]:
        active = 15 <= stamp.day <= 21
        return active, Direction.BULLISH, f"option expiration week (day {stamp.day})"


class FOMCDrift(SeasonalWindow):
    """Long into the scheduled FOMC announcement — the pre-FOMC drift.

    Lucca & Moench (2015) found most of the equity premium over their sample
    accrued in the 24 hours before scheduled FOMC announcements.

    **The dates here are approximated**, not sourced from the Fed calendar: the
    FOMC meets roughly eight times a year in specific months, mid-month. A real
    implementation must load the actual announcement schedule, and until it does
    this measures "mid-month in eight months", which is a different and much
    weaker claim. Do not read a result from this as a test of the paper.
    """

    name = "fomc_drift"

    #: Months containing a scheduled meeting in a typical year.
    MONTHS = (1, 3, 5, 6, 7, 9, 11, 12)

    def is_active(self, stamp: pd.Timestamp) -> tuple[bool, Direction, str]:
        active = stamp.month in self.MONTHS and 14 <= stamp.day <= 17
        return (
            active,
            Direction.BULLISH,
            f"approximate pre-FOMC window ({stamp:%b %d}) — dates not from the Fed calendar",
        )
