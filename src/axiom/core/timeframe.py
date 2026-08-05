"""Timeframe parsing and arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Self

_PATTERN = re.compile(r"^\s*(\d+)\s*([smhdwM])\s*$")

_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "M": 2_592_000,  # nominal 30d; only used for ordering, never for indexing
}

_PANDAS_UNIT: dict[str, str] = {
    "s": "s", "m": "min", "h": "h", "d": "D", "w": "W", "M": "MS",
}


@dataclass(frozen=True, slots=True, order=True)
class Timeframe:
    """A bar interval such as ``5m``, ``4h`` or ``1d``.

    Ordering is by duration, so ``Timeframe("1m") < Timeframe("1h")`` — which is
    what multi-timeframe logic needs when checking that an HTF bias series is
    genuinely higher timeframe than the entry series.
    """

    seconds: int
    label: str

    @classmethod
    def parse(cls, value: str | Timeframe) -> Self | Timeframe:
        if isinstance(value, Timeframe):
            return value
        match = _PATTERN.match(value)
        if not match:
            raise ValueError(
                f"unparseable timeframe {value!r}; expected forms like '1m', '15m', '4h', '1d'"
            )
        amount, unit = int(match.group(1)), match.group(2)
        if amount <= 0:
            raise ValueError(f"timeframe amount must be positive, got {value!r}")
        return cls(seconds=amount * _UNIT_SECONDS[unit], label=f"{amount}{unit}")

    @property
    def amount(self) -> int:
        return int(_PATTERN.match(self.label).group(1))  # type: ignore[union-attr]

    @property
    def unit(self) -> str:
        return _PATTERN.match(self.label).group(2)  # type: ignore[union-attr]

    @property
    def timedelta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    @property
    def pandas_freq(self) -> str:
        """The pandas offset alias for :meth:`~pandas.DataFrame.resample`."""
        return f"{self.amount}{_PANDAS_UNIT[self.unit]}"

    @property
    def is_intraday(self) -> bool:
        return self.seconds < _UNIT_SECONDS["d"]

    def bars_per(self, other: Timeframe) -> float:
        """How many of ``self`` fit inside one ``other`` bar."""
        return other.seconds / self.seconds

    def __str__(self) -> str:
        return self.label


M1 = Timeframe.parse("1m")
M5 = Timeframe.parse("5m")
M15 = Timeframe.parse("15m")
M30 = Timeframe.parse("30m")
H1 = Timeframe.parse("1h")
H4 = Timeframe.parse("4h")
D1 = Timeframe.parse("1d")
W1 = Timeframe.parse("1w")
