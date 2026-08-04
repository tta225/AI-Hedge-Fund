"""Positions and portfolio accounting.

Sign convention: :attr:`Position.quantity` is signed — positive is long,
negative is short — so netting is plain addition and a flat position is exactly
zero. Average price is maintained on the *open* quantity only; realising part of
a position does not disturb the basis of the remainder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from axiom.core.types import Instrument, Side


@dataclass(slots=True)
class Position:
    """A netted position in one instrument."""

    instrument: Instrument
    quantity: float = 0.0
    average_price: float = 0.0
    realised_pnl: float = 0.0
    commission_paid: float = 0.0
    opened_at: pd.Timestamp | None = None
    last_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.0

    @property
    def side(self) -> Side | None:
        if self.is_flat:
            return None
        return Side.BUY if self.is_long else Side.SELL

    @property
    def exposure(self) -> float:
        """Absolute notional at the last mark."""
        return abs(self.quantity) * self.last_price * self.instrument.point_value

    def unrealised_pnl(self, price: float | None = None) -> float:
        if self.is_flat:
            return 0.0
        mark = self.last_price if price is None else price
        return (mark - self.average_price) * self.quantity * self.instrument.point_value

    def mark(self, price: float) -> None:
        self.last_price = price

    def apply_fill(
        self, side: Side, quantity: float, price: float, commission: float,
        timestamp: pd.Timestamp | None = None,
    ) -> float:
        """Apply a fill and return the realised P&L it produced.

        Handles the three cases explicitly: opening/adding (weighted-average the
        basis), reducing/closing (realise against the basis), and reversal
        through zero (realise the whole old position, then open the remainder at
        the fill price).
        """
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")

        signed = quantity * side.sign
        self.commission_paid += commission
        realised = -commission

        if self.is_flat:
            self.quantity = signed
            self.average_price = price
            self.opened_at = timestamp
        elif (self.quantity > 0) == (signed > 0):
            # Adding to the position: blend the basis.
            total = self.quantity + signed
            self.average_price = (
                self.average_price * self.quantity + price * signed
            ) / total
            self.quantity = total
        else:
            closing = min(abs(signed), abs(self.quantity))
            realised += (
                (price - self.average_price)
                * closing
                * (1 if self.quantity > 0 else -1)
                * self.instrument.point_value
            )
            remaining = self.quantity + signed
            if remaining == 0.0:
                self.quantity = 0.0
                self.average_price = 0.0
                self.opened_at = None
            elif (remaining > 0) == (self.quantity > 0):
                self.quantity = remaining
            else:
                # Reversed through flat: the residual opens at the fill price.
                self.quantity = remaining
                self.average_price = price
                self.opened_at = timestamp

        self.realised_pnl += realised
        self.last_price = price
        return realised


@dataclass(slots=True)
class EquityPoint:
    timestamp: pd.Timestamp
    equity: float
    cash: float
    unrealised: float
    exposure: float


@dataclass(slots=True)
class Portfolio:
    """Cash, positions, and the equity curve.

    Cash accounting is float64. For a research and paper-trading platform the
    error is negligible relative to slippage assumptions; a live ledger should
    move to fixed-point. See ``docs/ARCHITECTURE.md``.
    """

    starting_cash: float
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    total_commission: float = 0.0
    #: Realised P&L bucketed by New York trading day, for the daily loss limit.
    daily_realised: dict[object, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.starting_cash

    def position(self, instrument: Instrument) -> Position:
        return self.positions.setdefault(instrument.symbol, Position(instrument))

    def apply_fill(
        self,
        instrument: Instrument,
        side: Side,
        quantity: float,
        price: float,
        commission: float,
        timestamp: pd.Timestamp,
    ) -> float:
        position = self.position(instrument)
        realised = position.apply_fill(side, quantity, price, commission, timestamp)
        self.cash += realised
        self.total_commission += commission
        day = timestamp.tz_convert("America/New_York").normalize()
        self.daily_realised[day] = self.daily_realised.get(day, 0.0) + realised
        return realised

    def mark_to_market(self, prices: dict[str, float]) -> None:
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].mark(price)

    @property
    def unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl() for p in self.positions.values())

    @property
    def realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.unrealised_pnl

    @property
    def gross_exposure(self) -> float:
        return sum(p.exposure for p in self.positions.values())

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if not p.is_flat]

    def realised_on(self, timestamp: pd.Timestamp) -> float:
        day = timestamp.tz_convert("America/New_York").normalize()
        return self.daily_realised.get(day, 0.0)

    def record(self, timestamp: pd.Timestamp) -> EquityPoint:
        point = EquityPoint(
            timestamp=timestamp,
            equity=self.equity,
            cash=self.cash,
            unrealised=self.unrealised_pnl,
            exposure=self.gross_exposure,
        )
        self.equity_curve.append(point)
        return point

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype="float64", name="equity")
        return pd.Series(
            [p.equity for p in self.equity_curve],
            index=pd.DatetimeIndex([p.timestamp for p in self.equity_curve]),
            name="equity",
        )
