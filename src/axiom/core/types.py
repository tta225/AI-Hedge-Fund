"""Core domain types shared by every AXIOM subsystem.

Numeric policy
--------------
Market data and price arithmetic use ``float`` (float64) so the whole pipeline
stays vectorisable through NumPy/pandas. Prices are snapped to an instrument's
tick grid via :meth:`Instrument.round_to_tick` at every boundary where a price
becomes an order or a fill, which bounds representation error well below one
tick. Cash ledger arithmetic is documented in ``docs/ARCHITECTURE.md``; moving
the ledger to fixed-point is tracked as a known limitation, not an oversight.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum, IntEnum
from typing import Self


class AssetClass(str, Enum):
    FUTURES = "futures"
    FX = "fx"
    EQUITY = "equity"
    CRYPTO = "crypto"
    OPTION = "option"
    INDEX = "index"


class Direction(IntEnum):
    """Directional bias. Signed so it can multiply a price delta directly."""

    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1

    @property
    def opposite(self) -> Direction:
        return Direction(-int(self))

    def __str__(self) -> str:
        return self.name.lower()


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @classmethod
    def from_direction(cls, direction: Direction) -> Side:
        if direction is Direction.NEUTRAL:
            raise ValueError("cannot derive a Side from Direction.NEUTRAL")
        return cls.BUY if direction is Direction.BULLISH else cls.SELL


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    @property
    def is_open(self) -> bool:
        return not self.is_terminal


class TradingMode(str, Enum):
    """Master operating mode. Ordered by increasing consequence."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

    @property
    def touches_real_money(self) -> bool:
        return self is TradingMode.LIVE


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable contract or symbol.

    ``point_value`` is the cash value of one full point of price movement for
    one unit (contract/share/lot). ``tick_value`` is derived, so a futures
    contract only needs ``tick_size`` and ``point_value`` to price correctly.
    """

    symbol: str
    asset_class: AssetClass
    exchange: str = ""
    currency: str = "USD"
    tick_size: float = 0.01
    point_value: float = 1.0
    description: str = ""
    #: IANA timezone whose calendar defines this instrument's trading day.
    session_tz: str = "America/New_York"

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise ValueError(f"{self.symbol}: tick_size must be positive")
        if self.point_value <= 0:
            raise ValueError(f"{self.symbol}: point_value must be positive")

    @property
    def tick_value(self) -> float:
        """Cash value of a one-tick move in one unit."""
        return self.tick_size * self.point_value

    def round_to_tick(self, price: float) -> float:
        """Snap ``price`` onto this instrument's tick grid."""
        if not math.isfinite(price):
            raise ValueError(f"{self.symbol}: cannot round non-finite price {price!r}")
        ticks = round(price / self.tick_size)
        # Re-round to kill float dust from the division (e.g. 0.1 + 0.2 cases).
        return round(ticks * self.tick_size, 10)

    def ticks_between(self, a: float, b: float) -> float:
        """Signed distance from ``a`` to ``b`` measured in ticks."""
        return (b - a) / self.tick_size

    def cash_pnl(self, entry: float, exit_: float, quantity: float, side: Side) -> float:
        """Cash P&L of closing ``quantity`` units opened at ``entry`` on ``side``."""
        return (exit_ - entry) * side.sign * quantity * self.point_value

    def __str__(self) -> str:
        return self.symbol


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar. ``timestamp`` is the bar's OPEN time, tz-aware UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Bar.timestamp must be timezone-aware (UTC)")
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(
                f"inconsistent bar at {self.timestamp.isoformat()}: "
                f"O={self.open} H={self.high} L={self.low} C={self.close}"
            )

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def midpoint(self) -> float:
        """Bar equilibrium — the 50% level ICT treats as consequent encroachment."""
        return (self.high + self.low) / 2.0

    @property
    def direction(self) -> Direction:
        if self.close > self.open:
            return Direction.BULLISH
        if self.close < self.open:
            return Direction.BEARISH
        return Direction.NEUTRAL

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @classmethod
    def from_mapping(cls, ts: datetime, row: dict[str, float]) -> Self:
        return cls(
            timestamp=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )


# --- Common instrument definitions ------------------------------------------
# Contract specs change; treat these as sensible defaults, not gospel. Anything
# routed to a live venue must be reconciled against that venue's contract spec.

ES = Instrument("ES", AssetClass.FUTURES, "CME", tick_size=0.25, point_value=50.0,
                description="E-mini S&P 500")
NQ = Instrument("NQ", AssetClass.FUTURES, "CME", tick_size=0.25, point_value=20.0,
                description="E-mini Nasdaq-100")
YM = Instrument("YM", AssetClass.FUTURES, "CBOT", tick_size=1.0, point_value=5.0,
                description="E-mini Dow")
CL = Instrument("CL", AssetClass.FUTURES, "NYMEX", tick_size=0.01, point_value=1000.0,
                description="Crude Oil")
GC = Instrument("GC", AssetClass.FUTURES, "COMEX", tick_size=0.10, point_value=100.0,
                description="Gold")

EURUSD = Instrument("EURUSD", AssetClass.FX, "FX", tick_size=0.00001, point_value=100_000.0,
                    description="Euro / US Dollar")
GBPUSD = Instrument("GBPUSD", AssetClass.FX, "FX", tick_size=0.00001, point_value=100_000.0,
                    description="British Pound / US Dollar")

SPY = Instrument("SPY", AssetClass.EQUITY, "ARCA", tick_size=0.01, point_value=1.0,
                 description="SPDR S&P 500 ETF")
QQQ = Instrument("QQQ", AssetClass.EQUITY, "NASDAQ", tick_size=0.01, point_value=1.0,
                 description="Invesco QQQ Trust")

# Crypto trades continuously, so its "trading day" is a convention rather than
# a fact. UTC is used because that is the boundary the venues report on; any
# ICT session concept applied here inherits that arbitrariness and should be
# read with suspicion. See docs/REAL_DATA_FINDINGS.md.
BTCUSD = Instrument("BTC-USD", AssetClass.CRYPTO, "COINBASE", tick_size=0.01,
                    point_value=1.0, session_tz="UTC", description="Bitcoin / US Dollar")
ETHUSD = Instrument("ETH-USD", AssetClass.CRYPTO, "COINBASE", tick_size=0.01,
                    point_value=1.0, session_tz="UTC", description="Ether / US Dollar")
SOLUSD = Instrument("SOL-USD", AssetClass.CRYPTO, "COINBASE", tick_size=0.01,
                    point_value=1.0, session_tz="UTC", description="Solana / US Dollar")

#: Registry of built-in instruments, keyed by symbol.
INSTRUMENTS: dict[str, Instrument] = {
    i.symbol: i
    for i in (ES, NQ, YM, CL, GC, EURUSD, GBPUSD, SPY, QQQ, BTCUSD, ETHUSD, SOLUSD)
}

#: Quote currencies that identify a crypto pair written as ``BASE-QUOTE``.
_CRYPTO_QUOTES = ("USD", "USDT", "USDC", "EUR", "GBP")


#: CME month codes, in calendar order: Jan→F, Feb→G, … Dec→Z.
_MONTH_CODES = "FGHJKMNQUVXZ"

#: A dated futures contract: root, month code, one or two year digits. ``ESH6``
#: is March 2026 E-mini S&P; ``GCJ5`` is April 2025 Gold.
_CONTRACT_PATTERN = re.compile(rf"^([A-Z]{{1,3}})([{_MONTH_CODES}])(\d{{1,2}})$")


def parse_contract(symbol: str) -> tuple[str, str, str] | None:
    """Split a dated futures ticker into ``(root, month_code, year)``.

    Returns None for anything that is not shaped like a contract. US equity
    tickers are alphabetic, so the trailing digit does most of the work here;
    the month code carries the rest.
    """
    match = _CONTRACT_PATTERN.match(symbol.upper())
    return match.groups() if match else None  # type: ignore[return-value]


def get_instrument(symbol: str) -> Instrument:
    """Look up a built-in instrument, falling back to a generic spec.

    An unknown ``BASE-QUOTE`` symbol is treated as crypto, and a symbol shaped
    like a dated futures contract is treated as a future.

    Getting the asset class wrong is not cosmetic, and the futures case is the
    expensive one. ``ESH6`` classified as an equity picks up the equity default
    of ``point_value=1.0``, but an E-mini point is **$50** — so a risk budget
    computed against it would size the position fifty times too large, silently
    and with no error anywhere. It also decides which providers are asked, and
    an equity provider handed a contract ticker returns either nothing or the
    wrong instrument.

    So a recognised contract inherits its **root's** specification — tick size,
    point value, exchange, session timezone — rather than a generic one.
    ``ESH6`` sizes exactly as ``ES`` does, which is the point: the March
    contract and the June contract are the same instrument on different dates.
    """
    key = symbol.upper()
    if key in INSTRUMENTS:
        return INSTRUMENTS[key]

    base, sep, quote = key.partition("-")
    if sep and base and quote in _CRYPTO_QUOTES:
        return Instrument(key, AssetClass.CRYPTO, tick_size=0.01, point_value=1.0,
                          session_tz="UTC")

    contract = parse_contract(key)
    if contract is not None:
        root, month, year = contract
        parent = INSTRUMENTS.get(root)
        if parent is not None and parent.asset_class is AssetClass.FUTURES:
            return replace(
                parent,
                symbol=key,
                description=f"{parent.description} ({month}{year} contract)",
            )
        # An unknown root is still recognisably a future. Generic specs are a
        # guess, but 'futures with the wrong point value' at least routes to a
        # provider that carries futures, where 'equity' does not.
        return Instrument(key, AssetClass.FUTURES, tick_size=0.01, point_value=1.0,
                          description=f"unrecognised futures contract ({root})")

    return Instrument(key, AssetClass.EQUITY, tick_size=0.01, point_value=1.0)
