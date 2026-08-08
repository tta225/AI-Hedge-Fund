"""NinjaTrader execution via the Automated Trading Interface file bridge.

NinjaTrader does not offer a REST API. The supported way for an outside program
to trade through it is the **ATI file interface**: you drop semicolon-delimited
*Order Instruction Files* into an ``incoming`` folder, NinjaTrader picks them up
"the instant they are written", and it reports state back by writing files into
an ``outgoing`` folder. There is also a DLL interface, but it is Win32-only and
buys nothing this does not already have.

Three properties of that design drive everything in this module.

**It is fire-and-forget.** Writing an OIF is not a request; nothing acknowledges
it. If NinjaTrader is closed, if ATI is switched off in the platform, if the
folder path is wrong by one character — the write succeeds and the order simply
never exists. A venue that reported SUBMITTED on a successful write would be
claiming something it has no evidence for, so :meth:`submit` writes the file and
then *waits for the outgoing folder to confirm the order*. No confirmation, no
SUBMITTED.

**The state files are the only feedback channel, and their names are not
verifiable from here.** NinjaTrader's documentation host is unreachable from
this environment, so the outgoing filename patterns below come from secondary
sources. That is a problem of a specific shape: a wrong pattern does not raise,
it silently reports "no fills, ever" — which reads as a quiet market rather than
a broken adapter. So the patterns are configuration with documented defaults,
:meth:`check` reports what it actually found on disk rather than what it hoped
for, and :meth:`discover` prints the real filenames so a mismatch is a two-minute
fix instead of an afternoon.

**It is Windows-only.** ATI is a Windows platform feature. On macOS this class
constructs, validates, and refuses to trade, saying why — see
:meth:`is_available`. For a Mac, Tradovate is the working path to the same
CME order flow; :mod:`axiom.execution.tradovate` implements it against a real
REST API with real acknowledgements.

This venue is live by construction. There is no NinjaTrader "paper folder" —
the same bridge trades a simulation account (``Sim101``) or a funded one, and
which is which is a string in a config file. `is_live` is therefore True
unconditionally, so the router demands live mode plus a confirmation phrase
before it will route anything here, whatever account is named.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from axiom.core.types import OrderStatus, OrderType, Side, TimeInForce
from axiom.execution.base import ExecutionError, ExecutionVenue, Fill, Order

__all__ = ["NinjaTraderPosition", "NinjaTraderVenue"]

#: Where NinjaTrader keeps the ATI folders, relative to the user's documents.
DEFAULT_ATI_ROOT = Path.home() / "Documents" / "NinjaTrader 8"

#: NinjaTrader only reads files matching ``oif*.txt``. Anything else is ignored
#: silently, which is the same failure as not writing the file at all.
OIF_PREFIX = "oif"

#: Position file: ``<instrument> <exchange>_<account>_Position.txt``.
POSITION_PATTERN = "{instrument}_{account}_Position.txt"

#: Order state file: ``<order id>_<account>_OrderState.txt``.
ORDER_STATE_PATTERN = "{order_id}_{account}_OrderState.txt"

#: Account value file: ``<account>_<value type>.txt``.
ACCOUNT_PATTERN = "{account}_{field}.txt"

_ORDER_TYPES: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP: "STOPMARKET",
    OrderType.STOP_LIMIT: "STOPLIMIT",
}

_TIF: dict[TimeInForce, str] = {
    TimeInForce.DAY: "DAY",
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "GTC",
    TimeInForce.FOK: "GTC",
}

_STATUS_MAP: dict[str, OrderStatus] = {
    "PENDINGSUBMIT": OrderStatus.PENDING,
    "PENDING": OrderStatus.PENDING,
    "SUBMITTED": OrderStatus.SUBMITTED,
    "ACCEPTED": OrderStatus.SUBMITTED,
    "WORKING": OrderStatus.SUBMITTED,
    "PARTFILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "UNKNOWN": OrderStatus.PENDING,
}

_TERMINAL = {"FILLED", "CANCELLED", "CANCELED", "REJECTED"}


@dataclass(slots=True)
class NinjaTraderPosition:
    """A position as NinjaTrader reports it in the outgoing folder."""

    symbol: str
    quantity: float
    average_price: float

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass(slots=True)
class _Bridge:
    """The two folders that make up the interface, and the checks on them."""

    root: Path
    incoming: Path = field(init=False)
    outgoing: Path = field(init=False)

    def __post_init__(self) -> None:
        self.incoming = self.root / "incoming"
        self.outgoing = self.root / "outgoing"

    def problems(self) -> list[str]:
        """Everything wrong with the bridge, rather than the first thing wrong.

        Reported together on purpose. Fixing a path, re-running, and finding the
        next path also wrong is a slow way to learn two facts that were both
        knowable at once.
        """
        found: list[str] = []
        if not self.root.is_dir():
            found.append(
                f"NinjaTrader folder not found at {self.root}. Set ati_root to the "
                f"folder that contains 'incoming' and 'outgoing'."
            )
            return found
        if not self.incoming.is_dir():
            found.append(f"no 'incoming' folder at {self.incoming}")
        elif not os.access(self.incoming, os.W_OK):
            found.append(f"'incoming' at {self.incoming} is not writable")
        if not self.outgoing.is_dir():
            found.append(
                f"no 'outgoing' folder at {self.outgoing}. Without it nothing can "
                f"be confirmed — orders would be sent blind."
            )
        return found


class NinjaTraderVenue(ExecutionVenue):
    """Routes orders to NinjaTrader through the ATI file bridge.

    Args:
        account: NinjaTrader account name, e.g. ``Sim101``. Must not contain
            spaces — the OIF format is semicolon-delimited and NinjaTrader
            rejects spaced account names.
        ati_root: folder containing ``incoming`` and ``outgoing``.
        allow_live: required to construct. This venue is always live-capable,
            so there is no safe default and no host string to infer one from.
        confirm_timeout: how long :meth:`submit` waits for the outgoing folder
            to confirm an order before reporting the write unconfirmed.
    """

    name = "ninjatrader"
    #: Unconditional. A NinjaTrader account is Sim or funded by configuration
    #: alone, with no structural difference the platform can see, so the safe
    #: reading is the pessimistic one.
    is_live = True

    def __init__(
        self,
        *,
        account: str = "Sim101",
        ati_root: Path | str | None = None,
        allow_live: bool = False,
        confirm_timeout: float = 5.0,
        poll_interval: float = 0.25,
    ) -> None:
        if not allow_live:
            raise PermissionError(
                f"{self.name}: this venue trades a real NinjaTrader account and "
                f"has no paper mode of its own — a Sim account and a funded one "
                f"differ only by the name in this config. Pass allow_live=True "
                f"deliberately. The router additionally requires live mode and a "
                f"confirmation phrase."
            )
        if " " in account:
            raise ValueError(
                f"{self.name}: account {account!r} contains a space. NinjaTrader "
                f"rejects spaced account names in OIF files — use 'Sim101', not "
                f"'Sim 101'."
            )

        self.account = account
        self.bridge = _Bridge(Path(ati_root) if ati_root else DEFAULT_ATI_ROOT)
        self.confirm_timeout = confirm_timeout
        self.poll_interval = poll_interval

        self._orders: dict[int, Order] = {}
        #: Our order id → the id written into the OIF, which NinjaTrader echoes
        #: back in the outgoing filename.
        self._ati_ids: dict[int, str] = {}
        self._sequence = 0

    # --- availability -------------------------------------------------------

    def is_available(self) -> bool:
        return platform.system() == "Windows" and not self.bridge.problems()

    def unavailable_reason(self) -> str:
        """Why this venue cannot trade, in the order the user would fix it."""
        if platform.system() != "Windows":
            return (
                f"NinjaTrader's Automated Trading Interface is a Windows-only "
                f"feature and this machine is {platform.system()}. For CME order "
                f"flow from a Mac, use the Tradovate venue instead — same "
                f"exchanges, a real REST API, and it acknowledges orders."
            )
        problems = self.bridge.problems()
        return "; ".join(problems) if problems else ""

    def _require_available(self) -> None:
        reason = self.unavailable_reason()
        if reason:
            raise ExecutionError(f"{self.name}: {reason}")

    # --- the file bridge ----------------------------------------------------

    def _next_ati_id(self) -> str:
        self._sequence += 1
        return f"AXIOM{int(time.time())}{self._sequence:04d}"

    def _write_oif(self, command: str) -> Path:
        """Write one instruction, atomically.

        NinjaTrader watches the folder and reads a file the moment it appears.
        Writing in place therefore races the reader: a partially flushed line
        gets consumed as a whole instruction, and a truncated PLACE is not a
        rejected order — it is an order for a quantity nobody chose. Writing to
        a name NinjaTrader ignores and then renaming makes the file appear
        complete or not at all.
        """
        self._require_available()
        stamp = f"{time.time_ns()}"
        scratch = self.bridge.incoming / f".axiom-{stamp}.tmp"
        final = self.bridge.incoming / f"{OIF_PREFIX}{stamp}.txt"
        scratch.write_text(command + "\n", encoding="ascii")
        scratch.replace(final)
        return final

    def _order_state(self, ati_id: str) -> tuple[str, float, float] | None:
        """Read one order's state file. None when NinjaTrader has not written it.

        Absence is the normal case for a few hundred milliseconds after a write
        and a permanent case when the bridge is misconfigured, which is why the
        caller distinguishes the two by waiting.
        """
        path = self.bridge.outgoing / ORDER_STATE_PATTERN.format(
            order_id=ati_id, account=self.account
        )
        try:
            raw = path.read_text(encoding="ascii", errors="replace").strip()
        except (OSError, FileNotFoundError):
            return None
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(";")]
        status = parts[0].upper() if parts else ""
        filled = _to_float(parts[1]) if len(parts) > 1 else 0.0
        price = _to_float(parts[2]) if len(parts) > 2 else 0.0
        return status, filled, price

    # --- venue contract -----------------------------------------------------

    def submit(self, order: Order, timestamp: pd.Timestamp) -> Order:
        """Write a PLACE instruction and wait for NinjaTrader to confirm it.

        The waiting is the point. A file write that succeeds tells you the disk
        works, not that an order exists — NinjaTrader could be closed, or ATI
        switched off in the platform, and the write looks identical. Reporting
        SUBMITTED on the strength of a successful write would put an order in
        the platform's book that the broker has never heard of, and the first
        symptom would be a position that never fills and a stop that never
        exists.
        """
        try:
            self._require_available()
        except ExecutionError as exc:
            return order.reject(str(exc))

        ati_id = self._next_ati_id()
        command = ";".join([
            "PLACE",
            self.account,
            order.instrument.symbol,
            "BUY" if order.side is Side.BUY else "SELL",
            str(int(order.quantity)),
            _ORDER_TYPES.get(order.order_type, "MARKET"),
            _fmt(order.limit_price),
            _fmt(order.stop_price),
            _TIF.get(order.time_in_force, "DAY"),
            "",  # OCO id — brackets are handled below, not here
            ati_id,
            order.strategy or "",
            "",  # strategy id
        ])

        try:
            self._write_oif(command)
        except OSError as exc:
            return order.reject(f"{self.name}: could not write instruction: {exc}")

        state = self._await_confirmation(ati_id)
        if state is None:
            return order.reject(
                f"{self.name}: wrote the order but NinjaTrader never acknowledged "
                f"it within {self.confirm_timeout:g}s. Nothing was confirmed, so "
                f"nothing is assumed. Check that NinjaTrader is running, that "
                f"'Enable ATI' is ticked under Tools → Options → Automated "
                f"trading interface, and that {self.bridge.root} is the folder it "
                f"is actually using. Run `check()` for what was found on disk."
            )

        status, filled, _price = state
        if status in _TERMINAL and status != "FILLED" and filled == 0:
            return order.reject(f"{self.name}: NinjaTrader reported {status}")

        order.created_at = timestamp
        order.status = _STATUS_MAP.get(status, OrderStatus.SUBMITTED)
        self._orders[order.order_id] = order
        self._ati_ids[order.order_id] = ati_id

        # Protection second, and only because the interface offers no way to
        # send it first. OIF has no bracket field: NinjaTrader attaches stops
        # and targets through an ATM strategy template configured inside the
        # platform, which this bridge cannot create. So there is a real window
        # here between entry and stop, unlike the Tradovate venue where both
        # travel in one OSO request. Anyone trading gap-prone instruments
        # through this venue should attach protection with an ATM template in
        # NinjaTrader rather than relying on these follow-up orders.
        if order.stop_loss is not None or order.take_profit is not None:
            self._write_protection(order, ati_id)
        return order

    def _await_confirmation(self, ati_id: str) -> tuple[str, float, float] | None:
        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            state = self._order_state(ati_id)
            if state is not None:
                return state
            time.sleep(self.poll_interval)
        return self._order_state(ati_id)

    def _write_protection(self, order: Order, entry_id: str) -> None:
        """Send the stop and target as an OCO pair against the entry."""
        opposite = "SELL" if order.side is Side.BUY else "BUY"
        oco = f"{entry_id}OCO"
        for price, order_type, field_name in (
            (order.stop_loss, "STOPMARKET", "stop"),
            (order.take_profit, "LIMIT", "target"),
        ):
            if price is None:
                continue
            self._write_oif(";".join([
                "PLACE",
                self.account,
                order.instrument.symbol,
                opposite,
                str(int(order.quantity)),
                order_type,
                _fmt(price) if order_type == "LIMIT" else "",
                _fmt(price) if order_type == "STOPMARKET" else "",
                "GTC",
                oco,
                f"{entry_id}{field_name}",
                order.strategy or "",
                "",
            ]))

    def cancel(self, order: Order) -> Order:
        """Write a CANCEL instruction, and only believe it once confirmed.

        Same reasoning as submit, mirrored: an unacknowledged cancel that gets
        recorded as success leaves a working order at the broker that this side
        has stopped tracking. The status is left alone unless NinjaTrader says
        the order is actually terminal.
        """
        ati_id = self._ati_ids.get(order.order_id)
        if ati_id is None:
            return order

        self._write_oif(
            ";".join(["CANCEL", self.account, "", "", "", "", "", "", "", "", ati_id, "", ""])
        )

        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            state = self._order_state(ati_id)
            if state and state[0] in _TERMINAL:
                order.status = _STATUS_MAP.get(state[0], OrderStatus.CANCELLED)
                return order
            time.sleep(self.poll_interval)

        raise ExecutionError(
            f"{self.name}: cancel for order {order.order_id} was written but "
            f"NinjaTrader did not report it terminal within "
            f"{self.confirm_timeout:g}s. The order may still be working — this "
            f"side is not marking it cancelled on a guess. Check the platform."
        )

    def flatten_everything(self) -> None:
        """Close every position and cancel every order, across all accounts.

        The panic button. Deliberately not wired to anything automatic.
        """
        self._write_oif("FLATTENEVERYTHING;;;;;;;;;;;;")

    def working_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_open]

    def poll(self, timestamp: pd.Timestamp) -> list[Fill]:
        """Read every tracked order's state file and emit fills seen since last."""
        fills: list[Fill] = []
        for order_id, ati_id in list(self._ati_ids.items()):
            order = self._orders.get(order_id)
            if order is None or not order.is_open:
                continue

            state = self._order_state(ati_id)
            if state is None:
                continue
            status, filled, price = state

            new_quantity = filled - order.filled_quantity
            if new_quantity > 0 and price > 0:
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        instrument=order.instrument,
                        side=order.side,
                        quantity=new_quantity,
                        price=price,
                        commission=0.0,  # NinjaTrader settles separately
                        timestamp=timestamp,
                        slippage=(
                            price - order.reference_price
                            if order.reference_price
                            else 0.0
                        ),
                        strategy=order.strategy,
                    )
                )
                order.filled_quantity = filled
                order.average_fill_price = price

            order.status = _STATUS_MAP.get(status, order.status)
            if status in _TERMINAL:
                self._ati_ids.pop(order_id, None)
        return fills

    # --- account state ------------------------------------------------------

    def positions(self) -> dict[str, NinjaTraderPosition]:
        """Every position NinjaTrader is reporting for this account.

        Reads the folder rather than asking about instruments we know, so a
        position opened by hand in the platform — or left over from a previous
        session — shows up here and lands in reconciliation as broker-only.
        Only asking about our own orders would hide exactly the positions that
        most need to be seen.
        """
        out: dict[str, NinjaTraderPosition] = {}
        if not self.bridge.outgoing.is_dir():
            return out

        suffix = f"_{self.account}_Position.txt"
        for path in self.bridge.outgoing.glob(f"*{suffix}"):
            symbol = path.name[: -len(suffix)].split(" ")[0]
            try:
                raw = path.read_text(encoding="ascii", errors="replace").strip()
            except OSError:
                continue
            parts = [p.strip() for p in raw.split(";")]
            if len(parts) < 3:
                continue
            market_position = parts[0].upper()
            quantity = _to_float(parts[1])
            if market_position == "FLAT" or quantity == 0:
                continue  # a flat row is not a position
            out[symbol] = NinjaTraderPosition(
                symbol=symbol,
                quantity=-quantity if market_position == "SHORT" else quantity,
                average_price=_to_float(parts[2]),
            )
        return out

    def account_value(self, field_name: str = "CashValue") -> float | None:
        path = self.bridge.outgoing / ACCOUNT_PATTERN.format(
            account=self.account, field=field_name
        )
        try:
            return _to_float(path.read_text(encoding="ascii", errors="replace").strip())
        except OSError:
            return None

    def reconcile(self, portfolio: Any, timestamp: pd.Timestamp | None = None) -> Any:
        """Compare the platform's book against NinjaTrader's.

        Shares `ReconciliationReport` with the other venues deliberately: an
        operator reading two brokers' reconciliations should not have to learn
        two report formats to spot the same class of problem.
        """
        from axiom.execution.alpaca_paper import ReconciliationReport

        broker = self.positions()
        report = ReconciliationReport(checked_at=timestamp or pd.Timestamp.now("UTC"))
        ours = {p.instrument.symbol: float(p.quantity) for p in portfolio.open_positions}

        for symbol, quantity in ours.items():
            theirs = broker.get(symbol)
            if theirs is None:
                report.platform_only[symbol] = quantity
            elif abs(theirs.quantity - quantity) > 1e-9:
                report.mismatched[symbol] = (quantity, theirs.quantity)

        for symbol, position in broker.items():
            if symbol not in ours:
                report.broker_only[symbol] = position.quantity

        return report

    # --- diagnostics --------------------------------------------------------

    def check(self) -> str:
        """One-line status for the CLI and console."""
        reason = self.unavailable_reason()
        if reason:
            return f"✗ {reason}"
        seen = len(list(self.bridge.outgoing.iterdir()))
        return (
            f"✓ LIVE — REAL MONEY: NinjaTrader bridge at {self.bridge.root}, "
            f"account {self.account!r}, {len(self.positions())} open position(s), "
            f"{seen} file(s) in outgoing"
        )

    def discover(self) -> list[str]:
        """Every filename NinjaTrader has written, unfiltered.

        The state-file naming below was taken from secondary sources, because
        NinjaTrader's own documentation is unreachable from this environment.
        A wrong pattern fails silently — it looks like a market with no fills —
        so this exists to make the real names visible in one command instead of
        leaving that failure to be inferred.
        """
        if not self.bridge.outgoing.is_dir():
            return []
        return sorted(p.name for p in self.bridge.outgoing.iterdir() if p.is_file())


def _fmt(value: float | None) -> str:
    """Prices in OIF fields; empty rather than '0' when absent.

    A literal 0 in the limit-price field is a limit order at zero, not an
    omitted field.
    """
    return "" if value is None else f"{value:g}"


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0
