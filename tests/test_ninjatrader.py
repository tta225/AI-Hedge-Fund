"""NinjaTrader ATI file bridge. Offline — a temporary folder plays NinjaTrader.

The bridge is fire-and-forget, and that shapes what is worth testing. There is
no HTTP status code, no response body, no exception on failure: writing an
instruction into a folder nobody is watching looks exactly like writing one into
a folder NinjaTrader is watching. Every meaningful bug in this module is a
variant of *claiming something happened because a file write returned*.

So the tests that carry weight are:

* **An unacknowledged order is not SUBMITTED.** If NinjaTrader is closed, the
  write still succeeds. A venue that reports SUBMITTED there puts an order in
  the platform's book that no broker has heard of.
* **An unacknowledged cancel is not CANCELLED.** The mirror image, and the
  worse one — it leaves a working order that this side has stopped tracking.
* **The OIF is never read half-written.** NinjaTrader consumes a file the
  instant it appears, so a torn write is not a rejected order, it is an order
  for a quantity nobody chose.
* **A flat row is not a position.** NinjaTrader leaves position files on disk
  after a close; counting them would make reconciliation dirty forever.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from axiom.core.types import OrderStatus, OrderType, Side, get_instrument
from axiom.execution.base import ExecutionError, Order
from axiom.execution.ninjatrader import (
    OIF_PREFIX,
    NinjaTraderVenue,
)

NOW = pd.Timestamp("2026-03-02 15:00", tz="UTC")
ES = get_instrument("ES")


@pytest.fixture(autouse=True)
def _pretend_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """ATI is Windows-only, and CI is not Windows.

    Without this every test would pass by hitting the platform refusal, which
    would assert nothing about the bridge itself. `TestPlatformSupport` opts
    back out to check the refusal deliberately.
    """
    monkeypatch.setattr(platform, "system", lambda: "Windows")


@pytest.fixture
def ati(tmp_path: Path) -> Path:
    (tmp_path / "incoming").mkdir()
    (tmp_path / "outgoing").mkdir()
    return tmp_path


def _venue(ati: Path, **kwargs: Any) -> NinjaTraderVenue:
    kwargs.setdefault("allow_live", True)
    kwargs.setdefault("confirm_timeout", 0.05)
    kwargs.setdefault("poll_interval", 0.01)
    return NinjaTraderVenue(ati_root=ati, **kwargs)


def _order(**kwargs: Any) -> Order:
    defaults: dict[str, Any] = {
        "instrument": ES,
        "side": Side.BUY,
        "quantity": 2.0,
        "order_type": OrderType.MARKET,
        "strategy": "test",
    }
    return Order(**{**defaults, **kwargs})


class _Acknowledging(NinjaTraderVenue):
    """A venue whose orders are acknowledged the moment they are written.

    Stands in for a running NinjaTrader. Without it every submit would take the
    full confirmation timeout and then reject, so nothing past the write could
    be tested at all.
    """

    def __init__(self, *args: Any, state: str = "WORKING;0;0", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state = state

    def _write_oif(self, command: str) -> Path:
        path = super()._write_oif(command)
        fields = command.split(";")
        if fields[0] == "PLACE" and fields[10]:
            (self.bridge.outgoing / f"{fields[10]}_{self.account}_OrderState.txt").write_text(
                self.state
            )
        return path


class TestConstruction:
    def test_it_refuses_to_construct_without_allow_live(self, ati: Path) -> None:
        """There is no NinjaTrader paper mode — Sim and funded differ by a
        string in a config file, so there is no safe default to fall back on."""
        with pytest.raises(PermissionError, match="allow_live=True"):
            NinjaTraderVenue(ati_root=ati)

    def test_it_is_always_live(self, ati: Path) -> None:
        assert _venue(ati).is_live is True

    def test_the_router_refuses_it_outside_confirmed_live_mode(self, ati: Path) -> None:
        from axiom.core.types import TradingMode
        from axiom.execution.router import OrderRouter
        from axiom.risk.manager import RiskManager

        with pytest.raises(PermissionError, match="live venue"):
            OrderRouter(venue=_venue(ati), risk=RiskManager(), mode=TradingMode.PAPER)

    def test_a_spaced_account_name_is_rejected_at_construction(self, ati: Path) -> None:
        """NinjaTrader silently ignores OIFs with spaced account names. Caught
        here rather than discovered as orders that never appear."""
        with pytest.raises(ValueError, match="contains a space"):
            _venue(ati, account="Sim 101")


class TestPlatformSupport:
    def test_a_mac_is_told_what_to_use_instead(
        self, ati: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        venue = _venue(ati)
        assert not venue.is_available()
        reason = venue.unavailable_reason()
        assert "Windows-only" in reason
        assert "Tradovate" in reason  # a dead end with no next step is not help

    def test_orders_are_rejected_on_an_unsupported_platform(
        self, ati: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        order = _venue(ati).submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED

    def test_nothing_is_written_on_an_unsupported_platform(
        self, ati: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        _venue(ati).submit(_order(), NOW)
        assert list((ati / "incoming").iterdir()) == []


class TestBridgeValidation:
    def test_a_missing_root_is_reported_with_the_path_it_looked_at(
        self, tmp_path: Path
    ) -> None:
        venue = _venue(tmp_path / "nowhere")
        assert "nowhere" in venue.unavailable_reason()

    def test_a_missing_outgoing_folder_is_fatal_not_cosmetic(
        self, tmp_path: Path
    ) -> None:
        """Without it nothing can ever be confirmed, so orders would go blind."""
        (tmp_path / "incoming").mkdir()
        assert "outgoing" in _venue(tmp_path).unavailable_reason()

    def test_both_missing_folders_are_reported_together(self, tmp_path: Path) -> None:
        """Fixing one, re-running, and finding the next also wrong is a slow way
        to learn two things that were both knowable at once."""
        reason = _venue(tmp_path).unavailable_reason()
        assert "incoming" in reason and "outgoing" in reason

    def test_check_reports_the_failure_rather_than_a_tick(self, tmp_path: Path) -> None:
        assert _venue(tmp_path).check().startswith("✗")


class TestSubmit:
    def test_an_unacknowledged_order_is_rejected_not_submitted(
        self, ati: Path
    ) -> None:
        """The failure this whole module is arranged around.

        NinjaTrader closed, or ATI unticked in the platform: the write succeeds
        identically. Reporting SUBMITTED here would put an order in the book
        that no broker has ever seen, and the first symptom would be a position
        that never fills and a stop that never exists.
        """
        order = _venue(ati).submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED
        assert "never acknowledged" in order.reject_reason

    def test_the_rejection_says_how_to_diagnose_it(self, ati: Path) -> None:
        reason = _venue(ati).submit(_order(), NOW).reject_reason
        assert "Enable ATI" in reason
        assert str(ati) in reason

    def test_an_acknowledged_order_is_submitted(self, ati: Path) -> None:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.SUBMITTED
        assert order.created_at == NOW

    def test_the_instruction_has_the_documented_field_order(self, ati: Path) -> None:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        venue.submit(_order(), NOW)
        written = next((ati / "incoming").glob(f"{OIF_PREFIX}*.txt")).read_text()
        fields = written.strip().split(";")
        assert fields[0] == "PLACE"
        assert fields[1] == "Sim101"
        assert fields[2] == "ES"
        assert fields[3] == "BUY"
        assert fields[4] == "2"
        assert fields[5] == "MARKET"
        assert fields[8] == "DAY"

    def test_an_absent_limit_price_is_empty_not_zero(self, ati: Path) -> None:
        """A literal 0 in the limit field is an order at zero, not an omission."""
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        venue.submit(_order(), NOW)
        fields = next((ati / "incoming").glob("oif*.txt")).read_text().split(";")
        assert fields[6] == "" and fields[7] == ""

    def test_a_limit_price_is_written(self, ati: Path) -> None:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        venue.submit(_order(order_type=OrderType.LIMIT, limit_price=5000.25), NOW)
        fields = next((ati / "incoming").glob("oif*.txt")).read_text().split(";")
        assert fields[5] == "LIMIT" and fields[6] == "5000.25"

    def test_the_file_is_only_named_oif_once_it_is_complete(self, ati: Path) -> None:
        """NinjaTrader reads a matching file the instant it appears, so writing
        in place races the reader and a torn PLACE is an order for a quantity
        nobody chose. The venue writes to an ignored name and renames."""
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        venue.submit(_order(), NOW)
        names = [p.name for p in (ati / "incoming").iterdir()]
        assert all(not n.endswith(".tmp") for n in names)
        assert all(n.startswith(OIF_PREFIX) for n in names)

    def test_a_rejection_from_ninjatrader_is_reported(self, ati: Path) -> None:
        venue = _Acknowledging(
            ati_root=ati,
            allow_live=True,
            confirm_timeout=0.05,
            poll_interval=0.01,
            state="REJECTED;0;0",
        )
        order = venue.submit(_order(), NOW)
        assert order.status is OrderStatus.REJECTED

    def test_protection_orders_are_written_as_an_oco_pair(self, ati: Path) -> None:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        venue.submit(_order(stop_loss=4990.0, take_profit=5050.0), NOW)
        instructions = [
            p.read_text().strip().split(";")
            for p in sorted((ati / "incoming").iterdir())
        ]
        protection = [f for f in instructions if f[9]]  # OCO id populated
        assert len(protection) == 2
        assert {f[9] for f in protection} == {protection[0][9]}  # same OCO group
        assert all(f[3] == "SELL" for f in protection)  # opposite the long entry

    def test_a_stop_only_order_writes_one_protection_leg(self, ati: Path) -> None:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        venue.submit(_order(stop_loss=4990.0), NOW)
        assert len(list((ati / "incoming").iterdir())) == 2  # entry + stop


class TestCancel:
    def _submitted(self, ati: Path) -> tuple[_Acknowledging, Order, str]:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        order = venue.submit(_order(), NOW)
        return venue, order, venue._ati_ids[order.order_id]

    def test_a_confirmed_cancel_is_recorded(self, ati: Path) -> None:
        venue, order, ati_id = self._submitted(ati)
        (ati / "outgoing" / f"{ati_id}_Sim101_OrderState.txt").write_text(
            "CANCELLED;0;0"
        )
        assert venue.cancel(order).status is OrderStatus.CANCELLED

    def test_an_unconfirmed_cancel_raises_rather_than_claiming_success(
        self, ati: Path
    ) -> None:
        """The order may still be working. Marking it cancelled on a guess
        leaves a live order nothing on this side is tracking."""
        venue, order, _ = self._submitted(ati)
        with pytest.raises(ExecutionError, match="may still be working"):
            venue.cancel(order)
        assert order.status is not OrderStatus.CANCELLED

    def test_a_cancel_that_races_a_fill_adopts_filled(self, ati: Path) -> None:
        venue, order, ati_id = self._submitted(ati)
        (ati / "outgoing" / f"{ati_id}_Sim101_OrderState.txt").write_text(
            "FILLED;2;5000.25"
        )
        assert venue.cancel(order).status is OrderStatus.FILLED

    def test_an_order_that_was_never_written_is_left_alone(self, ati: Path) -> None:
        venue = _venue(ati)
        order = _order()
        assert venue.cancel(order).status is order.status


class TestPolling:
    def _submitted(self, ati: Path) -> tuple[_Acknowledging, Order, str]:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        order = venue.submit(_order(), NOW)
        return venue, order, venue._ati_ids[order.order_id]

    def test_a_fill_is_emitted_once(self, ati: Path) -> None:
        venue, order, ati_id = self._submitted(ati)
        (ati / "outgoing" / f"{ati_id}_Sim101_OrderState.txt").write_text(
            "FILLED;2;5000.25"
        )
        fills = venue.poll(NOW)
        assert len(fills) == 1
        assert fills[0].quantity == 2.0 and fills[0].price == 5000.25
        assert order.status is OrderStatus.FILLED
        assert venue.poll(NOW) == []  # terminal orders stop being polled

    def test_a_partial_fill_emits_only_the_new_quantity(self, ati: Path) -> None:
        venue, order, ati_id = self._submitted(ati)
        state = ati / "outgoing" / f"{ati_id}_Sim101_OrderState.txt"
        state.write_text("PARTFILLED;1;5000.0")
        assert venue.poll(NOW)[0].quantity == 1.0
        assert order.status is OrderStatus.PARTIALLY_FILLED
        assert venue.poll(NOW) == []  # nothing new
        state.write_text("FILLED;2;5000.0")
        assert venue.poll(NOW)[0].quantity == 1.0  # only the increment

    def test_a_missing_state_file_is_not_treated_as_a_change(self, ati: Path) -> None:
        """Absent is unknown, not flat. Inventing a status from a missing file
        is how a working order gets quietly written off."""
        venue, order, ati_id = self._submitted(ati)
        (ati / "outgoing" / f"{ati_id}_Sim101_OrderState.txt").unlink()
        assert venue.poll(NOW) == []
        assert order.is_open

    def test_slippage_is_measured_against_the_reference_price(self, ati: Path) -> None:
        venue = _Acknowledging(
            ati_root=ati, allow_live=True, confirm_timeout=0.05, poll_interval=0.01
        )
        order = venue.submit(_order(reference_price=5000.0), NOW)
        ati_id = venue._ati_ids[order.order_id]
        (ati / "outgoing" / f"{ati_id}_Sim101_OrderState.txt").write_text(
            "FILLED;2;5001.0"
        )
        assert venue.poll(NOW)[0].slippage == pytest.approx(1.0)

    def test_a_malformed_state_file_does_not_crash_the_poll(self, ati: Path) -> None:
        venue, _, ati_id = self._submitted(ati)
        (ati / "outgoing" / f"{ati_id}_Sim101_OrderState.txt").write_text("garbage")
        assert venue.poll(NOW) == []


@dataclass
class _Position:
    instrument: Any
    quantity: float


@dataclass
class _Portfolio:
    open_positions: list[_Position]
    equity: float = 100_000.0


class TestPositions:
    def _write(self, ati: Path, name: str, body: str) -> None:
        (ati / "outgoing" / name).write_text(body)

    def test_a_long_position_is_read(self, ati: Path) -> None:
        self._write(ati, "ES 03-26_Sim101_Position.txt", "LONG;2;5000.25")
        positions = _venue(ati).positions()
        assert positions["ES"].quantity == 2.0
        assert positions["ES"].average_price == 5000.25

    def test_a_short_position_carries_a_negative_quantity(self, ati: Path) -> None:
        """NinjaTrader reports direction as a word and size as a positive
        number. Losing the word turns a short into a long."""
        self._write(ati, "ES 03-26_Sim101_Position.txt", "SHORT;3;5000.0")
        assert _venue(ati).positions()["ES"].quantity == -3.0
        assert _venue(ati).positions()["ES"].is_long is False

    def test_a_flat_row_is_not_a_position(self, ati: Path) -> None:
        """NinjaTrader leaves the file behind after a close. Counting it would
        make every reconciliation dirty forever."""
        self._write(ati, "ES 03-26_Sim101_Position.txt", "FLAT;0;0")
        assert _venue(ati).positions() == {}

    def test_another_accounts_positions_are_ignored(self, ati: Path) -> None:
        self._write(ati, "ES 03-26_Other_Position.txt", "LONG;9;5000.0")
        assert _venue(ati).positions() == {}

    def test_a_position_opened_by_hand_shows_up(self, ati: Path) -> None:
        """The folder is read wholesale rather than queried per known order,
        so a manual trade lands in reconciliation as broker-only instead of
        being invisible — which is the position that most needs to be seen."""
        self._write(ati, "NQ 03-26_Sim101_Position.txt", "LONG;1;18000.0")
        report = _venue(ati).reconcile(_Portfolio([]))
        assert report.broker_only == {"NQ": 1.0}
        assert not report.is_clean

    def test_agreement_is_clean(self, ati: Path) -> None:
        self._write(ati, "ES 03-26_Sim101_Position.txt", "LONG;2;5000.0")
        assert _venue(ati).reconcile(_Portfolio([_Position(ES, 2.0)])).is_clean

    def test_a_quantity_mismatch_reports_both_sides(self, ati: Path) -> None:
        self._write(ati, "ES 03-26_Sim101_Position.txt", "LONG;1;5000.0")
        report = _venue(ati).reconcile(_Portfolio([_Position(ES, 2.0)]))
        assert report.mismatched == {"ES": (2.0, 1.0)}

    def test_a_platform_only_position_is_reported(self, ati: Path) -> None:
        report = _venue(ati).reconcile(_Portfolio([_Position(ES, 2.0)]))
        assert report.platform_only == {"ES": 2.0}


class TestDiagnostics:
    def test_discover_lists_what_ninjatrader_actually_wrote(self, ati: Path) -> None:
        """The state-file naming came from secondary sources — a wrong pattern
        fails silently, looking like a market with no fills. This turns that
        into something visible in one command."""
        (ati / "outgoing" / "ES 03-26_Sim101_Position.txt").write_text("FLAT;0;0")
        (ati / "outgoing" / "Sim101_CashValue.txt").write_text("50000")
        assert _venue(ati).discover() == [
            "ES 03-26_Sim101_Position.txt",
            "Sim101_CashValue.txt",
        ]

    def test_account_value_is_read(self, ati: Path) -> None:
        (ati / "outgoing" / "Sim101_CashValue.txt").write_text("50000.75")
        assert _venue(ati).account_value() == pytest.approx(50000.75)

    def test_a_missing_account_file_is_none_not_zero(self, ati: Path) -> None:
        """Zero equity and unknown equity size positions very differently."""
        assert _venue(ati).account_value() is None

    def test_check_says_real_money_in_words(self, ati: Path) -> None:
        assert "REAL MONEY" in _venue(ati).check()

    def test_flatten_everything_writes_the_documented_command(self, ati: Path) -> None:
        _venue(ati).flatten_everything()
        written = next((ati / "incoming").glob("oif*.txt")).read_text().strip()
        assert written.startswith("FLATTENEVERYTHING;")
