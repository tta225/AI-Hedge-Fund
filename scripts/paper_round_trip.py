"""The equity round trip: prove the paths a crypto order could not.

The crypto round trip on 2026-09-01 found three bugs and left two paths
unproven, both of which only an equity order during market hours can exercise:

**Brackets.** No protective legs were attached to the crypto order, so the
bracket/OTO construction is stub-tested only. This is the path that matters most
operationally, because the whole reason the desk sends stops to the venue rather
than holding them in memory is that a stop held in this process does not exist
during a crash — and a bracket that silently fails to attach converts a bounded
loss into an unbounded one while looking like a successful entry.

**Equity mechanics.** Equities avoid all three crypto bugs by construction: DAY
is a valid time-in-force, positions come back as plain tickers, and fees are cash
rather than taken in kind. "Avoids them by construction" is exactly the reasoning
that produced those three bugs, so it is checked rather than assumed.

The script is deliberately paranoid, because it spends real (paper) money:

* It refuses to run against the live endpoint at all. Not a flag — a refusal.
* It refuses to run when the market is closed, asking the *broker's* clock
  rather than a local calendar. A market order into a closed session queues to
  the open and fills at a price nobody chose, which reads as slippage rather
  than as the scheduling mistake it is.
* One share. The point is to exercise a path, not to take a position.
* Teardown runs in a ``finally``. A script that leaves an armed bracket and an
  open position behind because it failed at step four is worse than one that
  never ran.
* It reports PASS/FAIL per step and exits non-zero on any failure, so it is
  usable from a scheduler without a human reading the output.

Run it at or after the open::

    python -m scripts.paper_round_trip --symbol XLF
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from axiom.core.types import AssetClass, OrderStatus, OrderType, Side, get_instrument
from axiom.execution.alpaca import AlpacaVenue
from axiom.execution.base import ExecutionError, Order
from axiom.store import Store, reconcile_positions

#: Seconds to wait for a market order to fill before calling it stuck. A liquid
#: ETF market order fills in well under a second; ten is a generous ceiling that
#: still fails fast enough to leave time for teardown.
FILL_TIMEOUT = 15.0
#: How far from the entry to place the protective legs, as a fraction. Wide
#: enough that neither triggers during the couple of minutes this runs — the
#: goal is to prove the legs are *held by the venue*, not to have them fire.
STOP_DISTANCE = 0.10
TARGET_DISTANCE = 0.10


@dataclass
class Report:
    """PASS/FAIL per step, so the exit code means something to a scheduler."""

    steps: list[tuple[str, bool, str]] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.steps.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    @property
    def failed(self) -> list[str]:
        return [name for name, ok, _ in self.steps if not ok]

    def render(self) -> str:
        passed = sum(1 for _, ok, _ in self.steps if ok)
        head = f"{passed}/{len(self.steps)} steps passed"
        if self.failed:
            return f"{head}\nFAILED: {', '.join(self.failed)}"
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": len(self.steps) - len(self.failed),
            "total": len(self.steps),
            "failed": self.failed,
            "steps": [
                {"name": n, "ok": ok, "detail": d} for n, ok, d in self.steps
            ],
        }


def _wait_for_fill(venue: AlpacaVenue, symbol: str, timeout: float) -> float:
    """Poll the broker's position until the entry shows up, or give up."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        held = venue.positions().get(symbol, 0.0)
        if held:
            return float(held)
        time.sleep(0.5)
    return 0.0


def _venue_legs(venue: AlpacaVenue, parent_venue_id: str) -> list[dict[str, Any]]:
    """The protective legs Alpaca is holding for this parent order.

    Read back from the venue rather than trusted from the submit response,
    because "the venue accepted my bracket" and "the venue is holding a stop"
    are different claims and only the second one protects anything.
    """
    payload = venue._request(
        "GET", f"/v2/orders/{parent_venue_id}?nested=true"
    )
    return list(payload.get("legs") or [])


def run(symbol: str, db: str, *, quantity: float = 1.0) -> Report:
    report = Report()
    venue = AlpacaVenue(paper=True)

    # --- refusals, before anything is spent -------------------------------
    if venue.is_live:
        raise SystemExit("refusing to run against the live endpoint")
    report.record("paper endpoint", not venue.is_live, venue.base_url)

    instrument = get_instrument(symbol)
    if instrument.asset_class is not AssetClass.EQUITY:
        raise SystemExit(f"{symbol} is {instrument.asset_class.value}; this proves equities")

    clock = venue.clock()
    if not report.record("market open", clock.is_open, clock.render()):
        raise SystemExit(
            f"market is closed. {clock.render()}. "
            "A market order sent now would queue to the open and fill at a "
            "price nobody chose."
        )

    account = venue.account()
    if not report.record(
        "account tradable",
        account.is_tradable,
        f"equity {account.equity:,.2f}, buying power {account.buying_power:,.2f}",
    ):
        raise SystemExit("account is blocked from trading")

    before = venue.positions()
    report.record("starting book read", True, json.dumps(before))
    if symbol in before:
        raise SystemExit(
            f"{symbol} is already held ({before[symbol]}). Pick a symbol the "
            "account is flat in, so the round trip is unambiguous."
        )

    parent_venue_id = ""
    filled = 0.0
    try:
        # --- bracket entry ------------------------------------------------
        # The trading API has no quote endpoint, so the legs are priced off the
        # data API's last trade. Only used to place them far enough away that
        # neither fires; the exact level is not the thing under test.
        last = _reference_price(symbol)
        report.record("reference price", last > 0, f"{last:,.2f}")

        stop = round(last * (1 - STOP_DISTANCE), 2)
        target = round(last * (1 + TARGET_DISTANCE), 2)
        now = pd.Timestamp.now("UTC").tz_localize(None)
        entry = Order(
            instrument=instrument,
            side=Side.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            strategy="paper_round_trip",
            tag=f"rt-{now.strftime('%Y%m%dT%H%M%S')}",
            stop_loss=stop,
            take_profit=target,
            reference_price=last,
        )
        print(f"\n  submitting bracket: buy {quantity} {symbol} "
              f"stop {stop:,.2f} target {target:,.2f}")
        sent = venue.submit(entry, now)
        parent_venue_id = venue._venue_ids.get(entry.order_id, "")
        report.record(
            "bracket accepted",
            sent.status is not OrderStatus.REJECTED and bool(parent_venue_id),
            f"venue id {parent_venue_id or 'MISSING'}, status {sent.status.value}",
        )

        filled = _wait_for_fill(venue, symbol, FILL_TIMEOUT)
        report.record("entry filled", filled == quantity, f"position {filled}")

        # --- the thing this script exists for -----------------------------
        legs = _venue_legs(venue, parent_venue_id)
        kinds = sorted(str(leg.get("type")) for leg in legs)
        report.record(
            "venue holds protective legs",
            len(legs) == 2,
            f"{len(legs)} leg(s): {kinds}",
        )
        report.record(
            "a stop is among them",
            any("stop" in str(leg.get("type", "")) for leg in legs),
            ", ".join(
                f"{leg.get('type')}@{leg.get('stop_price') or leg.get('limit_price')}"
                for leg in legs
            ),
        )

        # --- booking and reconciliation -----------------------------------
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        with Store(db) as store:
            from axiom.desk.fills import FillPoller

            poller = FillPoller(venue, store)
            outcome = poller.poll(now)
            report.record(
                "fill booked into the store",
                outcome.recorded >= 1 and not outcome.error,
                f"{outcome.recorded} recorded, {outcome.duplicates} already seen"
                + (f", error {outcome.error}" if outcome.error else ""),
            )
            again = poller.poll(pd.Timestamp.now("UTC").tz_localize(None))
            report.record(
                "re-poll books nothing twice",
                again.recorded == 0,
                f"{again.duplicates} recognised as duplicates",
            )
            local = store.positions()
            recon = reconcile_positions(local, venue.positions())
            # The pre-existing SPY is expected to disagree; this symbol is not.
            ours = [d for d in recon.discrepancies if d.symbol == symbol]
            report.record(
                f"{symbol} reconciles",
                not ours,
                "clean" if not ours else f"{ours[0].kind}: {ours[0].delta:+g}",
            )
            other = [d.symbol for d in recon.discrepancies if d.symbol != symbol]
            if other:
                print(f"  note: pre-existing disagreement on {other} — not from this run")

    finally:
        # --- teardown, whatever happened above ----------------------------
        print("\n  tearing down")
        cancelled = 0
        for working in venue.working_orders():
            try:
                venue.cancel(working)
                cancelled += 1
            except ExecutionError as exc:
                print(f"    cancel failed: {exc}")
        # Counting cancels is not the check. Alpaca's bracket legs are an OCO
        # pair, so cancelling one cancels its sibling and the count
        # under-reports; and `working_orders` reads top-level rows, under which
        # legs may be nested. What matters is that nothing is left armed, so
        # that is what is asserted.
        time.sleep(1)
        remaining = venue.working_orders()
        report.record(
            "no orders left working",
            not remaining,
            f"cancelled {cancelled}, {len(remaining)} still open"
            + (f": {[o.instrument.symbol for o in remaining]}" if remaining else ""),
        )

        held = venue.positions().get(symbol, 0.0)
        if held:
            close = Order(
                instrument=get_instrument(symbol),
                side=Side.SELL if held > 0 else Side.BUY,
                quantity=abs(held),
                order_type=OrderType.MARKET,
                strategy="paper_round_trip",
                tag=f"rt-close-{pd.Timestamp.now('UTC').strftime('%Y%m%dT%H%M%S')}",
            )
            try:
                venue.submit(close, pd.Timestamp.now("UTC").tz_localize(None))
                time.sleep(3)
            except ExecutionError as exc:
                print(f"    close failed: {exc}")
        flat = venue.positions().get(symbol, 0.0)
        report.record("position flat", flat == 0.0, f"{symbol} {flat}")

    return report


def _reference_price(symbol: str) -> float:
    """Last trade from the data API, for pricing the protective legs."""
    import os
    import urllib.request

    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest?feed=iex"
    request = urllib.request.Request(
        url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    return float(payload.get("trade", {}).get("p", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbol", default="XLF",
        help="A liquid equity the account is flat in. Default XLF: liquid, and "
             "cheap enough that one share is a rounding error.",
    )
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument(
        "--db",
        default="/tmp/claude-0/-home-user-AI-Hedge-Fund/"
                "ec100fa7-da0f-5aa1-851d-7949ff481da8/scratchpad/round_trip.db",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    print(f"EQUITY ROUND TRIP — {args.symbol}, {args.quantity} share(s), paper only\n")
    report = run(args.symbol, args.db, quantity=args.quantity)
    print(f"\n{report.render()}")

    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Wrote {args.out}")

    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
