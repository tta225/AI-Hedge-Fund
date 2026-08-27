"""Compare what we believe against what the broker says, and believe the broker.

Every live trading system eventually disagrees with its venue. A fill arrives
during a restart and is never booked. A websocket drops and reconnects past a
message. A partial fill is counted as complete. A manual intervention happens in
the broker's web UI. None of these are exotic; over enough sessions all of them
occur.

The rule that makes this survivable is short: **the broker is the source of
truth, and a disagreement halts trading.**

Not "logs a warning" — halts. A desk whose position belief is wrong is sizing
against a fiction, and its risk limits are computed from that fiction. Trading
through a discrepancy is how a small bookkeeping error becomes an unbounded
one. The correct response is to stop, surface it, and let a human decide.

Auto-correcting the local view and continuing is the tempting alternative and
it is worse: it destroys the evidence of *why* the drift happened, and the
cause is usually a bug that will recur. :func:`reconcile_positions` therefore
reports; it never silently writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Absolute quantity difference tolerated before a symbol is called broken.
#: Non-zero only because fractional-share and crypto venues round; it is not a
#: licence to ignore small drift.
DEFAULT_TOLERANCE = 1e-8


@dataclass(frozen=True, slots=True)
class Discrepancy:
    """One symbol on which the desk and the broker disagree."""

    symbol: str
    local: float
    broker: float

    @property
    def delta(self) -> float:
        """Broker minus local. Positive means the broker holds more."""
        return self.broker - self.local

    @property
    def kind(self) -> str:
        if self.local == 0.0:
            return "unknown_position"
        if self.broker == 0.0:
            return "phantom_position"
        if (self.local > 0) != (self.broker > 0):
            return "wrong_side"
        return "quantity_mismatch"

    def __str__(self) -> str:
        return (
            f"{self.symbol}: local {self.local:+g} vs broker {self.broker:+g} "
            f"({self.delta:+g}, {self.kind})"
        )


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Outcome of one comparison."""

    at: pd.Timestamp
    discrepancies: list[Discrepancy] = field(default_factory=list)
    symbols_checked: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.discrepancies

    @property
    def should_halt(self) -> bool:
        """Any disagreement at all. There is no safe amount of drift.

        A ``phantom_position`` — the desk thinks it holds something the broker
        does not — is the most dangerous kind, because the desk will keep
        managing a stop for a position that cannot be closed. But
        ``unknown_position`` is worse in a different way: the desk holds
        exposure it has no risk limit for. Both halt.
        """
        return bool(self.discrepancies)

    def render(self) -> str:
        if self.is_clean:
            return f"Reconciled {self.symbols_checked} symbols at {self.at}: clean"
        lines = [
            f"RECONCILIATION FAILED at {self.at} — "
            f"{len(self.discrepancies)} of {self.symbols_checked} symbols disagree.",
            "The broker is authoritative. Trading must halt until this is "
            "explained; do not resume by overwriting local state.",
            "",
        ]
        lines += [f"  {d}" for d in self.discrepancies]
        return "\n".join(lines)


def reconcile_positions(
    local: dict[str, float],
    broker: dict[str, float],
    *,
    at: pd.Timestamp | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Reconciliation:
    """Compare two position books.

    Args:
        local: symbol → signed quantity, e.g. from :meth:`Store.positions`.
        broker: symbol → signed quantity, from the venue.
        at: comparison timestamp. Defaults to now, UTC.
        tolerance: absolute quantity difference treated as equal.

    Returns:
        A :class:`Reconciliation`. Check :attr:`~Reconciliation.should_halt`
        before doing anything else with it.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    moment = pd.Timestamp.now(tz="UTC") if at is None else pd.Timestamp(at)

    # Union of both books: a symbol missing from either side is exactly the
    # case that matters, so iterating over one book's keys would miss half the
    # failures this function exists to catch.
    symbols = sorted(set(local) | set(broker))
    discrepancies = [
        Discrepancy(
            symbol=symbol,
            local=float(local.get(symbol, 0.0)),
            broker=float(broker.get(symbol, 0.0)),
        )
        for symbol in symbols
        if abs(float(broker.get(symbol, 0.0)) - float(local.get(symbol, 0.0))) > tolerance
    ]
    return Reconciliation(
        at=moment, discrepancies=discrepancies, symbols_checked=len(symbols)
    )
