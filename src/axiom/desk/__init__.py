"""The live desk: the loop that turns signals into managed positions.

:mod:`axiom.desk.guards` holds the preconditions that must be true before any
order is sent, and :mod:`axiom.desk.runner` is the loop that checks them, asks
the strategy, sizes through risk, and records everything before it acts.

The design is **crash-only**: there is no clean shutdown path, because a clean
shutdown path is a path that does not run when the machine is killed. Every
invariant is restored on startup by reading :mod:`axiom.store`, so a hard kill
and a graceful stop leave the desk in the same state.
"""

from axiom.desk.fills import FillPoller, PollOutcome
from axiom.desk.guards import (
    GuardReport,
    GuardStatus,
    check_data_freshness,
    check_guards,
)
from axiom.desk.runner import DeskConfig, DeskRunner, TickOutcome

__all__ = [
    "DeskConfig", "DeskRunner", "FillPoller", "GuardReport", "GuardStatus",
    "PollOutcome", "TickOutcome", "check_data_freshness", "check_guards",
]
