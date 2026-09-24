"""Durable state for a desk that has to survive being killed.

A backtest keeps everything in memory and that is correct — the run is
deterministic and reproducible from its inputs. A live desk cannot: the process
will be restarted, by a deploy, an OOM kill, or a machine going away, and when
it comes back the market is still there and so are the positions.

This package is what makes that survivable. :class:`~axiom.store.db.Store`
persists orders, fills, positions and equity to SQLite in WAL mode, and
:mod:`axiom.store.reconcile` compares what was persisted against what the
broker says is true — with the broker winning every disagreement.
"""

from axiom.store.db import Store, StoreError
from axiom.store.reconcile import (
    Discrepancy,
    Reconciliation,
    reconcile_positions,
)

__all__ = [
    "Discrepancy", "Reconciliation", "Store", "StoreError", "reconcile_positions",
]
