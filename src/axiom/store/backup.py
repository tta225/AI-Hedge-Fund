"""Copying the one file that cannot be reconstructed.

The store is the desk's memory: which orders were sent, which fills came back,
what the book therefore is. Prices can be re-downloaded and strategies
re-derived, but a lost fill history means the desk no longer knows what it
owns, and reconciliation against the broker is the only way back — assuming the
broker's records go back far enough.

Copying the file with ``cp`` is the obvious approach and it is wrong. SQLite in
WAL mode keeps recent committed transactions in ``-wal`` until a checkpoint, so
a naive copy taken mid-session is missing the most recent writes — exactly the
ones an incident would be about — and can be torn if a write lands during the
copy. :func:`backup_store` uses SQLite's own online backup API, which takes a
consistent snapshot of a live database without blocking writers.

Three things beyond the copy, because a backup nobody has restored is a
hypothesis:

:func:`verify_backup` opens the copy and runs an integrity check *and* a
schema-version check, so a corrupt or stale-schema backup is discovered on the
day it is taken rather than on the day it is needed.

:func:`restore_store` refuses to overwrite a database that already exists
unless told to, and when it does overwrite, it sets the displaced file aside
first. The failure mode of a restore procedure is restoring the wrong way
round, and it is unrecoverable by construction.

:func:`prune_backups` keeps a bounded number, because a backup job that fills
the disk takes the desk down by itself.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from axiom.store.db import SCHEMA_VERSION, Store, StoreError

#: How many snapshots to keep by default. Enough to cover a problem noticed a
#: few days late, few enough that they cannot silently fill a volume.
DEFAULT_KEEP = 14
#: Timestamp format in the filename. Sorts lexicographically in time order,
#: which is what makes pruning a slice rather than a parse.
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True, slots=True)
class BackupResult:
    """One snapshot, and enough to decide whether to trust it."""

    path: Path
    at: pd.Timestamp
    size_bytes: int
    #: Rows per table at snapshot time. A backup whose fill count is lower than
    #: the live database's is the signal that something copied the wrong file.
    row_counts: dict[str, int]
    verified: bool

    def render(self) -> str:
        counts = ", ".join(f"{table} {count:,}" for table, count in sorted(self.row_counts.items()))
        state = "verified" if self.verified else "UNVERIFIED"
        return (
            f"{self.path.name}  {self.size_bytes / 1024:,.0f} KiB  {state}\n"
            f"  taken {self.at}\n  {counts}"
        )


class BackupError(RuntimeError):
    """A backup could not be taken, verified, or restored."""


_TABLES = ("orders", "order_events", "fills", "equity", "halts")


def _row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _TABLES:
        try:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        except sqlite3.OperationalError:
            continue  # table absent in an older schema; the version check catches it
        counts[table] = int(row[0])
    return counts


def backup_store(
    store: Store | str | Path,
    directory: str | Path = "data/backups",
    *,
    at: pd.Timestamp | None = None,
    verify: bool = True,
) -> BackupResult:
    """Take a consistent snapshot of a live store.

    Accepts an open :class:`Store` or a path. Passing the open store is the
    normal case and the one worth supporting properly: the backup API works
    against a live connection with writers active, which is the whole reason
    for using it over a file copy.
    """
    at = at or pd.Timestamp.now("UTC").tz_localize(None)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    opened: Store | None = None
    if isinstance(store, Store):
        source = store
    else:
        source_path = Path(store)
        if not source_path.exists():
            raise BackupError(f"no store at {source_path}")
        source = opened = Store(source_path)

    # ``:memory:`` has no meaningful stem, and a file called ``:memory:-...db``
    # would be a confusing thing to find in a backup directory.
    stem = "desk" if source.path == ":memory:" else Path(source.path).stem
    target_path = directory / f"{stem}-{at.strftime(STAMP_FORMAT)}.db"
    if target_path.exists():
        raise BackupError(
            f"{target_path} already exists; two backups in the same second would "
            "overwrite each other silently"
        )

    try:
        destination = sqlite3.connect(target_path)
        try:
            source.snapshot_into(destination)
            counts = _row_counts(destination)
        finally:
            destination.close()
    except sqlite3.Error as exc:
        target_path.unlink(missing_ok=True)
        raise BackupError(f"backup of {source.path} failed: {exc}") from exc
    finally:
        if opened is not None:
            opened.close()

    verified = verify_backup(target_path) if verify else False
    return BackupResult(
        path=target_path,
        at=at,
        size_bytes=target_path.stat().st_size,
        row_counts=counts,
        verified=verified,
    )


def verify_backup(path: str | Path) -> bool:
    """Is this file a readable store at the schema this code expects?

    Two checks, because they fail for different reasons. ``integrity_check``
    catches a torn or truncated file — a disk that filled mid-copy. The schema
    version catches a backup taken by an older build, which is intact and still
    cannot be restored into without a migration.
    """
    path = Path(path)
    if not path.exists():
        raise BackupError(f"no backup at {path}")
    connection = sqlite3.connect(path)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            return False
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            return False
        return int(row[0]) == SCHEMA_VERSION
    except (sqlite3.DatabaseError, ValueError):
        return False
    finally:
        connection.close()


def restore_store(
    backup: str | Path, target: str | Path = "data/desk.db", *, overwrite: bool = False
) -> Path:
    """Put a snapshot back, having first checked that it is worth putting back.

    The displaced database is moved aside rather than deleted, with a
    ``.superseded-<stamp>`` suffix. Restoring the wrong snapshot is a recoverable
    mistake only if the thing it replaced still exists, and at the moment of a
    restore nobody is thinking clearly enough to have taken that precaution
    themselves.
    """
    backup = Path(backup)
    target = Path(target)

    if not verify_backup(backup):
        raise BackupError(
            f"{backup} failed verification (corrupt, or written at a different "
            f"schema version than {SCHEMA_VERSION}); refusing to restore it"
        )

    if target.exists():
        if not overwrite:
            raise BackupError(
                f"{target} already exists. Restoring over a live store is how a "
                "good database is replaced by a stale one; pass overwrite=True "
                "if that is genuinely what you mean."
            )
        stamp = pd.Timestamp.now("UTC").strftime(STAMP_FORMAT)
        superseded = target.with_suffix(f".superseded-{stamp}")
        shutil.move(str(target), str(superseded))
        # The WAL and shared-memory files belong to the database that was moved.
        # Leaving them behind next to a restored file is how SQLite is handed a
        # journal describing a database that is no longer there.
        for sidecar in ("-wal", "-shm"):
            companion = Path(str(target) + sidecar)
            if companion.exists():
                companion.rename(str(superseded) + sidecar)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, target)

    # Prove the restored file opens under the real Store, which runs the schema
    # checks. A restore that "succeeded" into an unopenable file is worse than
    # one that failed loudly.
    try:
        with Store(target):
            pass
    except StoreError as exc:
        raise BackupError(f"restored file at {target} does not open as a store: {exc}") from exc
    return target


def list_backups(directory: str | Path = "data/backups", stem: str = "desk") -> list[Path]:
    """Snapshots for one store, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{stem}-*.db"))


def prune_backups(
    directory: str | Path = "data/backups", *, keep: int = DEFAULT_KEEP, stem: str = "desk"
) -> list[Path]:
    """Delete all but the newest ``keep`` snapshots. Returns what was removed.

    Only files matching the naming convention are considered, so a directory
    shared with anything else is not a hazard.
    """
    if keep < 1:
        raise ValueError("keep at least one backup; pruning to zero deletes the last copy")
    backups = list_backups(directory, stem)
    doomed = backups[:-keep] if len(backups) > keep else []
    for path in doomed:
        path.unlink()
    return doomed
