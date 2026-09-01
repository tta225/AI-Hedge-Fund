"""Immutable, content-addressed record of every agent run.

A desk has to be able to answer, months later, *why* a position was proposed:
which data, which code, which model, which prompt, and what each seat actually
said. Scrollback is not that answer, and neither is a log line.

Each run produces a :class:`RunRecord` whose :attr:`RunRecord.digest` is the
SHA-256 of its own canonical content. That gives three things at once:

*Tamper evidence.* Recompute the digest; if it differs from the stored one, the
record was edited after the fact.

*Deduplication.* Two runs with the same digest are the same run. Replaying a
pipeline against a warm cache reproduces the digest exactly, which is what
turns "it should be deterministic" into a checkable claim.

*A citation.* A digest is short enough to paste into a ticket and precise
enough to retrieve the whole run.

Records append to a JSONL file. Append-only is deliberate: there is no update
path and no delete path, because an audit log you can rewrite is a diary.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.agents.base import AgentReport
from axiom.agents.runtime import Usage

#: Bumped when the record's shape changes, so an old record is never silently
#: reinterpreted under new field semantics.
SCHEMA_VERSION = 1


def _canonical(payload: dict[str, Any]) -> str:
    """Byte-stable JSON. Key order and separators must not vary between runs.

    Without ``sort_keys`` the digest would depend on dict insertion order, and
    two identical runs would hash differently on different Python builds.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One pipeline run, in full, as it will be stored."""

    #: What was analysed.
    symbol: str
    timeframe: str
    strategy: str
    #: Where the bars came from, and whether they may back a claim.
    data_source: str
    data_is_evidence: bool
    #: What ran.
    code_version: str
    model: str
    #: Every stage's output.
    reports: tuple[AgentReport, ...] = ()
    #: Aggregate consumption across the run.
    usage: Usage = field(default_factory=Usage)
    #: The approval request a human acts on.
    approval_summary: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: Free-form run context: git SHA, host, ticket id.
    context: dict[str, str] = field(default_factory=dict)

    def content(self) -> dict[str, Any]:
        """The exact payload the digest is taken over.

        Every clock reading is deliberately excluded — ``started_at`` here, and
        each report's ``produced_at``, ``latency_s``, ``usage`` and ``cached``
        flag via ``stable=True``. Two identical runs an hour apart are the same
        analysis; a digest that moved with the wall clock, the machine's speed,
        or a cache hit could never establish that, which is the only thing a
        digest is for.
        """
        return {
            "schema": SCHEMA_VERSION,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy": self.strategy,
            "data_source": self.data_source,
            "data_is_evidence": self.data_is_evidence,
            "code_version": self.code_version,
            "model": self.model,
            "approval_summary": self.approval_summary,
            "context": dict(sorted(self.context.items())),
            "reports": [r.as_dict(stable=True) for r in self.reports],
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.content()).encode()).hexdigest()

    @property
    def short_digest(self) -> str:
        """First 12 hex chars — the form meant for humans and tickets."""
        return self.digest[:12]

    @property
    def unsupported_figure_count(self) -> int:
        return sum(len(r.guardrail.unsupported) for r in self.reports)

    @property
    def failed_stages(self) -> tuple[str, ...]:
        return tuple(r.role.value for r in self.reports if r.failed)

    def as_record(self) -> dict[str, Any]:
        """Content plus the fields that are *about* the record rather than in it."""
        return {
            **self.content(),
            "digest": self.digest,
            "started_at": self.started_at.isoformat(),
            "usage": self.usage.as_dict(),
        }

    def render(self) -> str:
        lines = [
            f"run {self.short_digest}  {self.symbol} {self.timeframe} · {self.strategy}",
            f"  data      : {self.data_source}"
            + ("" if self.data_is_evidence else "  [NOT EVIDENCE]"),
            f"  model     : {self.model or 'deterministic (no LLM)'}",
            f"  code      : {self.code_version}",
            f"  cost      : ${self.usage.cost_usd:,.4f} over {self.usage.calls} call(s)"
            + (f", {self.usage.cached_calls} cached" if self.usage.cached_calls else ""),
        ]
        if self.failed_stages:
            lines.append(f"  failed    : {', '.join(self.failed_stages)}")
        if self.unsupported_figure_count:
            lines.append(
                f"  ⚠ unverified figures in narrative: {self.unsupported_figure_count}"
            )
        return "\n".join(lines)


class AuditLog:
    """Append-only JSONL store. One record per line, newest last.

    Deliberately not a database. A JSONL file survives this process, this
    machine, and this codebase; it is greppable by anyone with a terminal and
    needs no running service to read six months from now.
    """

    def __init__(self, path: str | Path = "data/audit/runs.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: RunRecord) -> str:
        """Write one record. Returns its digest.

        The write is atomic at the line level: the payload is serialised in full
        before the file is opened, so a serialisation error cannot leave a
        half-written line that breaks every later read.
        """
        line = _canonical(record.as_record())
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                # The record is the point. If the process dies here, a run that
                # happened must not read as a run that did not.
                os.fsync(handle.fileno())
        return record.digest

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def find(self, digest_prefix: str) -> dict[str, Any] | None:
        """Retrieve by digest or any unique prefix of one."""
        for record in self:
            if str(record.get("digest", "")).startswith(digest_prefix):
                return record
        return None

    def verify(self, record: dict[str, Any]) -> bool:
        """Recompute the digest over the stored content.

        False means the line was altered after it was written — or that it was
        produced under a different schema version, which is equally a reason
        not to trust a field-by-field reading of it.
        """
        stored = str(record.get("digest", ""))
        content = {
            k: v for k, v in record.items() if k not in {"digest", "started_at", "usage"}
        }
        return hashlib.sha256(_canonical(content).encode()).hexdigest() == stored


def code_version() -> str:
    """Best available identifier for the code that produced a run.

    Falls back to the package version when git is unavailable — a released
    wheel has no repository, and "unknown" would be worse than a version.
    """
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        # An uncommitted tree is not reproducible from the SHA alone, and a
        # record that implies otherwise is worse than one that admits it.
        return f"git:{sha}{'+dirty' if dirty else ''}"
    except Exception:
        from axiom import __version__

        return f"pkg:{__version__}"
