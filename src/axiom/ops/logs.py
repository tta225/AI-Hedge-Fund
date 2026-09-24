"""Structured logging with a correlation id that follows a decision.

A trading incident is almost never one log line. It is a tick that produced a
signal, a risk decision that sized it, an order that carried it, a fill that
opened it, and a reconciliation that later disagreed — spread across minutes
and interleaved with everything else the desk was doing. Reconstructing that
from prose log lines means grepping for a symbol and hoping.

So every line is JSON, and every line carries a **correlation id** set once at
the top of a decision and inherited by everything downstream. Filtering one id
gives the whole causal chain and nothing else.

Two deliberate choices worth stating:

*JSON on one line, always.* Multi-line output is unparseable by every log
shipper without a custom rule, and the moment a stack trace lands mid-file the
rule breaks. Exceptions are folded into a single ``error`` field.

*Timestamps in UTC with an explicit offset.* Local-time logs from a desk that
trades three sessions cannot be ordered against each other, and a daylight
saving transition makes an hour of them ambiguous.

The correlation id lives in a :class:`~contextvars.ContextVar`, so it is
correct under threads and asyncio without being passed through every function
signature — and, unlike a global, a background thread that never entered a
decision context gets no id rather than someone else's.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

#: The id in scope for the current decision, if any.
_CORRELATION: ContextVar[str | None] = ContextVar("axiom_correlation_id", default=None)

#: Record attributes that come from :mod:`logging` itself, not the caller.
#: Anything else on a record is treated as structured payload.
_STANDARD_FIELDS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)


def current_correlation_id() -> str | None:
    """The id in scope, or None outside a decision."""
    return _CORRELATION.get()


@contextmanager
def correlation_id(value: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of the block.

    Nested blocks inherit unless given an explicit value, so a helper that
    opens its own context does not sever the chain. The token is reset in a
    ``finally`` — leaking an id into whatever runs next would attribute an
    unrelated failure to this decision.

    Args:
        value: id to bind. Defaults to a fresh short uuid, or the inherited one
            when already inside a context.
    """
    resolved = value or _CORRELATION.get() or uuid.uuid4().hex[:12]
    token = _CORRELATION.set(resolved)
    try:
        yield resolved
    finally:
        _CORRELATION.reset(token)


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        correlation = getattr(record, "correlation_id", None) or current_correlation_id()
        if correlation:
            payload["correlation_id"] = correlation

        # Anything the caller attached via `extra=` becomes a top-level field,
        # which is what makes a log queryable rather than merely readable.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key != "correlation_id":
                payload[key] = _safe(value)

        if record.exc_info:
            # Folded into one field rather than appended as lines: a multi-line
            # record breaks every shipper that splits on newline.
            payload["error"] = self.formatException(record.exc_info).replace("\n", " | ")
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info).replace("\n", " | ")

        return json.dumps(payload, default=str, separators=(",", ":"))


def _safe(value: Any) -> Any:
    """Keep JSON-native types; stringify anything else.

    A log line that raises while being formatted destroys the record it was
    trying to preserve, which is the one moment it mattered most.
    """
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, list | tuple):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def setup_logging(
    level: int | str = logging.INFO,
    *,
    stream: Any = None,
    json_format: bool = True,
) -> None:
    """Configure the root logger once, replacing any existing handlers.

    Replacing rather than adding: calling this twice otherwise doubles every
    line, and a desk that logs everything twice is one whose log volume lies
    about its activity.

    Args:
        level: threshold.
        stream: destination. Defaults to stderr, so log output never
            contaminates stdout that something else is parsing.
        json_format: False gives human-readable lines, for a terminal session.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one structured event.

    ``event`` is a stable machine-readable name — ``order_submitted``,
    ``guard_failed`` — and the numbers go in ``fields``. Keeping the two apart
    is what lets an event be counted across runs; a message with the numbers
    interpolated into it is a unique string every time and can only be read,
    never aggregated.
    """
    logger.log(level, event, extra={"event": event, **fields})
