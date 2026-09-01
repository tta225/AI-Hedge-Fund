"""Read-only HTTP surface over the desk.

The uploaded architecture put this under ``platform/``, which shadows the
Python standard library's ``platform`` module — any dependency doing
``import platform`` inside that project gets the FastAPI app instead, and the
failure surfaces far from its cause. Renamed to ``service``.

**Every endpoint is read-only, and that is a design constraint rather than an
omission.** The whole platform terminates in a human approval request; an HTTP
endpoint that could arm, approve, or route would be a way around the one gate
the architecture exists to enforce. There is no POST here and there should
never be one.

FastAPI is an optional extra (``pip install 'axiom[service]'``).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from axiom import __version__
from axiom.agents.audit import AuditLog
from axiom.agents.governance import Mandate
from axiom.agents.registry import default_registry
from axiom.core.config import get_settings


def service_available() -> bool:
    return importlib.util.find_spec("fastapi") is not None


def create_app(audit_path: str = "data/audit/runs.jsonl") -> Any:
    """Build the read-only app.

    Args:
        audit_path: JSONL run log to serve.
    """
    if not service_available():
        raise ImportError(
            "the HTTP service needs FastAPI, an optional extra. "
            "Install it with: pip install 'axiom[service]'"
        )
    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="AXIOM",
        version=__version__,
        description=(
            "Read-only view of the agent desk and its audit log. This API "
            "cannot approve, arm, or route anything."
        ),
    )
    log = AuditLog(audit_path)

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness plus the two facts that decide whether output is usable.

        The uploaded version reported ``{"status": "healthy", "models_loaded":
        6}`` — a hardcoded count that would have said 6 with nothing loaded.
        These are read from the running configuration.
        """
        settings = get_settings()
        return {
            "status": "ok",
            "version": __version__,
            "trading_mode": settings.trading_mode.value,
            "kill_switch": settings.kill_switch,
            "orders_permitted": settings.orders_permitted,
            "agents_enabled": settings.agents.enabled,
        }

    @app.get("/desk")
    def desk() -> dict[str, Any]:
        """The seats on the desk and the mandate rules that gate them."""
        return {
            "seats": [role.value for role in default_registry().roles],
            "mandate_rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "blocking": rule.blocking,
                }
                for rule in Mandate().rules
            ],
        }

    @app.get("/runs")
    def runs(limit: int = 20) -> dict[str, Any]:
        """Recent runs, newest first."""
        records = list(log)[-max(limit, 1) :]
        records.reverse()
        return {
            "count": len(records),
            "runs": [
                {
                    "digest": record.get("digest", "")[:12],
                    "symbol": record.get("symbol"),
                    "timeframe": record.get("timeframe"),
                    "strategy": record.get("strategy"),
                    "data_source": record.get("data_source"),
                    "data_is_evidence": record.get("data_is_evidence"),
                    "started_at": record.get("started_at"),
                    "cost_usd": (record.get("usage") or {}).get("cost_usd"),
                }
                for record in records
            ],
        }

    @app.get("/runs/{digest}")
    def run(digest: str) -> dict[str, Any]:
        """One run in full, with its integrity independently verified.

        ``integrity_verified`` is recomputed on every read rather than trusted
        from the file. A stored flag would be edited by whoever edited the
        record.
        """
        record = log.find(digest)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no run matching {digest!r}")
        return {**record, "integrity_verified": log.verify(record)}

    return app
