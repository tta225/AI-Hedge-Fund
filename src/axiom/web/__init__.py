"""AXIOM web console — HTTP API and browser UI over the same engines as the CLI.

`server` imports FastAPI at module scope, so it is deliberately *not* imported
here. That keeps `import axiom.web` cheap and lets the CLI print an install hint
instead of a traceback when the optional extra is missing.
"""

from __future__ import annotations

__all__ = ["serve"]


def serve(*args: object, **kwargs: object) -> None:
    """Lazy proxy to `axiom.web.server.serve`."""
    from axiom.web.server import serve as _serve

    _serve(*args, **kwargs)  # type: ignore[arg-type]
