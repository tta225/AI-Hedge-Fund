"""Running the desk: structured logs, health checks, and alerts.

Everything else in AXIOM answers "should this trade?". This package answers
"is the thing that decides that still working?" — a separate question with a
separate failure mode, and one a research codebase normally forgets until the
first silent outage.

:mod:`axiom.ops.logs` emits JSON lines with a correlation id, so one decision
can be followed across the tick that made it, the order that carried it, and
the fill that closed it. :mod:`axiom.ops.health` answers whether the desk is
alive and current, from the store alone, so a watchdog needs no access to the
running process. :mod:`axiom.ops.alerts` decides what is worth waking someone
for and refuses to say it twice.
"""

from axiom.ops.alerts import (
    Alert,
    AlertRouter,
    Severity,
    console_sink,
)
from axiom.ops.health import HealthReport, HealthStatus, check_health
from axiom.ops.logs import (
    correlation_id,
    current_correlation_id,
    log_event,
    setup_logging,
)

__all__ = [
    "Alert", "AlertRouter", "HealthReport", "HealthStatus", "Severity",
    "check_health", "console_sink", "correlation_id", "current_correlation_id",
    "log_event", "setup_logging",
]
