"""The loop that runs the desk on a clock.

Every piece the desk needs already exists and deliberately owns no timer:
:meth:`~axiom.desk.runner.DeskRunner.tick` makes one decision,
:meth:`~axiom.desk.fills.FillPoller.poll` books what happened, and
:func:`~axiom.ops.health.check_health` says whether any of it is working. That
separation is what makes them testable without waiting in real time. This
module is the one place that owns a clock, and it is kept small for the same
reason.

The ordering within a cycle is the design:

1. **Poll fills first.** The desk sizes against current exposure, so booking
   what filled before deciding anything means the decision sees the real book.
   Deciding first and booking afterwards sizes against a position that already
   changed.
2. **Reconcile, and halt on disagreement.** Cheap, and the one check that
   catches everything the others miss.
3. **Tick, if the calendar says the market is open.** Guards inside the runner
   do the rest.
4. **Check health and alert.** Last, so the alert describes the cycle that just
   ran rather than the one before it.

Three properties that matter more than the sequence:

**Nothing in a cycle can kill the loop.** Every stage is wrapped. A supervisor
that dies on a transient venue error is worse than no supervisor, because the
positions it was managing are still open and now nothing is watching them.

**Failures back off, successes reset.** Consecutive failures widen the sleep
exponentially up to a cap, so a venue outage produces a handful of retries
rather than thousands of rate-limited requests. One success clears it.

**It stops when it should.** A persisted halt stops the *trading* stage but not
the loop — fills still need booking and health still needs reporting on a
halted desk, which is precisely when someone is looking at it.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from axiom.desk.fills import FillPoller, PollOutcome
from axiom.desk.runner import DeskRunner, TickOutcome
from axiom.ops.alerts import AlertRouter, alerts_for_health
from axiom.ops.health import HealthReport, HealthStatus, check_health
from axiom.ops.logs import correlation_id, log_event
from axiom.store.db import Store
from axiom.strategy.base import StrategyContext

logger = logging.getLogger(__name__)

#: Seconds between cycles when everything is working.
DEFAULT_INTERVAL = 60.0
#: Multiplier applied to the interval per consecutive failure.
BACKOFF_FACTOR = 2.0
#: Ceiling on the backed-off interval. Past this a human is the fix, not a
#: retry, and continuing to hammer a dead venue only delays that conversation.
MAX_INTERVAL = 900.0


@dataclass(slots=True)
class CycleOutcome:
    """What one supervisor cycle did."""

    at: pd.Timestamp
    poll: PollOutcome | None = None
    tick: TickOutcome | None = None
    health: HealthReport | None = None
    skipped_trading: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        parts = [f"[{self.at}]"]
        if self.poll is not None:
            parts.append(f"fills +{self.poll.recorded}")
        if self.skipped_trading:
            parts.append(f"no trade ({self.skipped_trading})")
        elif self.tick is not None:
            parts.append(f"{self.tick.signals} signal(s), {self.tick.orders_sent} sent")
        if self.health is not None:
            parts.append(f"health {self.health.status.value}")
        if self.errors:
            parts.append(f"ERRORS: {'; '.join(self.errors)}")
        return " | ".join(parts)


class Supervisor:
    """Runs the desk's stages on a clock, and keeps running.

    Args:
        runner: makes trading decisions.
        poller: books executions. Optional — a desk with no venue to poll is a
            legitimate configuration, and requiring one would force a stub.
        store: shared state, read for halts and health.
        context_factory: returns the current :class:`StrategyContext`, or None
            when there is no fresh bar to act on. Injected because building one
            means fetching data, and a supervisor that knows how to fetch data
            is a supervisor that cannot be tested without a network.
        is_open: whether the market is open now. Defaults to always open, which
            is correct for crypto and wrong for equities — pass a calendar.
        alerts: router for health alerts. One is created if not given.
        interval: seconds between cycles when healthy.
    """

    def __init__(
        self,
        *,
        runner: DeskRunner,
        store: Store,
        context_factory: Callable[[], StrategyContext | None],
        poller: FillPoller | None = None,
        is_open: Callable[[pd.Timestamp], bool] | None = None,
        alerts: AlertRouter | None = None,
        interval: float = DEFAULT_INTERVAL,
        max_interval: float = MAX_INTERVAL,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        if max_interval < interval:
            raise ValueError("max_interval must be at least interval")
        self.runner = runner
        self.store = store
        self.context_factory = context_factory
        self.poller = poller
        self.is_open = is_open or (lambda _: True)
        self.alerts = alerts if alerts is not None else AlertRouter()
        self.interval = interval
        self.max_interval = max_interval
        self._stop = threading.Event()
        self._failures = 0
        #: Health conditions currently alerting, so a recovery can be paired.
        self._alerting: set[str] = set()

    # --- lifecycle --------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the loop to finish its current cycle and return.

        Not a shutdown handler. Nothing is flushed and nothing is cancelled,
        because a stop that has work to do is a stop that does not happen when
        the process is killed instead. Everything the desk needs is already
        durable before this is called.
        """
        self._stop.set()

    def install_signal_handlers(self) -> None:
        """Make SIGTERM and SIGINT request a stop rather than kill mid-cycle.

        Only affects the tidy path. A SIGKILL, an OOM, or a machine
        disappearing still leaves the store consistent, because every stage
        commits before it acts.
        """
        for received in (signal.SIGTERM, signal.SIGINT):
            signal.signal(received, lambda *_: self.request_stop())

    def run(self, max_cycles: int | None = None) -> list[CycleOutcome]:
        """Run until stopped, or for ``max_cycles``.

        ``max_cycles`` exists for tests and for a supervised one-shot run; in
        production it is None and the loop ends only on a stop request.
        """
        findings = self.runner.resume(pd.Timestamp.now(tz="UTC"))
        for finding in findings:
            log_event(logger, "resume_finding", level=logging.WARNING, detail=finding)

        outcomes: list[CycleOutcome] = []
        cycles = 0
        while not self._stop.is_set():
            if max_cycles is not None and cycles >= max_cycles:
                break
            outcome = self.cycle()
            outcomes.append(outcome)
            cycles += 1

            if max_cycles is not None and cycles >= max_cycles:
                break
            # Interruptible: `Event.wait` returns immediately on a stop request,
            # so a SIGTERM does not have to wait out a fifteen-minute backoff.
            self._stop.wait(self.sleep_seconds())
        return outcomes

    def sleep_seconds(self) -> float:
        """Interval, widened by consecutive failures and capped."""
        if self._failures == 0:
            return self.interval
        widened = self.interval * (BACKOFF_FACTOR ** min(self._failures, 10))
        return float(min(widened, self.max_interval))

    # --- one cycle --------------------------------------------------------

    def cycle(self, now: pd.Timestamp | None = None) -> CycleOutcome:
        """One pass: poll, reconcile, trade, report. Never raises."""
        moment = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        with correlation_id():
            outcome = CycleOutcome(at=moment)
            self._poll(outcome)
            self._trade(outcome, moment)
            self._report(outcome, moment)

            if outcome.errors:
                self._failures += 1
                log_event(
                    logger, "cycle_failed", level=logging.ERROR,
                    failures=self._failures, errors=outcome.errors,
                    next_sleep_s=self.sleep_seconds(),
                )
            else:
                self._failures = 0
            return outcome

    def _poll(self, outcome: CycleOutcome) -> None:
        """Book what filled, before anything decides against the old book."""
        if self.poller is None:
            return
        try:
            outcome.poll = self.poller.poll()
            if outcome.poll.error:
                outcome.errors.append(f"fill poll: {outcome.poll.error}")
        except Exception as exc:
            outcome.errors.append(f"fill poll raised: {type(exc).__name__}: {exc}")
            logger.exception("fill poll raised")

    def _trade(self, outcome: CycleOutcome, moment: pd.Timestamp) -> None:
        """Decide, if the calendar and the store both permit it."""
        try:
            if self.store.is_halted():
                # The loop continues on a halted desk: fills still need
                # booking and health still needs reporting, and a halt is
                # exactly when someone is watching.
                outcome.skipped_trading = "desk is halted"
                return
            if not self.is_open(moment):
                outcome.skipped_trading = "market closed"
                return

            context = self.context_factory()
            if context is None:
                outcome.skipped_trading = "no fresh bar"
                return
            outcome.tick = self.runner.tick(context, moment)
        except Exception as exc:
            outcome.errors.append(f"tick raised: {type(exc).__name__}: {exc}")
            logger.exception("tick raised")

    def _report(self, outcome: CycleOutcome, moment: pd.Timestamp) -> None:
        """Assess health and alert on what changed."""
        try:
            report = check_health(self.store, now=moment)
            outcome.health = report

            failing = {
                alert.key for alert in alerts_for_health(report)
            }
            for alert in alerts_for_health(report):
                self.alerts.send(alert)
            # Pair a recovery with each condition that has cleared. Without
            # this the only sign a problem ended is that alerts stopped, which
            # is indistinguishable from the alerter having died.
            for key in sorted(self._alerting - failing):
                self.alerts.resolve(key, moment)
            self._alerting = failing

            if report.status is not HealthStatus.OK:
                log_event(
                    logger, "health_degraded", level=logging.WARNING,
                    status=report.status.value, failing=report.failing,
                )
        except Exception as exc:
            outcome.errors.append(f"health check raised: {type(exc).__name__}: {exc}")
            logger.exception("health check raised")


def sleep_until(deadline: pd.Timestamp, *, stop: threading.Event | None = None) -> None:
    """Wait until ``deadline``, interruptibly.

    A plain ``time.sleep`` cannot be interrupted, so a supervisor sleeping
    through a long backoff would ignore a SIGTERM for its whole duration.
    """
    remaining = (deadline - pd.Timestamp.now(tz="UTC")).total_seconds()
    if remaining <= 0:
        return
    if stop is not None:
        stop.wait(remaining)
    else:
        time.sleep(remaining)
