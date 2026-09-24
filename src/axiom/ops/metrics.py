"""Numbers a dashboard can scrape, and a histogram that does not lie.

:mod:`axiom.ops.logs` answers "what happened". This answers "how much, how
often, how slow" — the questions you ask *before* you know which log line to
look for. They are different tools: you cannot compute a p99 latency by reading
log lines, and you cannot debug one incident from a counter.

Three instrument types, and the third is the one that matters:

**Counter** — monotonic. Orders sent, fills booked, alerts raised. The value of
a counter is not the number; it is the *rate*, which a scraper derives.

**Gauge** — a level that moves both ways. Equity, open positions, gross
exposure.

**Histogram** — a distribution in fixed buckets. Averages hide the tail, and
the tail is the entire subject: a tick loop averaging 200 ms with a p99 of
30 seconds is a desk that misses one bar in a hundred, and the mean says it is
healthy. Buckets are cumulative (Prometheus convention), so quantiles can be
interpolated by the scraper rather than estimated here.

Deliberately in-process and unbounded-free: no background thread, no network,
no push. A metrics client that opens a socket is one more thing that can hang
inside the tick loop it is supposed to be measuring. :meth:`Registry.render`
produces the Prometheus text format, and getting it to a scraper is the
operator's business — an HTTP handler, a file the node exporter reads, a line
in the health report.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

#: Latency buckets in seconds, spanning a fast in-memory call to a wedged one.
#: Chosen so the interesting region for a trading tick — roughly 50 ms to 10 s —
#: has resolution, rather than being spread evenly across a range nobody cares
#: about.
DEFAULT_BUCKETS = (
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)

#: Label values are interpolated into the exposition format, so a value
#: containing a quote or newline would produce a corrupt scrape.
_ESCAPES = str.maketrans({"\\": r"\\", '"': r"\"", "\n": r"\n"})

Labels = Mapping[str, str]


def _key(labels: Labels | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _render_labels(key: tuple[tuple[str, str], ...], extra: str = "") -> str:
    parts = [f'{name}="{value.translate(_ESCAPES)}"' for name, value in key]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


@dataclass(slots=True)
class Counter:
    """A value that only goes up."""

    name: str
    help: str = ""
    _values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError(
                f"counter {self.name!r} cannot decrease; use a Gauge for a level "
                "that moves both ways"
            )
        key = _key(labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if not self._values:
            lines.append(f"{self.name} 0")
        lines.extend(
            f"{self.name}{_render_labels(key)} {value:g}"
            for key, value in sorted(self._values.items())
        )
        return lines


@dataclass(slots=True)
class Gauge:
    """A level."""

    name: str
    help: str = ""
    _values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def set(self, value: float, **labels: str) -> None:
        if not math.isfinite(value):
            raise ValueError(
                f"gauge {self.name!r} was set to {value}; a non-finite metric "
                "corrupts every aggregation downstream of it"
            )
        self._values[_key(labels)] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _key(labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        lines.extend(
            f"{self.name}{_render_labels(key)} {value:g}"
            for key, value in sorted(self._values.items())
        )
        return lines


@dataclass(slots=True)
class Histogram:
    """A distribution over fixed cumulative buckets.

    Stores counts, not samples, so memory is bounded regardless of how long the
    desk runs — which is the property that makes it safe to leave in a loop
    that ticks for months.
    """

    name: str
    help: str = ""
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    _counts: dict[tuple[tuple[str, str], ...], list[int]] = field(default_factory=dict)
    _sums: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("a histogram needs at least one bucket")
        if list(self.buckets) != sorted(self.buckets):
            raise ValueError(f"buckets for {self.name!r} must be ascending")

    def observe(self, value: float, **labels: str) -> None:
        if not math.isfinite(value):
            return
        key = _key(labels)
        counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
        self._sums[key] = self._sums.get(key, 0.0) + value
        # Cumulative: a sample lands in its own bucket and every wider one, so
        # bucket[i] means "count of samples <= buckets[i]". A sample above every
        # finite edge is counted only by +Inf, which is why +Inf is the total.
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                counts[i] += 1
        counts[-1] += 1

    def count(self, **labels: str) -> int:
        counts = self._counts.get(_key(labels))
        return counts[-1] if counts else 0

    def quantile(self, q: float, **labels: str) -> float:
        """Interpolation-free quantile: the first bucket edge that contains it.

        Reported as a bucket boundary rather than an interpolated value on
        purpose. Interpolating inside a bucket invents precision the histogram
        does not have, and a p99 quoted to three decimals invites someone to
        act on the third one.
        """
        if not 0 < q <= 1:
            raise ValueError("quantile must be in (0, 1]")
        counts = self._counts.get(_key(labels))
        if not counts or counts[-1] == 0:
            return math.nan
        target = q * counts[-1]
        for i, edge in enumerate(self.buckets):
            if counts[i] >= target:
                return edge
        return math.inf

    @contextmanager
    def time(self, **labels: str) -> Iterator[None]:
        """Observe how long the block took, including when it raises.

        Timing only the success path is how a latency panel stays green through
        an outage: the slow calls are exactly the ones that time out.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - started, **labels)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, counts in sorted(self._counts.items()):
            edges = [f"{edge:g}" for edge in self.buckets] + ["+Inf"]
            for i, edge in enumerate(edges):
                labels = _render_labels(key, 'le="' + edge + '"')
                lines.append(f"{self.name}_bucket{labels} {counts[i]}")
            lines.append(f"{self.name}_sum{_render_labels(key)} {self._sums.get(key, 0.0):g}")
            lines.append(f"{self.name}_count{_render_labels(key)} {counts[-1]}")
        return lines


class Registry:
    """The metrics the process holds.

    Instrument creation is idempotent by name, so a module can ask for its
    counter at import time or at call time and get the same object either way —
    which matters because the alternative is two counters with the same name
    and a scrape that reports whichever was rendered last.

    Guarded by a lock because the supervisor polls fills on one path and ticks
    on another, and a torn read of a dict during rendering is a real, if rare,
    way to lose a scrape.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def _get_or_create(
        self, name: str, kind: type[Counter] | type[Gauge] | type[Histogram], **kwargs: object
    ) -> Counter | Gauge | Histogram:
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, kind):
                    raise ValueError(
                        f"metric {name!r} is already registered as a "
                        f"{type(existing).__name__}, not a {kind.__name__}"
                    )
                return existing
            metric = kind(name=name, **kwargs)  # type: ignore[arg-type]
            self._metrics[name] = metric
            return metric

    def counter(self, name: str, help: str = "") -> Counter:
        metric = self._get_or_create(name, Counter, help=help)
        assert isinstance(metric, Counter)
        return metric

    def gauge(self, name: str, help: str = "") -> Gauge:
        metric = self._get_or_create(name, Gauge, help=help)
        assert isinstance(metric, Gauge)
        return metric

    def histogram(
        self, name: str, help: str = "", buckets: tuple[float, ...] = DEFAULT_BUCKETS
    ) -> Histogram:
        metric = self._get_or_create(name, Histogram, help=help, buckets=buckets)
        assert isinstance(metric, Histogram)
        return metric

    def render(self) -> str:
        """The Prometheus text exposition format."""
        with self._lock:
            metrics = list(self._metrics.values())
        lines: list[str] = []
        for metric in sorted(metrics, key=lambda m: m.name):
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()


#: The process-wide registry. A module-level singleton is the right shape here
#: for the same reason logging uses one: the alternative is threading a
#: registry through every constructor so that a counter can be incremented.
REGISTRY = Registry()


# The desk's standard instruments, named up front so they are consistent
# wherever they are incremented and so a dashboard can be built against them
# before the code that fills them exists.
ORDERS_SENT = REGISTRY.counter("axiom_orders_sent_total", "Orders accepted by a venue.")
ORDERS_REJECTED = REGISTRY.counter("axiom_orders_rejected_total", "Orders a venue refused.")
FILLS_BOOKED = REGISTRY.counter("axiom_fills_booked_total", "Executions recorded in the store.")
SIGNALS_GENERATED = REGISTRY.counter("axiom_signals_total", "Signals produced by strategies.")
SIGNALS_BLOCKED = REGISTRY.counter(
    "axiom_signals_blocked_total", "Signals a guard, risk check or compliance rule stopped."
)
HALTS = REGISTRY.counter("axiom_halts_total", "Times the desk halted itself.")
ALERTS_SENT = REGISTRY.counter("axiom_alerts_sent_total", "Alerts delivered to a sink.")

EQUITY = REGISTRY.gauge("axiom_equity", "Last marked account equity.")
OPEN_POSITIONS = REGISTRY.gauge("axiom_open_positions", "Instruments currently held.")
GROSS_EXPOSURE = REGISTRY.gauge("axiom_gross_exposure", "Sum of absolute position notional.")
IS_HALTED = REGISTRY.gauge("axiom_halted", "1 when trading is halted, 0 otherwise.")

TICK_SECONDS = REGISTRY.histogram("axiom_tick_seconds", "Wall time of one desk tick.")
VENUE_SECONDS = REGISTRY.histogram("axiom_venue_seconds", "Wall time of a venue call.")
