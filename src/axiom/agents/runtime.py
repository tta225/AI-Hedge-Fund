"""Governed access to the model. Nothing in this package calls Anthropic directly.

The agent layer's original ``_interpret`` was a bare SDK call: no timeout, no
retry, no spend ceiling, no record of what anything cost. That is fine for a
demo and unacceptable on a desk. One hung request stalls a pipeline stage
indefinitely; a loop with a bug spends without limit; and nobody can answer
"what did last night's research run cost" after the fact.

:class:`LLMRuntime` is the single chokepoint every agent goes through. It
enforces, in this order:

1. **Circuit breaker** — after repeated failures the provider is presumed down
   and calls fail immediately rather than each waiting out its own timeout.
2. **Cache** — an identical (model, system, prompt) triple returns the previous
   answer. This is what makes a run *reproducible*: replaying a pipeline against
   a warm cache produces byte-identical narratives, so a differing report means
   the facts changed, not that the model rolled differently.
3. **Budget** — a hard ceiling on calls, tokens, and dollars. Checked before the
   request (on the accumulated total) and again after, so a single expensive
   call cannot silently blow through a cap it started under.
4. **Timeout and retry** — bounded exponential backoff with jitter, and only on
   errors that are actually transient. A 400 is a bug in our request and
   retrying it just spends the budget three times.

Every call returns a :class:`LLMResult` carrying token counts, estimated cost,
attempt count, and wall-clock latency, so the metrics exist whether or not
anyone is watching.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

#: Published per-million-token prices, ``(input, output)``, in USD.
#:
#: Cost is *estimated* — it is what the published rate card implies for the
#: tokens the API reported. It is not a bill, and any negotiated rate makes it
#: an overestimate. It is here to make spend visible and boundable, which a
#: rate card is sufficient for; it is not an accounting record.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

#: Cache reads bill at a tenth of the input rate; writes at 1.25× (5-minute TTL).
_CACHE_READ_MULTIPLIER = 0.10
_CACHE_WRITE_MULTIPLIER = 1.25

#: Error types worth retrying. Matched by class name so the SDK stays an
#: optional import — the agent layer must be usable with no `anthropic` present.
_TRANSIENT_ERRORS = frozenset(
    {
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "OverloadedError",
    }
)


class BudgetExceededError(RuntimeError):
    """A call was refused because it would breach the run's spend ceiling."""


class CircuitOpenError(RuntimeError):
    """Calls are being refused because the provider is presumed unhealthy."""


@dataclass(frozen=True, slots=True)
class LLMBudget:
    """Hard ceilings for one pipeline run. Zero means unlimited.

    The dollar cap is the one that matters operationally; the token and call
    caps are there because a runaway loop hits them sooner and more obviously.
    """

    max_calls: int = 64
    max_input_tokens: int = 2_000_000
    max_output_tokens: int = 200_000
    max_cost_usd: float = 5.00


@dataclass(slots=True)
class Usage:
    """Accumulated consumption. Additive, so it aggregates across stages."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            calls=self.calls + other.calls,
            cached_calls=self.cached_calls + other.cached_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
        }


def estimate_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Published-rate cost in USD. Unknown models price at zero, not at a guess.

    A wrong number is worse than an absent one here: it would flow into the
    budget check and either throttle a run that was within its cap or let one
    through that was not.
    """
    prices = MODEL_PRICES_PER_MTOK.get(model)
    if prices is None:
        return 0.0
    price_in, price_out = prices
    return (
        input_tokens * price_in
        + output_tokens * price_out
        + cache_read_tokens * price_in * _CACHE_READ_MULTIPLIER
        + cache_write_tokens * price_in * _CACHE_WRITE_MULTIPLIER
    ) / 1_000_000


@dataclass(frozen=True, slots=True)
class LLMResult:
    """One completion, with everything needed to audit and bill it."""

    text: str
    model: str
    usage: Usage
    cached: bool = False
    attempts: int = 1
    #: Digest of (model, system, prompt). Two runs sharing a digest were asked
    #: exactly the same question — the anchor for reproducibility claims.
    prompt_digest: str = ""


class CircuitBreaker:
    """Fails fast once a provider looks down, and probes before reopening.

    Without this, a provider outage costs every stage its full timeout budget
    in sequence. A five-stage pipeline with a 60s timeout takes five minutes to
    tell you the API is unreachable.
    """

    def __init__(self, *, failure_threshold: int = 4, reset_after_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and not self._cooled_down()

    def _cooled_down(self) -> bool:
        return (
            self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.reset_after_s
        )

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if self._cooled_down():
                # Half-open: let one call through. If it fails, record_failure
                # re-opens the breaker on the spot.
                self._opened_at = None
                self._failures = self.failure_threshold - 1
                return
            waited = time.monotonic() - self._opened_at
            raise CircuitOpenError(
                f"provider circuit open after {self._failures} consecutive failures; "
                f"retrying in {self.reset_after_s - waited:.0f}s"
            )

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()


class ResponseCache:
    """In-memory completion cache keyed by the full request.

    Deterministic replay is the point, not the saved dollars. A run replayed
    against a warm cache produces identical narratives, so any difference in the
    report is attributable to the facts.
    """

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, digest: str) -> str | None:
        with self._lock:
            text = self._entries.get(digest)
            if text is None:
                self.misses += 1
            else:
                self.hits += 1
            return text

    def put(self, digest: str, text: str) -> None:
        with self._lock:
            self._entries[digest] = text

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def prompt_digest(model: str, system: str, prompt: str, max_tokens: int) -> str:
    """Content address for a request. Same digest ⇒ same question, verbatim."""
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "max_tokens": max_tokens},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMRuntime:
    """The only place in this package that talks to a model.

    Thread-safe: the orchestrator runs independent stages concurrently, and they
    share one runtime so budgets and usage are enforced across the whole run
    rather than per stage.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str = "",
        budget: LLMBudget | None = None,
        timeout_s: float = 90.0,
        max_attempts: int = 3,
        backoff_base_s: float = 1.0,
        cache: ResponseCache | None = None,
        breaker: CircuitBreaker | None = None,
        transport: Callable[..., tuple[str, int, int, int, int]] | None = None,
    ) -> None:
        """
        Args:
            transport: injection seam for tests. Receives the request and
                returns ``(text, input, output, cache_read, cache_write)``.
                Defaults to the real Anthropic client.
        """
        self.model = model
        self.api_key = api_key
        self.budget = budget or LLMBudget()
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.cache = cache if cache is not None else ResponseCache()
        self.breaker = breaker or CircuitBreaker()
        self._transport = transport
        self._usage = Usage()
        self._lock = threading.Lock()

    @property
    def usage(self) -> Usage:
        with self._lock:
            return replace(self._usage)

    def _check_budget(self, usage: Usage) -> None:
        budget = self.budget
        if budget.max_calls and usage.calls > budget.max_calls:
            raise BudgetExceededError(f"call budget exhausted ({budget.max_calls} calls)")
        if budget.max_input_tokens and usage.input_tokens > budget.max_input_tokens:
            raise BudgetExceededError(
                f"input token budget exhausted ({budget.max_input_tokens:,} tokens)"
            )
        if budget.max_output_tokens and usage.output_tokens > budget.max_output_tokens:
            raise BudgetExceededError(
                f"output token budget exhausted ({budget.max_output_tokens:,} tokens)"
            )
        if budget.max_cost_usd and usage.cost_usd > budget.max_cost_usd:
            raise BudgetExceededError(
                f"spend budget exhausted (${usage.cost_usd:,.2f} of "
                f"${budget.max_cost_usd:,.2f})"
            )

    def complete(
        self, *, system: str, prompt: str, max_tokens: int = 8_000, model: str | None = None
    ) -> LLMResult:
        """Run one completion under every governor. Raises rather than degrading.

        A caller that would rather have measured facts with no narrative than no
        report at all is expected to catch — :class:`~axiom.agents.base.Agent`
        does exactly that.
        """
        model = model or self.model
        digest = prompt_digest(model, system, prompt, max_tokens)

        if (hit := self.cache.get(digest)) is not None:
            with self._lock:
                self._usage.cached_calls += 1
            return LLMResult(
                text=hit,
                model=model,
                usage=Usage(cached_calls=1),
                cached=True,
                attempts=0,
                prompt_digest=digest,
            )

        # Budget is checked against the running total before spending anything,
        # so an exhausted budget costs nothing to discover.
        with self._lock:
            projected = replace(self._usage, calls=self._usage.calls + 1)
        self._check_budget(projected)

        self.breaker.before_call()
        started = time.monotonic()
        text, usage, attempts = self._call_with_retry(
            system=system, prompt=prompt, max_tokens=max_tokens, model=model
        )
        usage.latency_s = time.monotonic() - started

        self.cache.put(digest, text)
        with self._lock:
            self._usage = self._usage + usage
            total = replace(self._usage)
        # Re-checked after the fact: one call can cross a ceiling it started
        # under, and the run should stop at that point rather than one call later.
        self._check_budget(total)

        return LLMResult(
            text=text, model=model, usage=usage, attempts=attempts, prompt_digest=digest
        )

    def _call_with_retry(
        self, *, system: str, prompt: str, max_tokens: int, model: str
    ) -> tuple[str, Usage, int]:
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                text, tin, tout, cread, cwrite = self._invoke(
                    system=system, prompt=prompt, max_tokens=max_tokens, model=model
                )
            except Exception as exc:
                last = exc
                if type(exc).__name__ not in _TRANSIENT_ERRORS:
                    # A malformed request fails identically every time; retrying
                    # it burns budget and delays the error the caller needs.
                    self.breaker.record_failure()
                    raise
                if attempt == self.max_attempts:
                    break
                # Jitter matters with concurrent stages: without it, fan-out
                # stages that fail together retry together and re-collide.
                delay = self.backoff_base_s * (2 ** (attempt - 1))
                time.sleep(delay + random.uniform(0, delay * 0.25))
            else:
                self.breaker.record_success()
                return (
                    text,
                    Usage(
                        calls=1,
                        input_tokens=tin,
                        output_tokens=tout,
                        cache_read_tokens=cread,
                        cache_write_tokens=cwrite,
                        cost_usd=estimate_cost(
                            model,
                            input_tokens=tin,
                            output_tokens=tout,
                            cache_read_tokens=cread,
                            cache_write_tokens=cwrite,
                        ),
                    ),
                    attempt,
                )

        self.breaker.record_failure()
        raise RuntimeError(
            f"model call failed after {self.max_attempts} attempts: {last}"
        ) from last

    def _invoke(
        self, *, system: str, prompt: str, max_tokens: int, model: str
    ) -> tuple[str, int, int, int, int]:
        if self._transport is not None:
            return self._transport(
                system=system, prompt=prompt, max_tokens=max_tokens, model=model
            )
        return self._invoke_anthropic(
            system=system, prompt=prompt, max_tokens=max_tokens, model=model
        )

    def _invoke_anthropic(
        self, *, system: str, prompt: str, max_tokens: int, model: str
    ) -> tuple[str, int, int, int, int]:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the agents extra is not installed (pip install 'axiom[agents]')"
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_s)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            # The system prompt is byte-stable across every agent and every run,
            # which is what makes it worth a cache breakpoint: the facts vary in
            # the user turn, below the cached prefix.
            system=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        # A refusal is a successful HTTP response with no usable content. Reading
        # content[0] here would raise IndexError and be misreported as a bug.
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()
        usage = response.usage
        return (
            text,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
