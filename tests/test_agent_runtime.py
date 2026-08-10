"""The governors on model access. No network — the transport is injected.

The property under test throughout: every path that spends money, blocks, or
retries is bounded, and the bound is enforced rather than documented.
"""

from __future__ import annotations

import pytest

from axiom.agents.runtime import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitOpenError,
    LLMBudget,
    LLMRuntime,
    ResponseCache,
    Usage,
    estimate_cost,
    prompt_digest,
)


class _Transport:
    """Scripted stand-in for the Anthropic client."""

    def __init__(
        self,
        *,
        text: str = "ok",
        tokens: tuple[int, int] = (100, 50),
        fail_times: int = 0,
        error: type[Exception] = RuntimeError,
        error_name: str = "",
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.fail_times = fail_times
        # The runtime classifies retryability by exception class *name*, so the
        # SDK stays an optional import. Tests forge the name the same way.
        self.error = (
            type(error_name, (Exception,), {}) if error_name else error
        )
        self.calls = 0

    def __call__(self, **_: object) -> tuple[str, int, int, int, int]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error("transport failure")
        return (self.text, self.tokens[0], self.tokens[1], 0, 0)


def _runtime(transport: _Transport, **kwargs: object) -> LLMRuntime:
    kwargs.setdefault("backoff_base_s", 0.0)
    return LLMRuntime(transport=transport, **kwargs)  # type: ignore[arg-type]


class TestCostAccounting:
    def test_opus_5_priced_from_the_published_rate_card(self) -> None:
        # 1M in + 1M out at $5 / $25.
        assert estimate_cost(
            "claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000
        ) == pytest.approx(30.0)

    def test_cache_reads_bill_at_a_tenth_of_input(self) -> None:
        assert estimate_cost(
            "claude-opus-5", cache_read_tokens=1_000_000
        ) == pytest.approx(0.50)

    def test_cache_writes_bill_above_input(self) -> None:
        assert estimate_cost(
            "claude-opus-5", cache_write_tokens=1_000_000
        ) == pytest.approx(6.25)

    def test_an_unknown_model_costs_zero_rather_than_a_guess(self) -> None:
        """A fabricated price would flow into the budget check and mis-gate it."""
        assert estimate_cost("some-future-model", input_tokens=1_000_000) == 0.0

    def test_usage_is_additive_across_stages(self) -> None:
        total = Usage(calls=1, input_tokens=10, cost_usd=1.0) + Usage(
            calls=2, input_tokens=5, cost_usd=0.5
        )
        assert (total.calls, total.input_tokens, total.cost_usd) == (3, 15, 1.5)


class TestBudget:
    def test_call_budget_stops_the_run(self) -> None:
        transport = _Transport()
        runtime = _runtime(transport, budget=LLMBudget(max_calls=2))
        for _ in range(2):
            runtime.complete(system="s", prompt=f"p{_}", max_tokens=10)
        with pytest.raises(BudgetExceededError, match="call budget"):
            runtime.complete(system="s", prompt="p-final", max_tokens=10)

    def test_an_exhausted_budget_costs_nothing_to_discover(self) -> None:
        """The ceiling is checked before the request, not after paying for it."""
        transport = _Transport()
        runtime = _runtime(transport, budget=LLMBudget(max_calls=1))
        runtime.complete(system="s", prompt="a", max_tokens=10)
        with pytest.raises(BudgetExceededError):
            runtime.complete(system="s", prompt="b", max_tokens=10)
        assert transport.calls == 1

    def test_spend_budget_stops_the_run(self) -> None:
        transport = _Transport(tokens=(1_000_000, 1_000_000))
        # Zero disables the other ceilings, isolating the dollar cap.
        runtime = _runtime(
            transport,
            model="claude-opus-5",
            budget=LLMBudget(
                max_calls=0, max_input_tokens=0, max_output_tokens=0, max_cost_usd=1.0
            ),
        )
        # The first call is permitted — it starts under the cap — and raises on
        # the way out, so the run stops at the breach rather than one call later.
        with pytest.raises(BudgetExceededError, match="spend budget"):
            runtime.complete(system="s", prompt="expensive", max_tokens=10)
        assert runtime.usage.cost_usd == pytest.approx(30.0)


class TestCache:
    def test_an_identical_request_is_not_re_sent(self) -> None:
        transport = _Transport()
        runtime = _runtime(transport)
        first = runtime.complete(system="s", prompt="p", max_tokens=10)
        second = runtime.complete(system="s", prompt="p", max_tokens=10)
        assert transport.calls == 1
        assert second.cached and not first.cached
        assert second.text == first.text

    def test_a_different_prompt_is_a_different_entry(self) -> None:
        transport = _Transport()
        runtime = _runtime(transport)
        runtime.complete(system="s", prompt="a", max_tokens=10)
        runtime.complete(system="s", prompt="b", max_tokens=10)
        assert transport.calls == 2

    def test_max_tokens_participates_in_the_key(self) -> None:
        """Same question, different output ceiling, is a different request."""
        assert prompt_digest("m", "s", "p", 100) != prompt_digest("m", "s", "p", 200)

    def test_a_cached_call_bills_nothing(self) -> None:
        runtime = _runtime(_Transport(tokens=(1000, 1000)), model="claude-opus-5")
        runtime.complete(system="s", prompt="p", max_tokens=10)
        billed = runtime.usage.cost_usd
        runtime.complete(system="s", prompt="p", max_tokens=10)
        assert runtime.usage.cost_usd == billed

    def test_cache_records_hits_and_misses(self) -> None:
        cache = ResponseCache()
        assert cache.get("nope") is None
        cache.put("d", "text")
        assert cache.get("d") == "text"
        assert (cache.hits, cache.misses) == (1, 1)


class TestRetry:
    def test_transient_failures_are_retried(self) -> None:
        transport = _Transport(fail_times=2, error_name="RateLimitError")
        result = _runtime(transport).complete(system="s", prompt="p", max_tokens=10)
        assert transport.calls == 3
        assert result.attempts == 3

    def test_a_malformed_request_is_not_retried(self) -> None:
        """A 400 fails identically every time; retrying it just spends budget."""
        transport = _Transport(fail_times=99, error_name="BadRequestError")
        with pytest.raises(Exception, match="transport failure"):
            _runtime(transport).complete(system="s", prompt="p", max_tokens=10)
        assert transport.calls == 1

    def test_exhausted_retries_raise_with_the_cause(self) -> None:
        transport = _Transport(fail_times=99, error_name="APITimeoutError")
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            _runtime(transport).complete(system="s", prompt="p", max_tokens=10)
        assert transport.calls == 3


class TestCircuitBreaker:
    def test_opens_after_repeated_failures(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, reset_after_s=60.0)
        breaker.record_failure()
        assert not breaker.is_open
        breaker.record_failure()
        assert breaker.is_open

    def test_an_open_circuit_refuses_without_calling(self) -> None:
        transport = _Transport()
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=60.0)
        breaker.record_failure()
        runtime = _runtime(transport, breaker=breaker)
        with pytest.raises(CircuitOpenError, match="circuit open"):
            runtime.complete(system="s", prompt="p", max_tokens=10)
        assert transport.calls == 0

    def test_success_resets_the_failure_count(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open

    def test_reopens_after_the_cooldown(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=0.0)
        breaker.record_failure()
        breaker.before_call()  # half-open: permitted through
        assert not breaker.is_open

    def test_a_failing_run_trips_the_breaker(self) -> None:
        transport = _Transport(fail_times=99, error_name="APIConnectionError")
        breaker = CircuitBreaker(failure_threshold=1)
        runtime = _runtime(transport, breaker=breaker)
        with pytest.raises(RuntimeError):
            runtime.complete(system="s", prompt="p", max_tokens=10)
        assert breaker.is_open
