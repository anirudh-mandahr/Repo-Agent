"""Circuit breaker open/close behavior."""

from __future__ import annotations

import pytest

from core.exceptions import CircuitBreakerOpenError
from core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from core.settings import OrchestratorSettings


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_closes() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(
        "graph_query",
        failure_threshold=2,
        cooldown_s=10.0,
        clock=clock,
    )

    async def fail() -> str:
        raise TimeoutError("down")

    async def ok() -> str:
        return "ok"

    with pytest.raises(TimeoutError):
        await breaker.run(fail)
    assert breaker.state == "closed"

    with pytest.raises(TimeoutError):
        await breaker.run(fail)
    assert breaker.state == "open"
    snapshot = breaker.snapshot()
    assert snapshot.consecutive_failures == 2
    assert snapshot.cooldown_remaining_s == 10.0

    with pytest.raises(CircuitBreakerOpenError, match="circuit breaker open"):
        await breaker.run(ok)

    clock.advance(10.0)
    assert breaker.state == "half_open"
    assert await breaker.run(ok) == "ok"
    assert breaker.state == "closed"
    assert breaker.snapshot().consecutive_failures == 0


@pytest.mark.asyncio
async def test_half_open_failure_reopens_immediately() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(
        "code_analyst",
        failure_threshold=1,
        cooldown_s=5.0,
        clock=clock,
    )

    async def fail() -> str:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await breaker.run(fail)
    assert breaker.state == "open"
    clock.advance(5.0)
    with pytest.raises(ConnectionError):
        await breaker.run(fail)
    assert breaker.state == "open"


def test_registry_uses_per_agent_thresholds() -> None:
    clock = _Clock()
    settings = OrchestratorSettings(
        breaker_failure_threshold=3,
        breaker_cooldown_s=30.0,
        breaker_graph_query_failure_threshold=1,
        breaker_graph_query_cooldown_s=5.0,
    )
    registry = CircuitBreakerRegistry.from_orchestrator_settings(settings, clock=clock)
    graph = registry.get("graph_query")
    memory = registry.get("memory")
    assert graph._failure_threshold == 1
    assert graph._cooldown_s == 5.0
    assert memory._failure_threshold == 3
    assert memory._cooldown_s == 30.0
    graph.record_failure()
    snapshots = registry.snapshot()
    assert snapshots["graph_query"].state == "open"
    assert "memory" in snapshots
    assert snapshots["memory"].state == "closed"
