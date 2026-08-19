"""Pooled Graph Query lookup used by the Code Analyst."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from core.analysis.graph_lookup import GraphQueryLookup, build_code_analyst_pool
from core.exceptions import CircuitBreakerOpenError, GraphLookupError
from core.resilience.session_pool import AgentSessionPool


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0
        self.closed = False

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = name, arguments, meta
        self.calls += 1
        return self.payload

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_graph_query_lookup_reuses_pooled_session() -> None:
    opens = 0
    session = _FakeSession({"rows": [{"n": 1}], "cypher_executed": "RETURN 1"})

    async def opener(agent: str) -> _FakeSession:
        nonlocal opens
        _ = agent
        opens += 1
        return session

    pool = AgentSessionPool({"graph_query": "http://graph.invalid/mcp"}, open_session=opener)
    lookup = GraphQueryLookup(pool, timeout_s=1.0, retry_count=0)
    first = await lookup("RETURN 1", {})
    second = await lookup("RETURN 2", {"k": "v"})
    assert first == [{"n": 1}]
    assert second == [{"n": 1}]
    assert opens == 1
    assert session.calls == 2
    assert pool.open_counts == {"graph_query": 1}


@pytest.mark.asyncio
async def test_graph_query_lookup_propagates_open_breaker() -> None:
    async def opener(agent: str) -> _FakeSession:
        _ = agent
        raise AssertionError("session must not open when the breaker is already open")

    pool = AgentSessionPool({"graph_query": "http://graph.invalid/mcp"}, open_session=opener)
    breaker = pool.breakers.get("graph_query")
    breaker._failure_threshold = 1
    breaker.record_failure()
    lookup = GraphQueryLookup(pool, timeout_s=1.0, retry_count=0)
    with pytest.raises(CircuitBreakerOpenError, match="circuit breaker open"):
        await lookup("RETURN 1", {})


@pytest.mark.asyncio
async def test_graph_query_lookup_wraps_unavailable_as_graph_lookup_error() -> None:
    class _Boom:
        async def call_tool(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            _ = args, kwargs
            raise ConnectionError("refused")

        async def aclose(self) -> None:
            return None

    async def opener(agent: str) -> _Boom:
        _ = agent
        return _Boom()

    pool = AgentSessionPool({"graph_query": "http://graph.invalid/mcp"}, open_session=opener)
    lookup = GraphQueryLookup(pool, timeout_s=1.0, retry_count=0)
    with pytest.raises(GraphLookupError, match="execute_query failed"):
        await lookup("RETURN 1", {})


def test_build_code_analyst_pool_targets_graph_query() -> None:
    pool = build_code_analyst_pool()
    assert "graph_query" in pool._urls


def test_agent_package_does_not_host_graph_lookup() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "agents"
        / "code_analyst"
        / "src"
        / "code_analyst"
        / "graph_lookup.py"
    )
    assert not path.exists()
