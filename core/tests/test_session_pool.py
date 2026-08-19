"""Pooled MCP sessions are reused across calls in one query."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from core.exceptions import AgentUnavailableError
from core.mcp.client import PooledAgentClient
from core.resilience.session_pool import AgentSessionPool


class _FakeSession:
    def __init__(self, agent: str, calls: list[tuple[str, str]]) -> None:
        self.agent = agent
        self._calls = calls
        self.closed = False

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = arguments, meta
        self._calls.append((self.agent, name))
        return {"tool": name, "agent": self.agent}

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_pooled_sessions_reused_across_calls_in_one_query() -> None:
    opens: list[str] = []
    calls: list[tuple[str, str]] = []
    sessions: dict[str, _FakeSession] = {}

    async def opener(agent: str) -> _FakeSession:
        opens.append(agent)
        session = _FakeSession(agent, calls)
        sessions[agent] = session
        return session

    pool = AgentSessionPool(
        {"graph_query": "http://graph", "memory": "http://memory"},
        open_session=opener,
    )
    graph = PooledAgentClient(pool, "graph_query", timeout_s=1.0, retry_count=0)
    memory = PooledAgentClient(pool, "memory", timeout_s=1.0, retry_count=0)

    await graph.call("get_statistics", {}, correlation_id="q1")
    await graph.call("find_entity", {"name": "FastAPI"}, correlation_id="q1")
    await graph.call("get_dependents", {"name": "FastAPI"}, correlation_id="q1")
    await memory.call("get_context", {"session_id": "s1"}, correlation_id="q1")
    await memory.call("append_turn", {"session_id": "s1"}, correlation_id="q1")

    assert opens == ["graph_query", "memory"]
    assert pool.open_counts == {"graph_query": 1, "memory": 1}
    assert pool.call_counts == {"graph_query": 3, "memory": 2}
    assert [name for agent, name in calls if agent == "graph_query"] == [
        "get_statistics",
        "find_entity",
        "get_dependents",
    ]
    await pool.aclose()
    assert sessions["graph_query"].closed is True
    assert sessions["memory"].closed is True


@pytest.mark.asyncio
async def test_pool_drops_session_after_failure_and_reopens() -> None:
    opens = 0

    class _Flaky:
        def __init__(self, *, fail: bool) -> None:
            self._fail = fail

        async def call_tool(
            self,
            name: str,
            arguments: Mapping[str, Any] | None = None,
            *,
            meta: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            _ = name, arguments, meta
            if self._fail:
                raise ConnectionError("reset")
            return {"ok": True}

        async def aclose(self) -> None:
            return None

    async def opener(agent: str) -> _Flaky:
        nonlocal opens
        _ = agent
        opens += 1
        return _Flaky(fail=opens == 1)

    pool = AgentSessionPool({"graph_query": "http://graph"}, open_session=opener)
    client = PooledAgentClient(pool, "graph_query", timeout_s=1.0, retry_count=1)
    result = await client.call("find_entity", {"name": "FastAPI"}, correlation_id="q1")
    assert result == {"ok": True}
    assert opens == 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_closed_pool_rejects_calls() -> None:
    async def opener(agent: str) -> _FakeSession:
        return _FakeSession(agent, [])

    pool = AgentSessionPool({"graph_query": "http://graph"}, open_session=opener)
    await pool.aclose()
    with pytest.raises(AgentUnavailableError, match="pool closed"):
        await pool.call(
            "graph_query",
            "find_entity",
            {},
            correlation_id="q1",
            timeout_s=1.0,
        )
