"""Pooled MCP sessions are reused across calls in one query."""

from __future__ import annotations

import asyncio
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
async def test_pool_forwards_optional_progress_callback() -> None:
    seen: list[str] = []

    class _StreamingSession:
        async def call_tool(
            self,
            name: str,
            arguments: Mapping[str, Any] | None = None,
            *,
            meta: dict[str, Any] | None = None,
            progress_callback: Any | None = None,
        ) -> dict[str, str]:
            _ = name, arguments, meta
            if progress_callback is not None:
                await progress_callback(1.0, None, "live")
            return {"ok": "yes"}

        async def aclose(self) -> None:
            return None

    async def opener(agent: str) -> _StreamingSession:
        _ = agent
        return _StreamingSession()

    async def _progress(progress: float, total: float | None, message: str | None) -> None:
        _ = progress, total
        if message:
            seen.append(message)

    pool = AgentSessionPool({"orchestrator": "http://orch"}, open_session=opener)
    without = await pool.call(
        "orchestrator",
        "health",
        {},
        correlation_id="c1",
        timeout_s=1.0,
    )
    assert without == {"ok": "yes"}
    assert seen == []
    with_stream = await pool.call(
        "orchestrator",
        "handle_query",
        {},
        correlation_id="c1",
        timeout_s=1.0,
        progress_callback=_progress,
    )
    assert with_stream == {"ok": "yes"}
    assert seen == ["live"]
    await pool.aclose()


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
async def test_stale_reused_session_is_purged_and_call_retried_fresh() -> None:
    """Idle sessions broken by cancel-scope task affinity must not fail calls.

    A reused idle session raising a non-transient RuntimeError is dropped, the
    remaining idle sessions are purged, and the call retries once on a freshly
    opened session.
    """
    opens = 0

    class _Session:
        def __init__(self, *, stale: bool) -> None:
            self.stale = stale
            self.closed = False

        async def call_tool(
            self,
            name: str,
            arguments: Mapping[str, Any] | None = None,
            *,
            meta: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            _ = name, arguments, meta
            if self.stale:
                raise RuntimeError(
                    "Attempted to exit cancel scope in a different task than it was entered in"
                )
            await asyncio.sleep(0.01)
            return {"ok": True}

        async def aclose(self) -> None:
            self.closed = True

    created: list[_Session] = []

    async def opener(agent: str) -> _Session:
        nonlocal opens
        _ = agent
        opens += 1
        session = _Session(stale=False)
        created.append(session)
        return session

    pool = AgentSessionPool({"graph_query": "http://graph"}, open_session=opener)
    # Seed the pool with two healthy sessions, then mark them stale to model
    # sessions opened in an earlier (now finished) request task.
    await asyncio.gather(
        pool.call("graph_query", "warm_a", {}, correlation_id="q0", timeout_s=1.0),
        pool.call("graph_query", "warm_b", {}, correlation_id="q0", timeout_s=1.0),
    )
    assert opens == 2
    for session in created:
        session.stale = True

    result = await pool.call(
        "graph_query",
        "execute_query",
        {"cypher": "MATCH (n) RETURN n"},
        correlation_id="q1",
        timeout_s=1.0,
    )
    assert result == {"ok": True}
    # One stale session was dropped, the other purged, one fresh session opened.
    assert opens == 3
    assert created[0].closed and created[1].closed
    await pool.aclose()


@pytest.mark.asyncio
async def test_fresh_session_failure_still_propagates() -> None:
    """A brand-new session failing means real connectivity trouble: no retry."""

    class _Broken:
        async def call_tool(
            self,
            name: str,
            arguments: Mapping[str, Any] | None = None,
            *,
            meta: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            _ = name, arguments, meta
            raise RuntimeError("boom")

        async def aclose(self) -> None:
            return None

    async def opener(agent: str) -> _Broken:
        _ = agent
        return _Broken()

    pool = AgentSessionPool({"graph_query": "http://graph"}, open_session=opener)
    with pytest.raises(RuntimeError, match="boom"):
        await pool.call(
            "graph_query",
            "find_entity",
            {},
            correlation_id="q1",
            timeout_s=1.0,
        )
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


@pytest.mark.asyncio
async def test_concurrent_calls_to_same_agent_overlap() -> None:
    in_flight = 0
    max_in_flight = 0
    ready = asyncio.Event()
    entered = 0

    class _SlowSession:
        async def call_tool(
            self,
            name: str,
            arguments: Mapping[str, Any] | None = None,
            *,
            meta: dict[str, Any] | None = None,
        ) -> dict[str, bool]:
            nonlocal in_flight, max_in_flight, entered
            _ = name, arguments, meta
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            entered += 1
            if entered >= 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=1.0)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return {"ok": True}

        async def aclose(self) -> None:
            return None

    async def opener(agent: str) -> _SlowSession:
        _ = agent
        return _SlowSession()

    pool = AgentSessionPool(
        {"graph_query": "http://graph"},
        open_session=opener,
        max_sessions_per_agent=2,
    )
    await asyncio.gather(
        pool.call(
            "graph_query",
            "find_entity",
            {},
            correlation_id="c1",
            timeout_s=2.0,
        ),
        pool.call(
            "graph_query",
            "get_dependents",
            {},
            correlation_id="c2",
            timeout_s=2.0,
        ),
    )
    assert max_in_flight == 2
    assert pool.open_counts["graph_query"] == 2
    await pool.aclose()
