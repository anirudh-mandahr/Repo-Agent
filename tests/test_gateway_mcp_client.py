"""Gateway MCP client timeout, transport, and tool-call branches."""

from __future__ import annotations

from time import monotonic
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.exceptions import AgentUnavailableError
from core.health import HealthStatus
from core.memory import ConversationContext
from core.orchestration.models import ExecutionPlan, QueryIntent
from core.resilience.circuit_breaker import CircuitBreakerRegistry
from core.resilience.session_pool import AgentSessionPool
from core.settings import GatewaySettings
from gateway.mcp_client import (
    GatewayOrchestratorClient,
    GatewaySpecialistClients,
    call_agent_health,
    call_tool_json,
)


def _pool_that_raises(exc: BaseException) -> AgentSessionPool:
    async def opener(agent: str) -> Any:
        _ = agent
        raise exc

    return AgentSessionPool({"orchestrator": "http://example.invalid/mcp"}, open_session=opener)


@pytest.mark.asyncio
async def test_call_agent_health_timeout() -> None:
    pool = _pool_that_raises(TimeoutError("timed out"))
    status = await call_agent_health(
        "orchestrator",
        "http://example.invalid/mcp",
        pool=pool,
        timeout_s=0.01,
    )
    assert status.status == "error"
    assert "timed out" in (status.detail or "") or status.agent == "orchestrator"


@pytest.mark.asyncio
async def test_call_agent_health_transport_error() -> None:
    pool = _pool_that_raises(ConnectionError("connection reset"))
    status = await call_agent_health(
        "graph_query",
        "http://example.invalid/mcp",
        pool=pool,
    )
    assert status.status == "error"
    assert "connection reset" in (status.detail or "")


@pytest.mark.asyncio
async def test_call_agent_health_breaker_open() -> None:
    from time import monotonic

    registry = CircuitBreakerRegistry(
        default_failure_threshold=1, default_cooldown_s=30, clock=monotonic
    )
    breaker = registry.get("memory")
    breaker.record_failure()
    pool = AgentSessionPool(
        {"memory": "http://example.invalid/mcp"},
        breakers=registry,
        open_session=AsyncMock(side_effect=AssertionError("should not open")),
    )
    status = await call_agent_health("memory", "http://example.invalid/mcp", pool=pool)
    assert status.status == "error"
    assert "circuit breaker" in (status.detail or "")


@pytest.mark.asyncio
async def test_call_agent_health_without_pool_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(url: str) -> Any:
        _ = url
        raise ConnectionError("dns failed")

    monkeypatch.setattr("core.mcp.client.open_streamable_http_session", _boom)
    status = await call_agent_health("indexer", "http://example.invalid/mcp", pool=None)
    assert status.status == "error"


@pytest.mark.asyncio
async def test_call_tool_json_timeout_without_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(url: str) -> Any:
        _ = url
        raise TimeoutError("tool timeout")

    monkeypatch.setattr("core.mcp.client.open_streamable_http_session", _boom)
    with pytest.raises(AgentUnavailableError):
        await call_tool_json(
            "http://example.invalid/mcp",
            "find_entity",
            {"name": "FastAPI"},
            correlation_id="corr-timeout",
            timeout_s=0.01,
        )


@pytest.mark.asyncio
async def test_call_agent_health_parses_payloads() -> None:
    class _Pool:
        def __init__(self, payload: Any) -> None:
            self._payload = payload
            self.breakers = CircuitBreakerRegistry(
                default_failure_threshold=3, default_cooldown_s=1, clock=monotonic
            )

        async def call(self, *args: Any, **kwargs: Any) -> Any:
            _ = args, kwargs
            return self._payload

    ok = await call_agent_health(
        "memory",
        "http://x",
        pool=_Pool(HealthStatus(status="ok", agent="memory")),  # type: ignore[arg-type]
    )
    assert ok.status == "ok"
    nested = await call_agent_health(
        "memory",
        "http://x",
        pool=_Pool({"result": {"status": "ok", "agent": "memory"}}),  # type: ignore[arg-type]
    )
    assert nested.status == "ok"
    bad = await call_agent_health(
        "memory",
        "http://x",
        pool=_Pool({"result": {"nope": True}}),  # type: ignore[arg-type]
    )
    assert bad.status == "error"
    none = await call_agent_health("memory", "http://x", pool=_Pool(None))  # type: ignore[arg-type]
    assert none.status == "ok"
    text = await call_agent_health("memory", "http://x", pool=_Pool("plain"))  # type: ignore[arg-type]
    assert text.detail == "plain"


@pytest.mark.asyncio
async def test_call_agent_health_and_tool_without_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Session:
        async def call_tool(self, name: str, arguments: Any = None, *, meta: Any = None) -> Any:
            _ = name, arguments, meta
            return SimpleNamespace(
                isError=False,
                structuredContent={"status": "ok", "agent": "indexer"},
                content=[],
            )

        async def aclose(self) -> None:
            return None

    async def _open(url: str) -> _Session:
        _ = url
        return _Session()

    monkeypatch.setattr("core.mcp.client.open_streamable_http_session", _open)
    status = await call_agent_health("indexer", "http://example.invalid/mcp", pool=None)
    assert status.status == "ok"
    payload = await call_tool_json(
        "http://example.invalid/mcp",
        "get_statistics",
        {},
        correlation_id="c",
        timeout_s=1.0,
    )
    assert payload["status"] == "ok"


def test_build_gateway_pool() -> None:
    from gateway.mcp_client import build_gateway_pool

    pool = build_gateway_pool(GatewaySettings())
    assert "orchestrator" in pool._urls


@pytest.mark.asyncio
async def test_call_tool_json_uses_pool() -> None:
    class _Pool:
        async def call(self, *args: Any, **kwargs: Any) -> dict[str, str]:
            _ = args, kwargs
            return {"ok": "yes"}

    payload = await call_tool_json(
        "http://unused",
        "health",
        {},
        correlation_id="c1",
        timeout_s=1.0,
        pool=_Pool(),  # type: ignore[arg-type]
        agent="orchestrator",
    )
    assert payload == {"ok": "yes"}


@pytest.mark.asyncio
async def test_orchestrator_client_timeout_and_tools() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def call(
            self,
            agent: str,
            tool: str,
            arguments: dict[str, Any] | None,
            *,
            correlation_id: str,
            timeout_s: float,
            retry_count: int = 0,
        ) -> Any:
            _ = arguments, correlation_id, timeout_s, retry_count
            self.calls.append((agent, tool))
            if tool == "handle_query":
                return {"answer": "ok", "metadata": {"budget_exhausted": "tokens"}}
            if tool == "get_conversation_context":
                return ConversationContext().model_dump(mode="json")
            if tool == "analyze_query":
                return QueryIntent(
                    intent="lookup",
                    entities=["FastAPI"],
                    target_agents=["graph_query"],
                ).model_dump(mode="json")
            if tool == "route_to_agents":
                intent = QueryIntent(
                    intent="lookup",
                    entities=["FastAPI"],
                    target_agents=["graph_query"],
                )
                return ExecutionPlan(intent=intent, agents=["graph_query"]).model_dump(
                    mode="json"
                )
            if tool == "synthesize_response":
                return "synthesized"
            return {}

    settings = GatewaySettings()
    pool = _Pool()
    client = GatewayOrchestratorClient(settings, pool)  # type: ignore[arg-type]
    payload = await client.handle_query("q", "s", correlation_id="c")
    assert payload["metadata"]["budget_exhausted"] == "tokens"
    ctx = await client.get_conversation_context("s", 100, correlation_id="c")
    assert isinstance(ctx, ConversationContext)
    intent = await client.analyze_query("q", ctx, correlation_id="c")
    assert intent.intent == "lookup"
    plan = await client.route_to_agents(intent, correlation_id="c")
    assert plan.agents
    text = await client.synthesize_response("q", {}, ctx, correlation_id="c")
    assert text == "synthesized"


@pytest.mark.asyncio
async def test_orchestrator_client_transport_error() -> None:
    class _Pool:
        async def call(self, *args: Any, **kwargs: Any) -> Any:
            _ = args, kwargs
            raise ConnectionError("upstream down")

    client = GatewayOrchestratorClient(GatewaySettings(), _Pool())  # type: ignore[arg-type]
    with pytest.raises(ConnectionError):
        await client.handle_query("q", "s", correlation_id="c")


@pytest.mark.asyncio
async def test_specialist_clients_forward_tools() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.arguments: list[dict[str, Any]] = []

        async def call(self, agent: str, tool: str, *args: Any, **kwargs: Any) -> dict[str, str]:
            _ = kwargs
            self.calls.append(f"{agent}.{tool}")
            if args and isinstance(args[0], dict):
                self.arguments.append(args[0])
            return {"agent": agent, "tool": tool}

    pool = _Pool()
    clients = GatewaySpecialistClients(GatewaySettings(), pool)  # type: ignore[arg-type]
    await clients.graph_query.get_statistics(correlation_id="c")
    await clients.graph_query.find_entity("FastAPI", correlation_id="c")
    await clients.code_analyst.get_code_snippet(
        qualified_name="fastapi.applications.FastAPI",
        correlation_id="c",
    )
    await clients.indexer.index_repository(mode="full", correlation_id="c")
    await clients.indexer.index_file("fastapi/applications.py", correlation_id="c")
    assert "graph_query.find_entity" in pool.calls
    assert "code_analyst.get_code_snippet" in pool.calls
    assert "indexer.index_file" in pool.calls
    assert {"repo_url": None, "mode": "full"} in pool.arguments


@pytest.mark.asyncio
async def test_handle_query_non_dict_payload() -> None:
    class _Pool:
        async def call(self, *args: Any, **kwargs: Any) -> str:
            _ = args, kwargs
            return "plain-text-answer"

    client = GatewayOrchestratorClient(GatewaySettings(), _Pool())  # type: ignore[arg-type]
    payload = await client.handle_query("q", "s", correlation_id="c")
    assert payload == {"answer": "plain-text-answer", "metadata": {}}
