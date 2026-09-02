"""MCP client helpers used by the gateway."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from time import monotonic
from typing import Any, Literal

from core.exceptions import AgentUnavailableError, CircuitBreakerOpenError
from core.gateway import OrchestratorGatewayClient
from core.health import HealthStatus
from core.logging import get_logger
from core.mcp.streaming import progress_callback_for_stream
from core.memory import ConversationContext
from core.orchestration.models import ExecutionPlan, QueryIntent
from core.resilience.circuit_breaker import CircuitBreakerRegistry
from core.resilience.session_pool import AgentSessionPool
from core.settings import GatewaySettings

log = get_logger(__name__)


def build_gateway_pool(settings: GatewaySettings) -> AgentSessionPool:
    """Build the process-wide MCP session pool for the gateway.

    Args:
        settings: Gateway settings with agent URLs and breaker knobs.

    Returns:
        Lazy session pool covering all five agents.
    """
    breakers = CircuitBreakerRegistry(
        default_failure_threshold=settings.breaker_failure_threshold,
        default_cooldown_s=settings.breaker_cooldown_s,
        clock=monotonic,
    )
    return AgentSessionPool(settings.agent_urls(), breakers=breakers, clock=monotonic)


def _parse_health(agent: str, payload: Any) -> HealthStatus:
    if isinstance(payload, HealthStatus):
        return payload
    if isinstance(payload, dict):
        inner = payload.get("result", payload) if "result" in payload else payload
        if isinstance(inner, dict):
            try:
                return HealthStatus.model_validate(inner)
            except ValueError:
                return HealthStatus(
                    status="error",
                    agent=agent,
                    detail=str(inner),
                )
    if payload is None:
        return HealthStatus(status="ok", agent=agent)
    return HealthStatus(status="ok", agent=agent, detail=str(payload))


async def call_agent_health(
    agent: str,
    url: str,
    *,
    pool: AgentSessionPool | None = None,
    timeout_s: float = 2.0,
) -> HealthStatus:
    """Invoke the agent's health tool, reusing a pooled session when provided."""
    try:
        if pool is not None:
            payload = await pool.call(
                agent,
                "health",
                {},
                correlation_id="health",
                timeout_s=timeout_s,
                retry_count=0,
            )
            return _parse_health(agent, payload)
        from core.mcp.client import open_streamable_http_session, parse_call_tool_result

        session = await open_streamable_http_session(url)
        try:
            result = await session.call_tool("health", arguments={})
            payload = parse_call_tool_result(
                result,
                agent=agent,
                tool="health",
                correlation_id="health",
            )
            return _parse_health(agent, payload)
        finally:
            await session.aclose()
    except CircuitBreakerOpenError as exc:
        log.error("mcp.health_breaker_open", agent=agent, error=str(exc))
        snapshot = None
        if pool is not None:
            snapshot = pool.breakers.get(agent).snapshot()
        return HealthStatus(
            status="error",
            agent=agent,
            detail="circuit breaker open",
            circuit_breakers={agent: snapshot} if snapshot is not None else None,
        )
    except Exception as exc:
        log.error("mcp.health_failed", agent=agent, url=url, error=str(exc))
        return HealthStatus(status="error", agent=agent, detail=str(exc))


async def call_tool_json(
    url: str,
    tool: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    correlation_id: str,
    timeout_s: float,
    pool: AgentSessionPool | None = None,
    agent: str | None = None,
) -> Any:
    """Call one MCP tool and return JSON-ish structured content."""
    if pool is not None and agent is not None:
        return await pool.call(
            agent,
            tool,
            arguments,
            correlation_id=correlation_id,
            timeout_s=timeout_s,
            retry_count=0,
        )
    from core.mcp.client import open_streamable_http_session, parse_call_tool_result

    try:
        session = await open_streamable_http_session(url)
        try:
            result = await session.call_tool(
                tool,
                arguments=dict(arguments or {}),
                meta={"correlation_id": correlation_id},
            )
        finally:
            await session.aclose()
    except Exception as exc:
        raise AgentUnavailableError(
            agent=tool,
            correlation_id=correlation_id,
            message=str(exc),
        ) from exc
    return parse_call_tool_result(
        result,
        agent=agent or tool,
        tool=tool,
        correlation_id=correlation_id,
    )


class GatewayOrchestratorClient(OrchestratorGatewayClient):
    """Gateway-side adapter around orchestrator MCP tools."""

    def __init__(self, settings: GatewaySettings, pool: AgentSessionPool) -> None:
        self._url = settings.orchestrator_url
        self._timeout_s = settings.request_timeout_s
        self._chat_timeout_s = settings.chat_timeout_s
        self._pool = pool

    async def handle_query(
        self,
        query: str,
        session_id: str,
        *,
        correlation_id: str,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        progress_callback = None
        if on_token is not None or on_event is not None:
            progress_callback = progress_callback_for_stream(
                on_token=on_token,
                on_event=on_event,
            )
        if progress_callback is not None:
            payload = await self._pool.call(
                "orchestrator",
                "handle_query",
                {"query": query, "session_id": session_id},
                correlation_id=correlation_id,
                timeout_s=self._chat_timeout_s,
                progress_callback=progress_callback,
            )
        else:
            payload = await self._pool.call(
                "orchestrator",
                "handle_query",
                {"query": query, "session_id": session_id},
                correlation_id=correlation_id,
                timeout_s=self._chat_timeout_s,
            )
        if not isinstance(payload, dict):
            return {"answer": str(payload), "metadata": {}}
        return payload

    async def get_conversation_context(
        self,
        session_id: str,
        token_budget: int,
        *,
        correlation_id: str,
    ) -> ConversationContext:
        payload = await self._pool.call(
            "orchestrator",
            "get_conversation_context",
            {"session_id": session_id, "token_budget": token_budget},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )
        return ConversationContext.model_validate(payload)

    async def analyze_query(
        self,
        query: str,
        context: ConversationContext,
        *,
        correlation_id: str,
    ) -> QueryIntent:
        payload = await self._pool.call(
            "orchestrator",
            "analyze_query",
            {"query": query, "context": context.model_dump(mode="json")},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )
        return QueryIntent.model_validate(payload)

    async def route_to_agents(
        self,
        intent: QueryIntent,
        *,
        correlation_id: str,
    ) -> ExecutionPlan:
        payload = await self._pool.call(
            "orchestrator",
            "route_to_agents",
            {"intent": intent.model_dump(mode="json")},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )
        return ExecutionPlan.model_validate(payload)

    async def synthesize_response(
        self,
        query: str,
        agent_outputs: dict[str, Any],
        context: ConversationContext,
        *,
        correlation_id: str,
    ) -> str:
        payload = await self._pool.call(
            "orchestrator",
            "synthesize_response",
            {
                "query": query,
                "agent_outputs": agent_outputs,
                "context": context.model_dump(mode="json"),
            },
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )
        return str(payload)


class GatewaySpecialistClients:
    """Gateway-side adapter around the specialist MCP tools."""

    def __init__(self, settings: GatewaySettings, pool: AgentSessionPool) -> None:
        self._settings = settings
        self.graph_query = _GraphQueryGatewayClient(settings, pool)
        self.code_analyst = _CodeAnalystGatewayClient(settings, pool)
        self.indexer = _IndexerGatewayClient(settings, pool)


class _GraphQueryGatewayClient:
    def __init__(self, settings: GatewaySettings, pool: AgentSessionPool) -> None:
        self._timeout_s = settings.request_timeout_s
        self._pool = pool

    async def get_statistics(self, *, correlation_id: str) -> Any:
        return await self._pool.call(
            "graph_query",
            "get_statistics",
            {},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )

    async def find_entity(
        self,
        name: str,
        entity_type: str | None = None,
        *,
        correlation_id: str,
    ) -> Any:
        return await self._pool.call(
            "graph_query",
            "find_entity",
            {"name": name, "entity_type": entity_type},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )


class _CodeAnalystGatewayClient:
    def __init__(self, settings: GatewaySettings, pool: AgentSessionPool) -> None:
        self._timeout_s = settings.request_timeout_s
        self._pool = pool

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        correlation_id: str,
    ) -> Any:
        return await self._pool.call(
            "code_analyst",
            "get_code_snippet",
            {
                "qualified_name": qualified_name,
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
            },
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )


class _IndexerGatewayClient:
    def __init__(self, settings: GatewaySettings, pool: AgentSessionPool) -> None:
        self._timeout_s = settings.request_timeout_s
        self._pool = pool

    async def index_repository(
        self,
        repo_url: str | None = None,
        *,
        mode: Literal["full", "incremental"] = "incremental",
        correlation_id: str,
    ) -> Any:
        return await self._pool.call(
            "indexer",
            "index_repository",
            {"repo_url": repo_url, "mode": mode},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )

    async def index_file(self, path: str, *, correlation_id: str) -> Any:
        return await self._pool.call(
            "indexer",
            "index_file",
            {"path": path},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )

    async def get_index_status(self, *, correlation_id: str) -> Any:
        return await self._pool.call(
            "indexer",
            "get_index_status",
            {},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )
