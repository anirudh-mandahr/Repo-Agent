"""MCP client helpers used by the gateway."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from core.exceptions import AgentUnavailableError
from core.gateway import OrchestratorGatewayClient
from core.health import HealthStatus
from core.logging import get_logger
from core.memory import ConversationContext
from core.orchestration.models import ExecutionPlan, QueryIntent
from core.settings import GatewaySettings

log = get_logger(__name__)


def _parse_health(agent: str, result: CallToolResult) -> HealthStatus:
    structured = result.structuredContent
    if isinstance(structured, dict):
        payload = structured.get("result", structured)
        if isinstance(payload, dict):
            try:
                status = HealthStatus.model_validate(payload)
                if result.isError:
                    return status.model_copy(update={"status": "error"})
                return status
            except ValueError:
                pass

    if result.content:
        block = result.content[0]
        text = block.text if isinstance(block, TextContent) else str(block)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return HealthStatus.model_validate(parsed)
        except (json.JSONDecodeError, ValueError):
            return HealthStatus(
                status="error" if result.isError else "ok",
                agent=agent,
                detail=text,
            )

    if result.isError:
        return HealthStatus(status="error", agent=agent, detail="health tool failed")
    return HealthStatus(status="ok", agent=agent)


async def call_agent_health(agent: str, url: str) -> HealthStatus:
    """Open an MCP client session and invoke the agent's health tool."""
    try:
        async with _session(url) as session:
            result = await session.call_tool("health", arguments={})
        return _parse_health(agent, result)
    except Exception as exc:
        log.error("mcp.health_failed", agent=agent, url=url, error=str(exc))
        return HealthStatus(status="error", agent=agent, detail=str(exc))


@asynccontextmanager
async def _session(url: str) -> Any:
    async with streamable_http_client(url) as (read, write, _session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool_json(
    url: str,
    tool: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    correlation_id: str,
    timeout_s: float,
) -> Any:
    """Call one MCP tool and return JSON-ish structured content."""
    try:
        async with _session(url) as session:
            result = await asyncio.wait_for(
                session.call_tool(
                    tool,
                    arguments=dict(arguments or {}),
                    meta={"correlation_id": correlation_id},
                ),
                timeout=timeout_s,
            )
    except Exception as exc:
        raise AgentUnavailableError(
            agent=tool,
            correlation_id=correlation_id,
            message=str(exc),
        ) from exc

    if result.isError:
        raise AgentUnavailableError(
            agent=tool,
            correlation_id=correlation_id,
            message=str(result.content),
            data=result.structuredContent,
        )

    if isinstance(result.structuredContent, dict):
        payload = result.structuredContent
        return payload.get("result", payload)

    if result.content:
        block = result.content[0]
        text = block.text if isinstance(block, TextContent) else str(block)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


class GatewayOrchestratorClient(OrchestratorGatewayClient):
    """Gateway-side adapter around orchestrator MCP tools."""

    def __init__(self, settings: GatewaySettings) -> None:
        self._url = settings.orchestrator_url
        self._timeout_s = settings.request_timeout_s

    async def handle_query(
        self,
        query: str,
        session_id: str,
        *,
        correlation_id: str,
    ) -> dict[str, Any]:
        payload = await call_tool_json(
            self._url,
            "handle_query",
            {"query": query, "session_id": session_id},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
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
        payload = await call_tool_json(
            self._url,
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
        payload = await call_tool_json(
            self._url,
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
        payload = await call_tool_json(
            self._url,
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
        payload = await call_tool_json(
            self._url,
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

    def __init__(self, settings: GatewaySettings) -> None:
        self._settings = settings
        self.graph_query = _GraphQueryGatewayClient(settings)
        self.code_analyst = _CodeAnalystGatewayClient(settings)
        self.indexer = _IndexerGatewayClient(settings)


class _GraphQueryGatewayClient:
    def __init__(self, settings: GatewaySettings) -> None:
        self._url = settings.graph_query_url
        self._timeout_s = settings.request_timeout_s

    async def get_statistics(self, *, correlation_id: str) -> Any:
        return await call_tool_json(
            self._url,
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
        return await call_tool_json(
            self._url,
            "find_entity",
            {"name": name, "entity_type": entity_type},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )


class _CodeAnalystGatewayClient:
    def __init__(self, settings: GatewaySettings) -> None:
        self._url = settings.code_analyst_url
        self._timeout_s = settings.request_timeout_s

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        correlation_id: str,
    ) -> Any:
        return await call_tool_json(
            self._url,
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
    def __init__(self, settings: GatewaySettings) -> None:
        self._url = settings.indexer_url
        self._timeout_s = settings.request_timeout_s

    async def index_repository(self, repo_url: str | None = None, *, correlation_id: str) -> Any:
        return await call_tool_json(
            self._url,
            "index_repository",
            {"repo_url": repo_url},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )

    async def index_file(self, path: str, *, correlation_id: str) -> Any:
        return await call_tool_json(
            self._url,
            "index_file",
            {"path": path},
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
        )
