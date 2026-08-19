"""Graph query FastMCP adapter. Read-only Cypher (logic lives in core)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP

from core.health import HealthStatus, check_graph_query_health
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.mcp.context import bind_mcp_context
from core.mcp.server import run_agent_mcp
from core.querying.embeddings import default_embedding_provider
from core.querying.service import (
    EntityQueryResult,
    GraphQueryService,
    GraphStatistics,
    ImportTraceResult,
    NeighborQueryResult,
    QueryResult,
    RelatedQueryResult,
)
from core.querying.templates import DEFAULT_TRACE_DEPTH
from core.settings import AgentRuntimeSettings, GraphQuerySettings

AGENT_NAME = "graph_query"
DEFAULT_PORT = 8003

if TYPE_CHECKING:
    ToolContext = Context[Any, Any, Any]
else:
    ToolContext = Context

_settings = AgentRuntimeSettings.from_env(agent=AGENT_NAME, default_port=DEFAULT_PORT)
configure_logging(_settings.log_level)
log = get_logger(AGENT_NAME)

mcp = FastMCP(
    AGENT_NAME,
    host=_settings.host,
    port=_settings.port,
    log_level=_settings.log_level,
    stateless_http=True,
    json_response=True,
)

_gq_settings = GraphQuerySettings.from_env()
_service = GraphQueryService(
    embedding_provider=default_embedding_provider(),
    embeddings_enabled=_gq_settings.embeddings_enabled,
)


@mcp.tool()
async def health(ctx: ToolContext | None = None) -> HealthStatus:
    """Return agent liveness after probing Neo4j."""
    bind_mcp_context(ctx)
    status = await asyncio.to_thread(check_graph_query_health)
    log.info("health.check", agent=AGENT_NAME, status=status.status)
    return status


@mcp.tool()
async def find_entity(
    name: str,
    entity_type: str | None = None,
    ctx: ToolContext | None = None,
) -> EntityQueryResult:
    """Find Module/Class/Function/Method by exact name, full-text, or lexical hash."""
    bind_mcp_context(ctx)
    result = await asyncio.to_thread(_service.find_entity, name, entity_type)
    log.info(
        "query.find_entity",
        name=name,
        entity_type=entity_type,
        result_count=result.result_count,
        truncated=result.truncated,
        error=result.error,
    )
    return result


@mcp.tool()
async def get_dependencies(name: str, ctx: ToolContext | None = None) -> NeighborQueryResult:
    """Return outgoing IMPORTS, DEPENDS_ON, and CALLS neighbors."""
    bind_mcp_context(ctx)
    result = await asyncio.to_thread(_service.get_dependencies, name)
    log.info(
        "query.get_dependencies",
        name=name,
        result_count=result.result_count,
        truncated=result.truncated,
        total_count=result.total_count,
    )
    return result


@mcp.tool()
async def get_dependents(name: str, ctx: ToolContext | None = None) -> NeighborQueryResult:
    """Return incoming IMPORTS, DEPENDS_ON, and CALLS neighbors."""
    bind_mcp_context(ctx)
    result = await asyncio.to_thread(_service.get_dependents, name)
    log.info(
        "query.get_dependents",
        name=name,
        result_count=result.result_count,
        truncated=result.truncated,
        total_count=result.total_count,
    )
    return result


@mcp.tool()
async def trace_imports(
    module: str,
    depth: int = DEFAULT_TRACE_DEPTH,
    ctx: ToolContext | None = None,
) -> ImportTraceResult:
    """Follow IMPORTS and DEPENDS_ON chains from a module (depth cap default 5)."""
    bind_mcp_context(ctx)
    result = await asyncio.to_thread(_service.trace_imports, module, depth)
    log.info(
        "query.trace_imports",
        module=module,
        depth=result.depth,
        result_count=result.result_count,
        truncated=result.truncated,
    )
    return result


@mcp.tool()
async def find_related(
    name: str,
    relationship_type: str,
    ctx: ToolContext | None = None,
) -> RelatedQueryResult:
    """Return neighbors along a spec relationship type, with direction."""
    bind_mcp_context(ctx)
    result = await asyncio.to_thread(_service.find_related, name, relationship_type)
    log.info(
        "query.find_related",
        name=name,
        relationship_type=relationship_type,
        result_count=result.result_count,
        truncated=result.truncated,
        error=result.error,
    )
    return result


@mcp.tool()
async def execute_query(
    cypher: str,
    params: dict[str, Any] | None = None,
    ctx: ToolContext | None = None,
) -> QueryResult:
    """Run a read-only Cypher query. Write clauses are rejected."""
    bind_mcp_context(ctx)
    result = await asyncio.to_thread(_service.execute_query, cypher, params)
    log.info(
        "query.execute_query",
        result_count=result.result_count,
        truncated=result.truncated,
    )
    return result


@mcp.tool()
async def get_statistics(ctx: ToolContext | None = None) -> GraphStatistics:
    """Return label counts, relationship counts, and index metadata."""
    bind_mcp_context(ctx)
    result = await asyncio.to_thread(_service.get_statistics)
    log.info(
        "query.get_statistics",
        index_version=result.index_version,
        last_indexed_at=result.last_indexed_at,
    )
    return result


def main() -> None:
    """Run the FastMCP server with streamable HTTP transport."""
    bind_correlation_id()
    log.info("agent.start", agent=AGENT_NAME, host=_settings.host, port=_settings.port)
    run_agent_mcp(mcp, agent=AGENT_NAME)


if __name__ == "__main__":
    main()
