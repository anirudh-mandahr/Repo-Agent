"""Orchestrator FastMCP adapter.

Exposes the orchestrator core loop as MCP tools and delegates work to the
other agent MCP servers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from mcp.server.fastmcp import Context, FastMCP

from core.health import HealthStatus, check_orchestrator_health, collect_downstream_health
from core.llm.factory import build_llm_provider
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.mcp.context import bind_mcp_context
from core.mcp.server import run_agent_mcp
from core.memory import ConversationContext
from core.orchestration import ExecutionPlan, QueryIntent
from core.orchestration.mcp_clients import PooledOrchestratorClients, build_orchestrator_pool
from core.orchestration.router import analyze_query as analyze_query_core
from core.orchestration.router import route_to_agents as route_to_agents_core
from core.orchestration.synthesis import synthesize_response as synthesize_response_core
from core.resilience.session_pool import AgentSessionPool
from core.settings import AgentRuntimeSettings, OrchestratorSettings

from .service import OrchestratorClients, OrchestratorService

AGENT_NAME = "orchestrator"
DEFAULT_PORT = 8001

_settings = AgentRuntimeSettings.from_env(agent=AGENT_NAME, default_port=DEFAULT_PORT)
configure_logging(_settings.log_level)
log = get_logger(AGENT_NAME)

_orchestrator_settings = OrchestratorSettings.from_env()
_pool: AgentSessionPool | None = None


@asynccontextmanager
async def _mcp_lifespan(
    _server: FastMCP[dict[str, AgentSessionPool]],
) -> AsyncIterator[dict[str, AgentSessionPool]]:
    global _pool
    pool = build_orchestrator_pool()
    _pool = pool
    _service._breakers = pool.breakers
    try:
        yield {"mcp_pool": pool}
    finally:
        await pool.aclose()
        _pool = None


mcp = FastMCP(
    AGENT_NAME,
    host=_settings.host,
    port=_settings.port,
    log_level=_settings.log_level,
    stateless_http=True,
    json_response=True,
    lifespan=_mcp_lifespan,
)

_llm_provider = build_llm_provider()
_service = OrchestratorService(_llm_provider, settings=_orchestrator_settings)


if TYPE_CHECKING:
    ToolContext = Context[Any, Any, Any]
else:
    ToolContext = Context


def get_mcp_pool() -> AgentSessionPool:
    """Return the process-wide MCP session pool, creating it lazily if needed.

    Returns:
        Shared session pool for upstream agents.
    """
    global _pool
    if _pool is None:
        _pool = build_orchestrator_pool()
    return _pool


@mcp.tool()
async def health() -> HealthStatus:
    """Return agent liveness including upstream circuit-breaker state."""
    bind_correlation_id()
    pool = _pool
    snapshots = pool.breakers.snapshot() if pool is not None else {}
    downstream = await collect_downstream_health(pool) if pool is not None else {}
    status = await check_orchestrator_health(snapshots=snapshots, downstream=downstream)
    log.info("health.check", agent=AGENT_NAME, status=status.status)
    return status


@mcp.tool()
async def get_conversation_context(
    session_id: str,
    token_budget: int = 3000,
    ctx: ToolContext | None = None,
) -> ConversationContext:
    """Delegate `get_context` to the Memory agent."""
    correlation_id = bind_mcp_context(ctx)
    clients = PooledOrchestratorClients.from_pool(
        get_mcp_pool(),
        correlation_id=correlation_id,
        settings=_orchestrator_settings,
    )
    return await clients.get_context(session_id, token_budget=token_budget)


@mcp.tool()
async def analyze_query(
    query: str,
    context: ConversationContext,
    ctx: ToolContext | None = None,
) -> QueryIntent:
    """LLM routing intent (core delegation).

    Note: this MCP tool always performs full LLM classification when called
    directly; the rules-first shortcut exists only inside `handle_query`.
    """
    correlation_id = bind_mcp_context(ctx)
    return await analyze_query_core(
        query,
        context,
        llm_provider=_llm_provider,
        correlation_id=correlation_id,
    )


@mcp.tool()
def route_to_agents(intent: QueryIntent) -> ExecutionPlan:
    """Convert a QueryIntent into an execution plan."""
    return route_to_agents_core(intent)


@mcp.tool()
async def synthesize_response(
    query: str,
    agent_outputs: dict[str, Any],
    context: ConversationContext,
    ctx: ToolContext | None = None,
) -> str:
    """Delegate synthesis to the orchestration core (LLM, or evidence-only fallback)."""
    correlation_id = bind_mcp_context(ctx)
    result = await synthesize_response_core(
        query,
        agent_outputs,  # type: ignore[arg-type]
        context,
        llm_provider=_llm_provider,
        settings=_orchestrator_settings,
        correlation_id=correlation_id,
    )
    return result.answer


@mcp.tool()
async def handle_query(
    query: str,
    session_id: str,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    """Gateway-facing tool: run the full orchestrator loop."""
    correlation_id = bind_mcp_context(ctx)
    pool = get_mcp_pool()
    _service._breakers = pool.breakers
    clients = cast(
        OrchestratorClients,
        PooledOrchestratorClients.from_pool(
            pool,
            correlation_id=correlation_id,
            settings=_orchestrator_settings,
        ),
    )
    result = await _service.handle_query(
        query,
        session_id,
        clients=clients,
        correlation_id=correlation_id,
    )
    return {"answer": result.answer, "metadata": result.metadata}


def main() -> None:
    """Run the FastMCP server with streamable HTTP transport."""
    bind_correlation_id()
    log.info("agent.start", agent=AGENT_NAME, host=_settings.host, port=_settings.port)
    run_agent_mcp(mcp, agent=AGENT_NAME)


if __name__ == "__main__":
    main()
