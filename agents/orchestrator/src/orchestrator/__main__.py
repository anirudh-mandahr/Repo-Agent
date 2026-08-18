"""Orchestrator FastMCP adapter.

Exposes the orchestrator core loop as MCP tools and delegates work to the
other agent MCP servers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult

from core.health import HealthStatus, agent_health
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.provider import LLMProvider
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.memory import ConversationContext
from core.orchestration import ExecutionPlan, QueryIntent
from core.orchestration.router import analyze_query as analyze_query_core
from core.orchestration.router import route_to_agents as route_to_agents_core
from core.orchestration.synthesis import synthesize_response as synthesize_response_core
from core.settings import AgentRuntimeSettings, LLMSettings, OrchestratorSettings

from .service import OrchestratorClients, OrchestratorService

AGENT_NAME = "orchestrator"
DEFAULT_PORT = 8001

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

_orchestrator_settings = OrchestratorSettings.from_env()
_llm_provider: LLMProvider
llm_settings = LLMSettings.from_env()
if llm_settings.api_key:
    _llm_provider = OpenRouterProvider.from_env()
else:
    log.warning("orchestrator.offline_provider", reason="OPENROUTER_API_KEY unset")
    _llm_provider = OfflineProvider()

_service = OrchestratorService(_llm_provider, settings=_orchestrator_settings)


if TYPE_CHECKING:
    ToolContext = Context[Any, Any, Any]
else:
    ToolContext = Context


def _correlation_id_from_meta(ctx: ToolContext | None) -> str | None:
    if ctx is None:
        return None
    try:
        meta = ctx.request_context.meta
    except ValueError:
        return None
    if meta is None:
        return None
    raw = getattr(meta, "correlation_id", None)
    if raw is None and meta.model_extra:
        raw = meta.model_extra.get("correlation_id")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _bind_meta(ctx: ToolContext | None) -> str:
    return bind_correlation_id(_correlation_id_from_meta(ctx))


class _McpClient:
    def __init__(self, url: str) -> None:
        self._url = url

    async def call(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        correlation_id: str,
    ) -> Any:
        arguments = dict(arguments or {})
        async with streamable_http_client(self._url) as (read, write, _session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result: CallToolResult = await session.call_tool(
                    tool,
                    arguments=arguments,
                    meta={"correlation_id": correlation_id},
                )
        if result.isError:
            raise RuntimeError(f"mcp tool {tool} failed: {result.content}")
        if isinstance(result.structuredContent, dict):
            payload = result.structuredContent
            return payload.get("result", payload)
        return result.content


def _clients_from_env() -> Any:
    """Create simple per-call MCP clients for other agents."""
    from core.settings import GatewaySettings

    urls = GatewaySettings.from_env().agent_urls()
    return type(
        "_Clients",
        (),
        {
            "graph_query": _McpClient(urls["graph_query"]),
            "code_analyst": _McpClient(urls["code_analyst"]),
            "indexer": _McpClient(urls["indexer"]),
            "memory": _McpClient(urls["memory"]),
        },
    )()


@mcp.tool()
def health() -> HealthStatus:
    """Return agent liveness."""
    bind_correlation_id()
    status = agent_health(AGENT_NAME)
    log.info("health.check", agent=AGENT_NAME, status=status.status)
    return status


@mcp.tool()
async def get_conversation_context(
    session_id: str,
    token_budget: int = 3000,
    ctx: ToolContext | None = None,
) -> ConversationContext:
    """Delegate `get_context` to the Memory agent."""
    correlation_id = _bind_meta(ctx)
    clients = _clients_from_env()
    result = await clients.memory.call(
        "get_context",
        {"session_id": session_id, "token_budget": token_budget},
        correlation_id=correlation_id,
    )
    return ConversationContext.model_validate(result)


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
    correlation_id = _bind_meta(ctx)
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
    """Delegate synthesis (one LLM call) to the orchestration core."""
    correlation_id = _bind_meta(ctx)
    _ = correlation_id
    # `agent_outputs` comes in over MCP as a JSON dict, which is already
    # compatible with the synthesis prompt.
    return await synthesize_response_core(
        query,
        agent_outputs,  # type: ignore[arg-type]
        context,
        llm_provider=_llm_provider,
        settings=_orchestrator_settings,
        correlation_id=correlation_id,
    )


@mcp.tool()
async def handle_query(
    query: str,
    session_id: str,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    """Gateway-facing tool: run the full orchestrator loop."""
    correlation_id = _bind_meta(ctx)
    clients = _clients_from_env()

    # Adapt MCP client calls to OrchestratorService's expected interface.
    class _Adapter:
        def __init__(self, raw: Any) -> None:
            self._raw = raw
            # OrchestratorService expects `clients.memory`, `clients.graph_query`, etc.
            # Keep all tool implementations on this object and alias the attributes.
            self.memory = self
            self.graph_query = self
            self.code_analyst = self
            self.indexer = self

        async def get_context(
            self,
            session_id: str,
            token_budget: int = 3000,
        ) -> ConversationContext:
            result = await self._raw.memory.call(
                "get_context",
                {"session_id": session_id, "token_budget": token_budget},
                correlation_id=correlation_id,
            )
            return ConversationContext.model_validate(result)

        async def get_cached_response(self, cache_key: str) -> Any | None:
            return await self._raw.memory.call(
                "get_cached_response",
                {"cache_key": cache_key},
                correlation_id=correlation_id,
            )

        async def cache_response(self, cache_key: str, response_json: Any) -> None:
            await self._raw.memory.call(
                "cache_response",
                {"cache_key": cache_key, "response_json": response_json},
                correlation_id=correlation_id,
            )

        async def append_turn(self, session_id: str, role: str, content: str) -> None:
            await self._raw.memory.call(
                "append_turn",
                {"session_id": session_id, "role": role, "content": content},
                correlation_id=correlation_id,
            )

        async def get_statistics(self) -> Any:
            return await self._raw.graph_query.call(
                "get_statistics",
                {},
                correlation_id=correlation_id,
            )

        async def find_entity(self, name: str, entity_type: str | None = None) -> Any:
            args: dict[str, Any] = {"name": name, "entity_type": entity_type}
            return await self._raw.graph_query.call(
                "find_entity",
                args,
                correlation_id=correlation_id,
            )

        async def get_code_snippet(
            self,
            *,
            qualified_name: str | None = None,
            file_path: str | None = None,
            line_start: int | None = None,
            line_end: int | None = None,
        ) -> Any:
            args: dict[str, Any] = {
                "qualified_name": qualified_name,
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
            }
            return await self._raw.code_analyst.call(
                "get_code_snippet",
                args,
                correlation_id=correlation_id,
            )

        async def index_repository(self, repo_url: str | None = None) -> Any:
            return await self._raw.indexer.call(
                "index_repository",
                {"repo_url": repo_url},
                correlation_id=correlation_id,
            )

    adapted = _Adapter(clients)
    adapted_clients = cast(OrchestratorClients, adapted)
    result = await _service.handle_query(
        query,
        session_id,
        clients=adapted_clients,
        correlation_id=correlation_id,
    )
    return {"answer": result.answer, "metadata": result.metadata}


def main() -> None:
    """Run the FastMCP server with streamable HTTP transport."""
    bind_correlation_id()
    log.info("agent.start", agent=AGENT_NAME, host=_settings.host, port=_settings.port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
