"""Memory FastMCP adapter. Conversation and retrieval state (logic lives in core)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP

from core.health import HealthStatus, agent_health
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.provider import LLMProvider
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.memory import CachedResponse, ConversationContext, MemoryService
from core.settings import AgentRuntimeSettings, LLMSettings

AGENT_NAME = "memory"
DEFAULT_PORT = 8005

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


def _build_llm_provider() -> LLMProvider:
    llm_settings = LLMSettings.from_env()
    if llm_settings.api_key:
        return OpenRouterProvider.from_env()
    log.warning("memory.offline_provider", reason="OPENROUTER_API_KEY unset")
    return OfflineProvider()


_service = MemoryService(_build_llm_provider())


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


@mcp.tool()
def health(ctx: ToolContext | None = None) -> HealthStatus:
    """Return agent liveness."""
    _bind_meta(ctx)
    status = agent_health(AGENT_NAME)
    log.info("health.check", agent=AGENT_NAME, status=status.status)
    return status


@mcp.tool()
async def append_turn(
    session_id: str,
    role: str,
    content: str,
    ctx: ToolContext | None = None,
) -> dict[str, str]:
    """Append one user or assistant turn to session memory."""
    _bind_meta(ctx)
    await _service.append_turn(session_id, role, content)
    log.info("memory.append_turn", session_id=session_id, role=role)
    return {"status": "ok"}


@mcp.tool()
async def get_context(
    session_id: str,
    token_budget: int = 3000,
    ctx: ToolContext | None = None,
) -> ConversationContext:
    """Return rolling summary plus recent turns that fit the token budget."""
    _bind_meta(ctx)
    result = await _service.get_context(session_id, token_budget=token_budget)
    log.info(
        "memory.get_context",
        session_id=session_id,
        summary_present=bool(result.summary),
        recent_turns=len(result.recent_turns),
    )
    return result


@mcp.tool()
async def summarize_session(
    session_id: str,
    ctx: ToolContext | None = None,
) -> dict[str, str]:
    """Fold older turns into the rolling session summary."""
    _bind_meta(ctx)
    summary = await _service.summarize_session(session_id)
    log.info("memory.summarize_session", session_id=session_id, summary_present=bool(summary))
    return {"summary": summary}


@mcp.tool()
async def cache_response(
    cache_key: str,
    response_json: Any,
    ctx: ToolContext | None = None,
) -> dict[str, str]:
    """Store an opaque cached response by caller-computed key."""
    _bind_meta(ctx)
    await _service.cache_put(cache_key, response_json)
    log.info("memory.cache_put", cache_key=cache_key)
    return {"status": "ok"}


@mcp.tool()
async def get_cached_response(
    cache_key: str,
    ctx: ToolContext | None = None,
) -> CachedResponse | None:
    """Return a fresh cached response for the provided key, if present."""
    _bind_meta(ctx)
    result = await _service.cache_get(cache_key)
    log.info("memory.cache_get", cache_key=cache_key, hit=result is not None)
    return result


def main() -> None:
    """Run the FastMCP server with streamable HTTP transport."""
    bind_correlation_id()
    asyncio.run(_service.initialize())
    log.info("agent.start", agent=AGENT_NAME, host=_settings.host, port=_settings.port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
