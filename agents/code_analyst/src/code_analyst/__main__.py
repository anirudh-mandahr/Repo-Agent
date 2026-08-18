"""Code analyst FastMCP adapter. Reads source via graph results and /repo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP

from code_analyst.graph_lookup import GraphQueryLookup
from core.analysis.models import (
    ClassAnalysis,
    FunctionAnalysis,
    ImplementationComparison,
    ImplementationExplanation,
    PatternAnalysis,
    SnippetResult,
)
from core.analysis.service import CodeAnalystService
from core.health import HealthStatus, agent_health
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.provider import LLMProvider
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.settings import AgentRuntimeSettings, AnalysisSettings, LLMSettings

AGENT_NAME = "code_analyst"
DEFAULT_PORT = 8004

if TYPE_CHECKING:
    ToolContext = Context[Any, Any, Any]
else:
    ToolContext = Context

_settings = AgentRuntimeSettings.from_env(agent=AGENT_NAME, default_port=DEFAULT_PORT)
_analysis_settings = AnalysisSettings.from_env()
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
    log.warning("analysis.offline_provider", reason="OPENROUTER_API_KEY unset")
    return OfflineProvider()


_service = CodeAnalystService(
    _build_llm_provider(),
    GraphQueryLookup(_analysis_settings.graph_query_url),
    repo_root=_analysis_settings.repo_root,
)


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
async def analyze_function(
    qualified_name: str,
    ctx: ToolContext | None = None,
) -> FunctionAnalysis:
    """Analyze a function or method using graph context and its source snippet."""
    _bind_meta(ctx)
    result = await _service.analyze_function(qualified_name)
    log.info(
        "analysis.analyze_function",
        qualified_name=qualified_name,
        error=result.error,
    )
    return result


@mcp.tool()
async def analyze_class(
    qualified_name: str,
    ctx: ToolContext | None = None,
) -> ClassAnalysis:
    """Analyze a class using its methods, bases, decorators, and source snippet."""
    _bind_meta(ctx)
    result = await _service.analyze_class(qualified_name)
    log.info("analysis.analyze_class", qualified_name=qualified_name, error=result.error)
    return result


@mcp.tool()
async def find_patterns(pattern: str, ctx: ToolContext | None = None) -> PatternAnalysis:
    """Find decorator, dependency_injection, or factory instances and explain them."""
    _bind_meta(ctx)
    result = await _service.find_patterns(pattern)
    log.info(
        "analysis.find_patterns",
        pattern=pattern,
        instance_count=len(result.instances),
        error=result.error,
    )
    return result


@mcp.tool()
async def get_code_snippet(
    qualified_name: str | None = None,
    file_path: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
    ctx: ToolContext | None = None,
) -> SnippetResult:
    """Return numbered source for a qualified name or a file path and line range."""
    _bind_meta(ctx)
    result = await _service.get_code_snippet(
        qualified_name=qualified_name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
    )
    log.info(
        "analysis.get_code_snippet",
        qualified_name=qualified_name,
        file_path=file_path,
        error=result.error,
    )
    return result


@mcp.tool()
async def explain_implementation(
    qualified_name: str,
    ctx: ToolContext | None = None,
) -> ImplementationExplanation:
    """Explain how a function, method, or class is implemented."""
    _bind_meta(ctx)
    try:
        result = await _service.explain_implementation(qualified_name)
    except Exception as exc:
        log.error(
            "analysis.explain_implementation_failed",
            qualified_name=qualified_name,
            error=str(exc),
        )
        return ImplementationExplanation(qualified_name=qualified_name, error=str(exc))
    log.info(
        "analysis.explain_implementation",
        qualified_name=qualified_name,
        error=result.error,
    )
    return result


@mcp.tool()
async def compare_implementations(
    name_a: str,
    name_b: str,
    ctx: ToolContext | None = None,
) -> ImplementationComparison:
    """Compare two functions or methods side by side."""
    _bind_meta(ctx)
    result = await _service.compare_implementations(name_a, name_b)
    log.info(
        "analysis.compare_implementations",
        name_a=name_a,
        name_b=name_b,
        error=result.error,
    )
    return result


def main() -> None:
    """Run the FastMCP server with streamable HTTP transport."""
    bind_correlation_id()
    log.info("agent.start", agent=AGENT_NAME, host=_settings.host, port=_settings.port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
