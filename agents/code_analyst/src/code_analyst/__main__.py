"""Code analyst FastMCP adapter. Reads source via graph results and /repo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP

from core.analysis.graph_lookup import GraphQueryLookup, build_code_analyst_pool
from core.analysis.models import (
    ClassAnalysis,
    FunctionAnalysis,
    ImplementationComparison,
    ImplementationExplanation,
    PatternAnalysis,
    SnippetResult,
)
from core.analysis.service import CodeAnalystService
from core.health import HealthStatus, check_code_analyst_health
from core.llm.factory import build_llm_provider
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.mcp.context import bind_mcp_context
from core.mcp.server import run_agent_mcp
from core.settings import AgentRuntimeSettings, AnalysisSettings

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

_pool = build_code_analyst_pool()
_service = CodeAnalystService(
    build_llm_provider(),
    GraphQueryLookup(_pool),
    repo_root=_analysis_settings.repo_root,
)


@mcp.tool()
async def health(ctx: ToolContext | None = None) -> HealthStatus:
    """Return agent liveness after probing /repo and graph_query."""
    bind_mcp_context(ctx)
    status = await check_code_analyst_health(
        repo_root=_analysis_settings.repo_root,
        graph_query_url=_analysis_settings.graph_query_url,
    )
    log.info("health.check", agent=AGENT_NAME, status=status.status)
    return status


@mcp.tool()
async def analyze_function(
    qualified_name: str,
    ctx: ToolContext | None = None,
) -> FunctionAnalysis:
    """Analyze a function or method using graph context and its source snippet."""
    bind_mcp_context(ctx)
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
    bind_mcp_context(ctx)
    result = await _service.analyze_class(qualified_name)
    log.info("analysis.analyze_class", qualified_name=qualified_name, error=result.error)
    return result


@mcp.tool()
async def find_patterns(
    pattern: str,
    path_prefix: str | None = None,
    ctx: ToolContext | None = None,
) -> PatternAnalysis:
    """Find decorator, dependency_injection, or factory instances and explain them."""
    bind_mcp_context(ctx)
    result = await _service.find_patterns(pattern, path_prefix=path_prefix)
    log.info(
        "analysis.find_patterns",
        pattern=pattern,
        path_prefix=path_prefix,
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
    bind_mcp_context(ctx)
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
    bind_mcp_context(ctx)
    result = await _service.explain_implementation(qualified_name)
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
    bind_mcp_context(ctx)
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
    run_agent_mcp(mcp, agent=AGENT_NAME)


if __name__ == "__main__":
    main()
