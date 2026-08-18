"""Indexer FastMCP adapter. Writes the knowledge graph (logic lives in core)."""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from core.health import HealthStatus, agent_health
from core.indexing import (
    ExtractedGraph,
    IndexReport,
    IndexStatus,
    ParsedFile,
    already_running_report,
    clone_and_index,
    load_index_status,
    run_index_file,
)
from core.indexing import extract_entities as extract_entities_core
from core.indexing import parse_python_ast as parse_python_ast_core
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.settings import AgentRuntimeSettings

AGENT_NAME = "indexer"
DEFAULT_PORT = 8002

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

_index_lock = asyncio.Lock()
_last_report: IndexReport | None = None


@mcp.tool()
def health() -> HealthStatus:
    """Return agent liveness."""
    bind_correlation_id()
    status = agent_health(AGENT_NAME)
    log.info("health.check", agent=AGENT_NAME, status=status.status)
    return status


@mcp.tool()
async def index_repository(repo_url: str | None = None) -> IndexReport:
    """Clone the target repo and index it. Concurrent calls return already_running."""
    global _last_report
    bind_correlation_id()
    if _index_lock.locked():
        log.info("index.already_running")
        return already_running_report()
    async with _index_lock:
        report = await asyncio.to_thread(clone_and_index, repo_url)
        _last_report = report
        log.info(
            "index.tool_done",
            status=report.status,
            files_indexed=report.files_indexed,
            duration_s=report.duration_s,
        )
        return report


@mcp.tool()
async def index_file(path: str) -> IndexReport:
    """Hash-check and reindex a single file. Concurrent calls return already_running."""
    global _last_report
    bind_correlation_id()
    if _index_lock.locked():
        log.info("index.already_running")
        return already_running_report()
    async with _index_lock:
        report = await asyncio.to_thread(run_index_file, path)
        _last_report = report
        log.info(
            "index.file_tool_done",
            path=path,
            status=report.status,
            files_indexed=report.files_indexed,
        )
        return report


@mcp.tool()
def parse_python_ast(path_or_code: str) -> ParsedFile:
    """Parse a file path or source string into a ParsedFile."""
    bind_correlation_id()
    parsed = parse_python_ast_core(path_or_code)
    log.info(
        "index.parse_python_ast",
        path=parsed.path,
        module=parsed.module,
        error=parsed.error,
    )
    return parsed


@mcp.tool()
def extract_entities(path_or_code: str) -> ExtractedGraph:
    """Return flat entity and relationship lists derived from a ParsedFile."""
    bind_correlation_id()
    extracted = extract_entities_core(path_or_code)
    log.info(
        "index.extract_entities",
        entities=len(extracted.entities),
        relationships=len(extracted.relationships),
    )
    return extracted


@mcp.tool()
async def get_index_status() -> IndexStatus:
    """Return the last index report plus live Neo4j node and relationship counts."""
    bind_correlation_id()
    status = await asyncio.to_thread(
        load_index_status,
        last_report=_last_report,
        running=_index_lock.locked(),
    )
    log.info(
        "index.status",
        running=status.running,
        node_count=status.node_count,
        rel_count=status.rel_count,
        counts=status.counts,
    )
    return status


def main() -> None:
    """Run the FastMCP server with streamable HTTP transport."""
    bind_correlation_id()
    log.info("agent.start", agent=AGENT_NAME, host=_settings.host, port=_settings.port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
