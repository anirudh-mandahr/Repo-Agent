"""Health check models, real dependency probes, and the MCP healthcheck CLI."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.logging import get_logger

log = get_logger("health")

DOWNSTREAM_AGENTS: tuple[str, ...] = ("indexer", "graph_query", "code_analyst", "memory")


class HealthProbeError(RuntimeError):
    """Raised when a dependency probe fails."""


class CircuitBreakerSnapshot(BaseModel):
    """Public circuit-breaker state for health and degraded metadata."""

    state: Literal["closed", "open", "half_open"]
    consecutive_failures: int = 0
    cooldown_remaining_s: float | None = None


class HealthStatus(BaseModel):
    """Health payload returned by each agent MCP health tool."""

    status: Literal["ok", "error"]
    agent: str
    detail: str | None = None
    circuit_breakers: dict[str, CircuitBreakerSnapshot] | None = None


class AggregateHealth(BaseModel):
    """Gateway aggregate of all five agent health tools."""

    status: Literal["ok", "degraded"]
    agents: dict[str, HealthStatus] = Field(default_factory=dict)
    circuit_breakers: dict[str, CircuitBreakerSnapshot] = Field(default_factory=dict)


def agent_health(agent: str) -> HealthStatus:
    """Return process-liveness health for agents with no external dependency.

    Args:
        agent: Agent name.

    Returns:
        ``status="ok"`` payload for ``agent``.
    """
    return HealthStatus(status="ok", agent=agent)


def probe_neo4j() -> None:
    """Fail if Bolt connectivity cannot be verified in a single attempt.

    Raises:
        HealthProbeError: Neo4j did not respond.
    """
    from core.graph.client import GraphClient

    client = GraphClient()
    try:
        client.ping()
    except Exception as exc:
        raise HealthProbeError(f"neo4j unreachable: {exc}") from exc
    finally:
        client.close()


def probe_sqlite(db_path: str) -> None:
    """Fail if SQLite cannot execute ``SELECT 1`` at ``db_path``.

    Args:
        db_path: SQLite file path.

    Raises:
        HealthProbeError: The database is not reachable or writable.
    """
    path = Path(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=2.0)
        try:
            connection.execute("SELECT 1")
        finally:
            connection.close()
    except Exception as exc:
        raise HealthProbeError(f"sqlite unreachable: {exc}") from exc


def probe_repo_mount(repo_root: str) -> None:
    """Fail if the code-analyst repo volume is missing or unreadable.

    Args:
        repo_root: Expected mount path, typically ``/repo``.

    Raises:
        HealthProbeError: The mount is absent or not a readable directory.
    """
    path = Path(repo_root)
    if not path.is_dir():
        raise HealthProbeError(f"repo mount missing: {repo_root}")
    if not path.exists():
        raise HealthProbeError(f"repo mount missing: {repo_root}")
    try:
        next(path.iterdir(), None)
    except OSError as exc:
        raise HealthProbeError(f"repo mount unreadable: {exc}") from exc


async def probe_graph_query_mcp(url: str, timeout_s: float = 2.0) -> None:
    """Fail if the Graph Query MCP ``health`` tool cannot be called.

    Args:
        url: Streamable-HTTP MCP URL for graph_query.
        timeout_s: Overall timeout for the health tool call.

    Raises:
        HealthProbeError: Graph Query is unreachable or unhealthy.
    """
    from core.mcp.client import open_streamable_http_session, parse_call_tool_result

    try:
        session = await asyncio.wait_for(open_streamable_http_session(url), timeout=timeout_s)
        try:
            result = await asyncio.wait_for(
                session.call_tool("health", arguments={}),
                timeout=timeout_s,
            )
            payload = parse_call_tool_result(
                result,
                agent="graph_query",
                tool="health",
                correlation_id="health",
            )
        finally:
            await session.aclose()
    except HealthProbeError:
        raise
    except Exception as exc:
        raise HealthProbeError(f"graph_query unreachable: {exc}") from exc
    status = _coerce_health("graph_query", payload)
    if status.status != "ok":
        raise HealthProbeError(status.detail or "graph_query health returned error")


def check_indexer_health() -> HealthStatus:
    """Indexer process liveness (no extra dependency probe).

    Returns:
        ``ok`` health status for the indexer.
    """
    return agent_health("indexer")


def check_graph_query_health() -> HealthStatus:
    """Probe Neo4j connectivity for the Graph Query agent.

    Returns:
        ``ok`` when Bolt pings, otherwise ``error`` with a detail string.
    """
    try:
        probe_neo4j()
    except Exception as exc:
        log.warning("health.probe_failed", agent="graph_query", error=str(exc))
        return HealthStatus(status="error", agent="graph_query", detail=str(exc))
    return HealthStatus(status="ok", agent="graph_query")


def check_memory_health(db_path: str | None = None) -> HealthStatus:
    """Probe SQLite for the Memory agent.

    Args:
        db_path: Override for the configured SQLite path.

    Returns:
        ``ok`` when ``SELECT 1`` succeeds, otherwise ``error``.
    """
    from core.settings import MemorySettings

    path = db_path if db_path is not None else MemorySettings.from_env().db_path
    try:
        probe_sqlite(path)
    except Exception as exc:
        log.warning("health.probe_failed", agent="memory", error=str(exc))
        return HealthStatus(status="error", agent="memory", detail=str(exc))
    return HealthStatus(status="ok", agent="memory")


async def check_code_analyst_health(
    *,
    repo_root: str | None = None,
    graph_query_url: str | None = None,
) -> HealthStatus:
    """Probe the ``/repo`` mount and Graph Query reachability.

    Args:
        repo_root: Source mount path. Loaded from settings when omitted.
        graph_query_url: Graph Query MCP URL. Loaded from settings when omitted.

    Returns:
        ``ok`` when both probes pass, otherwise ``error``.
    """
    from core.settings import AnalysisSettings

    settings = AnalysisSettings.from_env()
    root = repo_root if repo_root is not None else settings.repo_root
    url = graph_query_url if graph_query_url is not None else settings.graph_query_url
    try:
        probe_repo_mount(root)
        await probe_graph_query_mcp(url)
    except Exception as exc:
        log.warning("health.probe_failed", agent="code_analyst", error=str(exc))
        return HealthStatus(status="error", agent="code_analyst", detail=str(exc))
    return HealthStatus(status="ok", agent="code_analyst")


async def collect_downstream_health(
    pool: Any,
    *,
    timeout_s: float = 2.0,
) -> dict[str, HealthStatus]:
    """Call ``health`` on each specialist through an MCP session pool.

    Args:
        pool: :class:`~core.resilience.session_pool.AgentSessionPool`.
        timeout_s: Per-agent timeout.

    Returns:
        Mapping of agent name to health payload.
    """

    async def _one(name: str) -> HealthStatus:
        try:
            payload = await asyncio.wait_for(
                pool.call(
                    name,
                    "health",
                    {},
                    correlation_id="health",
                    timeout_s=timeout_s,
                    retry_count=0,
                ),
                timeout=timeout_s,
            )
            return _coerce_health(name, payload)
        except Exception as exc:
            return HealthStatus(status="error", agent=name, detail=str(exc))

    results = await asyncio.gather(*[_one(name) for name in DOWNSTREAM_AGENTS])
    return {status.agent: status for status in results}


async def check_orchestrator_health(
    *,
    snapshots: Mapping[str, CircuitBreakerSnapshot] | None = None,
    downstream: Mapping[str, HealthStatus] | None = None,
) -> HealthStatus:
    """Aggregate downstream specialist health and circuit-breaker state.

    Args:
        snapshots: Optional breaker snapshots from the orchestrator pool.
        downstream: Optional per-agent health results. Empty means unchecked.

    Returns:
        ``ok`` when every known downstream agent is healthy and no breaker is open.
    """
    breakers = dict(snapshots or {})
    agents = dict(downstream or {})
    failing = [name for name, status in agents.items() if status.status != "ok"]
    open_breakers = [name for name, item in breakers.items() if item.state == "open"]
    if not failing and not open_breakers:
        return HealthStatus(
            status="ok",
            agent="orchestrator",
            circuit_breakers=breakers or None,
        )
    parts: list[str] = []
    if failing:
        parts.append("unhealthy: " + ", ".join(failing))
    if open_breakers:
        parts.append("open circuit breakers: " + ", ".join(open_breakers))
    return HealthStatus(
        status="error",
        agent="orchestrator",
        detail="; ".join(parts),
        circuit_breakers=breakers or None,
    )


def _coerce_health(agent: str, payload: Any) -> HealthStatus:
    if isinstance(payload, HealthStatus):
        return payload
    if isinstance(payload, dict):
        inner = payload.get("result", payload) if "result" in payload else payload
        if isinstance(inner, dict):
            try:
                return HealthStatus.model_validate(inner)
            except ValueError:
                return HealthStatus(status="error", agent=agent, detail=str(inner))
    if payload is None:
        return HealthStatus(status="ok", agent=agent)
    return HealthStatus(status="ok", agent=agent, detail=str(payload))


async def mcp_health_ok(url: str, *, timeout_s: float = 5.0) -> HealthStatus:
    """Call an agent's MCP ``health`` tool and return the parsed payload.

    Args:
        url: Streamable-HTTP MCP URL.
        timeout_s: Overall timeout.

    Returns:
        Parsed :class:`HealthStatus`, or ``error`` when the call fails.
    """
    from core.mcp.client import open_streamable_http_session, parse_call_tool_result

    try:
        session = await asyncio.wait_for(open_streamable_http_session(url), timeout=timeout_s)
        try:
            result = await asyncio.wait_for(
                session.call_tool("health", arguments={}),
                timeout=timeout_s,
            )
            payload = parse_call_tool_result(
                result,
                agent="healthcheck",
                tool="health",
                correlation_id="healthcheck",
            )
        finally:
            await session.aclose()
    except Exception as exc:
        return HealthStatus(status="error", agent="healthcheck", detail=str(exc))
    return _coerce_health("healthcheck", payload)


def main(argv: list[str] | None = None) -> int:
    """CLI used by Docker healthchecks: ``python -m core.health --url ...``.

    Args:
        argv: Optional argument list for tests.

    Returns:
        Process exit code (0 healthy, 1 unhealthy).
    """
    parser = argparse.ArgumentParser(description="Call an MCP health tool and exit 0/1.")
    parser.add_argument("--url", required=True, help="Streamable-HTTP MCP URL")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout in seconds")
    args = parser.parse_args(argv)
    status = asyncio.run(mcp_health_ok(args.url, timeout_s=args.timeout))
    if status.status == "ok":
        return 0
    log.error("healthcheck.failed", url=args.url, detail=status.detail)
    return 1


if __name__ == "__main__":
    sys.exit(main())
