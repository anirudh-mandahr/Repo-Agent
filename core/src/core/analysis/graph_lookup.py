"""Pooled MCP graph lookup used by the Code Analyst."""

from __future__ import annotations

from collections.abc import Mapping
from time import monotonic
from typing import Any

from core.exceptions import AgentUnavailableError, CircuitBreakerOpenError, GraphLookupError
from core.logging import get_correlation_id, get_logger
from core.mcp.client import PooledAgentClient
from core.querying.service import QueryResult
from core.resilience.circuit_breaker import CircuitBreakerRegistry
from core.resilience.session_pool import AgentSessionPool
from core.settings import AnalysisSettings

log = get_logger(__name__)


class GraphQueryLookup:
    """Async callable matching ``GraphLookup``: Cypher in, row dicts out.

    Calls Graph Query ``execute_query`` through a pooled, breaker-protected
    MCP session rather than opening a fresh HTTP session per lookup.
    """

    def __init__(
        self,
        pool: AgentSessionPool,
        *,
        timeout_s: float | None = None,
        retry_count: int | None = None,
        agent: str = "graph_query",
    ) -> None:
        """Bind this lookup to a shared session pool.

        Args:
            pool: MCP session pool that includes ``graph_query``.
            timeout_s: Per-attempt timeout. Defaults to analysis settings.
            retry_count: Transient retries. Defaults to analysis settings.
            agent: Upstream agent name in the pool.
        """
        settings = AnalysisSettings.from_env()
        self._agent = agent
        self._client = PooledAgentClient(
            pool,
            agent,
            timeout_s=timeout_s if timeout_s is not None else settings.request_timeout_s,
            retry_count=retry_count if retry_count is not None else settings.retry_count,
        )

    async def __call__(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run read-only Cypher via Graph Query and return row dicts.

        Args:
            cypher: Read-only Cypher query.
            params: Query parameters, if any.

        Returns:
            Matching rows as dictionaries.

        Raises:
            CircuitBreakerOpenError: The graph_query breaker is open.
            GraphLookupError: The pooled call failed.
        """
        correlation_id = get_correlation_id()
        arguments = {"cypher": cypher, "params": dict(params or {})}
        log.info(
            "analysis.graph_lookup",
            agent=self._agent,
            param_keys=sorted(arguments["params"]),
        )
        try:
            payload = await self._client.call(
                "execute_query",
                arguments,
                correlation_id=correlation_id,
            )
        except CircuitBreakerOpenError:
            raise
        except AgentUnavailableError as exc:
            log.error("analysis.graph_lookup_failed", agent=self._agent)
            raise GraphLookupError(
                agent="code_analyst",
                correlation_id=correlation_id,
                message=f"graph_query execute_query failed via pooled {self._agent} session",
                data=str(exc),
            ) from exc
        return _rows_from_payload(payload)


def build_code_analyst_pool() -> AgentSessionPool:
    """Build the Code Analyst's session pool for Graph Query.

    Returns:
        Pool covering ``graph_query`` with the analyst breaker settings.
    """
    settings = AnalysisSettings.from_env()
    breakers = CircuitBreakerRegistry(
        default_failure_threshold=settings.breaker_failure_threshold,
        default_cooldown_s=settings.breaker_cooldown_s,
        clock=monotonic,
    )
    return AgentSessionPool(
        {"graph_query": settings.graph_query_url},
        breakers=breakers,
        clock=monotonic,
    )


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        try:
            parsed = QueryResult.model_validate(payload)
        except ValueError:
            rows = payload.get("rows")
            return list(rows) if isinstance(rows, list) else []
        return parsed.rows
    return []
