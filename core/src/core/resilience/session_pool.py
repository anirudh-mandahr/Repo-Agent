"""Reusable MCP session per upstream agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from core.exceptions import AgentUnavailableError, CircuitBreakerOpenError
from core.logging import get_logger

from .circuit_breaker import CircuitBreakerRegistry
from .retry import await_with_timeout_retry

log = get_logger(__name__)

SessionOpener = Callable[[str], Awaitable["ToolSession"]]


class ToolSession(Protocol):
    """One live MCP client session."""

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke a tool on the live session."""
        ...

    async def aclose(self) -> None:
        """Close the session and its transport."""
        ...


class AgentSessionPool:
    """One reused session per agent, with per-agent locking and a circuit breaker."""

    def __init__(
        self,
        urls: Mapping[str, str] | None = None,
        *,
        open_session: SessionOpener | None = None,
        breakers: CircuitBreakerRegistry | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Create a pool.

        Args:
            urls: Agent name to streamable-HTTP MCP URL.
            open_session: Optional factory used in tests; defaults to streamable HTTP.
            breakers: Optional shared circuit-breaker registry.
            clock: Monotonic clock forwarded to a default registry.
        """
        self._urls = dict(urls or {})
        self._open_session = open_session
        if breakers is not None:
            self.breakers = breakers
        else:
            from time import monotonic

            self.breakers = CircuitBreakerRegistry(clock=clock or monotonic)
        self._sessions: dict[str, ToolSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._open_counts: dict[str, int] = {}
        self._call_counts: dict[str, int] = {}
        self._closed = False

    @property
    def open_counts(self) -> dict[str, int]:
        """How many times a session was opened per agent.

        Returns:
            Copy of per-agent open counters.
        """
        return dict(self._open_counts)

    @property
    def call_counts(self) -> dict[str, int]:
        """How many tool calls were issued per agent.

        Returns:
            Copy of per-agent call counters.
        """
        return dict(self._call_counts)

    async def call(
        self,
        agent: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        correlation_id: str,
        timeout_s: float,
        retry_count: int = 0,
    ) -> Any:
        """Call ``tool`` on ``agent`` using the pooled session.

        Args:
            agent: Upstream agent name.
            tool: MCP tool name.
            arguments: Tool arguments.
            correlation_id: Request correlation id.
            timeout_s: Per-attempt timeout.
            retry_count: Transient retries after the first attempt.

        Returns:
            Parsed tool result.

        Raises:
            CircuitBreakerOpenError: The agent breaker is open.
            AgentUnavailableError: The pool is closed or the tool failed.
        """
        if self._closed:
            raise AgentUnavailableError(
                agent=agent,
                correlation_id=correlation_id,
                message=f"mcp pool closed; cannot call {agent}.{tool}",
            )
        breaker = self.breakers.get(agent)

        async def _once() -> Any:
            return await self._call_locked(agent, tool, arguments, correlation_id)

        from core.observability.metrics import observe_request
        from core.observability.tracing import start_span

        try:
            with observe_request(agent, tool):
                with start_span(
                    f"mcp.{agent}.{tool}",
                    correlation_id=correlation_id,
                    agent=agent,
                    tool=tool,
                ):
                    result = await breaker.run(
                        lambda: await_with_timeout_retry(
                            _once,
                            timeout_s=timeout_s,
                            retry_count=retry_count,
                        )
                    )
        except CircuitBreakerOpenError:
            raise
        except (TimeoutError, ConnectionError, OSError, AgentUnavailableError) as exc:
            raise AgentUnavailableError(
                agent=agent,
                correlation_id=correlation_id,
                message=str(exc),
            ) from exc
        return _parse_tool_result(agent, tool, result, correlation_id)

    async def aclose(self) -> None:
        """Close every live session."""
        self._closed = True
        agents = list(self._sessions)
        for agent in agents:
            await self._drop(agent)
        self._sessions.clear()

    async def _call_locked(
        self,
        agent: str,
        tool: str,
        arguments: Mapping[str, Any] | None,
        correlation_id: str,
    ) -> Any:
        lock = self._locks.setdefault(agent, asyncio.Lock())
        async with lock:
            session = await self._ensure(agent)
            self._call_counts[agent] = self._call_counts.get(agent, 0) + 1
            try:
                from core.observability.tracing import inject_trace_carrier

                return await session.call_tool(
                    tool,
                    arguments=dict(arguments or {}),
                    meta=inject_trace_carrier(correlation_id),
                )
            except Exception:
                await self._drop(agent)
                raise

    async def _ensure(self, agent: str) -> ToolSession:
        existing = self._sessions.get(agent)
        if existing is not None:
            return existing
        session = await self._open(agent)
        self._sessions[agent] = session
        self._open_counts[agent] = self._open_counts.get(agent, 0) + 1
        log.info(
            "mcp.pool.session_open",
            agent=agent,
            opens=self._open_counts[agent],
        )
        return session

    async def _open(self, agent: str) -> ToolSession:
        if self._open_session is not None:
            return await self._open_session(agent)
        url = self._urls.get(agent)
        if not url:
            raise AgentUnavailableError(
                agent=agent,
                message=f"no mcp url configured for {agent}",
            )
        from core.mcp.client import open_streamable_http_session

        return await open_streamable_http_session(url)

    async def _drop(self, agent: str) -> None:
        session = self._sessions.pop(agent, None)
        if session is None:
            return
        try:
            await session.aclose()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("mcp.pool.close_failed", agent=agent, error=str(exc))


def _parse_tool_result(agent: str, tool: str, result: Any, correlation_id: str) -> Any:
    if not hasattr(result, "isError"):
        return result
    if getattr(result, "isError", False):
        raise AgentUnavailableError(
            agent=agent,
            correlation_id=correlation_id,
            message=f"mcp tool {tool} failed: {getattr(result, 'content', result)}",
            data=getattr(result, "structuredContent", None),
        )
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured.get("result", structured)
    content = getattr(result, "content", None)
    if content:
        block = content[0]
        text = getattr(block, "text", None)
        return text if text is not None else str(block)
    return None
