"""Bounded pool of reusable MCP sessions per upstream agent."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from core.exceptions import AgentUnavailableError, CircuitBreakerOpenError
from core.logging import get_logger

from .circuit_breaker import CircuitBreakerRegistry
from .retry import await_with_timeout_retry

log = get_logger(__name__)

DEFAULT_MAX_SESSIONS_PER_AGENT = 4
SessionOpener = Callable[[str], Awaitable["ToolSession"]]


class ProgressCallback(Protocol):
    """MCP ``call_tool(..., progress_callback=)`` signature."""

    async def __call__(
        self, progress: float, total: float | None, message: str | None
    ) -> None:
        """Handle one progress notification.

        Args:
            progress: Current progress value.
            total: Optional total, or ``None`` when unknown.
            message: Optional payload (stream events are JSON in this field).
        """
        ...


class ToolSession(Protocol):
    """One live MCP client session."""

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Invoke a tool on the live session."""
        ...

    async def aclose(self) -> None:
        """Close the session and its transport."""
        ...


def _resolve_max_sessions(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, explicit)
    raw = os.environ.get("MCP_SESSION_POOL_SIZE", "").strip()
    if not raw:
        return DEFAULT_MAX_SESSIONS_PER_AGENT
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_SESSIONS_PER_AGENT


class _AgentSessionBucket:
    """Idle-queue of up to ``max_size`` live sessions for one agent."""

    def __init__(
        self,
        agent: str,
        max_size: int,
        opener: Callable[[], Awaitable[ToolSession]],
    ) -> None:
        self._agent = agent
        self._max_size = max_size
        self._opener = opener
        self._idle: list[ToolSession] = []
        self._live: list[ToolSession] = []
        self._size = 0
        self._closed = False
        self._cond = asyncio.Condition()
        self._closing: set[asyncio.Task[None]] = set()

    async def acquire(
        self, *, correlation_id: str, timeout_s: float | None = None
    ) -> tuple[ToolSession, bool]:
        """Return ``(session, reused)``; ``reused`` is True for idle-queue hits.

        Waiting for a free slot is bounded by ``timeout_s``. A caller whose task
        is abandoned mid-call never returns its session, so without a bound the
        bucket would deadlock every later caller once enough slots had been
        stranded that way -- turning one dead specialist into a dead agent.

        Raises:
            AgentUnavailableError: The pool is closed, or no slot came free in
                ``timeout_s``.
        """
        created = False
        deadline = (
            None if timeout_s is None else asyncio.get_running_loop().time() + timeout_s
        )
        async with self._cond:
            while True:
                if self._closed:
                    raise AgentUnavailableError(
                        agent=self._agent,
                        correlation_id=correlation_id,
                        message=f"mcp pool closed; cannot acquire session for {self._agent}",
                    )
                if self._idle:
                    return self._idle.pop(), True
                if self._size < self._max_size:
                    self._size += 1
                    created = True
                    break
                if deadline is None:
                    await self._cond.wait()
                    continue
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise AgentUnavailableError(
                        agent=self._agent,
                        correlation_id=correlation_id,
                        message=(
                            f"mcp pool exhausted for {self._agent}; "
                            f"no session free after {timeout_s}s"
                        ),
                    )
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except TimeoutError:
                    continue
        if created:
            try:
                session = await self._opener()
            except Exception:
                async with self._cond:
                    self._size = max(0, self._size - 1)
                    self._cond.notify()
                raise
            async with self._cond:
                if self._closed:
                    self._size = max(0, self._size - 1)
                    self._cond.notify()
                    try:
                        await session.aclose()
                    except Exception as exc:  # pragma: no cover - defensive
                        log.warning(
                            "mcp.pool.close_failed",
                            agent=self._agent,
                            error=str(exc),
                        )
                    raise AgentUnavailableError(
                        agent=self._agent,
                        correlation_id=correlation_id,
                        message=f"mcp pool closed; cannot acquire session for {self._agent}",
                    )
                self._live.append(session)
            return session, False
        raise AgentUnavailableError(
            agent=self._agent,
            correlation_id=correlation_id,
            message=f"mcp pool failed to acquire session for {self._agent}",
        )

    async def release(self, session: ToolSession) -> None:
        async with self._cond:
            if self._closed:
                close_after = True
            else:
                self._idle.append(session)
                self._cond.notify()
                close_after = False
        if close_after:
            await self._close_quietly(session)

    async def drop(self, session: ToolSession) -> None:
        async with self._cond:
            if session in self._live:
                self._live.remove(session)
            if session in self._idle:
                self._idle.remove(session)
            self._size = max(0, self._size - 1)
            self._cond.notify()
        await self._close_quietly(session)

    def discard_nowait(self, session: ToolSession) -> None:
        """Reclaim ``session``'s slot without waiting for it to close.

        Used when a call timed out and its task was abandoned: that task will
        never release the session, so the slot would leak and the bucket would
        deadlock once every slot had been lost that way. ``drop`` frees the slot
        before awaiting the close, and closing an abandoned MCP session can
        itself hang, so it runs detached.
        """
        task = asyncio.create_task(self.drop(session))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def purge_idle(self) -> None:
        """Drop every idle session; used when a reused session proves stale."""
        async with self._cond:
            stale = list(self._idle)
            self._idle.clear()
            for session in stale:
                if session in self._live:
                    self._live.remove(session)
            self._size = max(0, self._size - len(stale))
            self._cond.notify_all()
        for session in stale:
            await self._close_quietly(session)

    async def aclose(self) -> None:
        async with self._cond:
            self._closed = True
            sessions = list(self._live)
            self._live.clear()
            self._idle.clear()
            self._size = 0
            self._cond.notify_all()
        for session in sessions:
            await self._close_quietly(session)

    async def _close_quietly(self, session: ToolSession) -> None:
        try:
            await session.aclose()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("mcp.pool.close_failed", agent=self._agent, error=str(exc))


class AgentSessionPool:
    """Bounded pool of reused MCP sessions per agent, with a circuit breaker."""

    def __init__(
        self,
        urls: Mapping[str, str] | None = None,
        *,
        open_session: SessionOpener | None = None,
        breakers: CircuitBreakerRegistry | None = None,
        clock: Callable[[], float] | None = None,
        max_sessions_per_agent: int | None = None,
    ) -> None:
        """Create a pool.

        Args:
            urls: Agent name to streamable-HTTP MCP URL.
            open_session: Optional factory used in tests; defaults to streamable HTTP.
            breakers: Optional shared circuit-breaker registry.
            clock: Monotonic clock forwarded to a default registry.
            max_sessions_per_agent: Concurrent live sessions per agent. ``None``
                reads ``MCP_SESSION_POOL_SIZE`` (default 4).
        """
        self._urls = dict(urls or {})
        self._open_session = open_session
        if breakers is not None:
            self.breakers = breakers
        else:
            from time import monotonic

            self.breakers = CircuitBreakerRegistry(clock=clock or monotonic)
        self._max_sessions_per_agent = _resolve_max_sessions(max_sessions_per_agent)
        self._buckets: dict[str, _AgentSessionBucket] = {}
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
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Call ``tool`` on ``agent`` using the pooled session.

        Args:
            agent: Upstream agent name.
            tool: MCP tool name.
            arguments: Tool arguments.
            correlation_id: Request correlation id.
            timeout_s: Per-attempt timeout.
            retry_count: Transient retries after the first attempt.
            progress_callback: Optional MCP progress consumer. Omitted for
                callers that do not negotiate streaming.

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
        # A timed-out call has its task abandoned mid-flight, so it can never
        # release the session it holds. Track it here to reclaim the slot.
        in_flight: list[ToolSession] = []

        async def _once() -> Any:
            return await self._call_with_session(
                agent,
                tool,
                arguments,
                correlation_id,
                progress_callback=progress_callback,
                in_flight=in_flight,
                acquire_timeout_s=timeout_s,
            )

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
            bucket = self._buckets.get(agent)
            if bucket is not None:
                for stranded in in_flight:
                    bucket.discard_nowait(stranded)
            raise AgentUnavailableError(
                agent=agent,
                correlation_id=correlation_id,
                message=str(exc),
            ) from exc
        return _parse_tool_result(agent, tool, result, correlation_id)

    async def aclose(self) -> None:
        """Close every live session."""
        self._closed = True
        buckets = list(self._buckets.values())
        self._buckets.clear()
        for bucket in buckets:
            await bucket.aclose()

    def _bucket(self, agent: str) -> _AgentSessionBucket:
        existing = self._buckets.get(agent)
        if existing is not None:
            return existing
        bucket = _AgentSessionBucket(
            agent,
            self._max_sessions_per_agent,
            lambda: self._open_and_count(agent),
        )
        self._buckets[agent] = bucket
        return bucket

    async def _call_with_session(
        self,
        agent: str,
        tool: str,
        arguments: Mapping[str, Any] | None,
        correlation_id: str,
        *,
        progress_callback: ProgressCallback | None = None,
        in_flight: list[ToolSession] | None = None,
        acquire_timeout_s: float | None = None,
    ) -> Any:
        bucket = self._bucket(agent)
        session, reused = await bucket.acquire(
            correlation_id=correlation_id, timeout_s=acquire_timeout_s
        )
        if in_flight is not None:
            in_flight.append(session)
        self._call_counts[agent] = self._call_counts.get(agent, 0) + 1

        def _untrack(done: ToolSession) -> None:
            """Stop tracking a session this call has already handed back."""
            if in_flight is not None and done in in_flight:
                in_flight.remove(done)

        try:
            result = await self._invoke_session(
                session,
                tool,
                arguments,
                correlation_id,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            await bucket.drop(session)
            _untrack(session)
            if not reused:
                raise
            # Idle sessions opened in an earlier request task can be broken by
            # anyio cancel-scope task affinity. Purge the idle queue and retry
            # once on a freshly opened session.
            log.warning(
                "mcp.pool.stale_session_retry",
                agent=agent,
                tool=tool,
                error=str(exc),
            )
            await bucket.purge_idle()
            session, _ = await bucket.acquire(
                correlation_id=correlation_id, timeout_s=acquire_timeout_s
            )
            if in_flight is not None:
                in_flight.append(session)
            try:
                result = await self._invoke_session(
                    session,
                    tool,
                    arguments,
                    correlation_id,
                    progress_callback=progress_callback,
                )
            except Exception:
                await bucket.drop(session)
                _untrack(session)
                raise
        await bucket.release(session)
        _untrack(session)
        return result

    async def _invoke_session(
        self,
        session: ToolSession,
        tool: str,
        arguments: Mapping[str, Any] | None,
        correlation_id: str,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        from core.observability.tracing import inject_trace_carrier

        meta = inject_trace_carrier(correlation_id)
        arguments_dict = dict(arguments or {})
        if progress_callback is not None:
            return await session.call_tool(
                tool,
                arguments=arguments_dict,
                meta=meta,
                progress_callback=progress_callback,
            )
        return await session.call_tool(
            tool,
            arguments=arguments_dict,
            meta=meta,
        )

    async def _open_and_count(self, agent: str) -> ToolSession:
        session = await self._open(agent)
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
