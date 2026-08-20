"""Streamable-HTTP MCP session factory and thin pooled client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from core.exceptions import AgentUnavailableError
from core.mcp.auth import mcp_request_headers
from core.resilience.session_pool import AgentSessionPool, ProgressCallback, ToolSession


class StreamableHttpSession:
    """Live ``ClientSession`` held open by an :class:`AsyncExitStack`."""

    def __init__(self, stack: AsyncExitStack, session: ClientSession) -> None:
        """Wrap an initialized MCP session.

        Args:
            stack: Exit stack that owns the HTTP streams and session.
            session: Initialized MCP client session.
        """
        self._stack = stack
        self._session = session

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Call a tool on the live session.

        Args:
            name: MCP tool name.
            arguments: Tool arguments.
            meta: Request metadata forwarded to the server.
            progress_callback: Optional MCP progress consumer for live tokens.

        Returns:
            Raw ``CallToolResult``.
        """
        return await self._session.call_tool(
            name,
            arguments=dict(arguments or {}),
            meta=meta,
            progress_callback=progress_callback,
        )

    async def aclose(self) -> None:
        """Close the session and HTTP streams."""
        await self._stack.aclose()


async def open_streamable_http_session(url: str) -> ToolSession:
    """Open and initialize one streamable-HTTP MCP session.

    Args:
        url: Streamable HTTP MCP endpoint.

    Returns:
        A reusable :class:`StreamableHttpSession`.
    """
    stack = AsyncExitStack()
    try:
        headers = mcp_request_headers()
        from mcp.client.streamable_http import (  # type: ignore[attr-defined]
            create_mcp_http_client as _create_client,
        )

        http_client = _create_client(headers=headers or None)
        await stack.enter_async_context(http_client)
        read, write, _session_id = await stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
    except BaseException as exc:
        # A dead endpoint does not fail here with a ConnectError. The task
        # group inside streamable_http_client sends the first request from a
        # child task; when that child hits a connect failure, anyio cancels
        # the group's scope, which surfaces in *this* task as a bare
        # CancelledError at ``session.initialize()``. An ``except Exception``
        # would miss it, leaving the scope orphaned -- and an orphaned scope
        # later cancels whatever this task is doing, killing the request
        # without a response. Unwind the stack in this same task: the task
        # group's exit uncancels the task and re-raises the real transport
        # error, which we translate into a transient ConnectionError.
        try:
            await stack.aclose()
        except asyncio.CancelledError:
            raise
        except BaseException as close_exc:
            raise ConnectionError(
                f"mcp session setup failed for {url}: {close_exc!r}"
            ) from close_exc
        current = asyncio.current_task()
        if isinstance(exc, asyncio.CancelledError) and (
            current is None or not current.cancelling()
        ):
            # Scope-driven cancellation whose transport error was already
            # consumed during unwind; the caller was not cancelled.
            raise ConnectionError(f"mcp session setup failed for {url}") from exc
        raise
    return StreamableHttpSession(stack=stack, session=session)


def parse_call_tool_result(
    result: CallToolResult,
    *,
    agent: str,
    tool: str,
    correlation_id: str,
) -> Any:
    """Parse a ``CallToolResult`` into JSON-ish content.

    Args:
        result: MCP tool result.
        agent: Agent that produced the result.
        tool: Tool name, used in error messages.
        correlation_id: Request correlation id.

    Returns:
        Structured payload or text content.

    Raises:
        AgentUnavailableError: The tool returned ``isError``.
    """
    if result.isError:
        raise AgentUnavailableError(
            agent=agent,
            correlation_id=correlation_id,
            message=f"mcp tool {tool} failed: {result.content}",
            data=result.structuredContent,
        )
    if isinstance(result.structuredContent, dict):
        payload = result.structuredContent
        return payload.get("result", payload)
    if result.content:
        block = result.content[0]
        text = block.text if isinstance(block, TextContent) else str(block)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


class PooledAgentClient:
    """Adapter-facing client that reuses one pooled session per agent."""

    def __init__(
        self,
        pool: AgentSessionPool,
        agent: str,
        *,
        timeout_s: float,
        retry_count: int,
    ) -> None:
        """Bind this client to one pooled agent.

        Args:
            pool: Shared session pool.
            agent: Upstream agent name.
            timeout_s: Per-attempt timeout.
            retry_count: Transient retries.
        """
        self._pool = pool
        self._agent = agent
        self._timeout_s = timeout_s
        self._retry_count = retry_count

    async def call(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        correlation_id: str,
    ) -> Any:
        """Call ``tool`` through the pool.

        Args:
            tool: MCP tool name.
            arguments: Tool arguments.
            correlation_id: Request correlation id.

        Returns:
            Parsed tool result.
        """
        return await self._pool.call(
            self._agent,
            tool,
            arguments,
            correlation_id=correlation_id,
            timeout_s=self._timeout_s,
            retry_count=self._retry_count,
        )
