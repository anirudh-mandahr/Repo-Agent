"""MCP progress notifications that carry live orchestrator stream events.

The MCP Python SDK (1.29) delivers interleaved notifications only when the
server uses streamable HTTP SSE (``json_response=False``) and the client
passes ``progress_callback`` to ``call_tool``. FastMCP then exposes
``Context.report_progress(..., message=...)``. This module is the codec
between those progress messages and the gateway's ``on_token`` / ``on_event``
callbacks.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from core.logging import get_correlation_id, get_logger
from core.resilience.session_pool import ProgressCallback

log = get_logger(__name__)

TokenCallback = Callable[[str], Awaitable[None]]
QueryEventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

_TOKEN_EVENT = "token"


def encode_stream_event(
    event: str,
    *,
    chunk: str | None = None,
    data: dict[str, Any] | None = None,
) -> str:
    """Serialize one stream event into an MCP progress ``message``.

    Args:
        event: Event kind (``token``, ``routing``, or ``agent_result``).
        chunk: Synthesis token text when ``event`` is ``token``.
        data: Event payload when ``event`` is ``routing`` or ``agent_result``.

    Returns:
        Compact JSON suitable for ``report_progress(..., message=...)``.
    """
    payload: dict[str, Any] = {"event": event}
    if chunk is not None:
        payload["chunk"] = chunk
    if data is not None:
        payload["data"] = data
    return json.dumps(payload, separators=(",", ":"))


def decode_stream_event(message: str | None) -> tuple[str, dict[str, Any]] | None:
    """Parse a progress ``message`` produced by :func:`encode_stream_event`.

    Args:
        message: Progress notification message, or ``None``.

    Returns:
        ``(event, payload)`` when the message is a known stream event, else
        ``None``.
    """
    if not message:
        return None
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    kind = payload.get("event")
    if not isinstance(kind, str) or not kind:
        return None
    return kind, payload


async def report_stream_event(
    ctx: Any | None,
    *,
    index: int,
    event: str,
    chunk: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Best-effort ``ctx.report_progress`` for one stream event.

    No-ops when there is no FastMCP context, no ``report_progress``, or the
    client omitted ``progressToken``. Failures never abort synthesis.

    Args:
        ctx: FastMCP tool ``Context``, or ``None`` for in-process calls.
        index: Monotonic progress value for this request.
        event: Event kind.
        chunk: Token text for ``token`` events.
        data: Payload for routing / agent_result events.
    """
    if ctx is None:
        return
    if event == _TOKEN_EVENT and not chunk:
        return
    report = getattr(ctx, "report_progress", None)
    if not callable(report):
        return
    try:
        await report(
            float(index),
            None,
            encode_stream_event(event, chunk=chunk, data=data),
        )
    except Exception as exc:
        log.debug(
            "mcp.stream_progress_failed",
            correlation_id=get_correlation_id(),
            error=str(exc),
            exception_type=type(exc).__name__,
        )


def stream_callbacks_from_mcp_context(
    ctx: Any | None,
) -> tuple[TokenCallback, QueryEventCallback]:
    """Build ``on_token`` / ``on_event`` callbacks that emit MCP progress.

    Args:
        ctx: FastMCP tool ``Context`` for the current ``handle_query``.

    Returns:
        Callbacks safe to pass into :meth:`OrchestratorService.handle_query`.
    """
    index = 0

    async def on_token(chunk: str) -> None:
        nonlocal index
        if not chunk:
            return
        index += 1
        await report_stream_event(ctx, index=index, event=_TOKEN_EVENT, chunk=chunk)

    async def on_event(event_type: str, data: dict[str, Any]) -> None:
        nonlocal index
        index += 1
        await report_stream_event(ctx, index=index, event=event_type, data=data)

    return on_token, on_event


def progress_callback_for_stream(
    *,
    on_token: TokenCallback | None = None,
    on_event: QueryEventCallback | None = None,
) -> ProgressCallback:
    """Adapt an MCP ``progress_callback`` into gateway stream callbacks.

    Args:
        on_token: Optional synthesis-chunk consumer.
        on_event: Optional ``routing`` / ``agent_result`` consumer.

    Returns:
        Callback matching ``ClientSession.call_tool(..., progress_callback=)``.
    """

    async def _progress(
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        _ = progress, total
        decoded = decode_stream_event(message)
        if decoded is None:
            return
        kind, payload = decoded
        if kind == _TOKEN_EVENT:
            chunk = payload.get("chunk")
            if on_token is not None and isinstance(chunk, str) and chunk:
                await on_token(chunk)
            return
        data = payload.get("data")
        if on_event is not None and isinstance(data, dict):
            await on_event(kind, data)

    return _progress
