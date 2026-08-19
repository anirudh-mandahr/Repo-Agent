"""Inbound FastMCP context helpers shared by every agent adapter."""

from __future__ import annotations

from typing import Any

from core.logging import bind_correlation_id, correlation_id_from_mcp_meta
from core.observability.tracing import attach_from_mcp_meta


def meta_from_mcp_context(ctx: Any | None) -> object | None:
    """Read ``request_context.meta`` from a FastMCP tool context.

    Args:
        ctx: FastMCP ``Context``, or ``None`` when invoked in-process.

    Returns:
        The meta object, or ``None`` when no request is bound.
    """
    if ctx is None:
        return None
    try:
        return ctx.request_context.meta  # type: ignore[no-any-return]
    except ValueError:
        return None


def bind_mcp_context(ctx: Any | None) -> str:
    """Propagate inbound MCP correlation_id and trace context.

    Args:
        ctx: FastMCP ``Context`` for the current tool call.

    Returns:
        The correlation id now bound on the logging context.
    """
    meta = meta_from_mcp_context(ctx)
    attach_from_mcp_meta(meta)
    return bind_correlation_id(correlation_id_from_mcp_meta(meta))
