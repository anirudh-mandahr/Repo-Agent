"""FastMCP HTTP server helpers: metrics route, auth, tracing, uvicorn."""

from __future__ import annotations

from collections.abc import MutableMapping
from time import perf_counter
from typing import Any, cast

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from core.mcp.auth import wrap_mcp_auth
from core.observability.metrics import METRICS_CONTENT_TYPE, record_http_result, render_metrics
from core.observability.tracing import configure_tracing, start_span


class _HttpObservabilityMiddleware:
    """Record Prometheus HTTP metrics and a root span per inbound MCP request."""

    def __init__(self, app: ASGIApp, *, agent: str) -> None:
        """Wrap ``app`` for one named agent process.

        Args:
            app: Downstream ASGI app.
            agent: Agent label applied to metrics and spans.
        """
        self.app = app
        self.agent = agent

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Instrument one ASGI HTTP request.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "/")
        started = perf_counter()
        status_code = 500

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(cast(Any, message))

        try:
            with start_span(f"{self.agent}.http", kind=_span_kind(), http_path=path):
                await self.app(scope, receive, send_wrapper)
        except Exception:
            record_http_result(
                self.agent,
                path,
                status_code=500,
                duration_s=perf_counter() - started,
            )
            raise
        record_http_result(
            self.agent,
            path,
            status_code=status_code,
            duration_s=perf_counter() - started,
        )


def _span_kind() -> Any:
    from opentelemetry.trace import SpanKind

    return SpanKind.SERVER


def register_metrics_route(mcp: FastMCP, *, agent: str) -> None:
    """Expose Prometheus ``GET /metrics`` on a FastMCP Starlette app.

    Args:
        mcp: FastMCP server instance.
        agent: Agent name (reserved for future per-process labels).
    """
    if getattr(mcp, "_repochat_metrics_route", False):
        return
    mcp._repochat_metrics_route = True  # type: ignore[attr-defined]
    _ = agent

    @mcp.custom_route("/metrics", methods=["GET"], include_in_schema=False)  # type: ignore[untyped-decorator]
    async def metrics_endpoint(request: Request) -> Response:
        _ = request
        return Response(content=render_metrics(), media_type=METRICS_CONTENT_TYPE)


def wrap_mcp_app(app: ASGIApp, *, agent: str) -> ASGIApp:
    """Apply shared-secret auth then HTTP observability.

    Args:
        app: FastMCP streamable-HTTP app.
        agent: Agent label.

    Returns:
        Wrapped ASGI app.
    """
    authenticated = wrap_mcp_auth(app)
    return _HttpObservabilityMiddleware(authenticated, agent=agent)


def run_agent_mcp(mcp: FastMCP, *, agent: str) -> None:
    """Serve FastMCP over streamable HTTP with auth, metrics, and tracing.

    Args:
        mcp: Configured FastMCP server.
        agent: Agent process name.
    """
    configure_tracing(agent)
    register_metrics_route(mcp, agent=agent)
    app = wrap_mcp_app(mcp.streamable_http_app(), agent=agent)
    import uvicorn

    config = uvicorn.Config(
        app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=str(mcp.settings.log_level).lower(),
    )
    uvicorn.Server(config).run()
