"""Shared-secret authentication for MCP streamable-HTTP servers and clients."""

from __future__ import annotations

import os
import secrets

from starlette.types import ASGIApp, Receive, Scope, Send

MCP_AUTH_HEADER = "x-api-key"
_EXEMPT_PATHS: frozenset[str] = frozenset()


def mcp_shared_secret() -> str | None:
    """Return the process MCP shared secret, if configured.

    ``MCP_SHARED_SECRET`` wins; ``GATEWAY_API_KEY`` is the fallback so a single
    compose secret can protect both the gateway and the agent mesh.

    Returns:
        The secret string, or ``None`` when auth is disabled.
    """
    raw = os.environ.get("MCP_SHARED_SECRET") or os.environ.get("GATEWAY_API_KEY")
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def compare_secrets(provided: str | None, expected: str) -> bool:
    """Compare API keys in constant time.

    Args:
        provided: Header value from the caller.
        expected: Configured secret.

    Returns:
        ``True`` when the values match.
    """
    given = (provided or "").encode("utf-8")
    want = expected.encode("utf-8")
    if len(given) != len(want):
        secrets.compare_digest(want, want)
        return False
    return secrets.compare_digest(given, want)


def mcp_request_headers() -> dict[str, str]:
    """Headers the MCP HTTP client must send to authenticated agents.

    Returns:
        ``X-API-Key`` mapping when a shared secret is configured.
    """
    secret = mcp_shared_secret()
    if secret is None:
        return {}
    return {"X-API-Key": secret}


class SharedSecretASGIMiddleware:
    """Reject MCP HTTP requests that do not present the shared secret."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        secret: str,
        exempt_paths: frozenset[str] | None = None,
    ) -> None:
        """Wrap ``app`` with shared-secret checks.

        Args:
            app: Downstream ASGI app.
            secret: Expected ``X-API-Key`` value.
            exempt_paths: Paths that skip authentication. Defaults to none;
                ``/metrics`` requires the shared secret when auth is enabled.
        """
        self.app = app
        self.secret = secret
        self.exempt_paths = exempt_paths if exempt_paths is not None else _EXEMPT_PATHS

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Filter HTTP requests missing a valid shared secret.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "")
        if path in self.exempt_paths:
            await self.app(scope, receive, send)
            return
        headers = _header_map(scope)
        provided = headers.get(MCP_AUTH_HEADER)
        if compare_secrets(provided, self.secret):
            await self.app(scope, receive, send)
            return
        await _send_json(send, 401, b'{"detail":"invalid api key"}')


def wrap_mcp_auth(app: ASGIApp) -> ASGIApp:
    """Wrap an MCP Starlette app with shared-secret auth when configured.

    Args:
        app: FastMCP streamable-HTTP ASGI app.

    Returns:
        The original app when no secret is set, otherwise the wrapped app.
    """
    secret = mcp_shared_secret()
    if secret is None:
        return app
    return SharedSecretASGIMiddleware(app, secret=secret)


def _header_map(scope: Scope) -> dict[str, str]:
    raw_headers = scope.get("headers") or []
    return {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in raw_headers}


async def _send_json(send: Send, status: int, body: bytes) -> None:
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
