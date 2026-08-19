"""Shared-secret MCP auth middleware."""

from __future__ import annotations

import asyncio
from typing import Any

from core.mcp.auth import SharedSecretASGIMiddleware, compare_secrets, mcp_request_headers


def test_compare_secrets_rejects_wrong_key() -> None:
    assert compare_secrets("secret", "secret") is True
    assert compare_secrets("nope", "secret") is False
    assert compare_secrets(None, "secret") is False


def test_mcp_request_headers_from_env(monkeypatch: Any) -> None:
    monkeypatch.delenv("MCP_SHARED_SECRET", raising=False)
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    assert mcp_request_headers() == {}
    monkeypatch.setenv("MCP_SHARED_SECRET", "mesh-secret")
    assert mcp_request_headers() == {"X-API-Key": "mesh-secret"}


def test_shared_secret_middleware_rejects_and_allows() -> None:
    async def inner(scope: dict[str, Any], receive: Any, send: Any) -> None:
        _ = scope, receive
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    app = SharedSecretASGIMiddleware(inner, secret="s3cret")

    async def _call(path: str, headers: list[tuple[bytes, bytes]]) -> int:
        status = 0

        async def send(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])

        async def receive() -> dict[str, Any]:
            return {"type": "http.request"}

        await app({"type": "http", "path": path, "headers": headers}, receive, send)
        return status

    assert asyncio.run(_call("/mcp", [])) == 401
    assert asyncio.run(_call("/mcp", [(b"x-api-key", b"wrong")])) == 401
    assert asyncio.run(_call("/mcp", [(b"x-api-key", b"s3cret")])) == 200
    assert asyncio.run(_call("/metrics", [])) == 401
    assert asyncio.run(_call("/metrics", [(b"x-api-key", b"s3cret")])) == 200
