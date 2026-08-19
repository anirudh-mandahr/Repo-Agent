"""Gateway security: constant-time API keys, rate limits, and message size."""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from core.gateway import (
    DEFAULT_MAX_MESSAGE_LENGTH,
    ChatEvent,
    ChatRequest,
    GatewayDependencies,
    SlidingWindowRateLimiter,
)
from core.health import HealthStatus
from core.mcp.auth import compare_secrets
from core.settings import GatewaySettings, OrchestratorSettings
from gateway.app import create_app


class _StubChatService:
    async def run(self, message: str, session_id: str, correlation_id: str) -> Any:
        _ = message
        return type(
            "Result",
            (),
            {
                "session_id": session_id,
                "correlation_id": correlation_id,
                "routing": {"mode": "rules", "agents": ["graph_query"]},
                "agent_results": [{"agent": "graph_query", "ok": True}],
                "answer": "hello",
                "done": {
                    "correlation_id": correlation_id,
                    "cached": False,
                    "degraded": False,
                    "routing_mode": "rules",
                    "latency_ms": 1,
                    "tokens": {
                        "total": 0,
                        "prompt": 0,
                        "completion": 0,
                        "llm_calls": 0,
                        "by_purpose": {},
                    },
                },
                "tools_invoked": [],
            },
        )()

    async def stream(self, message: str, session_id: str, correlation_id: str):
        _ = message, session_id
        yield ChatEvent(
            type="routing",
            correlation_id=correlation_id,
            data={"mode": "rules", "agents": ["graph_query"]},
        )
        yield ChatEvent(
            type="agent_result",
            correlation_id=correlation_id,
            data={"agent": "graph_query", "ok": True},
        )
        yield ChatEvent(
            type="answer",
            correlation_id=correlation_id,
            data={"chunk": "hello"},
        )
        yield ChatEvent(
            type="done",
            correlation_id=correlation_id,
            data={"correlation_id": correlation_id},
        )


class _StubSpecialists:
    def __init__(self) -> None:
        self.indexer = object()
        self.graph_query = object()
        self.code_analyst = object()


class _StubOrchestrator:
    async def get_conversation_context(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not used")

    async def analyze_query(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not used")

    async def route_to_agents(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not used")

    async def synthesize_response(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("not used")


def _app(*, api_key: str | None = None, rate_limit_requests: int = 60) -> TestClient:
    settings = GatewaySettings(
        host="127.0.0.1",
        port=8000,
        api_key=SecretStr(api_key) if api_key is not None else None,
        rate_limit_requests=rate_limit_requests,
        rate_limit_window_s=60.0,
    )
    deps = GatewayDependencies(
        orchestrator=_StubOrchestrator(),
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    app.state.chat_service = _StubChatService()
    return TestClient(app)


def test_compare_secrets_uses_compare_digest(monkeypatch: Any) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = secrets.compare_digest

    def _wrapped(left: bytes | str, right: bytes | str) -> bool:
        if isinstance(left, str):
            left_b, right_b = left.encode("utf-8"), str(right).encode("utf-8")
        else:
            left_b, right_b = left, right  # type: ignore[assignment]
        calls.append((left_b, right_b))
        return real(left_b, right_b)

    monkeypatch.setattr("core.mcp.auth.secrets.compare_digest", _wrapped)
    assert compare_secrets("abc", "abc") is True
    assert compare_secrets("abc", "xyz") is False
    assert compare_secrets("ab", "abcd") is False
    assert calls


def test_gateway_api_key_uses_compare_digest(monkeypatch: Any) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = secrets.compare_digest

    def _wrapped(left: bytes | str, right: bytes | str) -> bool:
        left_b = left.encode("utf-8") if isinstance(left, str) else left
        right_b = right.encode("utf-8") if isinstance(right, str) else right
        calls.append((left_b, right_b))
        return real(left_b, right_b)

    monkeypatch.setattr("core.mcp.auth.secrets.compare_digest", _wrapped)
    client = _app(api_key="shared-secret")
    rejected = client.post("/api/chat", json={"message": "hi"})
    assert rejected.status_code == 401
    accepted = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"x-api-key": "shared-secret"},
    )
    assert accepted.status_code == 200
    assert calls


def test_rate_limiter_rejects_after_cap() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_s=60.0)
    assert limiter.allow("k", now=1.0) is True
    assert limiter.allow("k", now=1.1) is True
    assert limiter.allow("k", now=1.2) is False
    assert limiter.allow("k", now=62.0) is True


def test_gateway_rate_limit_returns_429() -> None:
    client = _app(rate_limit_requests=2)
    assert client.post("/api/chat", json={"message": "one"}).status_code == 200
    assert client.post("/api/chat", json={"message": "two"}).status_code == 200
    limited = client.post("/api/chat", json={"message": "three"})
    assert limited.status_code == 429
    assert limited.json()["detail"] == "rate limit exceeded"


def test_metrics_requires_api_key_when_configured(monkeypatch: Any) -> None:
    async def fake_call_agent_health(agent: str, url: str, **kwargs: Any) -> HealthStatus:
        _ = url, kwargs
        return HealthStatus(status="ok", agent=agent)

    monkeypatch.setattr("gateway.app.call_agent_health", fake_call_agent_health)
    client = _app(api_key="shared-secret")
    rejected = client.get("/metrics")
    assert rejected.status_code == 401
    accepted = client.get("/metrics", headers={"x-api-key": "shared-secret"})
    assert accepted.status_code == 200
    health = client.get("/health")
    assert health.status_code == 200


def test_ws_rate_limit_rejects_handshake_when_exhausted() -> None:
    from starlette.websockets import WebSocketDisconnect

    client = _app(rate_limit_requests=1)
    assert client.post("/api/chat", json={"message": "one"}).status_code == 200
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()
    assert exc_info.value.code == 1008


def test_ws_rate_limit_rejects_per_message() -> None:
    from starlette.websockets import WebSocketDisconnect

    client = _app(rate_limit_requests=2)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "first"})
        assert ws.receive_json()["type"] == "routing"
        for _ in range(3):
            ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.send_json({"message": "second"})
            ws.receive_json()
    assert exc_info.value.code == 1008


def test_chat_request_rejects_oversized_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * (DEFAULT_MAX_MESSAGE_LENGTH + 1))


def test_gateway_rejects_oversized_message_with_422() -> None:
    client = _app()
    response = client.post(
        "/api/chat",
        json={"message": "x" * (DEFAULT_MAX_MESSAGE_LENGTH + 1)},
    )
    assert response.status_code == 422
