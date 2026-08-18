"""Gateway API tests for streaming, health, auth, and index jobs."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr

from core.gateway import ChatEvent, GatewayDependencies
from core.health import HealthStatus
from core.settings import GatewaySettings, OrchestratorSettings
from gateway.app import _run_index_job, create_app


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
                        "total": 120,
                        "prompt": 100,
                        "completion": 20,
                        "llm_calls": 1,
                        "by_purpose": {
                            "synthesis": {"prompt": 100, "completion": 20, "total": 120}
                        },
                    },
                },
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
            data={
                "correlation_id": correlation_id,
                "cached": False,
                "degraded": False,
                "routing_mode": "rules",
                "latency_ms": 1,
                "tokens": {
                    "total": 120,
                    "prompt": 100,
                    "completion": 20,
                    "llm_calls": 1,
                    "by_purpose": {
                        "synthesis": {"prompt": 100, "completion": 20, "total": 120}
                    },
                },
            },
        )


class _StubIndexer:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {"status": "ok", "files_indexed": 5}

    async def index_repository(
        self,
        repo_url: str | None = None,
        *,
        correlation_id: str,
    ) -> dict[str, Any]:
        _ = repo_url, correlation_id
        return self.result


class _StubSpecialists:
    def __init__(self, indexer: _StubIndexer | None = None) -> None:
        self.indexer = indexer or _StubIndexer()
        self.graph_query = object()
        self.code_analyst = object()


class _StubOrchestrator:
    async def get_conversation_context(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not used in these tests")

    async def analyze_query(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not used in these tests")

    async def route_to_agents(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not used in these tests")

    async def synthesize_response(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("not used in these tests")


def _settings(api_key: str | None = None) -> GatewaySettings:
    return GatewaySettings(
        host="127.0.0.1",
        port=8000,
        api_key=SecretStr(api_key) if api_key is not None else None,
    )


def _app(*, api_key: str | None = None, indexer: _StubIndexer | None = None) -> TestClient:
    settings = _settings(api_key)
    deps = GatewayDependencies(
        orchestrator=_StubOrchestrator(),
        specialists=_StubSpecialists(indexer=indexer),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    app.state.chat_service = _StubChatService()
    return TestClient(app)


def test_sse_chat_event_sequence() -> None:
    client = _app()
    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "hi", "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["x-correlation-id"]
        events: list[str] = []
        payloads: list[dict[str, Any]] = []
        current_event = ""
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
                events.append(current_event)
            if line.startswith("data:"):
                payloads.append(json.loads(line.split(":", 1)[1].strip()))
        assert events == ["routing", "agent_result", "answer", "done"]
        assert [payload["type"] for payload in payloads] == events


def test_health_aggregation_handles_one_agent_down(monkeypatch) -> None:
    from gateway import app as gateway_app

    async def fake_call_agent_health(agent: str, url: str) -> HealthStatus:
        _ = url
        if agent == "memory":
            return HealthStatus(status="error", agent=agent, detail="down")
        return HealthStatus(status="ok", agent=agent)

    monkeypatch.setattr(gateway_app, "call_agent_health", fake_call_agent_health)
    client = _app()
    response = client.get("/api/agents/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["agents"]["memory"]["status"] == "error"


def test_auth_disabled_allows_requests() -> None:
    client = _app()
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 200
    assert response.json()["answer"] == "hello"
    assert response.json()["done"]["tokens"]["total"] == 120


def test_auth_enabled_rejects_missing_key_and_accepts_valid_key() -> None:
    client = _app(api_key="shared-secret")
    rejected = client.post("/api/chat", json={"message": "hi"})
    assert rejected.status_code == 401

    accepted = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"x-api-key": "shared-secret"},
    )
    assert accepted.status_code == 200


def test_index_job_registry_lifecycle() -> None:
    client = _app(indexer=_StubIndexer({"status": "ok", "files_indexed": 7}))
    create = client.post("/api/index", json={"mode": "incremental"})
    assert create.status_code == 200
    job_id = create.json()["job_id"]

    status = client.get(f"/api/index/status/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "done"
    assert body["mode"] == "incremental"
    assert body["report"]["files_indexed"] == 7


def test_run_index_job_marks_failure() -> None:
    class _FailingIndexer:
        async def index_repository(
            self,
            repo_url: str | None = None,
            *,
            correlation_id: str,
        ) -> dict[str, Any]:
            _ = repo_url, correlation_id
            raise RuntimeError("boom")

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_StubOrchestrator(),
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    deps.specialists.indexer = _FailingIndexer()
    app = create_app(settings, deps=deps)
    record = app.state.job_registry.create(mode="full", correlation_id="corr-1")

    asyncio.run(_run_index_job(app, record.job_id))

    saved = app.state.job_registry.get(record.job_id)
    assert saved is not None
    assert saved.status == "failed"
    assert saved.error == "boom"
