"""Gateway API tests for streaming, health, auth, and index jobs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from core.exceptions import (
    AgentUnavailableError,
    CircuitBreakerOpenError,
    GraphLookupError,
    RoutingError,
    SchemaValidationError,
    SynthesisError,
)
from core.gateway import ChatEvent, ChatGatewayService, GatewayDependencies
from core.health import HealthStatus
from core.mcp.streaming import encode_stream_event
from core.orchestration.fallback import EVIDENCE_ONLY_HEADER
from core.orchestration.service import OrchestratorService
from core.resilience.session_pool import AgentSessionPool
from core.settings import GatewaySettings, OrchestratorSettings
from gateway.app import _run_index_job, create_app
from gateway.mcp_client import GatewayOrchestratorClient


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
        self.mode: str | None = None
        self.correlation_id: str | None = None

    async def index_repository(
        self,
        repo_url: str | None = None,
        *,
        mode: str = "incremental",
        correlation_id: str,
    ) -> dict[str, Any]:
        _ = repo_url
        self.mode = mode
        self.correlation_id = correlation_id
        return self.result


class _StubSpecialists:
    def __init__(self, indexer: _StubIndexer | None = None) -> None:
        self.indexer = indexer or _StubIndexer()
        self.graph_query = object()
        self.code_analyst = object()


class _LocalFailingOrchestrator:
    """Gateway orchestrator client that runs the real loop with a failing synthesizer."""

    def __init__(self, llm: object, *, settings: OrchestratorSettings | None = None) -> None:
        self._service = OrchestratorService(
            llm,  # type: ignore[arg-type]
            settings=settings or OrchestratorSettings(routing_strategy="rules_first"),
        )
        self._clients = SimpleNamespace(
            memory=_GatewayMemory(),
            graph_query=_GatewayGraphQuery(),
            code_analyst=_GatewayCodeAnalyst(),
            indexer=_GatewayIndexer(),
        )

    async def handle_query(
        self, query: str, session_id: str, *, correlation_id: str
    ) -> dict[str, Any]:
        result = await self._service.handle_query(
            query,
            session_id,
            clients=self._clients,  # type: ignore[arg-type]
            correlation_id=correlation_id,
        )
        return {"answer": result.answer, "metadata": result.metadata}


class _GatewayMemory:
    async def get_context(self, session_id: str, token_budget: int = 3000) -> object:
        _ = session_id, token_budget
        from core.memory import ConversationContext

        return ConversationContext()

    async def get_cached_response(self, cache_key: str) -> object | None:
        _ = cache_key
        return None

    async def cache_response(self, cache_key: str, response_json: object) -> None:
        _ = cache_key, response_json

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        _ = session_id, role, content


class _GatewayGraphQuery:
    async def get_statistics(self) -> object:
        return SimpleNamespace(index_version="idx-1")

    async def find_entity(self, name: str, entity_type: str | None = None) -> object:
        _ = entity_type
        return {
            "file_path": "fastapi/applications.py",
            "line_start": 10,
            "line_end": 40,
            "qualified_name": "fastapi.applications.FastAPI",
            "name": name,
        }

    async def get_dependencies(self, name: str) -> object:
        return {"name": name, "neighbors": []}

    async def get_dependents(self, name: str) -> object:
        return {"name": name, "neighbors": []}

    async def find_related(self, name: str, relationship_type: str) -> object:
        return {"name": name, "relationship_type": relationship_type, "neighbors": []}

    async def trace_imports(self, module: str, depth: int = 5) -> object:
        _ = depth
        return {"module": module, "paths": []}


class _GatewayCodeAnalyst:
    async def get_code_snippet(self, **kwargs: object) -> object:
        return {
            "file_path": kwargs.get("file_path") or "fastapi/applications.py",
            "line_start": kwargs.get("line_start") or 10,
            "line_end": kwargs.get("line_end") or 12,
            "text": "class FastAPI:\n    pass\n",
            "error": None,
        }

    async def explain_implementation(self, qualified_name: str) -> object:
        return {"qualified_name": qualified_name, "explanation": "explained", "error": None}

    async def analyze_function(self, qualified_name: str) -> object:
        return {"qualified_name": qualified_name, "summary": "analyzed", "error": None}

    async def analyze_class(self, qualified_name: str) -> object:
        return {"qualified_name": qualified_name, "summary": "analyzed class", "error": None}

    async def compare_implementations(self, name_a: str, name_b: str) -> object:
        return {"name_a": name_a, "name_b": name_b, "summary": "compared", "error": None}

    async def find_patterns(
        self, pattern: str, path_prefix: str | None = None
    ) -> object:
        return {"pattern": pattern, "instances": [], "error": None}


class _GatewayIndexer:
    async def index_repository(self, repo_url: str | None = None) -> object:
        return {"status": "ok", "repo_url": repo_url}


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


def test_ws_chat_event_sequence() -> None:
    client = _app()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi"})
        events = [ws.receive_json() for _ in range(4)]
    assert [event["type"] for event in events] == ["routing", "agent_result", "answer", "done"]
    assert events[0]["data"]["agents"] == ["graph_query"]
    assert events[2]["data"]["chunk"] == "hello"


def test_ws_chat_rejects_invalid_api_key() -> None:
    from starlette.websockets import WebSocketDisconnect

    client = _app(api_key="shared-secret")
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()
    assert exc_info.value.code == 1008


def test_health_aggregation_handles_one_agent_down(monkeypatch) -> None:
    from gateway import app as gateway_app

    async def fake_call_agent_health(agent: str, url: str, **kwargs: Any) -> HealthStatus:
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
    indexer = _StubIndexer({"status": "ok", "files_indexed": 7})
    client = _app(indexer=indexer)
    create = client.post("/api/index", json={"mode": "incremental"})
    assert create.status_code == 200
    job_id = create.json()["job_id"]

    status = client.get(f"/api/index/status/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "done"
    assert body["mode"] == "incremental"
    assert body["report"]["files_indexed"] == 7
    assert indexer.mode == "incremental"


def test_index_job_forwards_full_mode() -> None:
    indexer = _StubIndexer({"status": "ok", "files_indexed": 3})
    client = _app(indexer=indexer)
    create = client.post("/api/index", json={"mode": "full"})
    assert create.status_code == 200
    job_id = create.json()["job_id"]
    status = client.get(f"/api/index/status/{job_id}")
    assert status.json()["mode"] == "full"
    assert indexer.mode == "full"


class _LongRunningIndexer:
    """An indexer whose pass outlives the MCP request timeout.

    index_repository never answers -- the gateway's dispatch times out and is
    abandoned, exactly as it is against the real agent -- while
    get_index_status reports the run and then its report.
    """

    def __init__(self, *, polls_running: int = 2, report: dict[str, Any] | None = None) -> None:
        self.polls_running = polls_running
        self.report = report or {"status": "ok", "files_indexed": 1138, "duration_s": 26.0}
        self.polls = 0
        self.dispatched = False
        self.mode: str | None = None

    async def index_repository(
        self,
        repo_url: str | None = None,
        *,
        mode: str = "incremental",
        correlation_id: str,
    ) -> dict[str, Any]:
        _ = repo_url, correlation_id
        self.mode = mode
        self.dispatched = True
        await asyncio.sleep(3600)
        raise AssertionError("dispatch should have been abandoned")

    async def get_index_status(self, *, correlation_id: str) -> dict[str, Any]:
        _ = correlation_id
        if not self.dispatched:
            return {"running": False, "last_report": None}
        self.polls += 1
        if self.polls <= self.polls_running:
            return {"running": True, "last_report": None}
        return {"running": False, "last_report": self.report}


def _index_app(indexer: Any, **overrides: Any) -> Any:
    settings = GatewaySettings(
        host="127.0.0.1",
        port=8000,
        api_key=None,
        index_poll_interval_s=0.0,
        **overrides,
    )
    deps = GatewayDependencies(
        orchestrator=_StubOrchestrator(),
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    deps.specialists.indexer = indexer
    return create_app(settings, deps=deps)


def test_index_outliving_the_request_timeout_still_reports_success() -> None:
    indexer = _LongRunningIndexer()
    app = _index_app(indexer)
    record = app.state.job_registry.create(mode="full", correlation_id="corr-long")

    asyncio.run(_run_index_job(app, record.job_id))

    saved = app.state.job_registry.get(record.job_id)
    assert saved is not None
    assert saved.status == "done"
    assert saved.error is None
    assert saved.report == indexer.report
    assert indexer.mode == "full"


def test_a_finished_index_is_detected_even_if_running_was_never_observed() -> None:
    """The pass can end between the dispatch and the first poll."""
    indexer = _LongRunningIndexer(polls_running=0)
    app = _index_app(indexer)
    record = app.state.job_registry.create(mode="incremental", correlation_id="corr-fast")

    asyncio.run(_run_index_job(app, record.job_id))

    saved = app.state.job_registry.get(record.job_id)
    assert saved is not None
    assert saved.status == "done"
    assert saved.report == indexer.report


def test_an_index_that_never_finishes_fails_on_the_job_ceiling() -> None:
    indexer = _LongRunningIndexer(polls_running=10_000)
    app = _index_app(indexer, index_timeout_s=0.05)
    record = app.state.job_registry.create(mode="full", correlation_id="corr-stuck")

    asyncio.run(_run_index_job(app, record.job_id))

    saved = app.state.job_registry.get(record.job_id)
    assert saved is not None
    assert saved.status == "failed"
    assert "did not finish within" in (saved.error or "")


def test_an_index_that_never_starts_fails_on_the_start_grace() -> None:
    class _SilentIndexer(_LongRunningIndexer):
        async def get_index_status(self, *, correlation_id: str) -> dict[str, Any]:
            _ = correlation_id
            return {"running": False, "last_report": None}

    app = _index_app(_SilentIndexer(), index_start_grace_s=0.05)
    record = app.state.job_registry.create(mode="full", correlation_id="corr-silent")

    asyncio.run(_run_index_job(app, record.job_id))

    saved = app.state.job_registry.get(record.job_id)
    assert saved is not None
    assert saved.status == "failed"
    assert "did not start within" in (saved.error or "")


def test_already_running_is_reported_from_the_dispatch() -> None:
    refused = {"status": "already_running", "detail": "index already running"}
    indexer = _StubIndexer(refused)
    client = _app(indexer=indexer)
    job_id = client.post("/api/index", json={"mode": "full"}).json()["job_id"]

    body = client.get(f"/api/index/status/{job_id}").json()
    assert body["status"] == "done"
    assert body["report"] == refused


def test_run_index_job_marks_failure() -> None:
    class _FailingIndexer:
        async def index_repository(
            self,
            repo_url: str | None = None,
            *,
            mode: str = "incremental",
            correlation_id: str,
        ) -> dict[str, Any]:
            _ = repo_url, mode, correlation_id
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


def test_chat_gateway_surfaces_tools_invoked() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id, correlation_id
            return {
                "answer": "ok",
                "metadata": {
                    "routing_mode": "rules",
                    "tools_invoked": [
                        "graph_query.find_entity",
                        "graph_query.get_dependents",
                    ],
                    "cached": False,
                    "degraded": False,
                    "tokens": {
                        "total": 0,
                        "prompt": 0,
                        "completion": 0,
                        "llm_calls": 0,
                        "by_purpose": {},
                    },
                },
            }

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    result = asyncio.run(ChatGatewayService(deps).run("who depends on FastAPI?", "s1", "c1"))
    assert result.tools_invoked == [
        "graph_query.find_entity",
        "find_entity",
        "graph_query.get_dependents",
        "get_dependents",
    ]
    assert result.routing["tools_invoked"] == [
        "graph_query.find_entity",
        "graph_query.get_dependents",
    ]
    assert result.routing["agents"] == ["graph_query"]


@pytest.mark.asyncio
async def test_gateway_emits_answer_before_orchestrator_call_completes() -> None:
    released = asyncio.Event()
    call_started = asyncio.Event()
    saw_answer = asyncio.Event()

    class _Session:
        async def call_tool(
            self,
            name: str,
            arguments: Any = None,
            *,
            meta: Any = None,
            progress_callback: Any = None,
        ) -> dict[str, Any]:
            _ = name, arguments, meta
            if progress_callback is not None:
                await progress_callback(
                    1.0,
                    None,
                    encode_stream_event("token", chunk="Hel"),
                )
            call_started.set()
            await released.wait()
            return {
                "answer": "Hello",
                "metadata": {
                    "routing_mode": "rules",
                    "cached": False,
                    "degraded": False,
                    "partial": False,
                    "tokens": {
                        "total": 1,
                        "prompt": 1,
                        "completion": 0,
                        "llm_calls": 1,
                    },
                    "tools_invoked": ["graph_query.find_entity"],
                },
            }

        async def aclose(self) -> None:
            return None

    async def opener(agent: str) -> _Session:
        _ = agent
        return _Session()

    settings = _settings()
    pool = AgentSessionPool({"orchestrator": "http://orchestrator/mcp"}, open_session=opener)
    deps = GatewayDependencies(
        orchestrator=GatewayOrchestratorClient(settings, pool),
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    events: list[ChatEvent] = []

    async def consume() -> None:
        async for event in ChatGatewayService(deps).stream("What is FastAPI?", "s", "c"):
            events.append(event)
            if event.type == "answer":
                saw_answer.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(call_started.wait(), timeout=1.0)
    await asyncio.wait_for(saw_answer.wait(), timeout=1.0)
    assert task.done() is False
    assert [event.type for event in events] == ["answer"]
    assert events[0].data["chunk"] == "Hel"
    released.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert events[-1].type == "done"
    await pool.aclose()


@pytest.mark.asyncio
async def test_gateway_post_hoc_chunks_when_progress_never_arrives() -> None:
    class _Orch:
        async def handle_query(
            self,
            query: str,
            session_id: str,
            *,
            correlation_id: str,
            on_token: Any | None = None,
            on_event: Any | None = None,
        ) -> dict[str, Any]:
            _ = query, session_id, correlation_id, on_token, on_event
            return {
                "answer": "Hello world",
                "metadata": {
                    "routing_mode": "rules",
                    "cached": False,
                    "degraded": False,
                    "tokens": {"total": 1, "prompt": 1, "completion": 0, "llm_calls": 1},
                    "tools_invoked": ["graph_query.find_entity"],
                },
            }

    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=_settings(),
        orchestrator_settings=OrchestratorSettings(),
    )
    events = [
        event async for event in ChatGatewayService(deps).stream("What is FastAPI?", "s", "c")
    ]
    types = [event.type for event in events]
    assert types[0] == "routing"
    assert "answer" in types
    assert types[-1] == "done"
    chunks = [event.data.get("chunk") for event in events if event.type == "answer"]
    assert "".join(str(chunk) for chunk in chunks) == "Hello world"


@pytest.mark.asyncio
async def test_done_event_includes_memory_and_graph_statistics_flags() -> None:
    class _Orch:
        async def handle_query(
            self,
            query: str,
            session_id: str,
            *,
            correlation_id: str,
            on_token: Any | None = None,
            on_event: Any | None = None,
        ) -> dict[str, Any]:
            _ = query, session_id, correlation_id, on_token, on_event
            return {
                "answer": "Hello world",
                "metadata": {
                    "routing_mode": "rules",
                    "cached": False,
                    "degraded": False,
                    "memory_available": False,
                    "graph_statistics_available": False,
                    "tokens": {"total": 1, "prompt": 1, "completion": 0, "llm_calls": 1},
                    "tools_invoked": ["graph_query.find_entity"],
                },
            }

    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=_settings(),
        orchestrator_settings=OrchestratorSettings(),
    )
    events = [
        event async for event in ChatGatewayService(deps).stream("What is FastAPI?", "s", "c")
    ]
    done = next(event for event in events if event.type == "done")
    assert done.data["memory_available"] is False
    assert done.data["graph_statistics_available"] is False


def test_chat_gateway_raises_when_orchestrator_unavailable() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id
            raise AgentUnavailableError(
                agent="handle_query",
                correlation_id=correlation_id,
                message="unhandled errors in a TaskGroup (1 sub-exception)",
            )

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    with pytest.raises(AgentUnavailableError):
        asyncio.run(ChatGatewayService(deps).run("explain Depends", "s1", "c1"))


def test_chat_http_agent_unavailable_returns_503() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id
            raise AgentUnavailableError(
                agent="handle_query",
                correlation_id=correlation_id,
                message="orchestrator down",
            )

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    body = response.json()
    assert body["degraded"] is True
    assert body["agent"] == "handle_query"
    assert body["tools_invoked"] == []


def test_chat_http_synthesis_error_returns_200_evidence_only() -> None:
    class _Boom:
        async def complete(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("llm exploded")

    deps = GatewayDependencies(
        orchestrator=_LocalFailingOrchestrator(_Boom()),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=_settings(),
        orchestrator_settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    app = create_app(_settings(), deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "What is the FastAPI class?"})
    assert response.status_code == 200
    body = response.json()
    assert body["done"]["degraded"] is True
    assert body["done"]["evidence_only"] is True
    assert body["done"]["degraded_reason"] == "RuntimeError"
    assert EVIDENCE_ONLY_HEADER in body["answer"]
    assert "fastapi/applications.py" in body["answer"]


def test_chat_http_synthesis_timeout_returns_200_evidence_only() -> None:
    class _Hang:
        async def complete(self, *args: object, **kwargs: object) -> object:
            await asyncio.sleep(5)
            raise AssertionError("should have timed out")

    deps = GatewayDependencies(
        orchestrator=_LocalFailingOrchestrator(  # type: ignore[arg-type]
            _Hang(),
            settings=OrchestratorSettings(
                routing_strategy="rules_first",
                synthesis_timeout_s=0.05,
            ),
        ),
        specialists=_StubSpecialists(),
        gateway_settings=_settings(),
        orchestrator_settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    app = create_app(_settings(), deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "What is the FastAPI class?"})
    assert response.status_code == 200
    body = response.json()
    assert body["done"]["degraded"] is True
    assert body["done"]["evidence_only"] is True
    assert body["done"]["degraded_reason"] == "TimeoutError"
    assert EVIDENCE_ONLY_HEADER in body["answer"]


def test_chat_sse_and_ws_partial_synthesis_surfaces_partial_done() -> None:
    class _PartialThenHang:
        async def stream(self, *args: object, **kwargs: object) -> Any:
            _ = args, kwargs
            yield "FastAPI subclasses Starlette and ", None
            yield "APIRouter groups path operations.", None
            await asyncio.sleep(5)

    deps = GatewayDependencies(
        orchestrator=_LocalFailingOrchestrator(  # type: ignore[arg-type]
            _PartialThenHang(),
            settings=OrchestratorSettings(
                routing_strategy="rules_first",
                synthesis_timeout_s=0.05,
            ),
        ),
        specialists=_StubSpecialists(),
        gateway_settings=_settings(),
        orchestrator_settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    app = create_app(_settings(), deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    retained = "FastAPI subclasses Starlette"

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "What is the FastAPI class?", "stream": True},
    ) as response:
        assert response.status_code == 200
        sse_payloads: list[dict[str, Any]] = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                sse_payloads.append(json.loads(line.split(":", 1)[1].strip()))
    sse_answer = "".join(
        str(item.get("chunk") or "") for item in sse_payloads if item.get("type") == "answer"
    )
    sse_done = next(item for item in sse_payloads if item.get("type") == "done")
    assert sse_done["partial"] is True
    assert retained in sse_answer
    assert EVIDENCE_ONLY_HEADER not in sse_answer

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "What is the FastAPI class?"})
        ws_events: list[dict[str, Any]] = []
        for _ in range(32):
            event = ws.receive_json()
            ws_events.append(event)
            if event["type"] == "done":
                break
        else:
            raise AssertionError("websocket stream ended without a done event")
    ws_answer = "".join(
        str(event.get("data", {}).get("chunk") or "")
        for event in ws_events
        if event.get("type") == "answer"
    )
    ws_done = next(event for event in ws_events if event.get("type") == "done")
    assert ws_done["data"]["partial"] is True
    assert retained in ws_answer
    assert EVIDENCE_ONLY_HEADER not in ws_answer


def test_chat_http_leaked_synthesis_error_returns_503() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id
            raise SynthesisError(
                agent="orchestrator",
                correlation_id=correlation_id,
                message="synthesis failed",
            )

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert response.json()["degraded"] is True


def test_chat_http_routing_error_returns_422() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id
            raise RoutingError(
                agent="orchestrator",
                correlation_id=correlation_id,
                message="cannot route",
            )

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 422


def test_chat_http_schema_validation_error_returns_422() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id
            raise SchemaValidationError(
                agent="orchestrator",
                correlation_id=correlation_id,
                message="invalid structured output",
            )

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid structured output"


def test_chat_http_graph_lookup_error_returns_503() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id
            raise GraphLookupError(
                agent="code_analyst",
                correlation_id=correlation_id,
                message="graph_query execute_query failed",
            )

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    body = response.json()
    assert body["degraded"] is True
    assert body["agent"] == "code_analyst"


def test_chat_http_circuit_breaker_open_returns_503() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id
            raise CircuitBreakerOpenError(
                agent="graph_query",
                correlation_id=correlation_id,
                message="circuit breaker open for graph_query",
            )

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert response.json()["agent"] == "graph_query"


def test_chat_http_tools_invoked_includes_bare_tool_names() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, Any]:
            _ = query, session_id, correlation_id
            return {
                "answer": "ok",
                "metadata": {
                    "routing_mode": "rules",
                    "tools_invoked": [
                        "graph_query.find_entity",
                        "graph_query.find_related",
                    ],
                    "cached": False,
                    "degraded": False,
                    "tokens": {
                        "total": 0,
                        "prompt": 0,
                        "completion": 0,
                        "llm_calls": 0,
                        "by_purpose": {},
                    },
                },
            }

    settings = _settings()
    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=_StubSpecialists(),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(),
    )
    app = create_app(settings, deps=deps)
    client = TestClient(app)
    body = client.post(
        "/api/chat",
        json={"message": "What classes inherit from APIRouter?"},
    ).json()
    tools = body["tools_invoked"]
    assert "get_dependents" in tools or "find_related" in tools


def test_openapi_includes_api_key_security_scheme() -> None:
    client = _app()
    schema = client.get("/openapi.json").json()
    schemes = schema["components"]["securitySchemes"]
    assert schemes["ApiKeyAuth"]["type"] == "apiKey"
    assert schemes["ApiKeyAuth"]["name"] == "X-API-Key"
    assert schema["security"] == [{"ApiKeyAuth": []}]
    paths = schema["paths"]
    assert "/api/chat" in paths
    assert paths["/api/chat"]["post"]["tags"] == ["chat"]
    chat_responses = paths["/api/chat"]["post"]["responses"]
    assert "422" in chat_responses
    assert "503" in chat_responses
    schemas = schema["components"]["schemas"]
    assert "GatewayErrorBody" in schemas


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    client = _app()
    client.post("/api/chat", json={"message": "hi"})
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "repochat_requests_total" in body
    assert "repochat_request_duration_seconds" in body


def test_metrics_endpoint_requires_api_key(monkeypatch) -> None:
    async def fake_call_agent_health(agent: str, url: str, **kwargs: Any) -> HealthStatus:
        _ = url, kwargs
        return HealthStatus(status="ok", agent=agent)

    monkeypatch.setattr("gateway.app.call_agent_health", fake_call_agent_health)
    client = _app(api_key="shared-secret")
    assert client.get("/metrics").status_code == 401
    ok = client.get("/metrics", headers={"x-api-key": "shared-secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200
