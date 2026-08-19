"""Indexer (and sibling) FastMCP adapters bind inbound MCP correlation ids."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.exceptions import AgentUnavailableError
from core.health import HealthStatus
from core.logging import bind_correlation_id, correlation_id_from_mcp_meta, get_correlation_id
from core.mcp.context import bind_mcp_context


class _Meta:
    def __init__(self, correlation_id: str | None, extra: dict[str, Any] | None = None) -> None:
        self.correlation_id = correlation_id
        self.model_extra = extra


class _Ctx:
    def __init__(self, meta: _Meta | None) -> None:
        self.request_context = SimpleNamespace(meta=meta)


def test_correlation_id_from_mcp_meta_reads_field_and_extra() -> None:
    assert correlation_id_from_mcp_meta(None) is None
    assert correlation_id_from_mcp_meta(_Meta("abc-123")) == "abc-123"
    assert correlation_id_from_mcp_meta(_Meta(None, {"correlation_id": "from-extra"})) == (
        "from-extra"
    )
    assert correlation_id_from_mcp_meta(_Meta("  ")) is None
    bind_mcp_context(_Ctx(_Meta("corr-bind")))
    assert get_correlation_id() == "corr-bind"


def test_indexer_health_binds_inbound_correlation_id() -> None:
    from indexer.__main__ import health

    bind_correlation_id("unrelated")
    status = health(_Ctx(_Meta("corr-indexer-1")))  # type: ignore[arg-type]
    assert status.status == "ok"
    assert status.agent == "indexer"
    assert get_correlation_id() == "corr-indexer-1"


def test_indexer_parse_python_ast_binds_meta() -> None:
    from indexer.__main__ import parse_python_ast

    parsed = parse_python_ast("def ping():\n    return 1\n", _Ctx(_Meta("corr-parse")))  # type: ignore[arg-type]
    assert parsed.module
    assert get_correlation_id() == "corr-parse"


def test_graph_query_health_binds_inbound_correlation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from graph_query.__main__ import health

    bound: list[str] = []
    original = bind_mcp_context

    def _bind(ctx: Any) -> str:
        correlation_id = original(ctx)
        bound.append(correlation_id)
        return correlation_id

    monkeypatch.setattr("graph_query.__main__.bind_mcp_context", _bind)
    monkeypatch.setattr(
        "graph_query.__main__.check_graph_query_health",
        lambda: HealthStatus(status="ok", agent="graph_query"),
    )
    status = asyncio.run(health(_Ctx(_Meta("corr-gq-1"))))  # type: ignore[arg-type]
    assert status.status == "ok"
    assert bound == ["corr-gq-1"]


def test_memory_health_binds_inbound_correlation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from memory.__main__ import health

    monkeypatch.setattr(
        "memory.__main__.check_memory_health",
        lambda _path=None: HealthStatus(status="ok", agent="memory"),
    )
    status = health(_Ctx(_Meta("corr-mem-1")))  # type: ignore[arg-type]
    assert status.status == "ok"
    assert get_correlation_id() == "corr-mem-1"


def test_code_analyst_health_binds_inbound_correlation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from code_analyst.__main__ import health

    async def _ok(**kwargs: Any) -> HealthStatus:
        _ = kwargs
        return HealthStatus(status="ok", agent="code_analyst")

    bound: list[str] = []
    original = bind_mcp_context

    def _bind(ctx: Any) -> str:
        correlation_id = original(ctx)
        bound.append(correlation_id)
        return correlation_id

    monkeypatch.setattr("code_analyst.__main__.bind_mcp_context", _bind)
    monkeypatch.setattr("code_analyst.__main__.check_code_analyst_health", _ok)
    status = asyncio.run(health(_Ctx(_Meta("corr-ca-1"))))  # type: ignore[arg-type]
    assert status.status == "ok"
    assert bound == ["corr-ca-1"]


def test_orchestrator_health_ok() -> None:
    from orchestrator.__main__ import health

    status = asyncio.run(health())
    assert status.status == "ok"


def test_orchestrator_mcp_tool_error_raises_agent_unavailable() -> None:
    from mcp.types import CallToolResult, TextContent

    from core.mcp.client import PooledAgentClient
    from core.resilience.session_pool import AgentSessionPool

    error_result = CallToolResult(
        content=[TextContent(type="text", text="boom")],
        isError=True,
    )

    class _FakeSession:
        async def call_tool(
            self, name: str, arguments: Any = None, *, meta: Any = None
        ) -> CallToolResult:
            _ = name, arguments, meta
            return error_result

        async def aclose(self) -> None:
            return None

    async def opener(agent: str) -> _FakeSession:
        _ = agent
        return _FakeSession()

    pool = AgentSessionPool({"graph_query": "http://example.invalid/mcp"}, open_session=opener)
    client = PooledAgentClient(pool, "graph_query", timeout_s=1.0, retry_count=0)

    async def _run() -> None:
        await client.call("find_entity", {"name": "FastAPI"}, correlation_id="corr-mcp")

    with pytest.raises(AgentUnavailableError, match="find_entity"):
        asyncio.run(_run())


def test_indexer_tools_and_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.indexing import IndexReport, ParsedFile
    from indexer import __main__ as mod

    monkeypatch.setattr(mod, "clone_and_index", lambda url=None, **kwargs: IndexReport(status="ok"))
    monkeypatch.setattr(mod, "run_index_file", lambda path: IndexReport(status="ok"))
    monkeypatch.setattr(
        mod,
        "extract_entities_core",
        lambda path: SimpleNamespace(entities=[], relationships=[]),
    )
    monkeypatch.setattr(
        mod,
        "load_index_status",
        lambda **kwargs: SimpleNamespace(
            running=False, node_count=0, rel_count=0, counts={}
        ),
    )
    ctx = _Ctx(_Meta("corr-idx"))
    report = asyncio.run(mod.index_repository(None, ctx=ctx))  # type: ignore[arg-type]
    assert report.status == "ok"
    file_report = asyncio.run(mod.index_file("fastapi/applications.py", ctx))  # type: ignore[arg-type]
    assert file_report.status == "ok"
    extracted = mod.extract_entities("def ping():\n    return 1\n", ctx)  # type: ignore[arg-type]
    assert extracted.entities == []
    status = asyncio.run(mod.get_index_status(ctx))  # type: ignore[arg-type]
    assert status.running is False

    class _Locked:
        def locked(self) -> bool:
            return True

    monkeypatch.setattr(mod, "_index_lock", _Locked())
    monkeypatch.setattr(
        mod, "already_running_report", lambda: IndexReport(status="already_running")
    )
    skipped = asyncio.run(mod.index_repository(None, ctx=ctx))  # type: ignore[arg-type]
    assert skipped.status == "already_running"
    skipped_file = asyncio.run(mod.index_file("x.py", ctx))  # type: ignore[arg-type]
    assert skipped_file.status == "already_running"
    _ = ParsedFile


def test_indexer_index_repository_forwards_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.indexing import IndexReport
    from indexer import __main__ as mod

    captured: dict[str, object] = {}

    def _clone(url: str | None = None, **kwargs: object) -> IndexReport:
        captured["url"] = url
        captured.update(kwargs)
        return IndexReport(status="ok", mode=str(kwargs.get("mode") or "incremental"))

    monkeypatch.setattr(mod, "clone_and_index", _clone)
    ctx = _Ctx(_Meta("corr-idx-mode"))
    report = asyncio.run(mod.index_repository(None, "full", ctx))  # type: ignore[arg-type]
    assert report.status == "ok"
    assert captured["mode"] == "full"


def test_graph_query_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from graph_query import __main__ as mod

    class _Svc:
        def find_entity(self, name: str, entity_type: str | None = None) -> SimpleNamespace:
            return SimpleNamespace(
                result_count=1, truncated=False, error=None, name=name, entity_type=entity_type
            )

        def get_dependencies(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(result_count=0, truncated=False, total_count=0, name=name)

        def get_dependents(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(result_count=0, truncated=False, total_count=0, name=name)

        def trace_imports(self, module: str, depth: int = 5) -> SimpleNamespace:
            return SimpleNamespace(result_count=0, truncated=False, depth=depth, module=module)

        def find_related(self, name: str, relationship_type: str) -> SimpleNamespace:
            return SimpleNamespace(
                result_count=0,
                truncated=False,
                error=None,
                name=name,
                relationship_type=relationship_type,
            )

        def execute_query(self, cypher: str, params: Any = None) -> SimpleNamespace:
            _ = cypher, params
            return SimpleNamespace(result_count=0, truncated=False)

        def get_statistics(self) -> SimpleNamespace:
            return SimpleNamespace(index_version="v1", last_indexed_at=None)

    monkeypatch.setattr(mod, "_service", _Svc())
    ctx = _Ctx(_Meta("corr-gq"))
    assert asyncio.run(mod.find_entity("FastAPI", None, ctx)).result_count == 1  # type: ignore[arg-type]
    assert asyncio.run(mod.get_dependencies("FastAPI", ctx)).result_count == 0  # type: ignore[arg-type]
    assert asyncio.run(mod.get_dependents("FastAPI", ctx)).result_count == 0  # type: ignore[arg-type]
    assert asyncio.run(mod.trace_imports("fastapi", 2, ctx)).depth == 2  # type: ignore[arg-type]
    assert asyncio.run(mod.find_related("FastAPI", "CALLS", ctx)).result_count == 0  # type: ignore[arg-type]
    assert asyncio.run(mod.execute_query("RETURN 1", None, ctx)).result_count == 0  # type: ignore[arg-type]
    assert asyncio.run(mod.get_statistics(ctx)).index_version == "v1"  # type: ignore[arg-type]


def test_memory_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.memory import ConversationContext
    from memory import __main__ as mod

    class _Svc:
        async def append_turn(self, session_id: str, role: str, content: str) -> None:
            _ = session_id, role, content

        async def get_context(
            self, session_id: str, token_budget: int = 3000
        ) -> ConversationContext:
            _ = session_id, token_budget
            return ConversationContext()

        async def summarize_session(self, session_id: str) -> str:
            _ = session_id
            return "summary"

        async def cache_put(self, cache_key: str, response_json: Any) -> None:
            _ = cache_key, response_json

        async def cache_get(self, cache_key: str) -> None:
            _ = cache_key
            return None

    monkeypatch.setattr(mod, "_service", _Svc())
    ctx = _Ctx(_Meta("corr-mem"))
    assert asyncio.run(mod.append_turn("s", "user", "hi", ctx))["status"] == "ok"  # type: ignore[arg-type]
    ctx_out = asyncio.run(mod.get_context("s", 10, ctx))  # type: ignore[arg-type]
    assert isinstance(ctx_out, ConversationContext)
    assert asyncio.run(mod.summarize_session("s", ctx))["summary"] == "summary"  # type: ignore[arg-type]
    assert asyncio.run(mod.cache_response("k", {"a": 1}, ctx))["status"] == "ok"  # type: ignore[arg-type]
    assert asyncio.run(mod.get_cached_response("k", ctx)) is None  # type: ignore[arg-type]


def test_code_analyst_tools_and_explain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from code_analyst import __main__ as mod
    from core.analysis.models import (
        ClassAnalysis,
        FunctionAnalysis,
        ImplementationComparison,
        ImplementationExplanation,
        PatternAnalysis,
        SnippetResult,
    )

    class _Svc:
        async def analyze_function(self, qualified_name: str) -> FunctionAnalysis:
            return FunctionAnalysis(qualified_name=qualified_name, summary="ok")

        async def analyze_class(self, qualified_name: str) -> ClassAnalysis:
            return ClassAnalysis(qualified_name=qualified_name, summary="ok")

        async def find_patterns(self, pattern: str) -> PatternAnalysis:
            return PatternAnalysis(pattern=pattern, instances=[])

        async def get_code_snippet(self, **kwargs: Any) -> SnippetResult:
            return SnippetResult(file_path=str(kwargs.get("file_path") or "x.py"), text="pass")

        async def explain_implementation(self, qualified_name: str) -> ImplementationExplanation:
            return ImplementationExplanation(qualified_name=qualified_name, explanation="ok")

        async def compare_implementations(
            self, name_a: str, name_b: str
        ) -> ImplementationComparison:
            return ImplementationComparison(name_a=name_a, name_b=name_b, summary="ok")

    monkeypatch.setattr(mod, "_service", _Svc())
    ctx = _Ctx(_Meta("corr-ca"))
    assert asyncio.run(mod.analyze_function("fastapi.FastAPI", ctx)).summary == "ok"  # type: ignore[arg-type]
    assert asyncio.run(mod.analyze_class("fastapi.FastAPI", ctx)).summary == "ok"  # type: ignore[arg-type]
    assert asyncio.run(mod.find_patterns("decorator", ctx)).pattern == "decorator"  # type: ignore[arg-type]
    snippet = asyncio.run(mod.get_code_snippet("q", "fastapi/applications.py", 1, 2, ctx))  # type: ignore[arg-type]
    assert snippet.text == "pass"
    explained = asyncio.run(mod.explain_implementation("fastapi.FastAPI", ctx))  # type: ignore[arg-type]
    assert explained.explanation == "ok"
    compared = asyncio.run(mod.compare_implementations("a", "b", ctx))  # type: ignore[arg-type]
    assert compared.summary == "ok"


def test_code_analyst_explain_propagates_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from code_analyst import __main__ as mod
    from core.analysis.models import ImplementationExplanation

    class _Svc:
        async def explain_implementation(self, qualified_name: str) -> ImplementationExplanation:
            raise TimeoutError("analyst timeout")

    monkeypatch.setattr(mod, "_service", _Svc())
    with pytest.raises(TimeoutError, match="analyst timeout"):
        asyncio.run(
            mod.explain_implementation("fastapi.FastAPI", _Ctx(_Meta("corr-ca-timeout")))  # type: ignore[arg-type]
        )


def test_code_analyst_uses_core_graph_lookup() -> None:
    from code_analyst import __main__ as mod
    from core.analysis.graph_lookup import GraphQueryLookup

    assert isinstance(mod._service._graph_lookup, GraphQueryLookup)


def test_orchestrator_tools_budget_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.memory import ConversationContext
    from core.orchestration.models import ExecutionPlan, QueryIntent
    from orchestrator import __main__ as mod

    intent = QueryIntent(intent="lookup", entities=["FastAPI"], target_agents=["graph_query"])

    async def _analyze(*args: Any, **kwargs: Any) -> QueryIntent:
        _ = args, kwargs
        return intent

    monkeypatch.setattr(mod, "analyze_query_core", _analyze)
    monkeypatch.setattr(
        mod,
        "route_to_agents_core",
        lambda value: ExecutionPlan(intent=value, agents=["graph_query"]),
    )

    class _Synth:
        answer = "evidence"

    async def _synthesize(*args: Any, **kwargs: Any) -> _Synth:
        _ = args, kwargs
        return _Synth()

    monkeypatch.setattr(mod, "synthesize_response_core", _synthesize)

    class _Clients:
        async def get_context(
            self, session_id: str, token_budget: int = 3000
        ) -> ConversationContext:
            _ = session_id, token_budget
            return ConversationContext()

    monkeypatch.setattr(
        mod.PooledOrchestratorClients,
        "from_pool",
        classmethod(lambda cls, *a, **k: _Clients()),
    )
    ctx = _Ctx(_Meta("corr-orch"))
    out = asyncio.run(mod.get_conversation_context("s", 10, ctx))  # type: ignore[arg-type]
    assert isinstance(out, ConversationContext)
    routed = asyncio.run(mod.analyze_query("What is FastAPI?", ConversationContext(), ctx))  # type: ignore[arg-type]
    assert routed.intent == "lookup"
    plan = mod.route_to_agents(intent)
    assert plan.agents
    answer = asyncio.run(
        mod.synthesize_response("q", {}, ConversationContext(), ctx)  # type: ignore[arg-type]
    )
    assert answer == "evidence"

    class _Service:
        async def handle_query(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            _ = args, kwargs
            return SimpleNamespace(
                answer="partial",
                metadata={"budget_exhausted": "deadline", "degraded": False},
            )

    monkeypatch.setattr(mod, "_service", _Service())
    monkeypatch.setattr(mod, "get_mcp_pool", lambda: SimpleNamespace(breakers=SimpleNamespace()))
    payload = asyncio.run(mod.handle_query("q", "s", ctx))  # type: ignore[arg-type]
    assert payload["metadata"]["budget_exhausted"] == "deadline"


def test_orchestrator_uses_sse_json_response_false() -> None:
    from orchestrator.__main__ import mcp

    assert mcp.settings.json_response is False


def test_orchestrator_handle_query_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator import __main__ as mod

    reported: list[str | None] = []

    class _ProgressCtx:
        def __init__(self) -> None:
            self.request_context = SimpleNamespace(meta=_Meta("corr-stream"))

        async def report_progress(
            self, progress: float, total: float | None = None, message: str | None = None
        ) -> None:
            _ = progress, total
            reported.append(message)

    class _Service:
        async def handle_query(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            _ = args
            on_token = kwargs.get("on_token")
            if on_token is not None:
                await on_token("Hel")
            return SimpleNamespace(answer="Hello", metadata={"cached": False})

    monkeypatch.setattr(mod, "_service", _Service())
    monkeypatch.setattr(mod, "get_mcp_pool", lambda: SimpleNamespace(breakers=SimpleNamespace()))
    monkeypatch.setattr(
        mod.PooledOrchestratorClients,
        "from_pool",
        classmethod(lambda cls, *a, **k: object()),
    )
    payload = asyncio.run(mod.handle_query("q", "s", _ProgressCtx()))  # type: ignore[arg-type]
    assert payload["answer"] == "Hello"
    assert reported
    decoded = reported[0]
    assert decoded is not None
    assert "Hel" in decoded


def test_adapter_mains_start_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    def _run(mcp: Any, *, agent: str) -> None:
        _ = mcp
        started.append(agent)

    for mod_name in (
        "indexer.__main__",
        "graph_query.__main__",
        "code_analyst.__main__",
        "orchestrator.__main__",
    ):
        monkeypatch.setattr(f"{mod_name}.run_agent_mcp", _run)
        monkeypatch.setattr(f"{mod_name}.bind_correlation_id", lambda: None)
        module = __import__(mod_name, fromlist=["main"])
        module.main()
    assert set(started) >= {"indexer", "graph_query", "code_analyst", "orchestrator"}

    class _Mem:
        async def initialize(self) -> None:
            return None

    monkeypatch.setattr("memory.__main__.run_agent_mcp", _run)
    monkeypatch.setattr("memory.__main__.bind_correlation_id", lambda: None)
    monkeypatch.setattr("memory.__main__._service", _Mem())
    def _consume(coro: Any) -> None:
        if hasattr(coro, "close"):
            coro.close()

    monkeypatch.setattr("memory.__main__.asyncio.run", _consume)
    from memory.__main__ import main as mem_main

    mem_main()
    assert "memory" in started
