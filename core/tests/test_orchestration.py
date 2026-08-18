from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.llm.stub import StubProvider
from core.memory import ConversationContext, ConversationTurn
from core.orchestration.models import QueryIntent
from core.orchestration.router import rule_based_route
from core.orchestration.synthesis import synthesize_response
from core.settings import OrchestratorSettings
from orchestrator.service import OrchestratorService


class _MemoryClient:
    def __init__(
        self,
        *,
        ctx: ConversationContext,
        cached_response: object | None,
        expected_cache_key: str | None = None,
    ) -> None:
        self._ctx = ctx
        self._cached_response = cached_response
        self._expected_cache_key = expected_cache_key
        self.appended: list[tuple[str, str]] = []
        self.cached_put_keys: list[str] = []

    async def get_context(self, session_id: str, token_budget: int = 3000) -> ConversationContext:
        _ = session_id, token_budget
        return self._ctx

    async def get_cached_response(self, cache_key: str) -> object | None:
        if self._expected_cache_key is not None:
            assert cache_key == self._expected_cache_key
        return self._cached_response

    async def cache_response(self, cache_key: str, response_json: object) -> None:
        self.cached_put_keys.append(cache_key)
        _ = response_json

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        _ = session_id, content
        self.appended.append((role, content))


class _GraphQueryClient:
    def __init__(
        self,
        *,
        index_version: str | None = None,
        timeout_on_find_entity: bool = False,
        find_entity_payload: object | None = None,
    ) -> None:
        self._index_version = index_version
        self._timeout = timeout_on_find_entity
        self._payload = find_entity_payload or {
            "file_path": "sample.py",
            "line_start": 10,
            "line_end": 20,
        }

    async def get_statistics(self) -> object:
        return SimpleNamespace(index_version=self._index_version)

    async def find_entity(self, name: str, entity_type: str | None = None) -> object:
        _ = entity_type
        if self._timeout:
            raise TimeoutError("graph_query timeout")
        _ = name
        return self._payload


class _CodeAnalystClient:
    def __init__(self) -> None:
        self.snippets_requested: list[dict[str, object]] = []

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> object:
        _ = qualified_name
        self.snippets_requested.append(
            {"file_path": file_path, "line_start": line_start, "line_end": line_end}
        )
        return {
            "file_path": file_path or "",
            "line_start": line_start,
            "line_end": line_end,
            "text": f"snippet for {file_path}",
            "error": None,
        }


class _IndexerClient:
    async def index_repository(self, repo_url: str | None = None) -> object:
        return {"status": "ok", "repo_url": repo_url}


def _session_context() -> ConversationContext:
    return ConversationContext(
        summary="SESSION SUMMARY",
        recent_turns=[
            ConversationTurn(
                id=1,
                role="user",
                content="previous question",
                created_at="2026-01-01T00:00:00Z",
                token_estimate=10,
            )
        ],
    )


@pytest.mark.asyncio
async def test_routing_fallback_and_prompt_context_separation() -> None:
    # Two malformed routing outputs => LLM retries once => SchemaValidationError =>
    # rule-based fallback.
    llm = StubProvider(
        [
            "not-json",
            "also-not-json",
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)

    ctx = _session_context()
    memory = _MemoryClient(ctx=ctx, cached_response=None)
    graph_query = _GraphQueryClient(index_version="idx-1")
    code_analyst = _CodeAnalystClient()
    indexer = _IndexerClient()

    clients = SimpleNamespace(
        memory=memory,
        graph_query=graph_query,
        code_analyst=code_analyst,
        indexer=indexer,
    )

    result = await service.handle_query(
        "explain how FastAPI works",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-1",
    )

    assert result.metadata["routing_mode"] == "rules_fallback"
    assert result.answer == "FINAL ANSWER"

    # Routing prompt call(s) must not contain session summary.
    routing_calls = [c for c in llm.calls if c.response_model is QueryIntent]
    assert routing_calls
    routing_text = "\n".join(
        msg.content for call in routing_calls for msg in call.messages
    )
    assert "SESSION SUMMARY" not in routing_text

    # Synthesis prompt must contain session summary.
    synthesis_calls = [c for c in llm.calls if c.response_model is None]
    assert synthesis_calls
    synthesis_text = "\n".join(msg.content for msg in synthesis_calls[0].messages)
    assert "SESSION SUMMARY" in synthesis_text


@pytest.mark.asyncio
async def test_graph_query_timeout_degrades_response_without_exception() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=["sample.py"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="llm routing",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)

    memory = _MemoryClient(ctx=_session_context(), cached_response=None)
    graph_query = _GraphQueryClient(index_version="idx-1", timeout_on_find_entity=True)
    code_analyst = _CodeAnalystClient()
    indexer = _IndexerClient()

    clients = SimpleNamespace(
        memory=memory,
        graph_query=graph_query,
        code_analyst=code_analyst,
        indexer=indexer,
    )

    result = await service.handle_query(
        "explain sample.py",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-1",
    )

    assert result.answer == "FINAL ANSWER"
    assert result.metadata["degraded"] is True
    # Code analyst should be invoked in degraded raw-file mode.
    assert code_analyst.snippets_requested


@pytest.mark.asyncio
async def test_graph_only_lookup_skips_code_analyst_execution() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="lookup",
                entities=["FastAPI"],
                target_agents=["graph_query"],
                reasoning="lookup",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)

    memory = _MemoryClient(ctx=_session_context(), cached_response=None)
    graph_query = _GraphQueryClient(index_version="idx-1")
    code_analyst = _CodeAnalystClient()
    indexer = _IndexerClient()
    clients = SimpleNamespace(
        memory=memory,
        graph_query=graph_query,
        code_analyst=code_analyst,
        indexer=indexer,
    )

    result = await service.handle_query(
        "What is the FastAPI class?",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-lookup",
    )

    assert result.answer == "FINAL ANSWER"
    assert code_analyst.snippets_requested == []


@pytest.mark.asyncio
async def test_cache_hit_skips_llm_calls() -> None:
    settings = OrchestratorSettings.from_env()

    # If the orchestrator makes any LLM calls, StubProvider will raise.
    llm = StubProvider([])
    service = OrchestratorService(llm, settings=settings)

    # cache_key = orchestrator:v1:{index_version}:{normalized_query}
    normalized = "compare fastapi"
    expected_cache_key = f"orchestrator:v1:idx-2:{normalized}"

    cached = SimpleNamespace(response_json={"answer": "CACHED ANSWER"})
    memory = _MemoryClient(
        ctx=_session_context(),
        cached_response=cached,
        expected_cache_key=expected_cache_key,
    )
    graph_query = _GraphQueryClient(index_version="idx-2")
    code_analyst = _CodeAnalystClient()
    indexer = _IndexerClient()

    clients = SimpleNamespace(
        memory=memory,
        graph_query=graph_query,
        code_analyst=code_analyst,
        indexer=indexer,
    )

    result = await service.handle_query(
        "  Compare   FastAPI ",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-1",
    )

    assert result.answer == "CACHED ANSWER"
    assert result.metadata["cached"] is True
    assert llm.calls == []
    assert result.metadata["tokens"]["total"] == 0
    assert result.metadata["tokens"]["llm_calls"] == 0


@pytest.mark.asyncio
async def test_rules_first_simple_lookup_skips_routing_llm_call() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)

    memory = _MemoryClient(ctx=_session_context(), cached_response=None)
    graph_query = _GraphQueryClient(index_version="idx-1")
    code_analyst = _CodeAnalystClient()
    indexer = _IndexerClient()
    clients = SimpleNamespace(
        memory=memory,
        graph_query=graph_query,
        code_analyst=code_analyst,
        indexer=indexer,
    )

    result = await service.handle_query(
        "What classes inherit from APIRouter?",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-rules",
    )

    assert result.metadata["routing_mode"] == "rules"
    assert [call.purpose for call in llm.calls] == ["synthesis"]
    assert result.metadata["tokens"]["llm_calls"] == 1
    assert "routing" not in result.metadata["tokens"]["by_purpose"]


@pytest.mark.asyncio
async def test_rules_first_ambiguous_query_uses_one_routing_llm_call() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=["fastapi.dependencies.utils"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="llm routing",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    clients = SimpleNamespace(
        memory=_MemoryClient(ctx=_session_context(), cached_response=None),
        graph_query=_GraphQueryClient(index_version="idx-1"),
        code_analyst=_CodeAnalystClient(),
        indexer=_IndexerClient(),
    )

    result = await service.handle_query(
        "How does dependency injection work and show me examples from the codebase",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-ambiguous",
    )

    assert result.metadata["routing_mode"] == "llm"
    assert [call.purpose for call in llm.calls] == ["routing", "synthesis"]
    assert result.metadata["tokens"]["llm_calls"] == 2
    assert result.metadata["tokens"]["by_purpose"]["routing"]["total"] == 120


@pytest.mark.asyncio
async def test_conflicting_keywords_route_via_llm() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="mixed",
                entities=["FastAPI"],
                target_agents=["indexer", "graph_query", "code_analyst"],
                reasoning="llm routing",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    clients = SimpleNamespace(
        memory=_MemoryClient(ctx=_session_context(), cached_response=None),
        graph_query=_GraphQueryClient(index_version="idx-1"),
        code_analyst=_CodeAnalystClient(),
        indexer=_IndexerClient(),
    )

    result = await service.handle_query(
        "Reindex the repository and explain dependency injection examples in the codebase",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-conflict",
    )

    assert result.metadata["routing_mode"] == "llm"
    assert llm.calls[0].purpose == "routing"


@pytest.mark.asyncio
async def test_llm_first_strategy_restores_old_behavior() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="relationship",
                entities=["APIRouter"],
                target_agents=["graph_query"],
                reasoning="llm routing",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    clients = SimpleNamespace(
        memory=_MemoryClient(ctx=_session_context(), cached_response=None),
        graph_query=_GraphQueryClient(index_version="idx-1"),
        code_analyst=_CodeAnalystClient(),
        indexer=_IndexerClient(),
    )

    result = await service.handle_query(
        "What classes inherit from APIRouter?",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-llm-first",
    )

    assert result.metadata["routing_mode"] == "llm"
    assert llm.calls[0].purpose == "routing"


def test_rule_based_route_extracts_entities_for_dependency_injection_queries() -> None:
    route = rule_based_route(
        "How does dependency injection work and show me examples from the codebase"
    )

    assert route.ambiguous is True
    assert route.target_agents == []
    assert set(route.matched_rules) == {"explain", "examples"}


def test_rule_based_route_uses_graph_only_for_simple_lookup() -> None:
    route = rule_based_route("What is the FastAPI class?")

    assert route.ambiguous is False
    assert route.target_agents == ["graph_query"]
    assert route.matched_rules == ["lookup"]


def test_rule_based_route_uses_graph_only_for_relationship_lookup() -> None:
    route = rule_based_route("What classes inherit from APIRouter?")

    assert route.ambiguous is False
    assert route.target_agents == ["graph_query"]
    assert route.matched_rules == ["relationship", "lookup"]


def test_rule_based_route_uses_three_agents_for_index_plus_analysis() -> None:
    route = rule_based_route(
        "Reindex the repository and explain dependency injection examples in the codebase"
    )

    assert route.ambiguous is True
    assert set(route.matched_rules) == {"index", "explain", "examples"}


@pytest.mark.asyncio
async def test_synthesis_omits_empty_context_block() -> None:
    llm = StubProvider(["FINAL ANSWER"])

    answer = await synthesize_response(
        "Explain FastAPI",
        {"graph_query": {"output": {"entities": [{"file_path": "fastapi/applications.py"}]}}},
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert answer == "FINAL ANSWER"
    prompt = "\n".join(message.content for message in llm.calls[0].messages)
    assert "Conversation context (summary + recent turns):" not in prompt


@pytest.mark.asyncio
async def test_synthesis_returns_absent_entity_answer_without_llm_call() -> None:
    llm = StubProvider([])

    answer = await synthesize_response(
        "How does Django's ORM lazy-load querysets?",
        {
            "graph_query": {
                "output": {"entities": [], "queried_entities": ["Django", "ORM"]},
                "ok": True,
            },
            "code_analyst": {
                "output": {
                    "snippets": [
                        {"file_path": "Django", "error": "[Errno 2] No such file or directory"}
                    ]
                },
                "ok": True,
            },
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert "indexed FastAPI codebase" in answer
    assert llm.calls == []

