from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.exceptions import RoutingError
from core.llm.stub import StubProvider
from core.memory import ConversationContext, ConversationTurn
from core.orchestration.executor import _select_analysis_candidates, run_plan
from core.orchestration.models import ExecutionPlan, QueryIntent
from core.orchestration.router import (
    fallback_target_agents,
    normalize_grounded_intent,
    retrieval_search_terms,
    route_to_agents,
    rule_based_route,
)
from core.orchestration.service import OrchestratorService
from core.orchestration.synthesis import synthesize_response
from core.settings import OrchestratorSettings


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
        find_entity_by_name: dict[str, object] | None = None,
    ) -> None:
        self._index_version = index_version
        self._timeout = timeout_on_find_entity
        self._payload = find_entity_payload or {
            "file_path": "sample.py",
            "line_start": 10,
            "line_end": 20,
            "qualified_name": "sample.fn",
            "name": "fn",
        }
        self._by_name = dict(find_entity_by_name or {})
        self.find_entity_calls: list[str] = []
        self.relationship_calls: list[tuple[str, str]] = []
        self.find_related_calls: list[tuple[str, str]] = []

    async def get_statistics(self) -> object:
        return SimpleNamespace(index_version=self._index_version)

    async def find_entity(self, name: str, entity_type: str | None = None) -> object:
        _ = entity_type
        self.find_entity_calls.append(name)
        if self._timeout:
            raise TimeoutError("graph_query timeout")
        if name in self._by_name:
            return self._by_name[name]
        return self._payload

    async def get_dependencies(self, name: str) -> object:
        self.relationship_calls.append(("get_dependencies", name))
        return {"name": name, "neighbors": []}

    async def get_dependents(self, name: str) -> object:
        self.relationship_calls.append(("get_dependents", name))
        return {"name": name, "neighbors": []}

    async def find_related(self, name: str, relationship_type: str) -> object:
        self.relationship_calls.append(("find_related", name))
        self.find_related_calls.append((name, relationship_type))
        return {"name": name, "relationship_type": relationship_type, "neighbors": []}

    async def trace_imports(self, module: str, depth: int = 5) -> object:
        _ = depth
        self.relationship_calls.append(("trace_imports", module))
        return {"module": module, "paths": []}


class _CodeAnalystClient:
    def __init__(self) -> None:
        self.snippets_requested: list[dict[str, object]] = []
        self.explain_calls: list[str] = []
        self.analyze_calls: list[str] = []
        self.analyze_class_calls: list[str] = []
        self.compare_calls: list[tuple[str, str]] = []
        self.pattern_calls: list[str] = []

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
            {
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
                "qualified_name": qualified_name,
            }
        )
        return {
            "file_path": file_path or "",
            "line_start": line_start,
            "line_end": line_end,
            "text": f"snippet for {file_path}",
            "error": None,
        }

    async def explain_implementation(self, qualified_name: str) -> object:
        self.explain_calls.append(qualified_name)
        return {"qualified_name": qualified_name, "explanation": "explained", "error": None}

    async def analyze_function(self, qualified_name: str) -> object:
        self.analyze_calls.append(qualified_name)
        return {"qualified_name": qualified_name, "summary": "analyzed", "error": None}

    async def analyze_class(self, qualified_name: str) -> object:
        self.analyze_class_calls.append(qualified_name)
        return {"qualified_name": qualified_name, "summary": "analyzed class", "error": None}

    async def compare_implementations(self, name_a: str, name_b: str) -> object:
        self.compare_calls.append((name_a, name_b))
        return {"name_a": name_a, "name_b": name_b, "summary": "compared", "error": None}

    async def find_patterns(self, pattern: str) -> object:
        self.pattern_calls.append(pattern)
        return {"pattern": pattern, "instances": [], "error": None}


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


async def _synthesize(*args: object, **kwargs: object) -> str:
    result = await synthesize_response(*args, **kwargs)  # type: ignore[arg-type]
    return result.answer


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
async def test_routing_llm_timeout_falls_back_to_rules() -> None:
    class _TimeoutProvider(StubProvider):
        async def complete(self, messages, response_model=None, **kwargs):  # type: ignore[no-untyped-def,override]
            if kwargs.get("purpose") == "routing":
                raise TimeoutError("Request timed out.")
            return await StubProvider.complete(self, messages, response_model, **kwargs)

    llm = _TimeoutProvider(["FINAL ANSWER"])
    service = OrchestratorService(
        llm, settings=OrchestratorSettings(routing_strategy="llm_first")
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "session-1",
        clients=_clients(),  # type: ignore[arg-type]
        correlation_id="corr-routing-timeout",
    )
    assert result.metadata["routing_mode"] == "rules_fallback"
    assert result.answer == "FINAL ANSWER"


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

    assert "FINAL ANSWER" in result.answer
    assert "incomplete" in result.answer.lower()
    assert result.metadata["degraded"] is True
    assert code_analyst.explain_calls
    assert code_analyst.analyze_calls


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

    # cache_key = orchestrator:v1:{index_version}:{session_id}:{normalized_query}
    normalized = "compare fastapi"
    expected_cache_key = f"orchestrator:v1:idx-2:session-1:{normalized}"

    cached = SimpleNamespace(
        response_json={
            "answer": "CACHED ANSWER",
            "metadata": {"tools_invoked": ["graph_query.get_dependents"]},
        }
    )
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
    assert result.metadata["tools_invoked"] == ["graph_query.get_dependents"]
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


def test_rule_based_route_explain_depends_does_not_treat_class_as_relationship() -> None:
    route = rule_based_route("Explain how Depends resolves a dependency")

    assert route.ambiguous is False
    assert route.target_agents == ["graph_query", "code_analyst"]
    assert route.matched_rules == ["explain"]


def test_rule_based_route_who_depends_still_matches_relationship() -> None:
    route = rule_based_route("who depends on FastAPI")

    assert route.ambiguous is False
    assert route.target_agents == ["graph_query"]
    assert "relationship" in route.matched_rules


def test_related_relationship_type_maps_inheritance_wording() -> None:
    from core.orchestration.executor import related_relationship_type

    assert related_relationship_type("What classes inherit from APIRouter?") == "INHERITS_FROM"
    assert related_relationship_type("who calls FastAPI") == "CALLS"
    assert related_relationship_type("What is the FastAPI class?") is None


def test_rule_based_route_uses_three_agents_for_index_plus_analysis() -> None:
    route = rule_based_route(
        "Reindex the repository and explain dependency injection examples in the codebase"
    )

    assert route.ambiguous is True
    assert set(route.matched_rules) == {"index", "explain", "examples"}
    assert fallback_target_agents(
        "Reindex the repository and explain dependency injection examples in the codebase"
    ) == ["indexer", "graph_query", "code_analyst"]


def test_reindex_the_repository_routes_to_indexer() -> None:
    route = rule_based_route("reindex the repository")

    assert route.ambiguous is False
    assert route.target_agents == ["indexer"]
    assert route.matched_rules == ["index"]


@pytest.mark.asyncio
async def test_synthesis_omits_empty_context_block() -> None:
    llm = StubProvider(["FINAL ANSWER"])

    answer = await _synthesize(
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

    answer = await _synthesize(
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


@pytest.mark.asyncio
async def test_synthesis_uses_cascade_hits_instead_of_missing_entity_answer() -> None:
    llm = StubProvider(["FINAL ANSWER"])

    answer = await _synthesize(
        "How does dependency injection work and show me examples from the codebase",
        {
            "graph_query": {
                "output": {
                    "entities": [{"matches": [], "result_count": 0}],
                    "queried_entities": ["dependency injection"],
                    "candidates": [
                        {
                            "name": "get_dependant",
                            "qualified_name": "fastapi.dependencies.utils.get_dependant",
                            "file_path": "fastapi/dependencies/utils.py",
                            "tier": "fulltext",
                            "score": 3.1,
                        }
                    ],
                },
                "ok": True,
            }
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert "FINAL ANSWER" in answer
    assert llm.calls
    assert "fastapi/dependencies/utils.py" in answer


@pytest.mark.asyncio
async def test_synthesis_treats_empty_matches_as_missing() -> None:
    llm = StubProvider([])

    answer = await _synthesize(
        "How does Django's ORM lazy-load querysets?",
        {
            "graph_query": {
                "output": {
                    "entities": [{"matches": [], "result_count": 0, "truncated": False}],
                    "queried_entities": ["Django", "ORM"],
                    "candidates": [],
                },
                "ok": True,
            }
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert "indexed FastAPI codebase" in answer
    assert llm.calls == []


@pytest.mark.asyncio
async def test_postgresql_wal_incidental_fulltext_is_refused() -> None:
    llm = StubProvider(["should not answer from FastAPI hits"])

    answer = await _synthesize(
        "How does PostgreSQL write-ahead logging work?",
        {
            "graph_query": {
                "output": {
                    "entities": [{"matches": [], "result_count": 0}],
                    "queried_entities": ["PostgreSQL"],
                    "candidates": [
                        {
                            "name": "logging",
                            "qualified_name": "fastapi.logger.logging",
                            "file_path": "fastapi/logger.py",
                            "tier": "fulltext",
                            "score": 1.4,
                        }
                    ],
                },
                "ok": True,
            }
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert "not in the indexed FastAPI codebase" in answer
    assert llm.calls == []


def _clients(
    *,
    memory: _MemoryClient | None = None,
    graph_query: _GraphQueryClient | None = None,
    code_analyst: _CodeAnalystClient | None = None,
    indexer: _IndexerClient | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        memory=memory or _MemoryClient(ctx=_session_context(), cached_response=None),
        graph_query=graph_query or _GraphQueryClient(index_version="idx-1"),
        code_analyst=code_analyst or _CodeAnalystClient(),
        indexer=indexer or _IndexerClient(),
    )


@pytest.mark.asyncio
async def test_relationship_query_invokes_graph_neighbor_tools() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "file_path": "fastapi/routing.py",
            "line_start": 2255,
            "line_end": 6447,
            "qualified_name": "fastapi.routing.APIRouter",
            "name": "APIRouter",
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "What classes inherit from APIRouter?",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-rel",
    )

    assert result.metadata["routing_mode"] == "rules"
    assert graph_query.find_entity_calls
    assert {name for name, _target in graph_query.relationship_calls} == {
        "get_dependencies",
        "get_dependents",
        "trace_imports",
        "find_related",
    }
    assert graph_query.find_related_calls == [
        ("fastapi.routing.APIRouter", "INHERITS_FROM")
    ]
    assert code_analyst.snippets_requested == []
    assert "graph_query.find_entity" in result.metadata["tools_invoked"]
    assert "graph_query.get_dependents" in result.metadata["tools_invoked"]
    assert "graph_query.find_related" in result.metadata["tools_invoked"]
    assert "graph_query.trace_imports" in result.metadata["tools_invoked"]


@pytest.mark.asyncio
async def test_explanation_query_invokes_explain_and_analyze() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "explain how FastAPI works",
        "session-1",
        clients=_clients(code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-explain",
    )

    assert result.metadata["routing_mode"] == "rules"
    assert code_analyst.explain_calls
    assert code_analyst.analyze_calls
    assert "code_analyst.explain_implementation" in result.metadata["tools_invoked"]
    assert "code_analyst.analyze_function" in result.metadata["tools_invoked"]


@pytest.mark.asyncio
async def test_explain_depends_invokes_explain_implementation() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "file_path": "fastapi/dependencies.py",
            "line_start": 1,
            "line_end": 80,
            "qualified_name": "fastapi.dependencies.models.Depends",
            "name": "Depends",
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "Explain how Depends resolves a dependency",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-depends",
    )

    assert result.metadata["routing_mode"] == "rules"
    assert "Depends" in graph_query.find_entity_calls
    assert code_analyst.explain_calls
    assert "code_analyst.explain_implementation" in result.metadata["tools_invoked"]


@pytest.mark.asyncio
async def test_comparison_query_invokes_compare_implementations() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="comparison",
                entities=["FastAPI", "APIRouter"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="compare",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    code_analyst = _CodeAnalystClient()
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_by_name={
            "FastAPI": {
                "name": "FastAPI",
                "qualified_name": "fastapi.applications.FastAPI",
                "entity_type": "Class",
                "file_path": "fastapi/applications.py",
                "line_start": 42,
                "line_end": 4774,
            },
            "APIRouter": {
                "name": "APIRouter",
                "qualified_name": "fastapi.routing.APIRouter",
                "entity_type": "Class",
                "file_path": "fastapi/routing.py",
                "line_start": 2255,
                "line_end": 6447,
            },
        },
    )
    result = await service.handle_query(
        "compare FastAPI and APIRouter",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-compare",
    )

    assert code_analyst.compare_calls == [
        ("fastapi.applications.FastAPI", "fastapi.routing.APIRouter")
    ]
    assert "code_analyst.compare_implementations" in result.metadata["tools_invoked"]
    assert graph_query.find_entity_calls == ["FastAPI", "APIRouter"]


@pytest.mark.asyncio
async def test_pattern_query_invokes_find_patterns() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "find decorator patterns",
        "session-1",
        clients=_clients(code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-pattern",
    )

    assert result.metadata["routing_mode"] == "rules"
    assert code_analyst.pattern_calls == ["decorator"]
    assert result.metadata["tools_invoked"] == ["code_analyst.find_patterns"]


@pytest.mark.asyncio
async def test_lookup_with_analyst_invokes_find_entity_and_snippet() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="lookup",
                entities=["FastAPI"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="lookup",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "file_path": "fastapi/applications.py",
            "line_start": 42,
            "line_end": 4774,
            "qualified_name": "fastapi.applications.FastAPI",
            "name": "FastAPI",
            "entity_type": "Class",
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "What is the FastAPI class?",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-lookup-both",
    )

    assert graph_query.find_entity_calls == ["FastAPI"]
    assert code_analyst.snippets_requested
    for snippet in code_analyst.snippets_requested:
        assert snippet["file_path"] != "FastAPI"
        assert snippet["file_path"] == "fastapi/applications.py"
        assert snippet["qualified_name"] == "fastapi.applications.FastAPI"
    assert code_analyst.analyze_class_calls == ["fastapi.applications.FastAPI"]
    assert code_analyst.analyze_calls == []
    assert "graph_query.find_entity" in result.metadata["tools_invoked"]
    assert "code_analyst.get_code_snippet" in result.metadata["tools_invoked"]
    assert "code_analyst.analyze_class" in result.metadata["tools_invoked"]


@pytest.mark.asyncio
async def test_explanation_of_class_invokes_analyze_class() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=["FastAPI"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="explain class",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "file_path": "fastapi/applications.py",
            "line_start": 42,
            "line_end": 4774,
            "qualified_name": "fastapi.applications.FastAPI",
            "name": "FastAPI",
            "entity_type": "Class",
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "explain how FastAPI works",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-explain-class",
    )

    assert code_analyst.explain_calls == ["fastapi.applications.FastAPI"]
    assert code_analyst.analyze_class_calls == ["fastapi.applications.FastAPI"]
    assert code_analyst.analyze_calls == []
    assert "code_analyst.analyze_class" in result.metadata["tools_invoked"]
    assert "code_analyst.analyze_function" not in result.metadata["tools_invoked"]


def test_code_analyst_can_start_now_only_for_patterns() -> None:
    from core.orchestration.executor import code_analyst_can_start_now

    assert code_analyst_can_start_now(
        QueryIntent(
            routing_mode="rules",
            intent="pattern",
            entities=[],
            target_agents=["code_analyst"],
            reasoning="patterns",
        )
    )
    assert not code_analyst_can_start_now(
        QueryIntent(
            routing_mode="llm",
            intent="comparison",
            entities=["FastAPI", "APIRouter"],
            target_agents=["graph_query", "code_analyst"],
            reasoning="compare",
        )
    )
    assert not code_analyst_can_start_now(
        QueryIntent(
            routing_mode="llm",
            intent="lookup",
            entities=["FastAPI"],
            target_agents=["graph_query", "code_analyst"],
            reasoning="lookup",
        )
    )


@pytest.mark.asyncio
async def test_follow_up_query_reuses_prior_turn_entities() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    ctx = ConversationContext(
        recent_turns=[
            ConversationTurn(
                id=1,
                role="user",
                content="explain how FastAPI works",
                created_at="2026-01-01T00:00:00Z",
                token_estimate=10,
            )
        ]
    )
    graph_query = _GraphQueryClient(index_version="idx-1")
    result = await service.handle_query(
        "what about its parameters?",
        "session-1",
        clients=_clients(
            memory=_MemoryClient(ctx=ctx, cached_response=None),
            graph_query=graph_query,
        ),  # type: ignore[arg-type]
        correlation_id="corr-followup",
    )

    assert "FastAPI" in graph_query.find_entity_calls
    assert result.answer == "FINAL ANSWER"


@pytest.mark.asyncio
async def test_follow_up_uses_folded_summary_entities() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    ctx = ConversationContext(
        summary="User asked about the FastAPI class in fastapi/applications.py.",
        recent_turns=[],
    )
    graph_query = _GraphQueryClient(index_version="idx-1")
    result = await service.handle_query(
        "what about its parameters?",
        "session-1",
        clients=_clients(
            memory=_MemoryClient(ctx=ctx, cached_response=None),
            graph_query=graph_query,
        ),  # type: ignore[arg-type]
        correlation_id="corr-folded-summary",
    )

    assert "FastAPI" in graph_query.find_entity_calls
    assert result.answer == "FINAL ANSWER"


def test_entities_from_context_reads_summary_and_user_turns() -> None:
    from core.orchestration.router import entities_from_context

    ctx = ConversationContext(
        summary="Earlier we discussed APIRouter in fastapi/routing.py.",
        recent_turns=[
            ConversationTurn(
                id=1,
                role="assistant",
                content="APIRouter is a class.",
                created_at="2026-01-01T00:00:00Z",
                token_estimate=8,
            ),
            ConversationTurn(
                id=2,
                role="user",
                content="and FastAPI?",
                created_at="2026-01-01T00:00:01Z",
                token_estimate=4,
            ),
        ],
    )
    entities = entities_from_context(ctx)
    assert "FastAPI" in entities
    assert "APIRouter" in entities


def test_route_to_agents_emits_a_flat_agent_set() -> None:
    plan = route_to_agents(
        QueryIntent(
            intent="explanation",
            entities=["FastAPI"],
            target_agents=["graph_query", "code_analyst"],
        )
    )
    assert plan.agents == ["graph_query", "code_analyst"]
    assert "phases" not in type(plan).model_fields


@pytest.mark.asyncio
async def test_await_with_timeout_retry_retries_once() -> None:
    from core.resilience.retry import await_with_timeout_retry

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("once")
        return "ok"

    result = await await_with_timeout_retry(flaky, timeout_s=1.0, retry_count=1)
    assert result == "ok"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_empty_entities_still_retrieve_from_the_query_text() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=[],
                target_agents=["graph_query", "code_analyst"],
                reasoning="conceptual",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 10,
                    "line_end": 80,
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "name": "get_dependant",
                    "tier": "fulltext",
                    "score": 2.5,
                }
            ]
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "How does dependency injection work and show me examples from the codebase",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-conceptual",
    )

    assert graph_query.find_entity_calls
    assert any("dependency injection" in call.lower() for call in graph_query.find_entity_calls)
    assert "fastapi.dependencies.utils" not in graph_query.find_entity_calls
    assert code_analyst.explain_calls
    assert "get_dependant" in code_analyst.explain_calls[0]
    assert "FINAL ANSWER" in result.answer
    assert "fastapi/dependencies/utils.py" in result.answer


def test_normalize_does_not_expand_simple_lookup() -> None:
    intent = QueryIntent(
        routing_mode="llm",
        intent="lookup",
        entities=["FastAPI"],
        target_agents=["graph_query"],
        reasoning="lookup",
    )
    assert normalize_grounded_intent(intent, "What is the FastAPI class?") is intent


def test_normalize_adds_graph_query_when_llm_omits_it() -> None:
    intent = QueryIntent(
        routing_mode="llm",
        intent="explanation",
        entities=[],
        target_agents=["code_analyst"],
        reasoning="explain DI from training data",
    )
    out = normalize_grounded_intent(
        intent,
        "How does dependency injection work and show me examples from the codebase",
    )
    assert out.target_agents == ["graph_query", "code_analyst"]
    assert out.intent == "explanation"


def test_normalize_promotes_grounded_relationship_to_mixed() -> None:
    intent = QueryIntent(
        routing_mode="llm",
        intent="relationship",
        entities=["APIRouter", "get_openapi"],
        target_agents=["graph_query"],
        reasoning="connect across",
    )
    out = normalize_grounded_intent(
        intent,
        "Explain how dependency injection, APIRouter, and get_openapi connect across the codebase",
    )
    assert out.target_agents == ["graph_query", "code_analyst"]
    assert out.intent == "mixed"


def test_retrieval_search_terms_uses_query_text_for_conceptual_grounded_query() -> None:
    terms = retrieval_search_terms(
        "How does dependency injection work and show me examples from the codebase",
        [],
    )
    assert terms[-1].lower().startswith("how does dependency injection")
    assert "fastapi.dependencies.utils" not in terms


def test_retrieval_search_terms_keeps_simple_lookup_entity() -> None:
    terms = retrieval_search_terms("What is the FastAPI class?", ["FastAPI"])
    assert terms == ["FastAPI"]


def test_retrieval_search_terms_includes_full_validation_query() -> None:
    terms = retrieval_search_terms("How does FastAPI handle request validation?", ["FastAPI"])
    assert "FastAPI" in terms
    assert terms[-1] == "How does FastAPI handle request validation?"


@pytest.mark.asyncio
async def test_llm_omitting_graph_query_still_retrieves_codebase_examples() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=[],
                target_agents=["code_analyst"],
                reasoning="explain conceptually",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 10,
                    "line_end": 80,
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "name": "get_dependant",
                    "tier": "fulltext",
                    "score": 3.1,
                }
            ]
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "How does dependency injection work and show me examples from the codebase",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-llm-omit-graph",
    )

    assert graph_query.find_entity_calls
    assert "graph_query.find_entity" in result.metadata["tools_invoked"]
    assert "code_analyst.explain_implementation" in result.metadata["tools_invoked"]
    assert "get_dependant" in code_analyst.explain_calls[0]
    assert any(
        item["file_path"] == "fastapi/dependencies/utils.py"
        for item in code_analyst.snippets_requested
    )
    assert "fastapi/dependencies/utils.py" in result.answer


@pytest.mark.asyncio
async def test_ranked_fulltext_candidates_become_analyst_targets() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=["dependency injection"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="conceptual",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 370,
                    "line_end": 430,
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "name": "get_dependant",
                    "tier": "fulltext",
                    "score": 4.2,
                },
                {
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 540,
                    "line_end": 620,
                    "qualified_name": "fastapi.dependencies.utils.solve_dependencies",
                    "name": "solve_dependencies",
                    "tier": "fulltext",
                    "score": 3.8,
                },
                {
                    "file_path": "fastapi/dependencies/models.py",
                    "line_start": 1,
                    "line_end": 40,
                    "qualified_name": "fastapi.dependencies.models.Dependant",
                    "name": "Dependant",
                    "tier": "fulltext",
                    "score": 2.1,
                },
                {
                    "file_path": "docs/en/docs/tutorial/dependencies.md",
                    "line_start": 1,
                    "line_end": 20,
                    "qualified_name": "docs.tutorial.dependencies",
                    "name": "dependencies",
                    "tier": "lexical",
                    "score": 0.4,
                },
            ]
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "How does dependency injection work and show me examples from the codebase",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-ranked-candidates",
    )

    assert code_analyst.explain_calls == [
        "fastapi.dependencies.utils.get_dependant",
        "fastapi.dependencies.utils.solve_dependencies",
        "fastapi.dependencies.models.Dependant",
    ]
    assert "docs.tutorial.dependencies" not in code_analyst.explain_calls
    assert "graph_query.find_entity" in result.metadata["tools_invoked"]
    assert "fastapi/dependencies/utils.py" in result.answer


@pytest.mark.asyncio
async def test_exact_hits_are_preferred_over_fulltext_for_analysis() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=["Depends"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="explain",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "file_path": "fastapi/param_functions.py",
                    "line_start": 1,
                    "line_end": 20,
                    "qualified_name": "fastapi.param_functions.Depends",
                    "name": "Depends",
                    "tier": "exact",
                    "score": 1.0,
                },
                {
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 10,
                    "line_end": 80,
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "name": "get_dependant",
                    "tier": "fulltext",
                    "score": 9.9,
                },
            ]
        },
    )
    code_analyst = _CodeAnalystClient()
    await service.handle_query(
        "Explain how Depends is implemented in the codebase",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-exact-pref",
    )

    assert code_analyst.explain_calls == ["fastapi.param_functions.Depends"]


@pytest.mark.asyncio
async def test_synthesis_keeps_code_analyst_explanation_without_graph_hits() -> None:
    llm = StubProvider(["Depends resolves callables in fastapi/dependencies/utils.py"])

    answer = await _synthesize(
        "How does dependency injection work and show me examples from the codebase",
        {
            "graph_query": {
                "output": {
                    "entities": [{"matches": [], "result_count": 0}],
                    "queried_entities": ["dependency injection"],
                    "candidates": [],
                },
                "ok": True,
            },
            "code_analyst": {
                "output": {
                    "explanations": [
                        {
                            "qualified_name": "fastapi.dependencies.utils.get_dependant",
                            "explanation": "Builds a Dependant graph from a callable.",
                            "file_path": "fastapi/dependencies/utils.py",
                        }
                    ]
                },
                "ok": True,
            },
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert "won't invent" not in answer
    assert "indexed FastAPI codebase" not in answer
    assert llm.calls
    assert "fastapi/dependencies/utils.py" in answer or "Dependant" in answer


@pytest.mark.asyncio
async def test_synthesis_incomplete_retrieval_uses_partial_evidence() -> None:
    llm = StubProvider(["Partial answer from the snippet we did retrieve."])

    answer = await _synthesize(
        "How does FastAPI handle request validation?",
        {
            "graph_query": {
                "ok": False,
                "error": "graph_query timeout",
                "degraded_note": "graph_query unavailable/timeout; using raw-file fallback",
            },
            "code_analyst": {
                "output": {
                    "explanations": [
                        {
                            "qualified_name": "fastapi.exceptions.RequestValidationError",
                            "explanation": "Raised when request validation fails.",
                            "file_path": "fastapi/exceptions.py",
                        }
                    ]
                },
                "ok": True,
            },
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert "incomplete" in answer.lower()
    assert "indexed FastAPI codebase" not in answer
    assert llm.calls


@pytest.mark.asyncio
async def test_synthesis_grounded_query_without_evidence_does_not_invent_examples() -> None:
    llm = StubProvider(["EmailService and dependency-injector tutorial"])

    answer = await _synthesize(
        "How does dependency injection work and show me examples from the codebase",
        {
            "graph_query": {
                "output": {"entities": [], "queried_entities": [], "candidates": []},
                "ok": True,
            },
            "code_analyst": {
                "output": {
                    "snippets": [
                        {"file_path": "dependency injection", "error": "No such file"}
                    ]
                },
                "ok": True,
            },
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )

    assert "won't invent" in answer
    assert "EmailService" not in answer
    assert llm.calls == []


def test_route_to_agents_raises_routing_error_when_empty() -> None:
    with pytest.raises(RoutingError, match="no target agents"):
        route_to_agents(
            QueryIntent(intent="lookup", entities=["FastAPI"], target_agents=[], reasoning="")
        )


@pytest.mark.asyncio
async def test_synthesize_response_falls_back_on_llm_failure() -> None:
    class _Boom:
        async def complete(self, *args: object, **kwargs: object) -> object:
            raise TimeoutError("synthesis timeout")

    answer = await _synthesize(
        "What is the FastAPI class?",
        {
            "graph_query": {
                "ok": True,
                "output": {
                    "entities": [
                        {
                            "qualified_name": "fastapi.applications.FastAPI",
                            "file_path": "fastapi/applications.py",
                            "name": "FastAPI",
                        }
                    ],
                    "queried_entities": ["FastAPI"],
                    "candidates": [
                        {
                            "qualified_name": "fastapi.applications.FastAPI",
                            "file_path": "fastapi/applications.py",
                            "name": "FastAPI",
                        }
                    ],
                },
            }
        },
        ConversationContext(),
        llm_provider=_Boom(),  # type: ignore[arg-type]
        settings=OrchestratorSettings.from_env(),
        correlation_id="corr-synth",
    )

    assert "synthesis unavailable, showing retrieved evidence" in answer
    assert "fastapi/applications.py" in answer
    assert "fastapi.applications.FastAPI" in answer


def test_select_analysis_candidates_ranks_tests_below_package_source() -> None:
    selected = _select_analysis_candidates(
        [
            {
                "name": "Depends",
                "qualified_name": "tests.test_depends.Depends",
                "file_path": "tests/test_depends.py",
                "tier": "exact",
                "score": 1.0,
            },
            {
                "name": "Depends",
                "qualified_name": "fastapi.param_functions.Depends",
                "file_path": "fastapi/param_functions.py",
                "tier": "exact",
                "score": 1.0,
            },
        ]
    )
    assert selected[0]["file_path"] == "fastapi/param_functions.py"


@pytest.mark.asyncio
async def test_run_plan_records_status_when_tool_plan_is_empty() -> None:
    intent = QueryIntent(
        intent="relationship",
        entities=["APIRouter"],
        target_agents=["graph_query", "code_analyst"],
        reasoning="connect",
    )
    plan = ExecutionPlan(intent=intent, agents=["graph_query", "code_analyst"])
    outputs = await run_plan(
        plan,
        query="how do FastAPI and APIRouter connect",
        context=None,
        clients=_clients(graph_query=_GraphQueryClient(index_version="idx-1")),  # type: ignore[arg-type]
        settings=OrchestratorSettings(),
        correlation_id="corr-empty-plan",
    )
    assert "code_analyst" in outputs
    assert outputs["code_analyst"].ok is False
    assert outputs["code_analyst"].degraded_note
    assert outputs["graph_query"].ok is True


@pytest.mark.asyncio
async def test_grounded_relationship_invokes_code_analyst() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="relationship",
                entities=["APIRouter", "get_openapi"],
                target_agents=["graph_query"],
                reasoning="connect across",
            ).model_dump(),
            "FINAL ANSWER",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "file_path": "fastapi/routing.py",
                    "line_start": 2255,
                    "line_end": 6447,
                    "qualified_name": "fastapi.routing.APIRouter",
                    "name": "APIRouter",
                    "tier": "exact",
                    "score": 1.0,
                }
            ]
        },
    )
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "Explain how dependency injection, APIRouter, and get_openapi connect across the codebase",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-c06",
    )
    assert code_analyst.explain_calls
    assert "code_analyst.explain_implementation" in result.metadata["tools_invoked"]


@pytest.mark.asyncio
async def test_postgresql_wal_fulltext_hit_is_refused() -> None:
    llm = StubProvider(["should not synthesize a codebase answer"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "name": "logging",
                    "qualified_name": "fastapi.logger.logging",
                    "file_path": "fastapi/logger.py",
                    "line_start": 1,
                    "line_end": 20,
                    "tier": "fulltext",
                    "score": 1.4,
                }
            ]
        },
    )
    result = await service.handle_query(
        "How does PostgreSQL write-ahead logging work?",
        "session-1",
        clients=_clients(graph_query=graph_query),  # type: ignore[arg-type]
        correlation_id="corr-t05",
    )
    lowered = result.answer.lower()
    assert "not in the indexed fastapi codebase" in lowered
    assert llm.calls == [] or all(
        getattr(call, "purpose", None) != "synthesis" for call in getattr(llm, "calls", [])
    )


@pytest.mark.asyncio
async def test_postgresql_wal_exact_logging_hit_is_still_refused() -> None:
    llm = StubProvider(["should not synthesize a codebase answer"])
    settings = OrchestratorSettings(routing_strategy="rules_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "name": "logging",
                    "qualified_name": "fastapi.logger.logging",
                    "file_path": "fastapi/logger.py",
                    "line_start": 1,
                    "line_end": 20,
                    "tier": "exact",
                    "score": 1.0,
                }
            ]
        },
    )
    result = await service.handle_query(
        "How does PostgreSQL write-ahead logging work?",
        "session-1",
        clients=_clients(graph_query=graph_query),  # type: ignore[arg-type]
        correlation_id="corr-t05-exact",
    )
    assert "not in the indexed FastAPI codebase" in result.answer
    assert not graph_query.relationship_calls
    assert llm.calls == [] or all(
        getattr(call, "purpose", None) != "synthesis" for call in getattr(llm, "calls", [])
    )


def test_trap_set_covers_eval_trap_queries() -> None:
    from core.orchestration.scope import is_out_of_scope

    traps = [
        "How does Django's ORM lazy-load querysets?",
        "Explain React's useEffect cleanup",
        "How does Rails ActiveRecord implement callbacks?",
        "Explain Kubernetes pod scheduling",
        "How does PostgreSQL write-ahead logging work?",
        "Explain TensorFlow GradientTape internals",
    ]
    for query in traps:
        assert is_out_of_scope(query), query
    assert not is_out_of_scope("How does FastAPI handle middleware?")


@pytest.mark.asyncio
async def test_code_analyst_overlaps_relationship_queries() -> None:
    analyst_started = asyncio.Event()

    class _WaitingGraph(_GraphQueryClient):
        async def get_dependents(self, name: str) -> object:
            await asyncio.wait_for(analyst_started.wait(), timeout=2)
            return await super().get_dependents(name)

        async def get_dependencies(self, name: str) -> object:
            await asyncio.wait_for(analyst_started.wait(), timeout=2)
            return await super().get_dependencies(name)

        async def trace_imports(self, module: str, depth: int = 5) -> object:
            await asyncio.wait_for(analyst_started.wait(), timeout=2)
            return await super().trace_imports(module, depth)

        async def find_related(self, name: str, relationship_type: str) -> object:
            await asyncio.wait_for(analyst_started.wait(), timeout=2)
            return await super().find_related(name, relationship_type)

    class _SignallingCode(_CodeAnalystClient):
        async def explain_implementation(self, qualified_name: str) -> object:
            analyst_started.set()
            return await super().explain_implementation(qualified_name)

        async def get_code_snippet(self, **kwargs: object) -> object:
            analyst_started.set()
            return await super().get_code_snippet(**kwargs)

        async def analyze_function(self, qualified_name: str) -> object:
            analyst_started.set()
            return await super().analyze_function(qualified_name)

        async def analyze_class(self, qualified_name: str) -> object:
            analyst_started.set()
            return await super().analyze_class(qualified_name)

    llm = StubProvider(["FINAL ANSWER"])
    service = OrchestratorService(
        llm, settings=OrchestratorSettings(routing_strategy="rules_first")
    )
    graph_query = _WaitingGraph(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "file_path": "fastapi/routing.py",
                    "line_start": 10,
                    "line_end": 40,
                    "qualified_name": "fastapi.routing.APIRouter",
                    "name": "APIRouter",
                    "tier": "exact",
                    "labels": ["Class"],
                }
            ]
        },
    )
    code_analyst = _SignallingCode()
    result = await service.handle_query(
        (
            "Trace how APIRouter imports connect to FastAPI, "
            "then show who depends on them in the codebase"
        ),
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-overlap",
    )
    assert code_analyst.explain_calls or code_analyst.snippets_requested
    assert "code_analyst" in " ".join(result.metadata["tools_invoked"])
    assert "skipped_agents" not in result.metadata or all(
        item["agent"] != "code_analyst" for item in result.metadata.get("skipped_agents", [])
    )


@pytest.mark.asyncio
async def test_one_slow_analyst_tool_keeps_sibling_results() -> None:
    class _PartialCode(_CodeAnalystClient):
        async def explain_implementation(self, qualified_name: str) -> object:
            await asyncio.sleep(2)
            return await super().explain_implementation(qualified_name)

    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=["FastAPI"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="explain class",
            ).model_dump(),
            "FINAL ANSWER FROM SNIPPET",
        ]
    )
    service = OrchestratorService(
        llm,
        settings=OrchestratorSettings(
            routing_strategy="llm_first",
            retry_count=1,
            code_analyst_timeout_s=0.2,
            graph_query_timeout_s=1.0,
        ),
    )
    graph_query = _GraphQueryClient(
        index_version="idx-1",
        find_entity_payload={
            "matches": [
                {
                    "file_path": "fastapi/applications.py",
                    "line_start": 10,
                    "line_end": 40,
                    "qualified_name": "fastapi.applications.FastAPI",
                    "name": "FastAPI",
                    "tier": "exact",
                    "labels": ["Class"],
                }
            ]
        },
    )
    code_analyst = _PartialCode()
    started = asyncio.get_event_loop().time()
    result = await service.handle_query(
        "explain how FastAPI works",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-partial-analyst",
    )
    elapsed = asyncio.get_event_loop().time() - started
    analyst = result.agent_outputs["code_analyst"]
    output = analyst.get("output") or {}
    assert elapsed < 1.0
    assert analyst.get("ok") is True
    assert not analyst.get("degraded_note")
    assert output.get("snippets")
    assert result.metadata.get("evidence_only") is not True
    assert "FINAL ANSWER FROM SNIPPET" in result.answer


@pytest.mark.asyncio
async def test_skipped_agents_recorded_when_budget_blocks_specialists() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    service = OrchestratorService(
        llm,
        settings=OrchestratorSettings(
            routing_strategy="rules_first",
            request_budgets_enabled=True,
            request_deadline_s=0.0,
            plan_deadline_s=0.0,
            synthesis_reserve_s=0.0,
            request_token_budget=0,
            request_cost_usd_max=0.0,
            synthesis_safety_margin_s=0.0,
        ),
    )
    result = await service.handle_query(
        "Explain how get_openapi is implemented in the codebase",
        "session-1",
        clients=_clients(),  # type: ignore[arg-type]
        correlation_id="corr-skip",
    )
    skipped = {item["agent"]: item["reason"] for item in result.metadata.get("skipped_agents", [])}
    assert "graph_query" in skipped or "code_analyst" in skipped


