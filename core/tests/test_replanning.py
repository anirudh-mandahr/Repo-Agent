"""Bounded evidence-driven refinement loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.gateway import ChatGatewayService, GatewayDependencies
from core.llm.stub import StubProvider
from core.memory import ConversationContext
from core.orchestration.models import QueryIntent
from core.orchestration.service import OrchestratorService
from core.settings import GatewaySettings, OrchestratorSettings


class _MemoryClient:
    def __init__(self) -> None:
        self._ctx = ConversationContext()

    async def get_context(self, session_id: str, token_budget: int = 3000) -> ConversationContext:
        _ = session_id, token_budget
        return self._ctx

    async def get_cached_response(self, cache_key: str) -> object | None:
        _ = cache_key
        return None

    async def cache_response(self, cache_key: str, response_json: object) -> None:
        _ = cache_key, response_json

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        _ = session_id, role, content


class _GraphQueryClient:
    def __init__(self, *, index_version: str | None = "idx-1") -> None:
        self._index_version = index_version
        self.find_entity_calls: list[str] = []

    async def get_statistics(self) -> object:
        return SimpleNamespace(index_version=self._index_version)

    async def find_entity(self, name: str, entity_type: str | None = None) -> object:
        _ = entity_type
        self.find_entity_calls.append(name)
        if name == "NoSuchThing":
            return {"matches": [], "result_count": 0}
        return {
            "file_path": "fastapi/applications.py",
            "line_start": 1,
            "line_end": 40,
            "qualified_name": "fastapi.applications.FastAPI",
            "name": "FastAPI",
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


class _CodeAnalystClient:
    def __init__(self) -> None:
        self.explain_calls: list[str] = []
        self.analyze_calls: list[str] = []
        self.compare_calls: list[tuple[str, str]] = []
        self.snippets_requested: list[dict[str, object]] = []

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> object:
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

    async def explain_implementation(self, qualified_name: str) -> object:
        self.explain_calls.append(qualified_name)
        return {"qualified_name": qualified_name, "explanation": "explained", "error": None}

    async def analyze_function(self, qualified_name: str) -> object:
        self.analyze_calls.append(qualified_name)
        return {"qualified_name": qualified_name, "summary": "analyzed", "error": None}

    async def analyze_class(self, qualified_name: str) -> object:
        return {"qualified_name": qualified_name, "summary": "analyzed class", "error": None}

    async def compare_implementations(self, name_a: str, name_b: str) -> object:
        self.compare_calls.append((name_a, name_b))
        return {"name_a": name_a, "name_b": name_b, "summary": "compared", "error": None}

    async def find_patterns(
        self, pattern: str, path_prefix: str | None = None
    ) -> object:
        return {"pattern": pattern, "instances": [], "error": None}


class _IndexerClient:
    async def index_repository(self, repo_url: str | None = None) -> object:
        return {"status": "ok", "repo_url": repo_url}


def _clients(
    *,
    graph_query: _GraphQueryClient | None = None,
    code_analyst: _CodeAnalystClient | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        memory=_MemoryClient(),
        graph_query=graph_query or _GraphQueryClient(),
        code_analyst=code_analyst or _CodeAnalystClient(),
        indexer=_IndexerClient(),
    )


@pytest.mark.asyncio
async def test_first_round_miss_triggers_successful_second_round() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="lookup",
                entities=["NoSuchThing"],
                target_agents=["graph_query"],
                reasoning="lookup",
            ).model_dump(),
            "FOUND IT",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first", max_plan_iterations=2)
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient()
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "where is nosuchthing",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-replan",
    )

    assert result.metadata["plan_iteration_count"] == 2
    assert result.metadata["plan_iterations"][0]["sufficient"] is False
    assert result.metadata["plan_iterations"][1]["sufficient"] is True
    assert "NoSuchThing" in graph_query.find_entity_calls
    assert any(call != "NoSuchThing" for call in graph_query.find_entity_calls)
    assert "FOUND IT" in result.answer


@pytest.mark.asyncio
async def test_iteration_cap_enforced() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="lookup",
                entities=["NoSuchThing"],
                target_agents=["graph_query"],
                reasoning="lookup",
            ).model_dump(),
            "CAPPED",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first", max_plan_iterations=1)
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient()
    code_analyst = _CodeAnalystClient()
    result = await service.handle_query(
        "where is nosuchthing",
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-cap",
    )

    assert result.metadata["plan_iteration_count"] == 1
    assert result.metadata["plan_iterations"][0]["sufficient"] is False
    assert graph_query.find_entity_calls == ["NoSuchThing"]
    assert code_analyst.explain_calls == []
    assert "not in the indexed" in result.answer.lower() or "CAPPED" in result.answer


@pytest.mark.asyncio
async def test_plan_deadline_skips_follow_up_round() -> None:
    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="lookup",
                entities=["NoSuchThing"],
                target_agents=["graph_query"],
                reasoning="lookup",
            ).model_dump(),
            "DEADLINE",
        ]
    )
    settings = OrchestratorSettings(
        routing_strategy="llm_first",
        max_plan_iterations=2,
        plan_deadline_s=0.0,
    )
    service = OrchestratorService(llm, settings=settings)
    graph_query = _GraphQueryClient()
    result = await service.handle_query(
        "where is nosuchthing",
        "session-1",
        clients=_clients(graph_query=graph_query),  # type: ignore[arg-type]
        correlation_id="corr-deadline",
    )

    assert result.metadata["plan_iteration_count"] == 1
    assert graph_query.find_entity_calls == ["NoSuchThing"]


@pytest.mark.asyncio
async def test_compare_waits_for_graph_and_uses_resolved_names() -> None:
    order: list[str] = []

    class _OrderedGraph(_GraphQueryClient):
        async def find_entity(self, name: str, entity_type: str | None = None) -> object:
            _ = entity_type
            self.find_entity_calls.append(name)
            order.append("graph_start")
            payload = {
                "file_path": f"{name}.py",
                "line_start": 1,
                "line_end": 10,
                "qualified_name": f"fastapi.{name}",
                "name": name,
                "entity_type": "Class",
            }
            order.append("graph_end")
            return payload

    class _CompareAnalyst(_CodeAnalystClient):
        async def compare_implementations(self, name_a: str, name_b: str) -> object:
            order.append("compare")
            return await super().compare_implementations(name_a, name_b)

    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="comparison",
                entities=["FastAPI", "APIRouter"],
                target_agents=["graph_query", "code_analyst"],
                reasoning="compare",
            ).model_dump(),
            "COMPARED",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first")
    service = OrchestratorService(llm, settings=settings)
    graph_query = _OrderedGraph()
    code_analyst = _CompareAnalyst()
    result = await asyncio.wait_for(
        service.handle_query(
            "compare FastAPI and APIRouter",
            "session-1",
            clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
            correlation_id="corr-resolved-compare",
        ),
        timeout=2.0,
    )

    assert "compare" in order
    assert order.index("graph_end") < order.index("compare")
    assert code_analyst.compare_calls == [("fastapi.FastAPI", "fastapi.APIRouter")]
    assert "COMPARED" in result.answer


@pytest.mark.asyncio
async def test_chat_stream_emits_routing_event_per_plan_iteration() -> None:
    class _Orch:
        async def handle_query(
            self, query: str, session_id: str, *, correlation_id: str
        ) -> dict[str, object]:
            _ = query, session_id
            return {
                "answer": "ok",
                "metadata": {
                    "routing_mode": "llm",
                    "cached": False,
                    "degraded": False,
                    "tools_invoked": [
                        "graph_query.find_entity",
                        "code_analyst.explain_implementation",
                    ],
                    "plan_iteration_count": 2,
                    "plan_iterations": [
                        {
                            "iteration": 1,
                            "routing_mode": "llm",
                            "agents": ["graph_query"],
                            "tools_invoked": ["graph_query.find_entity"],
                            "sufficient": False,
                            "reason": "no_graph_hits",
                            "refinement": None,
                        },
                        {
                            "iteration": 2,
                            "routing_mode": "llm",
                            "agents": ["graph_query", "code_analyst"],
                            "tools_invoked": ["code_analyst.explain_implementation"],
                            "sufficient": True,
                            "reason": "analyst_evidence",
                            "refinement": "no_graph_hits",
                        },
                    ],
                    "tokens": {
                        "total": 0,
                        "prompt": 0,
                        "completion": 0,
                        "llm_calls": 0,
                        "by_purpose": {},
                    },
                },
            }

    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=SimpleNamespace(
            indexer=object(), graph_query=object(), code_analyst=object()
        ),
        gateway_settings=GatewaySettings(),
        orchestrator_settings=OrchestratorSettings(),
    )
    events = [
        event
        async for event in ChatGatewayService(deps).stream("Where is NoSuchThing?", "s1", "c1")
    ]
    routing = [event for event in events if event.type == "routing"]
    assert len(routing) == 2
    assert routing[0].data["iteration"] == 1
    assert routing[0].data["sufficient"] is False
    assert routing[1].data["iteration"] == 2
    assert routing[1].data["sufficient"] is True


def _analyst_call_count(tools: list[str]) -> int:
    return sum(1 for tool in tools if tool.startswith("code_analyst."))


@pytest.mark.asyncio
async def test_refinement_round_issues_fewer_analyst_calls() -> None:
    query = "How does dependency injection work and show me examples from the codebase"

    class _FailingAnalyst(_CodeAnalystClient):
        async def get_code_snippet(
            self,
            *,
            qualified_name: str | None = None,
            file_path: str | None = None,
            line_start: int | None = None,
            line_end: int | None = None,
        ) -> object:
            await super().get_code_snippet(
                qualified_name=qualified_name,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
            )
            raise TimeoutError("code_analyst timeout")

        async def explain_implementation(self, qualified_name: str) -> object:
            self.explain_calls.append(qualified_name)
            raise TimeoutError("code_analyst timeout")

        async def analyze_function(self, qualified_name: str) -> object:
            self.analyze_calls.append(qualified_name)
            raise TimeoutError("code_analyst timeout")

        async def analyze_class(self, qualified_name: str) -> object:
            self.analyze_calls.append(qualified_name)
            raise TimeoutError("code_analyst timeout")

    hits = {
        "matches": [
            {
                "file_path": "fastapi/params.py",
                "line_start": 1,
                "line_end": 40,
                "qualified_name": "fastapi.params.Depends",
                "name": "Depends",
                "tier": "exact",
            },
            {
                "file_path": "fastapi/dependencies/utils.py",
                "line_start": 10,
                "line_end": 80,
                "qualified_name": "fastapi.dependencies.utils.get_dependant",
                "name": "get_dependant",
                "tier": "exact",
            },
            {
                "file_path": "fastapi/dependencies/utils.py",
                "line_start": 200,
                "line_end": 260,
                "qualified_name": "fastapi.dependencies.utils.solve_dependencies",
                "name": "solve_dependencies",
                "tier": "exact",
            },
        ]
    }

    class _ConceptualGraph(_GraphQueryClient):
        async def find_entity(self, name: str, entity_type: str | None = None) -> object:
            _ = entity_type
            self.find_entity_calls.append(name)
            lowered = name.lower()
            if lowered in {"work", "examples", "codebase", "handle", "validation"}:
                return {
                    "matches": [
                        {
                            "file_path": "tests/test_dependency_duplicates.py",
                            "line_start": 1,
                            "line_end": 20,
                            "qualified_name": "tests.test_dependency_duplicates",
                            "name": name,
                            "tier": "fulltext",
                        }
                    ]
                }
            if lowered.startswith("how does dependency") or "dependenc" in lowered:
                return hits
            return {"matches": [], "result_count": 0}

    llm = StubProvider(
        [
            QueryIntent(
                routing_mode="llm",
                intent="explanation",
                entities=[],
                target_agents=["graph_query", "code_analyst"],
                reasoning="explain DI",
            ).model_dump(),
            "PARTIAL",
        ]
    )
    settings = OrchestratorSettings(routing_strategy="llm_first", max_plan_iterations=2)
    service = OrchestratorService(llm, settings=settings)
    graph_query = _ConceptualGraph()
    code_analyst = _FailingAnalyst()
    result = await service.handle_query(
        query,
        "session-1",
        clients=_clients(graph_query=graph_query, code_analyst=code_analyst),  # type: ignore[arg-type]
        correlation_id="corr-refine-fanout",
    )

    iterations = result.metadata["plan_iterations"]
    assert len(iterations) == 2
    round1 = _analyst_call_count(iterations[0]["tools_invoked"])
    round2 = _analyst_call_count(iterations[1]["tools_invoked"])
    assert round1 > 0
    assert round2 < round1
    assert any(
        str(term).lower().startswith("how does dependency injection")
        for term in graph_query.find_entity_calls
    )
    generic = {"work", "examples", "codebase", "handle", "validation"}
    assert generic.isdisjoint({call.lower() for call in graph_query.find_entity_calls})
    assert not any(
        "test_dependency_duplicates" in str(call) for call in code_analyst.explain_calls
    )
