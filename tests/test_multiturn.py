"""Multi-turn follow-ups must resolve referring expressions from prior turns."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.llm.stub import StubProvider
from core.memory import ConversationContext, ConversationTurn
from core.orchestration.service import OrchestratorService
from core.settings import OrchestratorSettings

_CLARIFICATION = (
    "which class",
    "what are you referring",
    "could you specify",
    "please clarify",
    "i need more context",
)

_GRAPH_LOOKUP_TOOLS = {
    "graph_query.find_entity",
    "graph_query.find_related",
    "graph_query.get_dependencies",
    "graph_query.get_dependents",
}


class _MemoryClient:
    def __init__(self) -> None:
        self._turns: list[ConversationTurn] = []
        self._cache: dict[str, object] = {}
        self.cache_get_keys: list[str] = []

    async def get_context(self, session_id: str, token_budget: int = 3000) -> ConversationContext:
        _ = session_id, token_budget
        return ConversationContext(recent_turns=list(self._turns))

    async def get_cached_response(self, cache_key: str) -> object | None:
        self.cache_get_keys.append(cache_key)
        payload = self._cache.get(cache_key)
        if payload is None:
            return None
        return SimpleNamespace(response_json=payload)

    async def cache_response(self, cache_key: str, response_json: object) -> None:
        self._cache[cache_key] = response_json

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        _ = session_id
        self._turns.append(
            ConversationTurn(
                id=len(self._turns) + 1,
                role=role,
                content=content,
                created_at="2026-01-01T00:00:00Z",
                token_estimate=max(1, len(content) // 4),
            )
        )


class _GraphQueryClient:
    def __init__(self) -> None:
        self.find_entity_calls: list[str] = []
        self.find_related_calls: list[tuple[str, str]] = []
        self.dependent_calls: list[str] = []

    async def get_statistics(self) -> object:
        return SimpleNamespace(index_version="idx-1")

    async def find_entity(self, name: str, entity_type: str | None = None) -> object:
        _ = entity_type
        self.find_entity_calls.append(name)
        if "APIRouter" in name:
            return {
                "file_path": "fastapi/routing.py",
                "line_start": 900,
                "line_end": 1400,
                "qualified_name": "fastapi.routing.APIRouter",
                "name": "APIRouter",
            }
        return {
            "file_path": "fastapi/applications.py",
            "line_start": 42,
            "line_end": 4774,
            "qualified_name": "fastapi.applications.FastAPI",
            "name": "FastAPI",
        }

    async def get_dependencies(self, name: str) -> object:
        return {"name": name, "neighbors": []}

    async def get_dependents(self, name: str) -> object:
        self.dependent_calls.append(name)
        if "APIRouter" in name:
            neighbors = [
                {"name": "app.include_router", "qualified_name": "app.include_router"}
            ]
        elif "FastAPI" in name:
            neighbors = [
                {
                    "name": "Starlette",
                    "qualified_name": "starlette.applications.Starlette",
                }
            ]
        else:
            neighbors = []
        return {"name": name, "neighbors": neighbors}

    async def find_related(self, name: str, relationship_type: str) -> object:
        self.find_related_calls.append((name, relationship_type))
        return {"name": name, "relationship_type": relationship_type, "neighbors": []}

    async def trace_imports(self, module: str, depth: int = 5) -> object:
        _ = depth
        return {"module": module, "paths": []}


class _CodeAnalystClient:
    async def get_code_snippet(self, **kwargs: Any) -> object:
        return {
            "file_path": kwargs.get("file_path") or "",
            "line_start": kwargs.get("line_start"),
            "line_end": kwargs.get("line_end"),
            "text": "class FastAPI:",
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

    async def find_patterns(self, pattern: str) -> object:
        return {"pattern": pattern, "instances": [], "error": None}


class _IndexerClient:
    async def index_repository(self, repo_url: str | None = None) -> object:
        return {"status": "ok", "repo_url": repo_url}


def _clients(
    *,
    memory: _MemoryClient,
    graph_query: _GraphQueryClient,
) -> SimpleNamespace:
    return SimpleNamespace(
        memory=memory,
        graph_query=graph_query,
        code_analyst=_CodeAnalystClient(),
        indexer=_IndexerClient(),
    )


@pytest.mark.asyncio
async def test_follow_up_resolves_antecedent_to_fastapi() -> None:
    llm = StubProvider(
        [
            "The FastAPI class is the main application class in fastapi/applications.py.",
            "FastAPI constructor parameters include title, debug, and routes.",
        ]
    )
    service = OrchestratorService(
        llm, settings=OrchestratorSettings(routing_strategy="rules_first")
    )
    memory = _MemoryClient()
    graph_query = _GraphQueryClient()
    clients = _clients(memory=memory, graph_query=graph_query)

    first = await service.handle_query(
        "What is the FastAPI class?",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-turn-1",
    )
    assert "FastAPI" in first.answer

    graph_query.find_entity_calls.clear()
    second = await service.handle_query(
        "What about its parameters?",
        "session-1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-turn-2",
    )

    tools = set(second.metadata["tools_invoked"])
    assert tools & _GRAPH_LOOKUP_TOOLS, tools
    assert "FastAPI" in graph_query.find_entity_calls
    assert "FastAPI" in second.answer
    lowered = second.answer.lower()
    assert not any(phrase in lowered for phrase in _CLARIFICATION)


@pytest.mark.asyncio
async def test_follow_up_cache_key_includes_resolved_entities() -> None:
    """Same follow-up text with a different antecedent must not reuse the cache."""
    llm = StubProvider(
        [
            "APIRouter is the routing class in fastapi/routing.py.",
            "app.include_router depends on APIRouter.",
            "FastAPI is the main application class in fastapi/applications.py.",
            "Starlette depends on FastAPI.",
        ]
    )
    service = OrchestratorService(
        llm, settings=OrchestratorSettings(routing_strategy="rules_first")
    )
    memory = _MemoryClient()
    graph_query = _GraphQueryClient()
    clients = _clients(memory=memory, graph_query=graph_query)

    first = await service.handle_query(
        "What is APIRouter?",
        "session-poison",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-poison-1",
    )
    assert first.metadata["cached"] is False
    assert "APIRouter" in first.answer

    second = await service.handle_query(
        "Who depends on it?",
        "session-poison",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-poison-2",
    )
    assert second.metadata["cached"] is False
    assert "APIRouter" in second.answer
    assert second.metadata["cache_key"] != first.metadata["cache_key"]

    third = await service.handle_query(
        "What is FastAPI?",
        "session-poison",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-poison-3",
    )
    assert third.metadata["cached"] is False
    assert "FastAPI" in third.answer

    graph_query.find_entity_calls.clear()
    graph_query.dependent_calls.clear()
    fourth = await service.handle_query(
        "Who depends on it?",
        "session-poison",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-poison-4",
    )

    assert fourth.metadata["cache_key"] != second.metadata["cache_key"]
    assert fourth.metadata["cached"] is False
    assert fourth.answer != second.answer
    assert "Starlette" in fourth.answer
    assert "APIRouter" not in fourth.answer
    looked_up = graph_query.find_entity_calls + graph_query.dependent_calls
    assert any("FastAPI" in name for name in looked_up)
