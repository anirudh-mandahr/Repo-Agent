"""Synthesis fallback, prompt budget, and evidence-only degradation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.llm.stub import StubProvider
from core.memory import ConversationContext
from core.orchestration.fallback import EVIDENCE_ONLY_HEADER
from core.orchestration.prompt_budget import (
    SYNTHESIS_TURN_TOKEN_CEILING,
    compact_neighbor_payloads,
    estimate_tokens,
    measure_synthesis_prompt,
)
from core.orchestration.service import OrchestratorService
from core.orchestration.synthesis import synthesize_response
from core.settings import OrchestratorSettings

_FASTAPI_HIT = {
    "ok": True,
    "output": {
        "queried_entities": ["FastAPI"],
        "entities": [
            {
                "qualified_name": "fastapi.applications.FastAPI",
                "file_path": "fastapi/applications.py",
                "line_start": 10,
                "line_end": 40,
                "name": "FastAPI",
            }
        ],
        "candidates": [
            {
                "qualified_name": "fastapi.applications.FastAPI",
                "file_path": "fastapi/applications.py",
                "line_start": 10,
                "line_end": 40,
                "name": "FastAPI",
            }
        ],
        "dependents": [
            {
                "neighbors": [
                    {
                        "name": "APIRouter",
                        "qualified_name": "fastapi.routing.APIRouter",
                        "file_path": "fastapi/routing.py",
                        "relationship_type": "DEPENDS_ON",
                    }
                ]
            }
        ],
    },
}

_SNIPPET_OUTPUT = {
    "ok": True,
    "output": {
        "snippets": [
            {
                "file_path": "fastapi/applications.py",
                "line_start": 10,
                "line_end": 12,
                "text": "class FastAPI:\n    pass\n",
                "error": None,
            }
        ]
    },
}


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
        return {
            "name": name,
            "neighbors": [
                {
                    "name": "APIRouter",
                    "qualified_name": "fastapi.routing.APIRouter",
                    "file_path": "fastapi/routing.py",
                    "relationship_type": "DEPENDS_ON",
                }
            ],
        }

    async def find_related(self, name: str, relationship_type: str) -> object:
        return {"name": name, "relationship_type": relationship_type, "neighbors": []}

    async def trace_imports(self, module: str, depth: int = 5) -> object:
        _ = depth
        return {"module": module, "paths": []}


class _CodeAnalystClient:
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

    async def find_patterns(self, pattern: str) -> object:
        return {"pattern": pattern, "instances": [], "error": None}


class _IndexerClient:
    async def index_repository(self, repo_url: str | None = None) -> object:
        return {"status": "ok", "repo_url": repo_url}


def _clients() -> SimpleNamespace:
    return SimpleNamespace(
        memory=_MemoryClient(),
        graph_query=_GraphQueryClient(),
        code_analyst=_CodeAnalystClient(),
        indexer=_IndexerClient(),
    )


def _agent_outputs() -> dict[str, Any]:
    return {"graph_query": _FASTAPI_HIT, "code_analyst": _SNIPPET_OUTPUT}


@pytest.mark.asyncio
async def test_synthesis_llm_error_returns_evidence_only_answer() -> None:
    class _Boom:
        async def complete(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("llm exploded")

    result = await synthesize_response(
        "What is the FastAPI class?",
        _agent_outputs(),
        ConversationContext(),
        llm_provider=_Boom(),  # type: ignore[arg-type]
        settings=OrchestratorSettings.from_env(),
        correlation_id="corr-raise",
    )

    assert result.evidence_only is True
    assert result.degraded_reason == "RuntimeError"
    assert EVIDENCE_ONLY_HEADER in result.answer
    assert "fastapi/applications.py" in result.answer
    assert "class FastAPI" in result.answer
    assert "APIRouter" in result.answer


@pytest.mark.asyncio
async def test_synthesis_llm_timeout_returns_evidence_only_answer() -> None:
    class _Hang:
        async def complete(self, *args: object, **kwargs: object) -> object:
            await asyncio.sleep(5)
            raise AssertionError("should have timed out")

    result = await synthesize_response(
        "What is the FastAPI class?",
        _agent_outputs(),
        ConversationContext(),
        llm_provider=_Hang(),  # type: ignore[arg-type]
        settings=OrchestratorSettings(synthesis_timeout_s=0.05),
        correlation_id="corr-timeout",
    )

    assert result.evidence_only is True
    assert result.degraded_reason == "TimeoutError"
    assert EVIDENCE_ONLY_HEADER in result.answer
    assert "fastapi.applications.FastAPI" in result.answer


@pytest.mark.asyncio
async def test_oversized_agent_outputs_are_truncated_within_budget() -> None:
    llm = StubProvider(["FINAL ANSWER"])
    budget = 500
    huge_text = "class FastAPI:\n    " + ("x" * 80_000)
    outputs: dict[str, Any] = {
        "graph_query": {
            "ok": True,
            "output": {
                "queried_entities": ["FastAPI"],
                "entities": [
                    {
                        "qualified_name": "fastapi.applications.FastAPI",
                        "file_path": "fastapi/applications.py",
                    }
                ],
                "candidates": [
                    {"qualified_name": f"Entity{i}", "file_path": f"mod{i}.py"}
                    for i in range(40)
                ],
            },
        },
        "code_analyst": {
            "ok": True,
            "output": {
                "snippets": [
                    {
                        "file_path": "fastapi/applications.py",
                        "line_start": 1,
                        "line_end": 4000,
                        "text": huge_text,
                        "error": None,
                    }
                ],
                "comparison": {
                    "name_a": "FastAPI",
                    "name_b": "APIRouter",
                    "snippet_a": "y" * 50_000,
                    "summary": "unclipped compare path",
                },
            },
        },
    }

    result = await synthesize_response(
        "Compare FastAPI and APIRouter implementations in the codebase",
        outputs,
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings(synthesis_prompt_token_budget=budget),
        correlation_id="corr-budget",
    )

    assert result.prompt_truncated is not None
    assert result.prompt_truncated.dropped
    assert result.estimated_tokens <= budget
    assert result.prompt_truncated.final_estimated_tokens <= budget
    assert result.prompt_truncated.original_estimated_tokens > budget
    assert llm.calls
    prompt = "\n".join(message.content for message in llm.calls[0].messages)
    assert estimate_tokens(prompt) <= budget
    kinds = {item["kind"] for item in result.prompt_truncated.dropped}
    assert "snippet" in kinds


@pytest.mark.asyncio
async def test_handle_query_synthesis_error_is_degraded_evidence_only() -> None:
    class _Boom:
        async def complete(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("llm exploded")

    service = OrchestratorService(
        _Boom(),  # type: ignore[arg-type]
        settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "session-1",
        clients=_clients(),  # type: ignore[arg-type]
        correlation_id="corr-handle-raise",
    )

    assert result.metadata["degraded"] is True
    assert result.metadata["evidence_only"] is True
    assert result.metadata["degraded_reason"] == "RuntimeError"
    assert EVIDENCE_ONLY_HEADER in result.answer
    assert "fastapi/applications.py" in result.answer


@pytest.mark.asyncio
async def test_handle_query_synthesis_timeout_is_degraded_evidence_only() -> None:
    class _Hang:
        async def complete(self, *args: object, **kwargs: object) -> object:
            await asyncio.sleep(5)
            raise AssertionError("should have timed out")

    service = OrchestratorService(
        _Hang(),  # type: ignore[arg-type]
        settings=OrchestratorSettings(
            routing_strategy="rules_first",
            synthesis_timeout_s=0.05,
        ),
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "session-1",
        clients=_clients(),  # type: ignore[arg-type]
        correlation_id="corr-handle-timeout",
    )

    assert result.metadata["degraded"] is True
    assert result.metadata["evidence_only"] is True
    assert result.metadata["degraded_reason"] == "TimeoutError"
    assert EVIDENCE_ONLY_HEADER in result.answer


@pytest.mark.asyncio
async def test_synthesis_failed_logs_structured_event(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class _Log:
        def warning(self, event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

    monkeypatch.setattr("core.orchestration.synthesis.log", _Log())

    class _Boom:
        async def complete(self, *args: object, **kwargs: object) -> object:
            raise ValueError("nope")

    await synthesize_response(
        "What is the FastAPI class?",
        _agent_outputs(),
        ConversationContext(),
        llm_provider=_Boom(),  # type: ignore[arg-type]
        settings=OrchestratorSettings.from_env(),
        correlation_id="corr-log",
    )

    assert events
    event, fields = events[0]
    assert event == "orchestrator.synthesis_failed"
    assert fields["correlation_id"] == "corr-log"
    assert fields["exception_type"] == "ValueError"
    assert fields["prompt_chars"] > 0
    assert fields["estimated_tokens"] > 0
    assert "elapsed_ms" in fields
    assert "graph_query" in fields["agents_invoked"]


def test_module_citations_omit_line_ranges() -> None:
    from core.orchestration.synthesis import _citation_line

    module_hit = {
        "entity_type": "Module",
        "qualified_name": "fastapi.applications",
        "file_path": "fastapi/applications.py",
        "line_start": 1,
        "line_end": 4774,
        "name": "applications",
    }
    class_hit = {
        "entity_type": "Class",
        "qualified_name": "fastapi.applications.FastAPI",
        "file_path": "fastapi/applications.py",
        "line_start": 42,
        "line_end": 120,
        "name": "FastAPI",
    }
    module_line = _citation_line(module_hit)
    class_line = _citation_line(class_hit)
    assert module_line is not None
    assert "1-4774" not in module_line
    assert "fastapi/applications.py" in module_line
    assert class_line is not None
    assert "42-120" in class_line


def test_compacted_neighbor_payload_stays_under_turn_token_ceiling() -> None:
    neighbors = [
        {
            "name": f"helper_{index}",
            "qualified_name": f"pkg.mod.helper_{index}",
            "relationship_type": "CALLS",
            "direction": "incoming",
            "file_path": f"pkg/mod_{index}.py",
            "hop": "Class->Method",
        }
        for index in range(400)
    ]
    payload = {
        "graph_query": {
            "ok": True,
            "output": {
                "queried_entities": ["APIRouter"],
                "dependents": [
                    {
                        "name": "APIRouter",
                        "neighbors": neighbors,
                        "result_count": 400,
                        "total_count": 12000,
                        "truncated": True,
                        "hops": ["Class->Method"],
                    }
                ],
            },
        }
    }
    raw_tokens = measure_synthesis_prompt(
        "What classes inherit from APIRouter?", "", payload
    )[1]
    compacted = compact_neighbor_payloads(payload)
    tokens = measure_synthesis_prompt(
        "What classes inherit from APIRouter?", "", compacted
    )[1]
    assert raw_tokens > SYNTHESIS_TURN_TOKEN_CEILING
    assert tokens <= SYNTHESIS_TURN_TOKEN_CEILING
    summarized = compacted["graph_query"]["output"]["dependents"][0]
    assert summarized["total_count"] == 12000
    assert summarized["truncated"] is True
    assert len(summarized["neighbors"]) <= 8
    assert "12000 neighbors" in summarized["summary"]
    assert estimate_tokens(summarized["summary"]) < 50
