"""Integration trap queries must not hallucinate FastAPI codebase answers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from core.analysis.service import CodeAnalystService
from core.graph.client import GraphClient
from core.llm.offline_provider import OfflineProvider
from core.llm.stub import StubProvider
from core.memory import ConversationContext
from core.orchestration.service import OrchestratorService
from core.orchestration.synthesis import synthesize_response
from core.querying.embeddings import default_embedding_provider
from core.querying.service import GraphQueryService
from core.settings import AnalysisSettings, OrchestratorSettings

EVAL_PATH = Path(__file__).resolve().parents[2] / "evals" / "traps.jsonl"
QA_PATH = Path(__file__).resolve().parents[2] / "evals" / "qa.jsonl"
PY_PATH_RE = re.compile(r"\b[\w./-]+\.py\b")
REPO_ROOT = Path(AnalysisSettings.from_env().repo_root)


def _t05_query() -> str:
    for raw in QA_PATH.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if payload.get("id") == "t05":
            return str(payload["query"])
    raise AssertionError("t05 missing from evals/qa.jsonl")


@pytest.mark.asyncio
async def test_t05_postgresql_wal_incidental_fulltext_is_refused() -> None:
    llm = StubProvider(["FastAPI logging internals should not be used as the answer"])
    result = await synthesize_response(
        _t05_query(),
        {
            "graph_query": {
                "ok": True,
                "output": {
                    "queried_entities": ["PostgreSQL"],
                    "candidates": [
                        {
                            "name": "logging",
                            "qualified_name": "fastapi.logger.logging",
                            "file_path": "fastapi/logger.py",
                            "tier": "fulltext",
                            "score": 2.0,
                        }
                    ],
                },
            }
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings.from_env(),
    )
    lowered = result.answer.lower()
    assert "not in the indexed fastapi codebase" in lowered
    assert llm.calls == []


def _load_queries() -> list[str]:
    queries: list[str] = []
    for raw in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        queries.append(str(payload["query"]))
    return queries


class _MemoryClient:
    async def get_context(self, session_id: str, token_budget: int = 3000) -> ConversationContext:
        _ = session_id, token_budget
        return ConversationContext()

    async def get_cached_response(self, cache_key: str) -> None:
        _ = cache_key
        return None

    async def cache_response(self, cache_key: str, response_json: Any) -> None:
        _ = cache_key, response_json

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        _ = session_id, role, content


class _GraphClientAdapter:
    def __init__(self, service: GraphQueryService) -> None:
        self._service = service

    async def get_statistics(self) -> Any:
        return self._service.get_statistics()

    async def find_entity(self, name: str, entity_type: str | None = None) -> dict[str, Any] | None:
        result = self._service.find_entity(name, entity_type)
        if not result.matches:
            return None
        return result.matches[0].model_dump(mode="json")

    async def get_dependencies(self, name: str) -> dict[str, Any]:
        return self._service.get_dependencies(name).model_dump(mode="json")

    async def get_dependents(self, name: str) -> dict[str, Any]:
        return self._service.get_dependents(name).model_dump(mode="json")

    async def find_related(self, name: str, relationship_type: str) -> dict[str, Any]:
        return self._service.find_related(name, relationship_type).model_dump(mode="json")

    async def trace_imports(self, module: str, depth: int = 5) -> dict[str, Any]:
        return self._service.trace_imports(module, depth).model_dump(mode="json")


class _CodeAnalystClientAdapter:
    def __init__(self, service: CodeAnalystService) -> None:
        self._service = service

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        result = await self._service.get_code_snippet(
            qualified_name=qualified_name,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
        )
        return result.model_dump(mode="json")

    async def explain_implementation(self, qualified_name: str) -> dict[str, Any]:
        result = await self._service.explain_implementation(qualified_name)
        return result.model_dump(mode="json")

    async def analyze_function(self, qualified_name: str) -> dict[str, Any]:
        result = await self._service.analyze_function(qualified_name)
        return result.model_dump(mode="json")

    async def analyze_class(self, qualified_name: str) -> dict[str, Any]:
        result = await self._service.analyze_class(qualified_name)
        return result.model_dump(mode="json")

    async def compare_implementations(self, name_a: str, name_b: str) -> dict[str, Any]:
        result = await self._service.compare_implementations(name_a, name_b)
        return result.model_dump(mode="json")

    async def find_patterns(self, pattern: str) -> dict[str, Any]:
        result = await self._service.find_patterns(pattern)
        return result.model_dump(mode="json")


async def _graph_lookup(
    service: GraphQueryService,
    cypher: str,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return service.execute_query(cypher, params).rows


def _graph_file_paths(client: GraphClient) -> set[str]:
    rows = client.run_read("MATCH (f:File) RETURN f.path AS path")
    return {str(row["path"]) for row in rows if row.get("path")}


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI not set")
@pytest.mark.skipif(not REPO_ROOT.exists(), reason="REPO_ROOT not available")
@pytest.mark.parametrize("query", _load_queries())
async def test_trap_query_reports_out_of_codebase_without_fake_paths(query: str) -> None:
    with GraphClient() as graph_client:
        graph_client.verify_connectivity()
        graph_service = GraphQueryService(
            graph_client,
            embedding_provider=default_embedding_provider(),
        )
        analysis_service = CodeAnalystService(
            OfflineProvider(),
            lambda cypher, params=None: _graph_lookup(graph_service, cypher, params),
            repo_root=REPO_ROOT,
        )
        orchestrator = OrchestratorService(
            OfflineProvider(),
            settings=OrchestratorSettings.from_env(),
        )
        clients = type(
            "Clients",
            (),
            {
                "memory": _MemoryClient(),
                "graph_query": _GraphClientAdapter(graph_service),
                "code_analyst": _CodeAnalystClientAdapter(analysis_service),
                "indexer": None,
            },
        )()

        result = await orchestrator.handle_query(
            query,
            "trap-session",
            clients=clients,  # type: ignore[arg-type]
            correlation_id="trap-eval",
        )

        lowered = result.answer.lower()
        mentioned_paths = set(PY_PATH_RE.findall(result.answer))
        assert mentioned_paths <= _graph_file_paths(graph_client)
        refused = any(
            needle in lowered
            for needle in (
                "not in the indexed fastapi codebase",
                "could not find indexed fastapi",
            )
        )
        if not refused:
            # Incidental graph hits may reach the synthesizer; the answer still
            # must not invent FastAPI paths that are not in the index.
            assert "fastapi/" not in lowered
