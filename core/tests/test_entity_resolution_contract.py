"""Contract tests: code_analyst must resolve every graph_query.find_entity hit.

These run against a real indexed Neo4j graph. Stubbed unit tests cannot catch
the name vs qualified_name mismatch that made class-level analysis fail.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from core.analysis.lookups import CLASS_CONTEXT, ENTITY_LOCATION, FUNCTION_CONTEXT
from core.analysis.models import ImplementationComparison
from core.analysis.service import CodeAnalystService
from core.graph.client import GraphClient
from core.llm.stub import StubProvider
from core.querying.service import GraphQueryService
from core.settings import AnalysisSettings

CONTRACT_NAMES = ("FastAPI", "APIRouter", "get_openapi")
REPO_ROOT = Path(AnalysisSettings.from_env().repo_root)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI not set"),
]


def _lookup(service: GraphQueryService) -> Any:
    async def graph_lookup(
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return service.execute_query(cypher, params).rows

    return graph_lookup


def test_lookups_resolve_every_find_entity_hit() -> None:
    with GraphClient() as client:
        client.verify_connectivity()
        query = GraphQueryService(client)
        for name in CONTRACT_NAMES:
            found = query.find_entity(name)
            assert found.error is None
            assert found.matches, f"find_entity({name!r}) returned no matches"
            for hit in found.matches:
                assert hit.qualified_name or hit.name
                by_name = query.execute_query(ENTITY_LOCATION, {"name": hit.name})
                by_qn = query.execute_query(ENTITY_LOCATION, {"name": hit.qualified_name})
                assert by_name.rows, f"ENTITY_LOCATION missed name={hit.name!r}"
                assert by_qn.rows, (
                    f"ENTITY_LOCATION missed qualified_name={hit.qualified_name!r}"
                )
                located = {row.get("qualified_name") for row in by_name.rows}
                assert hit.qualified_name in located
                if hit.entity_type == "Class":
                    klass = query.execute_query(CLASS_CONTEXT, {"name": hit.name})
                    assert klass.rows, f"CLASS_CONTEXT missed {hit.name!r}"
                    klass_qn = query.execute_query(
                        CLASS_CONTEXT, {"name": hit.qualified_name}
                    )
                    assert klass_qn.rows, f"CLASS_CONTEXT missed {hit.qualified_name!r}"
                elif hit.entity_type in {"Function", "Method"}:
                    fn = query.execute_query(FUNCTION_CONTEXT, {"name": hit.name})
                    assert fn.rows, f"FUNCTION_CONTEXT missed {hit.name!r}"
                    fn_qn = query.execute_query(
                        FUNCTION_CONTEXT, {"name": hit.qualified_name}
                    )
                    assert fn_qn.rows, f"FUNCTION_CONTEXT missed {hit.qualified_name!r}"


@pytest.mark.asyncio
@pytest.mark.skipif(not REPO_ROOT.exists(), reason="REPO_ROOT not available")
async def test_code_analyst_resolves_every_find_entity_hit() -> None:
    with GraphClient() as client:
        client.verify_connectivity()
        query = GraphQueryService(client)
        repo_root = REPO_ROOT if REPO_ROOT.exists() else Path(".")
        analyst = CodeAnalystService(StubProvider(), _lookup(query), repo_root=repo_root)
        for name in CONTRACT_NAMES:
            found = query.find_entity(name)
            assert found.matches, f"find_entity({name!r}) returned no matches"
            for hit in found.matches:
                snippet = await analyst.get_code_snippet(qualified_name=hit.name)
                assert "Entity not found" not in (snippet.error or ""), (
                    f"code_analyst missed find_entity hit {hit.name!r} "
                    f"({hit.qualified_name!r}, {hit.entity_type})"
                )
                snippet_qn = await analyst.get_code_snippet(
                    qualified_name=hit.qualified_name
                )
                assert "Entity not found" not in (snippet_qn.error or ""), (
                    f"code_analyst missed qualified_name {hit.qualified_name!r}"
                )
                if hit.name and "/" not in hit.name:
                    reresolved = await analyst.get_code_snippet(
                        file_path=hit.name,
                        line_start=1,
                        line_end=120,
                    )
                    assert "Entity not found" not in (reresolved.error or ""), (
                        f"get_code_snippet treated {hit.name!r} as a file_path"
                    )


@pytest.mark.asyncio
@pytest.mark.skipif(not REPO_ROOT.exists(), reason="REPO_ROOT not available")
async def test_compare_implementations_resolves_class_short_names() -> None:
    with GraphClient() as client:
        client.verify_connectivity()
        query = GraphQueryService(client)
        fastapi = query.find_entity("FastAPI")
        router = query.find_entity("APIRouter")
        assert fastapi.matches
        assert router.matches
        provider = StubProvider(
            [
                ImplementationComparison(
                    name_a="FastAPI",
                    name_b="APIRouter",
                    summary="Compared FastAPI and APIRouter.",
                ).model_dump()
            ]
        )
        analyst = CodeAnalystService(
            provider,
            _lookup(query),
            repo_root=REPO_ROOT if REPO_ROOT.exists() else Path("."),
        )
        result = await analyst.compare_implementations("FastAPI", "APIRouter")
        assert result.error is None, result.error


def test_class_neighbor_queries_record_hops() -> None:
    with GraphClient() as client:
        client.verify_connectivity()
        query = GraphQueryService(client)
        dependencies = query.get_dependencies("FastAPI")
        dependents = query.get_dependents("APIRouter")
        assert dependencies.neighbors, "class get_dependencies returned no neighbors"
        assert any(hit.hop for hit in dependencies.neighbors) or dependencies.hops
        assert "Class->Module" in dependencies.hops or "Class->Method" in dependencies.hops
        _ = dependents
        traces = query.trace_imports("FastAPI")
        assert traces.hop == "Class->Module" or traces.paths
