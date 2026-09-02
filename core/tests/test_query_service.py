"""Unit tests for GraphQueryService with a fake graph client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from core.graph.schema import FIND_ENTITY_LABELS, RELATIONSHIP_TYPES
from core.querying.embeddings import (
    HashingEmbeddingProvider,
    cosine_similarity,
    tokenize_for_embedding,
)
from core.querying.safety import QueryRejected
from core.querying.service import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    DEFAULT_NEIGHBOR_BRANCH_LIMIT,
    DEFAULT_RESULT_LIMIT,
    READ_TIMEOUT_S,
    STATISTICS_TIMEOUT_S,
    GraphQueryService,
    conceptual_entity_names,
    lucene_query,
    path_name_affinity,
    retrieval_sort_key,
)


class FakeClient:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        by_query: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.rows = rows if rows is not None else []
        self.by_query = by_query or {}
        self.queries: list[tuple[str, dict[str, Any], float]] = []

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        self.queries.append((query, dict(params or {}), timeout_s))
        for needle, rows in self.by_query.items():
            if needle in query:
                return list(rows)
        if "UNWIND labels(n)" in query and "count(*) AS count" in query:
            return [{"label": "Module", "count": 2}, {"label": "Meta", "count": 1}]
        if "sum(c) AS total" in query:
            return [{"total": len(self.rows)}]
        if "MATCH ()-[r]->()" in query and "count(*) AS count" in query:
            return [{"relationship_type": "CONTAINS", "count": 4}]
        if "MATCH (m:Meta {key: 'index_version'})" in query:
            return [{"index_version": "abc123", "last_indexed_at": "2026-08-18T00:00:00+00:00"}]
        return list(self.rows)


def test_execute_query_appends_limit_and_uses_timeout() -> None:
    client = FakeClient([{"n": 1}])
    result = GraphQueryService(client).execute_query("MATCH (n) RETURN n")
    query, params, timeout_s = client.queries[0]
    assert query.endswith(f"LIMIT {DEFAULT_RESULT_LIMIT}")
    assert params == {}
    assert timeout_s == READ_TIMEOUT_S
    assert result.cypher_executed == query
    assert result.params == {}
    assert result.result_count == 1
    assert result.truncated is False


def test_execute_query_does_not_duplicate_existing_limit() -> None:
    client = FakeClient()
    result = GraphQueryService(client).execute_query("MATCH (n) RETURN n LIMIT 5")
    query, _params, _timeout = client.queries[0]
    assert query == "MATCH (n) RETURN n LIMIT 5"
    assert query.count("LIMIT") == 1
    assert result.cypher_executed == query
    assert result.params == {}


def test_execute_query_rejects_writes_before_read() -> None:
    client = FakeClient()
    with pytest.raises(QueryRejected) as exc_info:
        GraphQueryService(client).execute_query("CREATE (n:X)")
    assert exc_info.value.clause == "CREATE"
    assert client.queries == []


def test_find_entity_maps_rows() -> None:
    client = FakeClient(
        [
            {
                "labels": ["Function"],
                "name": "get_openapi",
                "qualified_name": "fastapi.openapi.utils.get_openapi",
                "file_path": "fastapi/openapi/utils.py",
                "path": None,
                "line_start": 10,
                "line_end": 40,
            }
        ]
    )
    result = GraphQueryService(client).find_entity("get_openapi")
    assert result.result_count == 1
    assert result.truncated is False
    assert result.error is None
    hit = result.matches[0]
    assert hit.qualified_name == "fastapi.openapi.utils.get_openapi"
    assert hit.entity_type == "Function"
    assert hit.file_path == "fastapi/openapi/utils.py"
    assert hit.line_start == 10
    assert hit.docstring_text is None
    assert hit.docstring_summary is None


def test_find_entity_invalid_type_returns_structured_error() -> None:
    client = FakeClient([{"qualified_name": "should.not.run"}])
    result = GraphQueryService(client).find_entity("FastAPI", entity_type="Widget")
    assert result.error is not None
    assert "Widget" in result.error
    assert result.valid_types == list(FIND_ENTITY_LABELS)
    for label in FIND_ENTITY_LABELS:
        assert label in result.error
    assert result.result_count == 0
    assert result.matches == []
    assert client.queries == []


def test_get_dependencies_maps_outgoing_neighbors() -> None:
    client = FakeClient(
        [
            {
                "relationship_type": "DEPENDS_ON",
                "direction": "outgoing",
                "labels": ["Module"],
                "name": "routing",
                "qualified_name": "fastapi.routing",
                "module": None,
                "path": None,
                "file_path": "fastapi/routing.py",
            }
        ]
    )
    result = GraphQueryService(client).get_dependencies("fastapi.applications")
    assert result.name == "fastapi.applications"
    assert result.result_count == 1
    assert result.total_count == 1
    assert result.truncated is False
    hit = result.neighbors[0]
    assert hit.relationship_type == "DEPENDS_ON"
    assert hit.direction == "outgoing"
    assert hit.qualified_name == "fastapi.routing"
    assert hit.hop is None


def test_get_dependencies_records_class_hops() -> None:
    client = FakeClient(
        [
            {
                "relationship_type": "IMPORTS",
                "direction": "outgoing",
                "labels": ["Import"],
                "name": "APIRouter",
                "qualified_name": "fastapi.routing.APIRouter",
                "module": None,
                "path": None,
                "file_path": "fastapi/routing.py",
                "hop": "Class->Module",
            },
            {
                "relationship_type": "CALLS",
                "direction": "outgoing",
                "labels": ["Method"],
                "name": "add_api_route",
                "qualified_name": "fastapi.routing.APIRouter.add_api_route",
                "module": None,
                "path": None,
                "file_path": "fastapi/routing.py",
                "hop": "Class->Method",
            },
        ]
    )
    result = GraphQueryService(client).get_dependencies("FastAPI")
    assert result.hops == ["Class->Module", "Class->Method"]
    assert result.neighbors[0].hop == "Class->Module"
    assert result.neighbors[1].hop == "Class->Method"


def test_trace_imports_records_class_to_module_hop() -> None:
    client = FakeClient(
        [
            {
                "nodes": ["fastapi.applications", "fastapi.routing"],
                "hop": "Class->Module",
            }
        ]
    )
    result = GraphQueryService(client).trace_imports("FastAPI", depth=3)
    assert result.hop == "Class->Module"
    assert result.paths == [["fastapi.applications", "fastapi.routing"]]


def test_get_dependents_maps_incoming_neighbors() -> None:
    client = FakeClient(
        [
            {
                "relationship_type": "CALLS",
                "direction": "incoming",
                "labels": ["Function"],
                "name": "ping",
                "qualified_name": "sample_module.ping",
                "module": None,
                "path": None,
                "file_path": "sample_module.py",
            }
        ]
    )
    result = GraphQueryService(client).get_dependents("sample_module.helper")
    assert result.neighbors[0].direction == "incoming"
    assert result.neighbors[0].relationship_type == "CALLS"


def test_trace_imports_returns_paths_and_caps_depth() -> None:
    client = FakeClient([{"nodes": ["fastapi.applications", "fastapi.routing"]}])
    result = GraphQueryService(client).trace_imports("fastapi.applications", depth=99)
    assert result.depth == 5
    assert result.paths == [["fastapi.applications", "fastapi.routing"]]
    assert client.queries[0][1] == {"module": "fastapi.applications", "depth": 5}


def test_find_related_invalid_type_returns_structured_error() -> None:
    client = FakeClient([{"name": "should.not.run"}])
    result = GraphQueryService(client).find_related("FastAPI", "DEFINES")
    assert result.error is not None
    assert "DEFINES" in result.error
    assert result.valid_types == list(RELATIONSHIP_TYPES)
    assert client.queries == []


def test_find_related_maps_neighbors_with_direction() -> None:
    client = FakeClient(
        [
            {
                "direction": "outgoing",
                "relationship_type": "CONTAINS",
                "labels": ["Class"],
                "name": "FastAPI",
                "qualified_name": "fastapi.applications.FastAPI",
                "module": None,
                "path": None,
                "file_path": "fastapi/applications.py",
            }
        ]
    )
    result = GraphQueryService(client).find_related("fastapi.applications", "CONTAINS")
    assert result.error is None
    assert result.neighbors[0].direction == "outgoing"
    assert result.neighbors[0].relationship_type == "CONTAINS"


def test_truncated_when_default_limit_is_hit() -> None:
    rows = [
        {
            "labels": ["Function"],
            "name": f"fn_{i}",
            "qualified_name": f"fn_{i}",
            "file_path": "x.py",
        }
        for i in range(DEFAULT_RESULT_LIMIT)
    ]
    result = GraphQueryService(FakeClient(rows)).find_entity("fn")
    assert result.result_count == DEFAULT_RESULT_LIMIT
    assert result.truncated is True


def test_neighbor_query_reports_total_and_truncated_when_count_exceeds_rows() -> None:
    rows = [
        {
            "relationship_type": "CALLS",
            "direction": "incoming",
            "labels": ["Function"],
            "name": f"fn_{i}",
            "qualified_name": f"mod.fn_{i}",
            "file_path": "x.py",
        }
        for i in range(DEFAULT_NEIGHBOR_BRANCH_LIMIT)
    ]
    client = FakeClient(rows, by_query={"sum(c) AS total": [{"total": 87}]})
    result = GraphQueryService(client).get_dependents("APIRouter")
    assert result.result_count == DEFAULT_NEIGHBOR_BRANCH_LIMIT
    assert result.total_count == 87
    assert result.truncated is True
    params = [item[1] for item in client.queries]
    assert any(item.get("branch_limit") == DEFAULT_NEIGHBOR_BRANCH_LIMIT for item in params)


def test_get_statistics_returns_counts_and_index_metadata() -> None:
    client = FakeClient()
    result = GraphQueryService(client).get_statistics()
    assert result.node_counts["Module"] == 2
    assert result.node_counts["Meta"] == 1
    assert result.node_counts["Function"] == 0
    assert result.relationship_counts["CONTAINS"] == 4
    assert result.relationship_counts["CALLS"] == 0
    assert result.index_version == "abc123"
    assert result.last_indexed_at == "2026-08-18T00:00:00+00:00"
    assert all(timeout == STATISTICS_TIMEOUT_S for _query, _params, timeout in client.queries[-3:])


def test_lucene_query_uses_name_fields_for_identifiers() -> None:
    query = lucene_query("Django")
    assert "name:" in query
    assert "qualified_name:" in query
    assert "text:" not in query


def test_lucene_query_ors_conceptual_tokens() -> None:
    query = lucene_query("How does dependency injection work?")
    assert " AND " not in query
    assert " OR " in query
    assert "dependency" in query
    assert "injection" in query


def test_lucene_query_drops_question_boilerplate() -> None:
    query = lucene_query(
        "How does dependency injection work and show me examples from the codebase"
    )
    assert "dependency" in query
    assert "injection" in query
    assert " OR " in query
    assert " AND " not in query
    assert "codebase" not in query.lower()
    assert "How" not in query


def test_lucene_query_requires_foreign_proper_nouns() -> None:
    query = lucene_query("How does Django's ORM lazy-load querysets?")
    assert "Django" in query
    assert " AND " in query
    assert "name:\"Django\"" in query or 'name:"Django"' in query


def test_lucene_query_ors_request_validation_tokens() -> None:
    query = lucene_query("How does FastAPI handle request validation?")
    assert "FastAPI" in query
    assert "request" in query
    assert "validation" in query
    assert " OR " in query


def test_cosine_similarity_is_1_for_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0


def test_retrieve_falls_through_to_fulltext_when_exact_misses() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "db.index.fulltext.queryNodes": [
                {
                    "labels": ["Function"],
                    "name": "get_dependant",
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "file_path": "fastapi/dependencies/utils.py",
                    "path": None,
                    "line_start": 10,
                    "line_end": 80,
                    "score": 4.2,
                }
            ],
        }
    )
    result = GraphQueryService(client).retrieve("dependency injection")
    assert result.result_count == 1
    hit = result.matches[0]
    assert hit.tier == "fulltext"
    assert hit.qualified_name.endswith("get_dependant")
    assert hit.score == 4.2
    assert any("lucene_query" in params for _query, params, _timeout in client.queries)
    lucene = next(
        params["lucene_query"]
        for _query, params, _timeout in client.queries
        if "lucene_query" in params
    )
    assert " OR " in lucene


def test_retrieve_skips_embeddings_when_flag_is_off() -> None:
    client = FakeClient(rows=[])
    result = GraphQueryService(
        client,
        embeddings_enabled=False,
        embedding_provider=_TokenEmbedder(),
    ).retrieve("dependency injection")
    assert result.matches == []
    assert all("vector.queryNodes" not in query for query, _params, _timeout in client.queries)


def test_retrieve_ignores_injected_provider_when_flag_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GQ_EMBEDDINGS_ENABLED", "0")
    client = FakeClient(rows=[])
    GraphQueryService(client, embedding_provider=_TokenEmbedder()).retrieve(
        "dependency injection"
    )
    assert all("vector.queryNodes" not in query for query, _params, _timeout in client.queries)


def test_retrieve_lexical_tier_uses_vector_index() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "db.index.fulltext.queryNodes": [],
            "db.index.vector.queryNodes": [
                {
                    "labels": ["Function"],
                    "name": "get_dependant",
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 1,
                    "line_end": 20,
                    "score": 0.91,
                },
                {
                    "labels": ["Function"],
                    "name": "get_openapi",
                    "qualified_name": "fastapi.openapi.utils.get_openapi",
                    "file_path": "fastapi/openapi/utils.py",
                    "line_start": 1,
                    "line_end": 20,
                    "score": 0.11,
                },
            ],
        }
    )
    result = GraphQueryService(
        client,
        embeddings_enabled=True,
        embedding_provider=_TokenEmbedder(),
        embedding_min_score=DEFAULT_EMBEDDING_MIN_SCORE,
    ).retrieve("how does dependency injection work")
    assert result.matches
    assert result.matches[0].tier == "lexical"
    assert result.matches[0].name == "get_dependant"
    vector_params = [
        params for query, params, _timeout in client.queries if "vector.queryNodes" in query
    ]
    assert vector_params
    assert len(vector_params[0]["query_vector"]) == 2
    assert vector_params[0]["min_score"] == DEFAULT_EMBEDDING_MIN_SCORE


def test_vectors_from_another_backend_disable_the_tier_instead_of_ranking_nonsense() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "db.index.fulltext.queryNodes": [],
            "m.value AS value": [{"value": "openrouter:openai/text-embedding-3-small:256"}],
            "db.index.vector.queryNodes": [
                {
                    "labels": ["Function"],
                    "name": "get_dependant",
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 1,
                    "line_end": 20,
                    "score": 0.91,
                }
            ],
        }
    )
    service = GraphQueryService(
        client,
        embeddings_enabled=True,
        embedding_provider=HashingEmbeddingProvider(),
    )
    result = service.retrieve("how does dependency injection work")

    assert not [hit for hit in result.matches if hit.tier == "lexical"]
    assert not [query for query, _p, _t in client.queries if "vector.queryNodes" in query]


def test_a_matching_backend_leaves_the_tier_enabled() -> None:
    provider = HashingEmbeddingProvider()
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "db.index.fulltext.queryNodes": [],
            "m.value AS value": [{"value": provider.fingerprint}],
            "db.index.vector.queryNodes": [],
        }
    )
    GraphQueryService(
        client,
        embeddings_enabled=True,
        embedding_provider=provider,
    ).retrieve("how does dependency injection work")

    assert [query for query, _p, _t in client.queries if "vector.queryNodes" in query]


def test_the_score_floor_follows_the_configured_backend() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "db.index.fulltext.queryNodes": [],
            "db.index.vector.queryNodes": [],
        }
    )
    GraphQueryService(
        client,
        embeddings_enabled=True,
        embedding_provider=_TokenEmbedder(),
        embedding_min_score=0.7,
    ).retrieve("how does dependency injection work")
    vector_params = [
        params for query, params, _timeout in client.queries if "vector.queryNodes" in query
    ]
    assert vector_params
    assert all(params["min_score"] == 0.7 for params in vector_params)


def test_retrieve_request_validation_and_di_queries_return_fastapi_entities() -> None:
    validation_hit = {
        "labels": ["Class"],
        "name": "RequestValidationError",
        "qualified_name": "fastapi.exceptions.RequestValidationError",
        "file_path": "fastapi/exceptions.py",
        "line_start": 10,
        "line_end": 40,
        "score": 5.0,
    }
    di_hit = {
        "labels": ["Function"],
        "name": "get_dependant",
        "qualified_name": "fastapi.dependencies.utils.get_dependant",
        "file_path": "fastapi/dependencies/utils.py",
        "line_start": 10,
        "line_end": 80,
        "score": 4.2,
    }
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "db.index.fulltext.queryNodes": [validation_hit, di_hit],
            "db.index.vector.queryNodes": [],
        }
    )
    service = GraphQueryService(client)
    validation = service.retrieve("How does FastAPI handle request validation?")
    di = service.retrieve(
        "How does dependency injection work and show me examples from the codebase"
    )
    assert any(hit.qualified_name.endswith("RequestValidationError") for hit in validation.matches)
    assert any(hit.file_path == "fastapi/exceptions.py" for hit in validation.matches)
    assert any(hit.qualified_name.endswith("get_dependant") for hit in di.matches)
    assert any(hit.file_path == "fastapi/dependencies/utils.py" for hit in di.matches)


def test_retrieve_django_and_react_traps_return_no_fastapi_hits() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "db.index.fulltext.queryNodes": [],
            "db.index.vector.queryNodes": [],
        }
    )
    service = GraphQueryService(
        client,
        embeddings_enabled=True,
        embedding_provider=_TokenEmbedder(),
    )
    django = service.retrieve("How does Django's ORM lazy-load querysets?")
    react = service.retrieve("Explain React's useEffect cleanup")
    assert django.matches == []
    assert react.matches == []


def test_retrieve_ranks_package_source_above_tests_for_exact_hits() -> None:
    test_hit = {
        "labels": ["Function"],
        "name": "Depends",
        "qualified_name": "tests.test_dependency_duplicates.Depends",
        "file_path": "tests/test_dependency_duplicates.py",
        "line_start": 10,
        "line_end": 20,
        "score": 1.0,
    }
    package_hit = {
        "labels": ["Function"],
        "name": "Depends",
        "qualified_name": "fastapi.param_functions.Depends",
        "file_path": "fastapi/param_functions.py",
        "line_start": 80,
        "line_end": 120,
        "score": 1.0,
    }
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [test_hit, package_hit],
            "db.index.fulltext.queryNodes": [],
        }
    )
    result = GraphQueryService(client).retrieve("Depends")
    assert result.matches
    assert result.matches[0].file_path == "fastapi/param_functions.py"
    assert any(hit.file_path.startswith("tests/") for hit in result.matches)


def test_retrieve_dependency_injection_prefers_package_source() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [
                {
                    "labels": ["Function"],
                    "name": "Depends",
                    "qualified_name": "tests.test_dependency_duplicates.Depends",
                    "file_path": "tests/test_dependency_duplicates.py",
                    "line_start": 14,
                    "line_end": 25,
                    "score": 1.0,
                },
                {
                    "labels": ["Function"],
                    "name": "Depends",
                    "qualified_name": "fastapi.param_functions.Depends",
                    "file_path": "fastapi/param_functions.py",
                    "line_start": 80,
                    "line_end": 120,
                    "score": 1.0,
                },
                {
                    "labels": ["Function"],
                    "name": "get_dependant",
                    "qualified_name": "fastapi.dependencies.utils.get_dependant",
                    "file_path": "fastapi/dependencies/utils.py",
                    "line_start": 370,
                    "line_end": 430,
                    "score": 1.0,
                },
            ],
            "db.index.fulltext.queryNodes": [],
        }
    )
    result = GraphQueryService(client).retrieve("Depends")
    assert result.matches[0].file_path in {
        "fastapi/dependencies/utils.py",
        "fastapi/param_functions.py",
    }


def _hit(
    *,
    name: str,
    qualified_name: str,
    file_path: str,
    labels: list[str] | None = None,
    score: float = 1.0,
) -> dict[str, object]:
    return {
        "labels": labels or ["Class"],
        "name": name,
        "qualified_name": qualified_name,
        "file_path": file_path,
        "line_start": 1,
        "line_end": 8,
        "score": score,
    }


def test_path_name_affinity_prefers_defining_module() -> None:
    assert path_name_affinity("WebSocket", "fastapi/websockets.py") < path_name_affinity(
        "WebSocket", "fastapi/routing.py"
    )
    assert path_name_affinity("CORSMiddleware", "fastapi/middleware/cors.py") < path_name_affinity(
        "CORSMiddleware", "fastapi/applications.py"
    )
    assert path_name_affinity("TestClient", "fastapi/testclient.py") < path_name_affinity(
        "TestClient", "fastapi/__init__.py"
    )
    assert path_name_affinity("JSONResponse", "fastapi/responses.py") == 0
    assert path_name_affinity("HTMLResponse", "fastapi/responses.py") == 0


def test_retrieval_sort_key_ranks_reexport_above_unrelated_and_tests() -> None:
    local = retrieval_sort_key(
        "exact",
        "fastapi/websockets.py",
        1.0,
        name="WebSocket",
        qualified_name="fastapi.websockets.WebSocket",
    )
    unrelated = retrieval_sort_key(
        "exact",
        "fastapi/routing.py",
        1.0,
        name="WebSocket",
        qualified_name="fastapi.routing.WebSocket",
    )
    test_hit = retrieval_sort_key(
        "exact",
        "tests/test_ws.py",
        9.0,
        name="WebSocket",
        qualified_name="tests.test_ws.WebSocket",
    )
    assert local < unrelated < test_hit


def test_find_entity_maps_docstring_fields() -> None:
    client = FakeClient(
        [
            {
                "labels": ["Function"],
                "name": "Depends",
                "qualified_name": "fastapi.param_functions.Depends",
                "file_path": "fastapi/param_functions.py",
                "path": None,
                "line_start": 80,
                "line_end": 120,
                "docstring_text": "Declare a FastAPI dependency.\n\nIt takes a single callable.",
                "docstring_summary": "Declare a FastAPI dependency.",
            }
        ]
    )
    result = GraphQueryService(client).find_entity("Depends")
    hit = result.matches[0]
    assert hit.docstring_summary == "Declare a FastAPI dependency."
    assert hit.docstring_text is not None
    assert "single callable" in hit.docstring_text
    assert "dependency_overrides_provider" not in hit.docstring_text


def test_get_docstring_maps_rows() -> None:
    client = FakeClient(
        [
            {
                "labels": ["Function"],
                "name": "Depends",
                "qualified_name": "fastapi.param_functions.Depends",
                "file_path": "fastapi/param_functions.py",
                "line_start": 80,
                "line_end": 120,
                "text": "Declare a FastAPI dependency.",
                "summary": "Declare a FastAPI dependency.",
            }
        ]
    )
    result = GraphQueryService(client).get_docstring("Depends")
    assert result.qualified_name == "Depends"
    assert result.result_count == 1
    assert result.truncated is False
    hit = result.matches[0]
    assert hit.qualified_name == "fastapi.param_functions.Depends"
    assert hit.entity_type == "Function"
    assert hit.text == "Declare a FastAPI dependency."
    assert hit.summary == "Declare a FastAPI dependency."
    query, params, _timeout = client.queries[0]
    assert "Depends" not in query
    assert params == {"qualified_name": "Depends"}
    assert "$qualified_name" in query
    assert "DOCUMENTED_BY" in query


def test_get_docstring_ranks_package_source_first() -> None:
    client = FakeClient(
        [
            {
                "labels": ["Function"],
                "name": "Depends",
                "qualified_name": "tests.test_depends.Depends",
                "file_path": "tests/test_depends.py",
                "text": "test double",
                "summary": "test double",
            },
            {
                "labels": ["Function"],
                "name": "Depends",
                "qualified_name": "fastapi.param_functions.Depends",
                "file_path": "fastapi/param_functions.py",
                "text": "Declare a FastAPI dependency.",
                "summary": "Declare a FastAPI dependency.",
            },
        ]
    )
    result = GraphQueryService(client).get_docstring("Depends")
    assert result.matches[0].file_path == "fastapi/param_functions.py"
    assert result.matches[0].text == "Declare a FastAPI dependency."


def test_conceptual_entity_names_covers_dependency_resolution() -> None:
    names = conceptual_entity_names("How does dependency resolution work in the codebase")
    assert names == ["Depends", "get_dependant", "solve_dependencies"]
    assert conceptual_entity_names("What is WebSocket?") == []


def test_conceptual_entity_names_covers_request_lifecycle_and_validation() -> None:
    lifecycle = conceptual_entity_names("Explain the complete lifecycle of a FastAPI request")
    assert lifecycle == [
        "APIRoute.get_request_handler",
        "run_endpoint_function",
        "serialize_response",
    ]
    validation = conceptual_entity_names("How does FastAPI handle request validation?")
    assert validation == [
        "request_params_to_args",
        "request_body_to_args",
        "RequestValidationError",
    ]
    assert conceptual_entity_names("What is WebSocket?") == []


def test_retrieve_request_lifecycle_looks_up_conceptual_entities() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "$name IN i.names": [],
            "db.index.fulltext.queryNodes": [],
        }
    )
    service = GraphQueryService(client)
    service.retrieve("Explain the complete lifecycle of a FastAPI request")
    lifecycle_names = [
        params["name"]
        for query, params, _timeout in client.queries
        if "n.name = $name OR n.qualified_name = $name" in query
    ]
    assert "APIRoute.get_request_handler" in lifecycle_names
    assert "run_endpoint_function" in lifecycle_names
    assert "serialize_response" in lifecycle_names

    client.queries.clear()
    service.retrieve("How does FastAPI handle request validation?")
    validation_names = [
        params["name"]
        for query, params, _timeout in client.queries
        if "n.name = $name OR n.qualified_name = $name" in query
    ]
    assert "request_params_to_args" in validation_names
    assert "request_body_to_args" in validation_names
    assert "RequestValidationError" in validation_names


def test_retrieve_reexport_ranks_websocket_module_above_imports_and_tests() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [
                _hit(
                    name="WebSocket",
                    qualified_name="tests.test_ws.WebSocket",
                    file_path="tests/test_ws.py",
                )
            ],
            "$name IN i.names": [
                _hit(
                    name="WebSocket",
                    qualified_name="fastapi.WebSocket",
                    file_path="fastapi/__init__.py",
                ),
                _hit(
                    name="WebSocket",
                    qualified_name="fastapi.routing.WebSocket",
                    file_path="fastapi/routing.py",
                ),
                _hit(
                    name="WebSocket",
                    qualified_name="fastapi.websockets.WebSocket",
                    file_path="fastapi/websockets.py",
                ),
            ],
            "db.index.fulltext.queryNodes": [],
        }
    )
    result = GraphQueryService(client).retrieve("WebSocket")
    assert result.matches[0].file_path == "fastapi/websockets.py"
    assert result.matches[0].name == "WebSocket"


def test_retrieve_reexport_ranks_corsmiddleware_local_module() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "$name IN i.names": [
                _hit(
                    name="CORSMiddleware",
                    qualified_name="fastapi.CORSMiddleware",
                    file_path="fastapi/__init__.py",
                ),
                _hit(
                    name="CORSMiddleware",
                    qualified_name="fastapi.middleware.cors.CORSMiddleware",
                    file_path="fastapi/middleware/cors.py",
                ),
            ],
            "db.index.fulltext.queryNodes": [],
        }
    )
    result = GraphQueryService(client).retrieve("CORSMiddleware")
    assert result.matches[0].file_path == "fastapi/middleware/cors.py"


def test_retrieve_reexport_ranks_testclient_local_module() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [
                _hit(
                    name="TestClient",
                    qualified_name="tests.test_starlette_testclient.TestClient",
                    file_path="tests/test_starlette_testclient.py",
                )
            ],
            "$name IN i.names": [
                _hit(
                    name="TestClient",
                    qualified_name="fastapi.testclient.TestClient",
                    file_path="fastapi/testclient.py",
                )
            ],
            "db.index.fulltext.queryNodes": [],
        }
    )
    result = GraphQueryService(client).retrieve("TestClient")
    assert result.matches[0].file_path == "fastapi/testclient.py"


def test_retrieve_reexport_ranks_json_and_html_response_module() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "$name IN i.names": [
                _hit(
                    name="JSONResponse",
                    qualified_name="fastapi.responses.JSONResponse",
                    file_path="fastapi/responses.py",
                ),
                _hit(
                    name="HTMLResponse",
                    qualified_name="fastapi.routing.HTMLResponse",
                    file_path="fastapi/routing.py",
                ),
            ],
            "db.index.fulltext.queryNodes": [],
        }
    )
    json_hits = GraphQueryService(client).retrieve("JSONResponse")
    html_client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [],
            "$name IN i.names": [
                _hit(
                    name="HTMLResponse",
                    qualified_name="fastapi.responses.HTMLResponse",
                    file_path="fastapi/responses.py",
                ),
                _hit(
                    name="HTMLResponse",
                    qualified_name="fastapi.HTMLResponse",
                    file_path="fastapi/__init__.py",
                ),
            ],
            "db.index.fulltext.queryNodes": [],
        }
    )
    html_hits = GraphQueryService(html_client).retrieve("HTMLResponse")
    assert json_hits.matches[0].file_path == "fastapi/responses.py"
    assert html_hits.matches[0].file_path == "fastapi/responses.py"


def test_retrieve_dependency_resolution_hits_utils_and_param_functions() -> None:
    client = FakeClient(
        by_query={
            "n.name = $name OR n.qualified_name = $name": [
                _hit(
                    name="Depends",
                    qualified_name="tests.test_dependency_duplicates.Depends",
                    file_path="tests/test_dependency_duplicates.py",
                    labels=["Function"],
                ),
                _hit(
                    name="Depends",
                    qualified_name="fastapi.param_functions.Depends",
                    file_path="fastapi/param_functions.py",
                    labels=["Function"],
                ),
                _hit(
                    name="get_dependant",
                    qualified_name="fastapi.dependencies.utils.get_dependant",
                    file_path="fastapi/dependencies/utils.py",
                    labels=["Function"],
                ),
                _hit(
                    name="solve_dependencies",
                    qualified_name="fastapi.dependencies.utils.solve_dependencies",
                    file_path="fastapi/dependencies/utils.py",
                    labels=["Function"],
                ),
            ],
            "db.index.fulltext.queryNodes": [],
        }
    )
    result = GraphQueryService(client).retrieve(
        "How does dependency resolution work in the codebase"
    )
    paths = {hit.file_path for hit in result.matches}
    assert "fastapi/dependencies/utils.py" in paths
    assert "fastapi/param_functions.py" in paths
    assert result.matches[0].file_path in {
        "fastapi/dependencies/utils.py",
        "fastapi/param_functions.py",
    }


class _TokenEmbedder:
    """Two-d embedder: axis 0 is DI language, axis 1 is OpenAPI language."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if ("dependency" in lowered or "injection" in lowered) else 0.0,
                    1.0 if "openapi" in lowered else 0.0,
                ]
            )
        return vectors


def test_hashing_embedder_ranks_dependency_injection_above_openapi() -> None:
    provider = HashingEmbeddingProvider()
    query = "How does dependency injection work and show me examples from the codebase"
    di_doc = (
        "fastapi.dependencies.utils.get_dependant(call)\n"
        "get_dependant\n"
        "Build a Dependant from a callable for dependency injection.\n"
        "def get_dependant(call: Callable[..., Any]) -> Dependant:"
    )
    openapi_doc = (
        "fastapi.openapi.utils.get_openapi(title)\n"
        "get_openapi\n"
        "Generate an OpenAPI schema.\n"
        "def get_openapi(title: str) -> dict[str, Any]:"
    )
    django_doc = "django.db.models.query.QuerySet\nlazy-load querysets in the Django ORM."
    query_tokens = set(tokenize_for_embedding(query))
    di_tokens = set(tokenize_for_embedding(di_doc))
    django_tokens = set(tokenize_for_embedding(django_doc))
    assert query_tokens & di_tokens
    assert len(query_tokens & di_tokens) > len(query_tokens & django_tokens)
    vectors = provider.embed([query, di_doc, openapi_doc, django_doc])
    di_score = cosine_similarity(vectors[0], vectors[1])
    openapi_score = cosine_similarity(vectors[0], vectors[2])
    assert di_score > openapi_score
    assert vectors[0] == provider.embed([query])[0]

