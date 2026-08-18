"""Unit tests for GraphQueryService with a fake graph client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from core.graph.schema import FIND_ENTITY_LABELS, RELATIONSHIP_TYPES
from core.querying.safety import QueryRejected
from core.querying.service import (
    DEFAULT_RESULT_LIMIT,
    READ_TIMEOUT_S,
    STATISTICS_TIMEOUT_S,
    GraphQueryService,
)


class FakeClient:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.queries: list[tuple[str, dict[str, Any], float]] = []

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        self.queries.append((query, dict(params or {}), timeout_s))
        if "UNWIND labels(n)" in query and "count(*) AS count" in query:
            return [{"label": "Module", "count": 2}, {"label": "Meta", "count": 1}]
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
    hit = result.neighbors[0]
    assert hit.relationship_type == "DEPENDS_ON"
    assert hit.direction == "outgoing"
    assert hit.qualified_name == "fastapi.routing"


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
