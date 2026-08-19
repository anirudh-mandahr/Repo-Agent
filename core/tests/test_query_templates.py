"""Template tests: user values travel as Cypher parameters, never f-strings."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.querying import patterns, templates
from core.querying.service import GraphQueryService
from core.querying.templates import (
    DEFAULT_NEIGHBOR_BRANCH_LIMIT,
    FIND_ENTITY,
    FIND_FULLTEXT,
    FIND_IMPORTED_NAME,
    FIND_RELATED,
    GET_DEPENDENCIES,
    GET_DEPENDENTS,
    GET_DOCSTRING,
    TRACE_IMPORTS,
    VECTOR_SEARCH,
)


class CapturingClient:
    """Records the Cypher and params passed to run_read."""

    def __init__(self) -> None:
        self.query: str = ""
        self.params: dict[str, Any] = {}
        self.timeout_s: float | None = None
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        self.query = query
        self.params = dict(params or {})
        self.timeout_s = timeout_s
        self.queries.append((query, dict(params or {})))
        return []


def _assert_no_fstrings(source: str) -> None:
    tree = ast.parse(source)
    joined = [node for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)]
    assert joined == []


def test_templates_module_has_no_fstrings() -> None:
    _assert_no_fstrings(Path(templates.__file__).read_text())


def test_patterns_module_has_no_fstrings() -> None:
    _assert_no_fstrings(Path(patterns.__file__).read_text())


def test_templates_use_parameter_placeholders() -> None:
    assert "$name" in FIND_ENTITY
    assert "$entity_type" in FIND_ENTITY
    assert "DOCUMENTED_BY" in FIND_ENTITY
    assert "d.text AS docstring_text" in FIND_ENTITY
    assert "d.summary AS docstring_summary" in FIND_ENTITY
    assert "$qualified_name" in GET_DOCSTRING
    assert "DOCUMENTED_BY" in GET_DOCSTRING
    assert "d.text AS text" in GET_DOCSTRING
    assert "d.summary AS summary" in GET_DOCSTRING
    assert "$name" in FIND_IMPORTED_NAME
    assert "$name IN i.names" in FIND_IMPORTED_NAME
    assert "i.alias = $name" in FIND_IMPORTED_NAME
    assert "target.qualified_name" in FIND_IMPORTED_NAME
    assert "m.qualified_name + '.' + $name" not in FIND_IMPORTED_NAME
    assert "$index_name" in FIND_FULLTEXT
    assert "$lucene_query" in FIND_FULLTEXT
    assert "$top_k" in FIND_FULLTEXT
    assert "source_rank" in FIND_FULLTEXT
    assert "d.text AS docstring_text" in FIND_FULLTEXT
    assert "d.summary AS docstring_summary" in FIND_FULLTEXT
    assert "$query_vector" in VECTOR_SEARCH
    assert "$min_score" in VECTOR_SEARCH
    assert "$index_name" in VECTOR_SEARCH
    assert "d.text AS docstring_text" in VECTOR_SEARCH
    assert "d.summary AS docstring_summary" in VECTOR_SEARCH
    assert "$name" in GET_DEPENDENCIES
    assert "$name" in GET_DEPENDENTS
    assert "$branch_limit" in GET_DEPENDENCIES
    assert "$branch_limit" in GET_DEPENDENTS
    assert GET_DEPENDENCIES.count("LIMIT $branch_limit") == 3
    assert GET_DEPENDENTS.count("LIMIT $branch_limit") == 3
    assert "$module" in TRACE_IMPORTS
    assert "$depth" in TRACE_IMPORTS
    assert "$relationship_type" in FIND_RELATED
    assert "IMPORTS|DEPENDS_ON|CALLS" in GET_DEPENDENCIES
    assert "IMPORTS|DEPENDS_ON|CALLS" in GET_DEPENDENTS
    assert "Class->Module" in GET_DEPENDENCIES
    assert "Class->Method" in GET_DEPENDENCIES
    assert "Class->Module" in GET_DEPENDENTS
    assert "Class->Method" in GET_DEPENDENTS
    assert "Class->Module" in TRACE_IMPORTS


def test_find_entity_passes_name_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).find_entity("get_openapi")
    exact_query, exact_params = client.queries[0]
    assert "get_openapi" not in exact_query
    assert exact_params == {"name": "get_openapi", "entity_type": None}
    assert "$name" in exact_query
    imported = next(params for query, params in client.queries if "$name IN i.names" in query)
    assert imported == {"name": "get_openapi", "entity_type": None}
    fulltext_query, fulltext_params = next(
        (query, params)
        for query, params in client.queries
        if "lucene_query" in params
    )
    assert "get_openapi" not in fulltext_query
    assert fulltext_params["lucene_query"]
    assert "$lucene_query" in fulltext_query
    assert "$index_name" in fulltext_query


def test_find_entity_passes_entity_type_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).find_entity("FastAPI", entity_type="Class")
    assert "FastAPI" not in client.queries[0][0]
    assert client.queries[0][1] == {"name": "FastAPI", "entity_type": "Class"}
    assert any(params.get("entity_type") == "Class" for _query, params in client.queries[1:])


def test_get_docstring_passes_qualified_name_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).get_docstring("fastapi.param_functions.Depends")
    assert "fastapi.param_functions.Depends" not in client.query
    assert client.params == {"qualified_name": "fastapi.param_functions.Depends"}
    assert "$qualified_name" in client.query
    assert "DOCUMENTED_BY" in client.query
    assert "d.text AS text" in client.query


def test_get_dependencies_passes_name_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).get_dependencies("fastapi.applications")
    limited = next(
        params
        for query, params in client.queries
        if "$branch_limit" in query or "branch_limit" in params
    )
    assert limited["name"] == "fastapi.applications"
    assert limited["branch_limit"] == DEFAULT_NEIGHBOR_BRANCH_LIMIT
    for query, _params in client.queries:
        assert "fastapi.applications" not in query


def test_get_dependents_passes_name_as_parameter() -> None:
    client = CapturingClient()
    qn = "fastapi.openapi.utils.get_openapi"
    GraphQueryService(client).get_dependents(qn)
    limited = next(params for query, params in client.queries if "branch_limit" in params)
    assert limited["name"] == qn
    assert limited["branch_limit"] == DEFAULT_NEIGHBOR_BRANCH_LIMIT
    for query, _params in client.queries:
        assert qn not in query


def test_trace_imports_passes_module_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).trace_imports("fastapi.applications")
    assert "fastapi.applications" not in client.query
    assert client.params == {"module": "fastapi.applications", "depth": 5}
    assert "$module" in client.query


def test_find_related_passes_relationship_type_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).find_related("FastAPI", "INHERITS_FROM")
    assert "FastAPI" not in client.query
    assert "INHERITS_FROM" not in client.query or "$relationship_type" in client.query
    assert client.params == {"name": "FastAPI", "relationship_type": "INHERITS_FROM"}
