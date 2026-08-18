"""Template tests: user values travel as Cypher parameters, never f-strings."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.querying import patterns, templates
from core.querying.service import GraphQueryService
from core.querying.templates import (
    FIND_ENTITY,
    FIND_RELATED,
    GET_DEPENDENCIES,
    GET_DEPENDENTS,
    TRACE_IMPORTS,
)


class CapturingClient:
    """Records the Cypher and params passed to run_read."""

    def __init__(self) -> None:
        self.query: str = ""
        self.params: dict[str, Any] = {}
        self.timeout_s: float | None = None

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        self.query = query
        self.params = dict(params or {})
        self.timeout_s = timeout_s
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
    assert "$name" in GET_DEPENDENCIES
    assert "$name" in GET_DEPENDENTS
    assert "$module" in TRACE_IMPORTS
    assert "$depth" in TRACE_IMPORTS
    assert "$relationship_type" in FIND_RELATED
    assert "IMPORTS|DEPENDS_ON|CALLS" in GET_DEPENDENCIES
    assert "IMPORTS|DEPENDS_ON|CALLS" in GET_DEPENDENTS


def test_find_entity_passes_name_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).find_entity("get_openapi")
    assert "get_openapi" not in client.query
    assert client.params == {"name": "get_openapi", "entity_type": None}
    assert "$name" in client.query


def test_find_entity_passes_entity_type_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).find_entity("FastAPI", entity_type="Class")
    assert "FastAPI" not in client.query
    assert client.params == {"name": "FastAPI", "entity_type": "Class"}


def test_get_dependencies_passes_name_as_parameter() -> None:
    client = CapturingClient()
    GraphQueryService(client).get_dependencies("fastapi.applications")
    assert "fastapi.applications" not in client.query
    assert client.params == {"name": "fastapi.applications"}


def test_get_dependents_passes_name_as_parameter() -> None:
    client = CapturingClient()
    qn = "fastapi.openapi.utils.get_openapi"
    GraphQueryService(client).get_dependents(qn)
    assert qn not in client.query
    assert client.params == {"name": qn}


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
