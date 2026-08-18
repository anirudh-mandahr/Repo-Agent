"""Integration: find_entity('get_openapi') against an indexed Neo4j graph."""

from __future__ import annotations

import os

import pytest

from core.graph.client import GraphClient
from core.querying.service import GraphQueryService


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI not set")
def test_find_entity_get_openapi() -> None:
    with GraphClient() as client:
        client.verify_connectivity()
        service = GraphQueryService(client)
        result = service.find_entity("get_openapi")
    assert result.error is None
    assert result.result_count >= 1
    assert any(
        hit.qualified_name == "get_openapi" or hit.qualified_name.endswith(".get_openapi")
        for hit in result.matches
    )
    hit = next(
        item
        for item in result.matches
        if item.qualified_name.endswith("get_openapi")
    )
    assert hit.file_path
    assert hit.line_start is not None
    assert hit.line_end is not None
    assert hit.entity_type in {"Function", "Method"}
