"""Schema statement unit tests and an optional Neo4j idempotency check."""

from __future__ import annotations

import os

import pytest

from core.graph.client import GraphClient
from core.graph.schema import (
    CONSTRAINT_STATEMENTS,
    INDEX_STATEMENTS,
    RELATIONSHIP_TYPES,
    SCHEMA_STATEMENTS,
    ensure_schema,
)


def test_schema_statements_are_idempotent_and_cover_keys() -> None:
    joined = "\n".join(SCHEMA_STATEMENTS)
    for statement in SCHEMA_STATEMENTS:
        assert "IF NOT EXISTS" in statement
    assert "FOR (n:Module) REQUIRE n.qualified_name IS UNIQUE" in joined
    assert "FOR (n:Class) REQUIRE n.qualified_name IS UNIQUE" in joined
    assert "FOR (n:Function) REQUIRE n.qualified_name IS UNIQUE" in joined
    assert "FOR (n:Method) REQUIRE n.qualified_name IS UNIQUE" in joined
    assert "FOR (n:File) REQUIRE n.path IS UNIQUE" in joined
    assert "FOR (n:Function) ON (n.name)" in joined
    assert "FOR (n:Class) ON (n.name)" in joined
    assert "FOR (n:Method) ON (n.name)" in joined
    assert "FOR (n:Decorator) ON (n.name)" in joined
    assert len(CONSTRAINT_STATEMENTS) == 5
    assert len(INDEX_STATEMENTS) == 4
    assert "DEFINES" not in joined
    assert "INHERITS_FROM" in RELATIONSHIP_TYPES
    assert "HAS_PARAMETER" in RELATIONSHIP_TYPES
    assert "DECORATED_BY" in RELATIONSHIP_TYPES
    assert "DOCUMENTED_BY" in RELATIONSHIP_TYPES
    assert "DEPENDS_ON" in RELATIONSHIP_TYPES


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI not set")
def test_ensure_schema_is_idempotent() -> None:
    with GraphClient() as client:
        client.verify_connectivity()
        ensure_schema(client)
        ensure_schema(client)
        constraints = client.run_read(
            "SHOW CONSTRAINTS YIELD labelsOrTypes, properties "
            "RETURN labelsOrTypes, properties"
        )
        by_label = {
            labels[0]: set(properties)
            for item in constraints
            if (labels := item["labelsOrTypes"]) and (properties := item["properties"])
        }
        assert by_label["Module"] == {"qualified_name"}
        assert by_label["Class"] == {"qualified_name"}
        assert by_label["Function"] == {"qualified_name"}
        assert by_label["Method"] == {"qualified_name"}
        assert by_label["File"] == {"path"}

        indexes = client.run_read(
            "SHOW INDEXES YIELD labelsOrTypes, properties "
            "RETURN labelsOrTypes, properties"
        )
        name_indexes = {
            labels[0]
            for item in indexes
            if (labels := item["labelsOrTypes"]) and item["properties"] == ["name"]
        }
        assert {"Function", "Class", "Method", "Decorator"} <= name_indexes
