"""Table-driven tests for guard_readonly write-clause detection."""

from __future__ import annotations

import pytest

from core.querying.safety import QueryRejected, guard_readonly, has_limit


@pytest.mark.parametrize(
    ("cypher", "clause"),
    [
        ("CREATE (n:X)", "CREATE"),
        ("create (n:X)", "CREATE"),
        ("MATCH (n) MERGE (m:Y {id: n.id})", "MERGE"),
        ("MATCH (n) DELETE n", "DELETE"),
        ("MATCH (n) DETACH DELETE n", "DETACH"),
        ("MATCH (n) SET n.x = 1", "SET"),
        ("MATCH (n) REMOVE n.x", "REMOVE"),
        ("DROP CONSTRAINT module_qualified_name", "DROP"),
        ("CALL db.labels()", "CALL db.*"),
        ("CALL db.index.fulltext.createNodeIndex('x', ['Y'], ['z'])", "CALL db.*"),
        ("LOAD CSV FROM 'file:///x.csv' AS row RETURN row", "LOAD CSV"),
        ("load csv FROM 'file:///x.csv' AS row RETURN row", "LOAD CSV"),
        ("MATCH (n) /* comment */ DELETE n", "DELETE"),
        ("MATCH (n)\n// innocuous\nDELETE n", "DELETE"),
        ("MATCH (n) DEL/*hidden*/ETE n", "DELETE"),
        ("MATCH (n) CRE/*x*/ATE (m)", "CREATE"),
    ],
)
def test_guard_readonly_rejects_write_clauses(cypher: str, clause: str) -> None:
    with pytest.raises(QueryRejected) as exc_info:
        guard_readonly(cypher)
    assert exc_info.value.clause == clause
    assert clause in str(exc_info.value)


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) RETURN n",
        "MATCH (n) WHERE n.name = 'CREATE' RETURN n",
        'MATCH (n) WHERE n.name = "MERGE" RETURN n',
        "MATCH (n) WHERE n.name = 'DELETE' RETURN n",
        "MATCH (n) RETURN n.merged AS merged",
        "MATCH (n) WHERE n.state = 'merged' RETURN n",
        "MATCH (n) RETURN n // DELETE",
        "MATCH (n) RETURN 'LOAD CSV' AS hint",
        "CALL { MATCH (n) RETURN n } RETURN n",
        "CALL db.index.fulltext.queryNodes($index_name, $lucene_query) "
        "YIELD node, score RETURN node",
        "CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) "
        "YIELD node, score RETURN node",
        "MATCH (n) RETURN n LIMIT 10",
        "RETURN 1",
    ],
)
def test_guard_readonly_allows_reads(cypher: str) -> None:
    assert guard_readonly(cypher) == cypher


def test_query_rejected_exposes_clause() -> None:
    err = QueryRejected("SET")
    assert err.clause == "SET"
    assert "SET" in str(err)


def test_has_limit_ignores_limit_inside_strings_and_comments() -> None:
    assert has_limit("MATCH (n) RETURN n LIMIT 10") is True
    assert has_limit("MATCH (n) RETURN n") is False
    assert has_limit("MATCH (n) RETURN 'LIMIT' AS x") is False
    assert has_limit("MATCH (n) RETURN n // LIMIT 5") is False


def test_guard_readonly_rejects_earliest_write_clause() -> None:
    with pytest.raises(QueryRejected) as exc_info:
        guard_readonly("MATCH (n) SET n.x = 1 DELETE n")

    assert exc_info.value.clause == "SET"


def test_guard_readonly_still_rejects_other_db_calls_beside_fulltext_query() -> None:
    with pytest.raises(QueryRejected) as exc_info:
        guard_readonly(
            "CALL db.index.fulltext.queryNodes($index_name, $lucene_query) "
            "YIELD node CALL db.labels() YIELD label RETURN label"
        )
    assert exc_info.value.clause == "CALL db.*"


def test_guard_readonly_ignores_write_words_inside_identifiers() -> None:
    cypher = "MATCH (n) RETURN n.createdAt AS created_at, n.deletedFlag AS deleted_flag"
    assert guard_readonly(cypher) == cypher
