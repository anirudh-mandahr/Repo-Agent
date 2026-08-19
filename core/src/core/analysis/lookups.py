"""Read-only Cypher used by CodeAnalystService via the injected graph_lookup."""

from __future__ import annotations

from textwrap import dedent

# Keep the exact ``n.name = $name OR n.qualified_name = $name`` predicate so
# unit tests can pin it. The extra clause resolves re-export coordinates such
# as ``fastapi.FastAPI`` to ``fastapi.applications.FastAPI``.
_NAME_OR_QN = "n.name = $name OR n.qualified_name = $name"
_REEXPORT_ALIAS = (
    "($name CONTAINS '.' AND n.name = last(split($name, '.')) "
    "AND n.qualified_name STARTS WITH head(split($name, '.')) + '.')"
)
_LOOKUP_PREDICATE = f"{_NAME_OR_QN} OR {_REEXPORT_ALIAS}"
_LOOKUP_ORDER = (
    "CASE WHEN n.qualified_name = $name THEN 0 "
    "WHEN n.name = $name THEN 1 ELSE 2 END, "
    "size(coalesce(n.qualified_name, ''))"
)


def _cypher(template: str) -> str:
    """Fill lookup predicate/order placeholders without Cypher-map f-strings."""
    return (
        dedent(template)
        .strip()
        .replace("__LOOKUP__", _LOOKUP_PREDICATE)
        .replace("__ORDER__", _LOOKUP_ORDER)
    )


FUNCTION_CONTEXT = _cypher(
    """\
    MATCH (n)
    WHERE (n:Function OR n:Method) AND (__LOOKUP__)
    OPTIONAL MATCH (n)-[:HAS_PARAMETER]->(p:Parameter)
    WITH n, collect(DISTINCT p { .name, .annotation, .default, .position }) AS parameters
    OPTIONAL MATCH (n)-[:DECORATED_BY]->(d:Decorator)
    WITH n, parameters, collect(DISTINCT d.name) AS decorators
    OPTIONAL MATCH (mod:Module)-[:CONTAINS]->(n)
    WITH n, parameters, decorators, mod
    OPTIONAL MATCH (cls:Class)-[:CONTAINS]->(n)
    WITH n, parameters, decorators, mod, cls
    OPTIONAL MATCH (dep)-[:CALLS|DEPENDS_ON|IMPORTS]->(n)
    RETURN n.qualified_name AS qualified_name,
           n.name AS name,
           labels(n) AS labels,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end,
           parameters,
           decorators,
           mod.qualified_name AS module,
           cls.qualified_name AS class_name,
           collect(DISTINCT coalesce(dep.qualified_name, dep.name)) AS dependents
    ORDER BY __ORDER__
    """
)

CLASS_CONTEXT = _cypher(
    """\
    MATCH (n:Class)
    WHERE __LOOKUP__
    OPTIONAL MATCH (n)-[:CONTAINS]->(m:Method)
    WITH n, collect(DISTINCT m.qualified_name) AS methods
    OPTIONAL MATCH (n)-[:INHERITS_FROM]->(b)
    WITH n, methods, collect(DISTINCT coalesce(b.qualified_name, b.name)) AS inherited_from
    OPTIONAL MATCH (n)-[:DECORATED_BY]->(d:Decorator)
    WITH n, methods, inherited_from, collect(DISTINCT d.name) AS decorators
    OPTIONAL MATCH (mod:Module)-[:CONTAINS]->(n)
    RETURN n.qualified_name AS qualified_name,
           n.name AS name,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end,
           n.bases AS bases,
           methods,
           inherited_from,
           decorators,
           mod.qualified_name AS module
    ORDER BY __ORDER__
    """
)

ENTITY_LOCATION = _cypher(
    """\
    MATCH (n)
    WHERE __LOOKUP__
    RETURN n.qualified_name AS qualified_name,
           n.name AS name,
           labels(n) AS labels,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end
    ORDER BY __ORDER__
    """
)
