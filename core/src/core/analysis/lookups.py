"""Read-only Cypher used by CodeAnalystService via the injected graph_lookup."""

from __future__ import annotations

from textwrap import dedent

FUNCTION_CONTEXT = dedent(
    """\
    MATCH (n)
    WHERE (n:Function OR n:Method) AND (n.name = $name OR n.qualified_name = $name)
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
    """
).strip()

CLASS_CONTEXT = dedent(
    """\
    MATCH (n:Class)
    WHERE n.name = $name OR n.qualified_name = $name
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
    """
).strip()

ENTITY_LOCATION = dedent(
    """\
    MATCH (n)
    WHERE n.name = $name OR n.qualified_name = $name
    RETURN n.qualified_name AS qualified_name,
           n.name AS name,
           labels(n) AS labels,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end
    """
).strip()
