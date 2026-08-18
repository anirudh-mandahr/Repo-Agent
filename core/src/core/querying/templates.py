"""Parameterized Cypher templates. User input is never interpolated into Cypher."""

from __future__ import annotations

from textwrap import dedent

DEFAULT_TRACE_DEPTH = 5

FIND_ENTITY = dedent(
    """\
    MATCH (n)
    WHERE (
            (n:Module OR n:Class OR n:Function OR n:Method)
            AND (n.name = $name OR n.qualified_name = $name)
          )
       OR (n:File AND n.path = $name)
    WITH n, labels(n) AS labels
    WHERE $entity_type IS NULL OR $entity_type IN labels
    RETURN labels AS labels,
           n.name AS name,
           n.qualified_name AS qualified_name,
           n.file_path AS file_path,
           n.path AS path,
           n.line_start AS line_start,
           n.line_end AS line_end
    """
).strip()

GET_DEPENDENCIES = dedent(
    """\
    MATCH (n)
    WHERE n.name = $name
       OR n.qualified_name = $name
       OR (n:File AND n.path = $name)
    MATCH (n)-[r:IMPORTS|DEPENDS_ON|CALLS]->(m)
    RETURN type(r) AS relationship_type,
           'outgoing' AS direction,
           labels(m) AS labels,
           m.name AS name,
           m.qualified_name AS qualified_name,
           m.module AS module,
           m.path AS path,
           m.file_path AS file_path
    """
).strip()

GET_DEPENDENTS = dedent(
    """\
    MATCH (n)
    WHERE n.name = $name
       OR n.qualified_name = $name
       OR (n:File AND n.path = $name)
    MATCH (m)-[r:IMPORTS|DEPENDS_ON|CALLS]->(n)
    RETURN type(r) AS relationship_type,
           'incoming' AS direction,
           labels(m) AS labels,
           m.name AS name,
           m.qualified_name AS qualified_name,
           m.module AS module,
           m.path AS path,
           m.file_path AS file_path
    """
).strip()

TRACE_IMPORTS = dedent(
    """\
    MATCH (src:Module)
    WHERE src.name = $module OR src.qualified_name = $module
    MATCH path = (src)-[:IMPORTS|DEPENDS_ON*1..5]->(dst)
    WHERE length(path) <= $depth
    RETURN [n IN nodes(path) | coalesce(n.qualified_name, n.module, n.name, n.path)] AS nodes
    """
).strip()

FIND_RELATED = dedent(
    """\
    MATCH (n)
    WHERE n.name = $name
       OR n.qualified_name = $name
       OR (n:File AND n.path = $name)
    MATCH (n)-[out]->(dst)
    WHERE type(out) = $relationship_type
    RETURN 'outgoing' AS direction,
           type(out) AS relationship_type,
           labels(dst) AS labels,
           dst.name AS name,
           dst.qualified_name AS qualified_name,
           dst.module AS module,
           dst.path AS path,
           dst.file_path AS file_path
    UNION
    MATCH (n)
    WHERE n.name = $name
       OR n.qualified_name = $name
       OR (n:File AND n.path = $name)
    MATCH (src)-[inc]->(n)
    WHERE type(inc) = $relationship_type
    RETURN 'incoming' AS direction,
           type(inc) AS relationship_type,
           labels(src) AS labels,
           src.name AS name,
           src.qualified_name AS qualified_name,
           src.module AS module,
           src.path AS path,
           src.file_path AS file_path
    """
).strip()
