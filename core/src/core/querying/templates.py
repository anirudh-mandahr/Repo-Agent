"""Parameterized Cypher templates. User input is never interpolated into Cypher."""

from __future__ import annotations

from textwrap import dedent

DEFAULT_TRACE_DEPTH = 5
DEFAULT_NEIGHBOR_BRANCH_LIMIT = 20

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
    OPTIONAL MATCH (n)-[:DOCUMENTED_BY]->(d:Docstring)
    RETURN labels AS labels,
           n.name AS name,
           n.qualified_name AS qualified_name,
           n.file_path AS file_path,
           n.path AS path,
           n.line_start AS line_start,
           n.line_end AS line_end,
           d.text AS docstring_text,
           d.summary AS docstring_summary
    """
).strip()

GET_DOCSTRING = dedent(
    """\
    MATCH (n)
    WHERE (n:Module OR n:Class OR n:Function OR n:Method)
      AND (n.qualified_name = $qualified_name OR n.name = $qualified_name)
    OPTIONAL MATCH (n)-[:DOCUMENTED_BY]->(d:Docstring)
    RETURN labels(n) AS labels,
           n.name AS name,
           n.qualified_name AS qualified_name,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end,
           d.text AS text,
           d.summary AS summary
    """
).strip()

FIND_IMPORTED_NAME = dedent(
    """\
    MATCH (m:Module)-[:IMPORTS]->(i:Import)
    WHERE $name IN i.names OR i.alias = $name
    OPTIONAL MATCH (target)
    WHERE (target:Module OR target:Class OR target:Function OR target:Method)
      AND target.name = $name
      AND (
            target.qualified_name = i.module + '.' + $name
            OR (
                i.module STARTS WITH '.'
                AND size(i.module) > 1
                AND target.qualified_name ENDS WITH ltrim(i.module, '.') + '.' + $name
            )
          )
    WITH collect(DISTINCT target) AS resolved, collect(DISTINCT m) AS importers
    WITH resolved,
         importers,
         [mod IN importers WHERE mod.file_path STARTS WITH 'fastapi/'] AS local_importers
    WITH size(resolved) > 0 AS exact,
         CASE
           WHEN size(resolved) > 0 THEN resolved
           WHEN size(local_importers) > 0 THEN local_importers
           ELSE importers
         END AS hits
    UNWIND hits AS hit
    WITH DISTINCT hit, exact
    WHERE $entity_type IS NULL OR $entity_type IN labels(hit)
    RETURN labels(hit) AS labels,
           hit.name AS name,
           hit.qualified_name AS qualified_name,
           hit.file_path AS file_path,
           null AS path,
           hit.line_start AS line_start,
           hit.line_end AS line_end,
           CASE WHEN exact THEN 1.0 ELSE 0.6 END AS score
    ORDER BY score DESC, size(coalesce(hit.file_path, ''))
    """
).strip()

GET_DEPENDENCIES_COUNT = dedent(
    """\
    CALL {
        MATCH (n)
        WHERE n.name = $name
           OR n.qualified_name = $name
           OR (n:File AND n.path = $name)
        MATCH (n)-[r:IMPORTS|DEPENDS_ON|CALLS]->(m)
        RETURN count(*) AS c
        UNION ALL
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (mod:Module)-[:CONTAINS]->(n)
        MATCH (mod)-[r:IMPORTS|DEPENDS_ON]->(m)
        RETURN count(*) AS c
        UNION ALL
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (n)-[:CONTAINS]->(meth:Method)
        MATCH (meth)-[r:CALLS]->(m)
        RETURN count(*) AS c
    }
    RETURN sum(c) AS total
    """
).strip()

GET_DEPENDENCIES = dedent(
    """\
    CALL {
        MATCH (n)
        WHERE n.name = $name
           OR n.qualified_name = $name
           OR (n:File AND n.path = $name)
        MATCH (n)-[r:IMPORTS|DEPENDS_ON|CALLS]->(m)
        WITH type(r) AS relationship_type,
             'outgoing' AS direction,
             labels(m) AS labels,
             m.name AS name,
             m.qualified_name AS qualified_name,
             m.module AS module,
             m.path AS path,
             m.file_path AS file_path,
             null AS hop,
             COUNT { (m)<-[:IMPORTS|DEPENDS_ON|CALLS]-() } AS in_degree
        ORDER BY in_degree DESC, coalesce(qualified_name, name)
        RETURN relationship_type, direction, labels, name, qualified_name,
               module, path, file_path, hop
        LIMIT $branch_limit
        UNION
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (mod:Module)-[:CONTAINS]->(n)
        MATCH (mod)-[r:IMPORTS|DEPENDS_ON]->(m)
        WITH type(r) AS relationship_type,
             'outgoing' AS direction,
             labels(m) AS labels,
             m.name AS name,
             m.qualified_name AS qualified_name,
             m.module AS module,
             m.path AS path,
             m.file_path AS file_path,
             'Class->Module' AS hop,
             COUNT { (m)<-[:IMPORTS|DEPENDS_ON|CALLS]-() } AS in_degree
        ORDER BY in_degree DESC, coalesce(qualified_name, name)
        RETURN relationship_type, direction, labels, name, qualified_name,
               module, path, file_path, hop
        LIMIT $branch_limit
        UNION
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (n)-[:CONTAINS]->(meth:Method)
        MATCH (meth)-[r:CALLS]->(m)
        WITH type(r) AS relationship_type,
             'outgoing' AS direction,
             labels(m) AS labels,
             m.name AS name,
             m.qualified_name AS qualified_name,
             m.module AS module,
             m.path AS path,
             m.file_path AS file_path,
             'Class->Method' AS hop,
             COUNT { (m)<-[:IMPORTS|DEPENDS_ON|CALLS]-() } AS in_degree
        ORDER BY in_degree DESC, coalesce(qualified_name, name)
        RETURN relationship_type, direction, labels, name, qualified_name,
               module, path, file_path, hop
        LIMIT $branch_limit
    }
    RETURN relationship_type, direction, labels, name, qualified_name, module, path, file_path, hop
    ORDER BY CASE
               WHEN hop IS NULL THEN 0
               WHEN hop = 'Class->Module' THEN 1
               ELSE 2
             END,
             name
    """
).strip()

GET_DEPENDENTS_COUNT = dedent(
    """\
    CALL {
        MATCH (n)
        WHERE n.name = $name
           OR n.qualified_name = $name
           OR (n:File AND n.path = $name)
        MATCH (m)-[r:IMPORTS|DEPENDS_ON|CALLS]->(n)
        RETURN count(*) AS c
        UNION ALL
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (mod:Module)-[:CONTAINS]->(n)
        MATCH (m)-[r:IMPORTS|DEPENDS_ON]->(mod)
        RETURN count(*) AS c
        UNION ALL
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (n)-[:CONTAINS]->(meth:Method)
        MATCH (m)-[r:CALLS]->(meth)
        RETURN count(*) AS c
    }
    RETURN sum(c) AS total
    """
).strip()

GET_DEPENDENTS = dedent(
    """\
    CALL {
        MATCH (n)
        WHERE n.name = $name
           OR n.qualified_name = $name
           OR (n:File AND n.path = $name)
        MATCH (m)-[r:IMPORTS|DEPENDS_ON|CALLS]->(n)
        WITH type(r) AS relationship_type,
             'incoming' AS direction,
             labels(m) AS labels,
             m.name AS name,
             m.qualified_name AS qualified_name,
             m.module AS module,
             m.path AS path,
             m.file_path AS file_path,
             null AS hop,
             COUNT { (m)<-[:IMPORTS|DEPENDS_ON|CALLS]-() } AS in_degree
        ORDER BY in_degree DESC, coalesce(qualified_name, name)
        RETURN relationship_type, direction, labels, name, qualified_name,
               module, path, file_path, hop
        LIMIT $branch_limit
        UNION
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (mod:Module)-[:CONTAINS]->(n)
        MATCH (m)-[r:IMPORTS|DEPENDS_ON]->(mod)
        WITH type(r) AS relationship_type,
             'incoming' AS direction,
             labels(m) AS labels,
             m.name AS name,
             m.qualified_name AS qualified_name,
             m.module AS module,
             m.path AS path,
             m.file_path AS file_path,
             'Class->Module' AS hop,
             COUNT { (m)<-[:IMPORTS|DEPENDS_ON|CALLS]-() } AS in_degree
        ORDER BY in_degree DESC, coalesce(qualified_name, name)
        RETURN relationship_type, direction, labels, name, qualified_name,
               module, path, file_path, hop
        LIMIT $branch_limit
        UNION
        MATCH (n:Class)
        WHERE n.name = $name OR n.qualified_name = $name
        MATCH (n)-[:CONTAINS]->(meth:Method)
        MATCH (m)-[r:CALLS]->(meth)
        WITH type(r) AS relationship_type,
             'incoming' AS direction,
             labels(m) AS labels,
             m.name AS name,
             m.qualified_name AS qualified_name,
             m.module AS module,
             m.path AS path,
             m.file_path AS file_path,
             'Class->Method' AS hop,
             COUNT { (m)<-[:IMPORTS|DEPENDS_ON|CALLS]-() } AS in_degree
        ORDER BY in_degree DESC, coalesce(qualified_name, name)
        RETURN relationship_type, direction, labels, name, qualified_name,
               module, path, file_path, hop
        LIMIT $branch_limit
    }
    RETURN relationship_type, direction, labels, name, qualified_name, module, path, file_path, hop
    ORDER BY CASE
               WHEN hop IS NULL THEN 0
               WHEN hop = 'Class->Module' THEN 1
               ELSE 2
             END,
             name
    """
).strip()

TRACE_IMPORTS = dedent(
    """\
    MATCH (n)
    WHERE n.name = $module OR n.qualified_name = $module
    OPTIONAL MATCH (owner:Module)-[:CONTAINS]->(n)
    WHERE n:Class
    WITH CASE
           WHEN n:Module THEN n
           ELSE owner
         END AS src,
         CASE
           WHEN n:Class AND owner IS NOT NULL THEN 'Class->Module'
           ELSE null
         END AS hop
    WHERE src IS NOT NULL
    MATCH path = (src)-[:IMPORTS|DEPENDS_ON*1..5]->(dst)
    WHERE length(path) <= $depth
    RETURN [node IN nodes(path) | coalesce(
             node.qualified_name, node.module, node.name, node.path
           )] AS nodes,
           hop AS hop
    """
).strip()

FIND_FULLTEXT = dedent(
    """\
    CALL db.index.fulltext.queryNodes($index_name, $lucene_query)
    YIELD node, score
    WITH node, score,
         CASE WHEN node:Docstring THEN NULL ELSE node END AS direct
    OPTIONAL MATCH (owner)-[:DOCUMENTED_BY]->(node)
    WHERE node:Docstring
    WITH coalesce(direct, owner) AS n, score
    WHERE n IS NOT NULL
      AND (n:Module OR n:Class OR n:Function OR n:Method)
      AND ($entity_type IS NULL OR $entity_type IN labels(n))
    WITH n, max(score) AS score
    OPTIONAL MATCH (n)-[:DOCUMENTED_BY]->(d:Docstring)
    WITH n, score, d,
         CASE
           WHEN n.file_path STARTS WITH 'tests/' OR n.file_path CONTAINS '/tests/' THEN 2
           WHEN n.file_path STARTS WITH 'docs_src/' OR n.file_path CONTAINS '/docs_src/' THEN 1
           ELSE 0
         END AS source_rank
    RETURN labels(n) AS labels,
           n.name AS name,
           n.qualified_name AS qualified_name,
           n.file_path AS file_path,
           n.path AS path,
           n.line_start AS line_start,
           n.line_end AS line_end,
           score AS score,
           d.text AS docstring_text,
           d.summary AS docstring_summary
    ORDER BY source_rank, score DESC
    LIMIT $top_k
    """
).strip()

VECTOR_SEARCH = dedent(
    """\
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
    YIELD node, score
    WITH node, score
    WHERE ($entity_type IS NULL OR $entity_type IN labels(node))
      AND (node:Class OR node:Function OR node:Method)
      AND score >= $min_score
    OPTIONAL MATCH (node)-[:DOCUMENTED_BY]->(d:Docstring)
    RETURN labels(node) AS labels,
           node.name AS name,
           node.qualified_name AS qualified_name,
           node.file_path AS file_path,
           node.path AS path,
           node.line_start AS line_start,
           node.line_end AS line_end,
           score AS score,
           d.text AS docstring_text,
           d.summary AS docstring_summary
    ORDER BY score DESC
    LIMIT $top_k
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
