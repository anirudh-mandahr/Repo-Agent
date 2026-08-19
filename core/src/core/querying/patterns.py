"""Named structural-pattern Cypher templates for the Code Analyst graph lookup.

User input is never interpolated into these strings. Patterns walk the spec
schema (Decorator / Parameter nodes) rather than legacy node properties.
"""

from __future__ import annotations

from textwrap import dedent

SUPPORTED_PATTERNS: tuple[str, ...] = (
    "decorator",
    "dependency_injection",
    "factory",
)

PATTERN_DECORATOR = dedent(
    """\
    MATCH (n)-[:DECORATED_BY]->(:Decorator)
    WHERE n:Function OR n:Method
    RETURN DISTINCT n.qualified_name AS qualified_name,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end
    """
).strip()

PATTERN_DEPENDENCY_INJECTION = dedent(
    """\
    MATCH (n)-[:HAS_PARAMETER]->(p:Parameter)
    WHERE (n:Function OR n:Method)
      AND (
        p.name CONTAINS 'Depends'
        OR coalesce(p.annotation, '') CONTAINS 'Depends'
        OR coalesce(p.default, '') CONTAINS 'Depends'
      )
    RETURN DISTINCT n.qualified_name AS qualified_name,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end
    """
).strip()

PATTERN_FACTORY = dedent(
    """\
    MATCH (n)
    WHERE (n:Function OR n:Method)
      AND (
        toLower(n.name) CONTAINS 'factory'
        OR toLower(n.name) STARTS WITH 'create_'
        OR toLower(n.name) ENDS WITH '_factory'
      )
    RETURN n.qualified_name AS qualified_name,
           n.file_path AS file_path,
           n.line_start AS line_start,
           n.line_end AS line_end
    """
).strip()

PATTERN_TEMPLATES: dict[str, str] = {
    "decorator": PATTERN_DECORATOR,
    "dependency_injection": PATTERN_DEPENDENCY_INJECTION,
    "factory": PATTERN_FACTORY,
}


def pattern_cypher(pattern: str) -> str | None:
    """Return the fixed Cypher template for ``pattern``, or ``None`` if unknown.
    
    Args:
        pattern: str.

    Returns:
        str | None.
    """
    return PATTERN_TEMPLATES.get(pattern)
