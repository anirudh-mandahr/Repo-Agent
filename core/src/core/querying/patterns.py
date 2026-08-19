"""Named structural-pattern Cypher templates for the Code Analyst graph lookup.

User input is never interpolated into these strings. Patterns walk the spec
schema (Decorator / Parameter nodes) rather than legacy node properties.
"""

from __future__ import annotations

import re
from textwrap import dedent

SUPPORTED_PATTERNS: tuple[str, ...] = (
    "decorator",
    "dependency_injection",
    "factory",
)

PATTERN_DECORATOR = dedent(
    """\
    MATCH (n)-[:DECORATED_BY]->(:Decorator)
    WHERE (n:Function OR n:Method)
      AND (
        $path_prefix IS NULL
        OR n.file_path STARTS WITH $path_prefix
        OR n.file_path ENDS WITH '/' + $path_prefix
        OR n.file_path = $path_prefix
      )
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


def pattern_params(
    pattern: str,
    path_prefix: str | None = None,
) -> dict[str, str | None]:
    """Cypher parameters for ``pattern``. User values are never interpolated.

    Args:
        pattern: One of :data:`SUPPORTED_PATTERNS`.
        path_prefix: Optional module or file-path prefix (decorator scoping).

    Returns:
        Parameter map for the template, empty when the pattern is unknown.
    """
    cypher = pattern_cypher(pattern)
    if cypher is None:
        return {}
    params: dict[str, str | None] = {}
    if "$path_prefix" in cypher:
        params["path_prefix"] = path_prefix
    return params


_MODULE_SCOPE_RE = re.compile(
    r"\b(?:in|from|of)\s+(?:the\s+)?(?P<module>[A-Za-z][\w./]*)\s+module\b"
    r"|\b(?:in|from)\s+(?P<path>(?:[\w]+/)*[\w]+\.py)\b",
    re.IGNORECASE,
)


def pattern_path_prefix(query: str) -> str | None:
    """Optional file-path prefix implied by ``in the X module`` / ``in X.py``.

    Args:
        query: User question.

    Returns:
        A path fragment such as ``routing.py`` or ``fastapi/routing.py``, or
        ``None`` when the query does not scope the search.
    """
    match = _MODULE_SCOPE_RE.search(query)
    if match is None:
        return None
    token = (match.group("module") or match.group("path") or "").strip()
    if not token:
        return None
    if "/" in token or token.endswith(".py"):
        return token
    return token + ".py"
