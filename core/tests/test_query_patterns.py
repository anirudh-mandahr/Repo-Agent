"""Pattern Cypher templates live in core.querying.patterns for the Code Analyst."""

from __future__ import annotations

from core.querying.patterns import (
    PATTERN_TEMPLATES,
    SUPPORTED_PATTERNS,
    pattern_cypher,
    pattern_params,
    pattern_path_prefix,
)


def test_supported_patterns_match_template_keys() -> None:
    assert set(PATTERN_TEMPLATES) == set(SUPPORTED_PATTERNS)
    assert set(SUPPORTED_PATTERNS) == {"decorator", "dependency_injection", "factory"}


def test_pattern_cypher_selects_fixed_template() -> None:
    assert pattern_cypher("decorator") == PATTERN_TEMPLATES["decorator"]
    assert "DECORATED_BY" in PATTERN_TEMPLATES["decorator"]
    assert "HAS_PARAMETER" in PATTERN_TEMPLATES["dependency_injection"]
    assert pattern_cypher("singleton") is None


def test_decorator_pattern_does_not_use_legacy_property() -> None:
    assert "n.decorators" not in PATTERN_TEMPLATES["decorator"]
    assert "n.args" not in PATTERN_TEMPLATES["dependency_injection"]


def test_decorator_pattern_scopes_with_path_prefix_parameter() -> None:
    cypher = PATTERN_TEMPLATES["decorator"]
    assert "$path_prefix" in cypher
    assert "n.file_path STARTS WITH $path_prefix" in cypher
    assert "n.file_path ENDS WITH '/' + $path_prefix" in cypher
    assert pattern_params("decorator", "routing.py") == {"path_prefix": "routing.py"}
    assert pattern_params("decorator") == {"path_prefix": None}
    assert pattern_params("factory") == {}
    assert pattern_params("singleton") == {}
    assert "{" not in cypher.replace("$path_prefix", "")


def test_pattern_path_prefix_extracts_routing_module() -> None:
    assert (
        pattern_path_prefix("Find all decorators used in the routing module") == "routing.py"
    )
    assert pattern_path_prefix("decorators in fastapi/routing.py") == "fastapi/routing.py"
    assert pattern_path_prefix("find decorator patterns") is None
    assert pattern_path_prefix("What design patterns are used in FastAPI's core and why?") is None
