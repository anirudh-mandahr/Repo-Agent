"""Pattern Cypher templates live in core.querying.patterns for the Code Analyst."""

from __future__ import annotations

from core.querying.patterns import (
    PATTERN_TEMPLATES,
    SUPPORTED_PATTERNS,
    pattern_cypher,
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
