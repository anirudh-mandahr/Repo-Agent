"""AST parse and graph write stubs."""

from __future__ import annotations

from core.ast_parse import parse_python_source
from core.graph.client import unwind_write


def test_parse_python_source() -> None:
    module = parse_python_source("x = 1\n")
    assert module.body


def test_unwind_write_rejects_non_unwind() -> None:
    class _Tx:
        def run(self, query: str, **kwargs: object) -> None:
            raise AssertionError("should not run")

    try:
        unwind_write(_Tx(), "CREATE (n:Node)", [])  # type: ignore[arg-type]
    except ValueError as exc:
        assert "UNWIND" in str(exc)
    else:
        raise AssertionError("expected ValueError")
