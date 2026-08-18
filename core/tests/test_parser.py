"""Exact parser output for the sample fixture, plus parse-error handling."""

from __future__ import annotations

import hashlib
from pathlib import Path

from core.indexing.parser import (
    ParsedCall,
    ParsedCallable,
    ParsedClass,
    ParsedDocstring,
    ParsedFile,
    ParsedImport,
    ParsedParameter,
    extract_entities,
    parse_code,
    parse_file,
    parse_python_ast,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_module.py"
BROKEN = FIXTURES / "broken.py"


def test_parse_sample_module_exactly() -> None:
    parsed = parse_file(SAMPLE, FIXTURES)
    expected_hash = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    expected = ParsedFile(
        path="sample_module.py",
        module="sample_module",
        content_hash=expected_hash,
        line_start=1,
        line_end=38,
        docstring=ParsedDocstring(
            text="Fixture module for indexer parser tests.",
            summary="Fixture module for indexer parser tests.",
        ),
        classes=[
            ParsedClass(
                name="Base",
                qualified_name="sample_module.Base",
                bases=[],
                decorators=[],
                line_start=7,
                line_end=10,
                docstring=ParsedDocstring(text="Base class.", summary="Base class."),
            ),
            ParsedClass(
                name="Worker",
                qualified_name="sample_module.Worker",
                bases=["Base"],
                decorators=[],
                line_start=17,
                line_end=27,
                docstring=ParsedDocstring(text="Does work.", summary="Does work."),
            ),
        ],
        functions=[
            ParsedCallable(
                name="helper",
                qualified_name="sample_module.helper",
                parameters=[
                    ParsedParameter(name="value", annotation="int", default=None, position=0)
                ],
                decorators=[],
                line_start=13,
                line_end=14,
                is_async=False,
                calls=[],
                docstring=None,
            ),
            ParsedCallable(
                name="fetch_all",
                qualified_name="sample_module.fetch_all",
                parameters=[
                    ParsedParameter(name="limit", annotation="int", default=None, position=0)
                ],
                decorators=[],
                line_start=30,
                line_end=32,
                is_async=True,
                calls=["helper"],
                docstring=ParsedDocstring(text="Load records.", summary="Load records."),
            ),
            ParsedCallable(
                name="ping",
                qualified_name="sample_module.ping",
                parameters=[],
                decorators=["app.get"],
                line_start=35,
                line_end=38,
                is_async=False,
                calls=["helper"],
                docstring=None,
            ),
        ],
        methods=[
            ParsedCallable(
                name="run",
                qualified_name="sample_module.Worker.run",
                parameters=[
                    ParsedParameter(name="count", annotation="int", default=None, position=0)
                ],
                decorators=["staticmethod"],
                line_start=20,
                line_end=23,
                is_async=False,
                calls=["helper"],
                docstring=None,
            ),
            ParsedCallable(
                name="label",
                qualified_name="sample_module.Worker.label",
                parameters=[
                    ParsedParameter(name="self", annotation=None, default=None, position=0)
                ],
                decorators=["property"],
                line_start=25,
                line_end=27,
                is_async=False,
                calls=[],
                docstring=None,
            ),
        ],
        imports=[
            ParsedImport(module="typing", names=["Any"], alias="Anything"),
            ParsedImport(module="os", names=["os"], alias=None),
        ],
        calls=[
            ParsedCall(caller_qualified_name="sample_module.Worker.run", callee="helper"),
            ParsedCall(caller_qualified_name="sample_module.fetch_all", callee="helper"),
            ParsedCall(caller_qualified_name="sample_module.ping", callee="helper"),
        ],
        error=None,
    )
    assert parsed == expected


def test_parse_file_records_syntax_error_without_raising() -> None:
    parsed = parse_file(BROKEN, FIXTURES)
    assert parsed.error is not None
    assert parsed.path == "broken.py"
    assert parsed.module == "broken"
    assert parsed.classes == []
    assert parsed.functions == []
    assert parsed.methods == []
    assert parsed.content_hash == hashlib.sha256(BROKEN.read_bytes()).hexdigest()


def test_init_module_name(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    init = pkg / "__init__.py"
    init.write_text("X = 1\n")
    parsed = parse_file(init, tmp_path)
    assert parsed.module == "pkg"
    assert parsed.path == "pkg/__init__.py"


def test_parse_python_ast_accepts_path_or_code() -> None:
    from_path = parse_python_ast(str(SAMPLE), repo_root=FIXTURES)
    assert from_path.module == "sample_module"
    assert from_path.error is None
    from_code = parse_python_ast("def ping() -> str:\n    return 'ok'\n")
    assert from_code.functions[0].name == "ping"
    assert from_code.error is None
    assert parse_code("def ping() -> str:\n    return 'ok'\n").functions[0].name == "ping"


def test_extract_entities_emits_spec_nodes_and_relationships() -> None:
    extracted = extract_entities(str(SAMPLE), repo_root=FIXTURES)
    types = {item["type"] for item in extracted.entities}
    rel_types = {item["type"] for item in extracted.relationships}
    assert {
        "File",
        "Module",
        "Class",
        "Function",
        "Method",
        "Parameter",
        "Decorator",
        "Import",
        "Docstring",
    } <= types
    assert {
        "CONTAINS",
        "IMPORTS",
        "INHERITS_FROM",
        "CALLS",
        "DECORATED_BY",
        "HAS_PARAMETER",
        "DOCUMENTED_BY",
        "DEPENDS_ON",
    } <= rel_types
    params = [item for item in extracted.entities if item["type"] == "Parameter"]
    assert any(item["name"] == "value" and item["annotation"] == "int" for item in params)
    decorators = {item["name"] for item in extracted.entities if item["type"] == "Decorator"}
    assert {"app.get", "staticmethod", "property"} <= decorators
    docs = [item for item in extracted.entities if item["type"] == "Docstring"]
    assert any(item["summary"] == "Load records." for item in docs)
