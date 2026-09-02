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
        line_end=1,
        docstring=ParsedDocstring(
            text="Fixture module for indexer parser tests.",
            summary="Fixture module for indexer parser tests.",
        ),
        classes=[
            ParsedClass(
                name="Base",
                qualified_name="sample_module.Base",
                parent_qualified_name="sample_module",
                bases=[],
                decorators=[],
                line_start=7,
                line_end=10,
                docstring=ParsedDocstring(text="Base class.", summary="Base class."),
            ),
            ParsedClass(
                name="Worker",
                qualified_name="sample_module.Worker",
                parent_qualified_name="sample_module",
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
                parent_qualified_name="sample_module",
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
                parent_qualified_name="sample_module",
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
                parent_qualified_name="sample_module",
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
                parent_qualified_name="sample_module.Worker",
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
                parent_qualified_name="sample_module.Worker",
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
    parsed_dump = parsed.model_dump()
    expected_dump = expected.model_dump()
    for collection in ("classes", "functions", "methods"):
        for item in parsed_dump[collection]:
            assert item["source_summary"]
            item["source_summary"] = ""
        for item in expected_dump[collection]:
            item["source_summary"] = ""
    assert parsed_dump == expected_dump


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


def test_nested_function_is_its_own_node_owning_its_calls() -> None:
    parsed = parse_code(
        "\n".join(
            [
                "def outer() -> None:",
                "    def inner() -> None:",
                "        helper()",
                "    inner()",
            ]
        )
    )
    assert parsed.error is None
    outer, inner = parsed.functions
    assert outer.name == "outer"
    assert inner.name == "inner"
    assert inner.qualified_name == "<string>.outer.inner"
    assert inner.parent_qualified_name == "<string>.outer"
    # Each callable owns only the calls it makes directly.
    assert outer.calls == ["inner"]
    assert inner.calls == ["helper"]
    assert any(
        call.caller_qualified_name == inner.qualified_name and call.callee == "helper"
        for call in parsed.calls
    )


def test_calls_on_opaque_receivers_are_not_recorded_by_trailing_name() -> None:
    """``super().__init__()`` must not be recorded as a call to ``__init__``."""
    parsed = parse_code(
        "\n".join(
            [
                "class Child(Base):",
                "    def __init__(self):",
                "        super().__init__()",
                "        make()().run()",
                "        items[0].get()",
                "        self.router.get('/')",
                "        helper(self.value)",
            ]
        )
    )
    assert parsed.error is None
    (init,) = parsed.methods
    # ``super`` and ``make`` are themselves plain-name calls and stay; the calls
    # hanging off their results are dropped instead of reduced to ``__init__``
    # / ``run`` / ``get``.
    assert init.calls == ["super", "make", "self.router.get", "helper"]


def test_class_nested_in_function_is_captured_with_its_bases() -> None:
    parsed = parse_code(
        "\n".join(
            [
                "def make_router() -> None:",
                "    class HeaderRouter(APIRouter):",
                "        def matches(self, scope): ...",
            ]
        )
    )
    assert parsed.error is None
    nested = parsed.classes[0]
    assert nested.qualified_name == "<string>.make_router.HeaderRouter"
    assert nested.parent_qualified_name == "<string>.make_router"
    assert nested.bases == ["APIRouter"]
    assert parsed.methods[0].qualified_name == "<string>.make_router.HeaderRouter.matches"


def test_type_checking_imports_are_collected() -> None:
    parsed = parse_code(
        "\n".join(
            [
                "from typing import TYPE_CHECKING",
                "if TYPE_CHECKING:",
                "    from collections.abc import Mapping",
                "    import os",
            ]
        )
    )
    modules = {item.module for item in parsed.imports}
    assert "typing" in modules
    assert "collections.abc" in modules
    assert "os" in modules


def test_typing_type_checking_attribute_imports_are_collected() -> None:
    parsed = parse_code(
        "\n".join(
            [
                "import typing",
                "if typing.TYPE_CHECKING:",
                "    from collections.abc import Sequence",
            ]
        )
    )
    modules = {item.module for item in parsed.imports}
    assert "typing" in modules
    assert "collections.abc" in modules


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
    modules = [item for item in extracted.entities if item["type"] == "Module"]
    assert modules[0]["line_start"] == 1
    assert modules[0]["line_end"] == 1
    classes = [item for item in extracted.entities if item["type"] == "Class"]
    worker = next(item for item in classes if item["name"] == "Worker")
    assert worker["line_start"] == 17
    assert worker["line_end"] == 27


def test_module_span_is_header_not_file_length() -> None:
    parsed = parse_code(
        "\n".join(
            [
                '"""Module docs."""',
                "",
                "class Huge:",
                "    pass",
                "",
                "def tail() -> None:",
                "    return None",
            ]
        )
    )
    assert parsed.line_start == 1
    assert parsed.line_end == 1
    assert parsed.classes[0].line_start == 3
    assert parsed.classes[0].line_end == 4
    assert parsed.functions[0].line_start == 6
    assert parsed.functions[0].line_end == 7


def test_module_span_without_docstring_uses_leading_imports() -> None:
    parsed = parse_code("import os\nimport sys\n\nclass Box:\n    pass\n")
    assert parsed.line_start == 1
    assert parsed.line_end == 2
    assert parsed.classes[0].line_start == 4
    assert parsed.classes[0].line_end == 5


def test_definitions_inside_non_scoping_blocks_are_collected() -> None:
    """``if``/``try``/``with`` nest statements without changing the owner."""
    parsed = parse_code(
        "\n".join(
            [
                "if FLAG:",
                "    def guarded() -> None:",
                "        pass",
                "else:",
                "    class Fallback:",
                "        pass",
                "try:",
                "    def attempted() -> None:",
                "        pass",
                "except ValueError:",
                "    def recovered() -> None:",
                "        pass",
                "finally:",
                "    def cleaned() -> None:",
                "        pass",
            ]
        )
    )
    assert parsed.error is None
    assert {fn.name for fn in parsed.functions} == {
        "guarded",
        "attempted",
        "recovered",
        "cleaned",
    }
    assert [cls.name for cls in parsed.classes] == ["Fallback"]
    # The block does not open a scope, so the module stays the owner.
    assert all(fn.parent_qualified_name == "<string>" for fn in parsed.functions)
    assert parsed.classes[0].parent_qualified_name == "<string>"


def test_deeply_nested_definition_inside_a_guarded_closure() -> None:
    parsed = parse_code(
        "\n".join(
            [
                "def handler():",
                "    async def app():",
                "        if streaming:",
                "            @asynccontextmanager",
                "            async def producer():",
                "                yield 1",
                "        return app",
            ]
        )
    )
    assert parsed.error is None
    producer = next(fn for fn in parsed.functions if fn.name == "producer")
    assert producer.qualified_name == "<string>.handler.app.producer"
    assert producer.parent_qualified_name == "<string>.handler.app"
    assert producer.decorators == ["asynccontextmanager"]
    assert producer.is_async
