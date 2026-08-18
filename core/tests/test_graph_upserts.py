"""Unit tests for UNWIND MERGE builders. No database required."""

from __future__ import annotations

from core.graph.upserts import (
    ClassRecord,
    ContainsRow,
    FileRecord,
    FunctionRecord,
    MetaRecord,
    MethodRecord,
    ModuleRecord,
    delete_file_subtree,
    upsert_calls,
    upsert_classes,
    upsert_contains,
    upsert_contains_classes,
    upsert_contains_functions,
    upsert_contains_methods,
    upsert_decorators,
    upsert_depends_on,
    upsert_docstrings,
    upsert_files,
    upsert_functions,
    upsert_import_depends_on,
    upsert_imports,
    upsert_inherits,
    upsert_meta,
    upsert_methods,
    upsert_modules,
    upsert_parameters,
)


def test_upsert_files_merges_on_path() -> None:
    files = [FileRecord(path="fastapi/app.py", content_hash="abc123")]
    cypher, rows = upsert_files(files)
    assert "UNWIND $rows AS row" in cypher
    assert "MERGE (f:File {path: row.path})" in cypher
    assert "SET f.content_hash = row.content_hash" in cypher
    assert rows == [{"path": "fastapi/app.py", "content_hash": "abc123"}]


def test_upsert_modules_merges_on_qualified_name() -> None:
    modules = [
        ModuleRecord(
            qualified_name="fastapi.applications",
            name="applications",
            file_path="fastapi/applications.py",
            line_start=1,
            line_end=400,
        )
    ]
    cypher, rows = upsert_modules(modules)
    assert "MERGE (n:Module {qualified_name: row.qualified_name})" in cypher
    assert "SET n.name = row.name" in cypher
    assert "n.file_path = row.file_path" in cypher
    assert "n.line_start = row.line_start" in cypher
    assert "n.line_end = row.line_end" in cypher
    assert rows == [
        {
            "qualified_name": "fastapi.applications",
            "name": "applications",
            "file_path": "fastapi/applications.py",
            "line_start": 1,
            "line_end": 400,
        }
    ]


def test_upsert_classes_functions_methods_merge_on_qualified_name() -> None:
    classes = [
        ClassRecord(
            qualified_name="fastapi.FastAPI",
            name="FastAPI",
            file_path="fastapi/applications.py",
            line_start=10,
            line_end=80,
            bases=["Starlette"],
        )
    ]
    functions = [
        FunctionRecord(
            qualified_name="fastapi.routing.APIRouter.add_api_route",
            name="add_api_route",
            file_path="fastapi/routing.py",
            line_start=20,
            line_end=30,
        )
    ]
    methods = [
        MethodRecord(
            qualified_name="fastapi.FastAPI.get",
            name="get",
            file_path="fastapi/applications.py",
            line_start=100,
            line_end=120,
        )
    ]
    class_cypher, class_rows = upsert_classes(classes)
    fn_cypher, fn_rows = upsert_functions(functions)
    method_cypher, method_rows = upsert_methods(methods)

    assert "MERGE (n:Class {qualified_name: row.qualified_name})" in class_cypher
    assert "n.bases = row.bases" in class_cypher
    assert "MERGE (n:Function {qualified_name: row.qualified_name})" in fn_cypher
    assert "MERGE (n:Method {qualified_name: row.qualified_name})" in method_cypher
    assert class_rows[0]["name"] == "FastAPI"
    assert class_rows[0]["bases"] == ["Starlette"]
    assert fn_rows[0]["qualified_name"] == "fastapi.routing.APIRouter.add_api_route"
    assert method_rows[0]["line_start"] == 100


def test_upsert_contains_merges_file_to_module() -> None:
    cypher, rows = upsert_contains(
        [ContainsRow(path="fastapi/app.py", module_qualified_name="fastapi.app")]
    )
    assert "MATCH (f:File {path: row.path})" in cypher
    assert "MATCH (m:Module {qualified_name: row.module_qualified_name})" in cypher
    assert "MERGE (f)-[:CONTAINS]->(m)" in cypher
    assert rows == [{"path": "fastapi/app.py", "module_qualified_name": "fastapi.app"}]


def test_upsert_contains_relationships_merge_on_qualified_names() -> None:
    class_cypher, class_rows = upsert_contains_classes(
        [
            {
                "parent_qualified_name": "fastapi.applications",
                "child_qualified_name": "fastapi.FastAPI",
            }
        ]
    )
    fn_cypher, _fn_rows = upsert_contains_functions(
        [
            {
                "parent_qualified_name": "fastapi.routing",
                "child_qualified_name": "fastapi.routing.serialize",
            }
        ]
    )
    method_cypher, _method_rows = upsert_contains_methods(
        [
            {
                "parent_qualified_name": "fastapi.FastAPI",
                "child_qualified_name": "fastapi.FastAPI.get",
            }
        ]
    )
    assert "MATCH (parent:Module {qualified_name: row.parent_qualified_name})" in class_cypher
    assert "MATCH (child:Class {qualified_name: row.child_qualified_name})" in class_cypher
    assert "MERGE (parent)-[:CONTAINS]->(child)" in class_cypher
    assert "MATCH (child:Function {qualified_name: row.child_qualified_name})" in fn_cypher
    assert "MATCH (parent:Class {qualified_name: row.parent_qualified_name})" in method_cypher
    assert "DEFINES" not in class_cypher
    assert class_rows[0]["child_qualified_name"] == "fastapi.FastAPI"


def test_upsert_parameters_decorators_docstrings_imports() -> None:
    param_cypher, param_rows = upsert_parameters(
        [
            {
                "owner_qualified_name": "sample.helper",
                "name": "value",
                "annotation": "int",
                "default": None,
                "position": 0,
                "file_path": "sample.py",
            }
        ]
    )
    dec_cypher, dec_rows = upsert_decorators(
        [{"owner_qualified_name": "sample.ping", "name": "app.get"}]
    )
    doc_cypher, doc_rows = upsert_docstrings(
        [
            {
                "owner_qualified_name": "sample.Base",
                "text": "Base class.",
                "summary": "Base class.",
                "file_path": "sample.py",
            }
        ]
    )
    import_cypher, import_rows = upsert_imports(
        [
            {
                "owner_qualified_name": "sample",
                "file_path": "sample.py",
                "position": 0,
                "module": "os",
                "names": ["os"],
                "alias": None,
            }
        ]
    )
    assert "MERGE (p:Parameter" in param_cypher
    assert "MERGE (owner)-[:HAS_PARAMETER]->(p)" in param_cypher
    assert "MERGE (d:Decorator {name: row.name})" in dec_cypher
    assert "MERGE (owner)-[:DECORATED_BY]->(d)" in dec_cypher
    assert "MERGE (d:Docstring" in doc_cypher
    assert "MERGE (owner)-[:DOCUMENTED_BY]->(d)" in doc_cypher
    assert "MERGE (i:Import" in import_cypher
    assert "MERGE (m)-[:IMPORTS]->(i)" in import_cypher
    assert param_rows[0]["name"] == "value"
    assert dec_rows[0]["name"] == "app.get"
    assert doc_rows[0]["summary"] == "Base class."
    assert import_rows[0]["module"] == "os"


def test_upsert_calls_inherits_depends_on() -> None:
    calls_cypher, call_rows = upsert_calls(
        [
            {
                "caller_qualified_name": "fastapi.FastAPI.__init__",
                "callee_qualified_name": "fastapi.routing.get",
            }
        ]
    )
    inherits_cypher, inherit_rows = upsert_inherits(
        [
            {
                "child_qualified_name": "fastapi.FastAPI",
                "parent_qualified_name": "starlette.Starlette",
            }
        ]
    )
    depends_cypher, depends_rows = upsert_depends_on(
        [
            {
                "from_qualified_name": "fastapi.applications",
                "to_qualified_name": "fastapi.routing",
            }
        ]
    )
    import_dep_cypher, import_dep_rows = upsert_import_depends_on(
        [
            {
                "file_path": "fastapi/applications.py",
                "position": 0,
                "module_qualified_name": "fastapi.routing",
            }
        ]
    )
    assert "MERGE (caller)-[:CALLS]->(callee)" in calls_cypher
    assert "caller:Function OR caller:Method" in calls_cypher
    assert "MERGE (child)-[:INHERITS_FROM]->(parent)" in inherits_cypher
    assert "INHERITS]" not in inherits_cypher.replace("INHERITS_FROM", "")
    assert "MERGE (src)-[:DEPENDS_ON]->(dst)" in depends_cypher
    assert "MERGE (i)-[:DEPENDS_ON]->(dst)" in import_dep_cypher
    assert call_rows[0]["caller_qualified_name"] == "fastapi.FastAPI.__init__"
    assert inherit_rows[0]["parent_qualified_name"] == "starlette.Starlette"
    assert depends_rows[0]["to_qualified_name"] == "fastapi.routing"
    assert import_dep_rows[0]["position"] == 0


def test_delete_file_subtree_matches_file_path_and_spares_decorators() -> None:
    cypher, rows = delete_file_subtree("fastapi/applications.py")
    assert "UNWIND $rows AS row" in cypher
    assert "n.file_path = row.path" in cypher
    assert "n:File AND n.path = row.path" in cypher
    assert "NOT n:Decorator" in cypher
    assert "DETACH DELETE n" in cypher
    assert rows == [{"path": "fastapi/applications.py"}]


def test_builders_accept_plain_mappings() -> None:
    cypher, rows = upsert_files([{"path": "a.py", "content_hash": "x"}])
    assert "MERGE (f:File {path: row.path})" in cypher
    assert rows == [{"path": "a.py", "content_hash": "x"}]


def test_upsert_meta_sets_value_and_updated_at() -> None:
    cypher, rows = upsert_meta(
        [MetaRecord(key="index_version", value="v1", updated_at="2026-01-01T00:00:00Z")]
    )

    assert "MERGE (m:Meta {key: row.key})" in cypher
    assert "SET m.value = row.value" in cypher
    assert "m.updated_at = row.updated_at" in cypher
    assert rows == [
        {"key": "index_version", "value": "v1", "updated_at": "2026-01-01T00:00:00Z"}
    ]


def test_delete_file_subtree_uses_single_row_payload() -> None:
    cypher, rows = delete_file_subtree("pkg/module.py")

    assert cypher.startswith("UNWIND $rows AS row")
    assert rows == [{"path": "pkg/module.py"}]
