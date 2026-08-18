"""UNWIND MERGE builders for knowledge-graph nodes and relationships.

Each public function is a pure ``(cypher, rows)`` factory. Callers persist with
``GraphClient.run_write_batch``. Nodes MERGE on the uniqueness key and SET mutable
properties. Relationships MATCH both ends then MERGE the edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, Field

from core.graph.schema import (
    LABEL_CLASS,
    LABEL_DECORATOR,
    LABEL_DOCSTRING,
    LABEL_FILE,
    LABEL_FUNCTION,
    LABEL_IMPORT,
    LABEL_META,
    LABEL_METHOD,
    LABEL_MODULE,
    LABEL_PARAMETER,
    REL_CALLS,
    REL_CONTAINS,
    REL_DECORATED_BY,
    REL_DEPENDS_ON,
    REL_DOCUMENTED_BY,
    REL_HAS_PARAMETER,
    REL_IMPORTS,
    REL_INHERITS_FROM,
)

CypherBatch = tuple[str, list[dict[str, Any]]]


class FileRecord(BaseModel):
    """A ``:File`` node keyed by ``path``."""

    path: str
    content_hash: str


class CodeEntity(BaseModel):
    """Shared properties for ``Module``, ``Class``, ``Function``, and ``Method`` nodes."""

    qualified_name: str
    name: str
    file_path: str
    line_start: int
    line_end: int


class ModuleRecord(CodeEntity):
    """A ``:Module`` node keyed by ``qualified_name``."""


class ClassRecord(CodeEntity):
    """A ``:Class`` node keyed by ``qualified_name``."""

    bases: list[str] = Field(default_factory=list)


class FunctionRecord(CodeEntity):
    """A ``:Function`` node keyed by ``qualified_name``."""


class MethodRecord(CodeEntity):
    """A ``:Method`` node keyed by ``qualified_name``."""


class ParameterRecord(BaseModel):
    """A ``:Parameter`` node owned by a function or method."""

    owner_qualified_name: str
    name: str
    annotation: str | None = None
    default: str | None = None
    position: int
    file_path: str


class DecoratorRelRow(BaseModel):
    """``(:Function|:Method|:Class)-[:DECORATED_BY]->(:Decorator)``."""

    owner_qualified_name: str
    name: str


class ImportRecord(BaseModel):
    """A ``:Import`` node plus ``(:Module)-[:IMPORTS]->(:Import)``."""

    owner_qualified_name: str
    file_path: str
    position: int
    module: str
    names: list[str]
    alias: str | None = None


class DocstringRecord(BaseModel):
    """A ``:Docstring`` node owned by a documented entity."""

    owner_qualified_name: str
    text: str
    summary: str
    file_path: str


class ContainsRow(BaseModel):
    """``(:File)-[:CONTAINS]->(:Module)``."""

    path: str
    module_qualified_name: str


class ContainsEntityRow(BaseModel):
    """``(:Module|:Class)-[:CONTAINS]->(:Class|:Function|:Method)``."""

    parent_qualified_name: str
    child_qualified_name: str


class DependsOnRow(BaseModel):
    """``(:Module)-[:DEPENDS_ON]->(:Module)``."""

    from_qualified_name: str
    to_qualified_name: str


class ImportDependsOnRow(BaseModel):
    """``(:Import)-[:DEPENDS_ON]->(:Module)``."""

    file_path: str
    position: int
    module_qualified_name: str


class CallsRow(BaseModel):
    """``(:Function|:Method)-[:CALLS]->(:Function|:Method)``."""

    caller_qualified_name: str
    callee_qualified_name: str


class InheritsRow(BaseModel):
    """``(:Class)-[:INHERITS_FROM]->(:Class)``."""

    child_qualified_name: str
    parent_qualified_name: str = Field(description="Base class qualified_name")


class MetaRecord(BaseModel):
    """A singleton-ish ``:Meta`` node keyed by ``key``."""

    key: str
    value: str
    updated_at: str


def _as_dicts(records: Sequence[BaseModel | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, BaseModel):
            rows.append(record.model_dump())
        else:
            rows.append(dict(record))
    return rows


def _code_node_cypher(label: str) -> str:
    extra = ""
    if label == LABEL_CLASS:
        extra = ",\n            n.bases = row.bases"
    return dedent(
        f"""\
        UNWIND $rows AS row
        MERGE (n:{label} {{qualified_name: row.qualified_name}})
        SET n.name = row.name,
            n.file_path = row.file_path,
            n.line_start = row.line_start,
            n.line_end = row.line_end{extra}
        """
    ).strip()


def upsert_files(files: Sequence[FileRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:File`` on ``path`` and SET ``content_hash``."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MERGE (f:{LABEL_FILE} {{path: row.path}})
        SET f.content_hash = row.content_hash
        """
    ).strip()
    return query, _as_dicts(files)


def upsert_meta(rows: Sequence[MetaRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Meta`` on ``key`` and SET ``value`` plus ``updated_at``."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MERGE (m:{LABEL_META} {{key: row.key}})
        SET m.value = row.value,
            m.updated_at = row.updated_at
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_modules(modules: Sequence[ModuleRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Module`` on ``qualified_name`` and SET mutable props."""
    return _code_node_cypher(LABEL_MODULE), _as_dicts(modules)


def upsert_classes(classes: Sequence[ClassRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Class`` on ``qualified_name`` and SET mutable props."""
    return _code_node_cypher(LABEL_CLASS), _as_dicts(classes)


def upsert_functions(functions: Sequence[FunctionRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Function`` on ``qualified_name`` and SET mutable props."""
    return _code_node_cypher(LABEL_FUNCTION), _as_dicts(functions)


def upsert_methods(methods: Sequence[MethodRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Method`` on ``qualified_name`` and SET mutable props."""
    return _code_node_cypher(LABEL_METHOD), _as_dicts(methods)


def upsert_parameters(rows: Sequence[ParameterRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Parameter`` nodes and ``HAS_PARAMETER`` edges."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (owner)
        WHERE owner.qualified_name = row.owner_qualified_name
          AND (owner:{LABEL_FUNCTION} OR owner:{LABEL_METHOD})
        MERGE (p:{LABEL_PARAMETER} {{
            owner_qualified_name: row.owner_qualified_name,
            position: row.position
        }})
        SET p.name = row.name,
            p.annotation = row.annotation,
            p.default = row.default,
            p.file_path = row.file_path
        MERGE (owner)-[:{REL_HAS_PARAMETER}]->(p)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_decorators(rows: Sequence[DecoratorRelRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE shared ``:Decorator`` nodes on ``name`` and ``DECORATED_BY`` edges."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (owner)
        WHERE owner.qualified_name = row.owner_qualified_name
          AND (owner:{LABEL_FUNCTION} OR owner:{LABEL_METHOD} OR owner:{LABEL_CLASS})
        MERGE (d:{LABEL_DECORATOR} {{name: row.name}})
        MERGE (owner)-[:{REL_DECORATED_BY}]->(d)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_imports(rows: Sequence[ImportRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Import`` nodes and ``(:Module)-[:IMPORTS]->(:Import)``."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (m:{LABEL_MODULE} {{qualified_name: row.owner_qualified_name}})
        MERGE (i:{LABEL_IMPORT} {{file_path: row.file_path, position: row.position}})
        SET i.module = row.module,
            i.names = row.names,
            i.alias = row.alias
        MERGE (m)-[:{REL_IMPORTS}]->(i)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_docstrings(rows: Sequence[DocstringRecord | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``:Docstring`` nodes and ``DOCUMENTED_BY`` edges."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (owner {{qualified_name: row.owner_qualified_name}})
        MERGE (d:{LABEL_DOCSTRING} {{owner_qualified_name: row.owner_qualified_name}})
        SET d.text = row.text,
            d.summary = row.summary,
            d.file_path = row.file_path
        MERGE (owner)-[:{REL_DOCUMENTED_BY}]->(d)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_contains(rows: Sequence[ContainsRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:File)-[:CONTAINS]->(:Module)``."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (f:{LABEL_FILE} {{path: row.path}})
        MATCH (m:{LABEL_MODULE} {{qualified_name: row.module_qualified_name}})
        MERGE (f)-[:{REL_CONTAINS}]->(m)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_contains_classes(rows: Sequence[ContainsEntityRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:Module)-[:CONTAINS]->(:Class)``."""
    return _contains_cypher(LABEL_MODULE, LABEL_CLASS), _as_dicts(rows)


def upsert_contains_functions(rows: Sequence[ContainsEntityRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:Module)-[:CONTAINS]->(:Function)``."""
    return _contains_cypher(LABEL_MODULE, LABEL_FUNCTION), _as_dicts(rows)


def upsert_contains_methods(rows: Sequence[ContainsEntityRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:Class)-[:CONTAINS]->(:Method)``."""
    return _contains_cypher(LABEL_CLASS, LABEL_METHOD), _as_dicts(rows)


def _contains_cypher(parent_label: str, child_label: str) -> str:
    return dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (parent:{parent_label} {{qualified_name: row.parent_qualified_name}})
        MATCH (child:{child_label} {{qualified_name: row.child_qualified_name}})
        MERGE (parent)-[:{REL_CONTAINS}]->(child)
        """
    ).strip()


def upsert_depends_on(rows: Sequence[DependsOnRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:Module)-[:DEPENDS_ON]->(:Module)``."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (src:{LABEL_MODULE} {{qualified_name: row.from_qualified_name}})
        MATCH (dst:{LABEL_MODULE} {{qualified_name: row.to_qualified_name}})
        MERGE (src)-[:{REL_DEPENDS_ON}]->(dst)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_import_depends_on(rows: Sequence[ImportDependsOnRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:Import)-[:DEPENDS_ON]->(:Module)`` when the imported module exists."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (i:{LABEL_IMPORT} {{file_path: row.file_path, position: row.position}})
        MATCH (dst:{LABEL_MODULE} {{qualified_name: row.module_qualified_name}})
        MERGE (i)-[:{REL_DEPENDS_ON}]->(dst)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_calls(rows: Sequence[CallsRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:Function|:Method)-[:CALLS]->(:Function|:Method)``."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (caller)
        WHERE caller.qualified_name = row.caller_qualified_name
          AND (caller:{LABEL_FUNCTION} OR caller:{LABEL_METHOD})
        MATCH (callee)
        WHERE callee.qualified_name = row.callee_qualified_name
          AND (callee:{LABEL_FUNCTION} OR callee:{LABEL_METHOD})
        MERGE (caller)-[:{REL_CALLS}]->(callee)
        """
    ).strip()
    return query, _as_dicts(rows)


def upsert_inherits(rows: Sequence[InheritsRow | Mapping[str, Any]]) -> CypherBatch:
    """MERGE ``(:Class)-[:INHERITS_FROM]->(:Class)``."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (child:{LABEL_CLASS} {{qualified_name: row.child_qualified_name}})
        MATCH (parent:{LABEL_CLASS} {{qualified_name: row.parent_qualified_name}})
        MERGE (child)-[:{REL_INHERITS_FROM}]->(parent)
        """
    ).strip()
    return query, _as_dicts(rows)


def delete_file_subtree(path: str) -> CypherBatch:
    """DETACH DELETE the ``:File`` and file-owned entities. Shared ``:Decorator`` nodes stay."""
    query = dedent(
        f"""\
        UNWIND $rows AS row
        MATCH (n)
        WHERE (n.file_path = row.path OR (n:{LABEL_FILE} AND n.path = row.path))
          AND NOT n:{LABEL_DECORATOR}
        DETACH DELETE n
        """
    ).strip()
    return query, [{"path": path}]
