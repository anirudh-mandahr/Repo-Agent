"""Walk a repository, parse Python files, and upsert the knowledge graph."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from core.graph.client import GraphClient
from core.graph.schema import NODE_LABELS, ensure_schema
from core.graph.upserts import (
    CallsRow,
    ClassRecord,
    ContainsEntityRow,
    ContainsRow,
    DecoratorRelRow,
    DependsOnRow,
    DocstringRecord,
    FileRecord,
    FunctionRecord,
    ImportDependsOnRow,
    ImportRecord,
    InheritsRow,
    MetaRecord,
    MethodRecord,
    ModuleRecord,
    ParameterRecord,
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
from core.indexing.cloner import clone_repo
from core.indexing.parser import (
    ParsedCallable,
    ParsedDocstring,
    ParsedFile,
    ParseError,
    hash_file,
    parse_file,
)
from core.logging import bind_correlation_id, get_correlation_id, get_logger
from core.settings import IndexingSettings

log = get_logger(__name__)

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

LOAD_FILE_HASHES_QUERY = "MATCH (f:File) RETURN f.path AS path, f.content_hash AS content_hash"
LOAD_CALLABLES_QUERY = (
    "MATCH (n) WHERE n:Function OR n:Method "
    "RETURN n.qualified_name AS qualified_name, n.name AS name"
)
LOAD_CLASSES_QUERY = "MATCH (n:Class) RETURN n.qualified_name AS qualified_name, n.name AS name"
LOAD_MODULES_QUERY = "MATCH (n:Module) RETURN n.qualified_name AS qualified_name, n.name AS name"
LOAD_INDEX_META_QUERY = (
    "MATCH (m:Meta {key: 'index_version'}) "
    "RETURN m.value AS index_version, m.updated_at AS last_indexed_at"
)
GRAPH_COUNTS_QUERY = (
    "MATCH (n) WITH count(n) AS nodes "
    "OPTIONAL MATCH ()-[r]->() "
    "RETURN nodes, count(r) AS rels"
)
LABEL_COUNTS_QUERY = (
    "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count"
)

IndexStatusName = Literal["ok", "already_running", "error"]


class IndexGraphClient(Protocol):
    """Read/write surface used by the indexer. Tests may supply a fake."""

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]: ...

    def run_write_batch(self, query: str, rows: Sequence[Mapping[str, Any]]) -> None: ...


class IndexReport(BaseModel):
    """Summary of one index pass."""

    status: IndexStatusName = "ok"
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    nodes_written: int = 0
    rels_written: int = 0
    parse_errors: list[ParseError] = Field(default_factory=list)
    unresolved_calls: int = 0
    duration_s: float = 0.0
    detail: str | None = None


class IndexStatus(BaseModel):
    """Last index report plus live Neo4j counts."""

    running: bool = False
    last_report: IndexReport | None = None
    node_count: int = 0
    rel_count: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    detail: str | None = None


def already_running_report() -> IndexReport:
    """Return the payload used when an index pass is already in progress."""
    return IndexReport(status="already_running", detail="index already running")


def walk_python_files(
    repo_root: str | Path,
    *,
    skip_tests: bool = True,
    skip_docs: bool = True,
) -> Iterator[Path]:
    """Yield ``*.py`` files under ``repo_root``, optionally skipping tests/ and docs/."""
    root = Path(repo_root)
    skip_dirs = set(SKIP_DIR_NAMES)
    if skip_tests:
        skip_dirs.add("tests")
    if skip_docs:
        skip_dirs.add("docs")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in skip_dirs and not name.startswith(".")
        ]
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def load_file_hashes(client: IndexGraphClient) -> dict[str, str]:
    """Load existing ``{path: content_hash}`` from the graph in one query."""
    rows = client.run_read(LOAD_FILE_HASHES_QUERY)
    hashes: dict[str, str] = {}
    for row in rows:
        path = row.get("path")
        content_hash = row.get("content_hash")
        if isinstance(path, str) and isinstance(content_hash, str):
            hashes[path] = content_hash
    return hashes


def query_graph_counts(client: IndexGraphClient) -> tuple[int, int]:
    """Return ``(node_count, rel_count)`` from a live read-only query."""
    rows = client.run_read(GRAPH_COUNTS_QUERY)
    if not rows:
        return 0, 0
    row = rows[0]
    return int(row.get("nodes") or 0), int(row.get("rels") or 0)


def query_label_counts(client: IndexGraphClient) -> dict[str, int]:
    """Return live node counts keyed by label."""
    rows = client.run_read(LABEL_COUNTS_QUERY)
    counts = {label: 0 for label in NODE_LABELS}
    for row in rows:
        label = row.get("label")
        if isinstance(label, str):
            counts[label] = int(row.get("count") or 0)
    return counts


def index_repository(
    client: IndexGraphClient,
    repo_root: str | Path,
    *,
    skip_tests: bool | None = None,
    skip_docs: bool | None = None,
) -> IndexReport:
    """Parse ``*.py`` files and upsert new/changed files into the graph."""
    bind_correlation_id(get_correlation_id() if get_correlation_id() != "-" else None)
    settings = IndexingSettings.from_env()
    skip_tests = settings.skip_tests if skip_tests is None else skip_tests
    skip_docs = settings.skip_docs if skip_docs is None else skip_docs
    root = Path(repo_root)
    started = time.perf_counter()

    paths = list(walk_python_files(root, skip_tests=skip_tests, skip_docs=skip_docs))
    log.info(
        "index.start",
        repo_root=str(root),
        file_count=len(paths),
        skip_tests=skip_tests,
        skip_docs=skip_docs,
    )

    existing = load_file_hashes(client)
    parse_errors: list[ParseError] = []
    to_upsert: list[ParsedFile] = []
    changed_paths: list[str] = []
    files_skipped = 0
    files_seen = 0

    for path in paths:
        files_seen += 1
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        content_hash = hash_file(path)
        previous = existing.get(rel)
        if previous == content_hash:
            files_skipped += 1
        else:
            parsed = parse_file(path, root)
            if parsed.error:
                parse_errors.append(ParseError(path=rel, message=parsed.error))
            else:
                if previous is not None:
                    changed_paths.append(rel)
                to_upsert.append(parsed)
        if files_seen % 100 == 0:
            log.info(
                "index.progress",
                files_seen=files_seen,
                files_indexed=len(to_upsert),
                files_skipped=files_skipped,
                parse_errors=len(parse_errors),
            )

    nodes_written = 0
    rels_written = 0
    unresolved_calls = 0
    if to_upsert:
        if changed_paths:
            query, _rows = delete_file_subtree(changed_paths[0])
            client.run_write_batch(query, [{"path": p} for p in changed_paths])
        nodes_written, rels_written, unresolved_calls = _upsert_parsed_files(client, to_upsert)
    _update_index_version(client)

    duration_s = time.perf_counter() - started
    report = IndexReport(
        status="ok",
        files_seen=files_seen,
        files_indexed=len(to_upsert),
        files_skipped=files_skipped,
        nodes_written=nodes_written,
        rels_written=rels_written,
        parse_errors=parse_errors,
        unresolved_calls=unresolved_calls,
        duration_s=duration_s,
    )
    log.info(
        "index.done",
        files_seen=report.files_seen,
        files_indexed=report.files_indexed,
        files_skipped=report.files_skipped,
        nodes_written=report.nodes_written,
        rels_written=report.rels_written,
        unresolved_calls=report.unresolved_calls,
        parse_errors=len(report.parse_errors),
        duration_s=report.duration_s,
    )
    return report


def index_file(
    client: IndexGraphClient,
    repo_root: str | Path,
    path: str,
) -> IndexReport:
    """Hash-check and reindex a single file, reusing delete_file_subtree + upsert."""
    bind_correlation_id(get_correlation_id() if get_correlation_id() != "-" else None)
    root = Path(repo_root)
    started = time.perf_counter()
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = root / path
    if not file_path.is_file():
        report = IndexReport(
            status="error",
            files_seen=0,
            detail=f"file not found: {path}",
            duration_s=time.perf_counter() - started,
        )
        log.warning("index.file_missing", path=path)
        return report

    rel = file_path.resolve().relative_to(root.resolve()).as_posix()
    content_hash = hash_file(file_path)
    existing = load_file_hashes(client)
    previous = existing.get(rel)
    if previous == content_hash:
        report = IndexReport(
            status="ok",
            files_seen=1,
            files_indexed=0,
            files_skipped=1,
            duration_s=time.perf_counter() - started,
        )
        log.info("index.file_skipped", path=rel)
        return report

    parsed = parse_file(file_path, root)
    if parsed.error:
        report = IndexReport(
            status="ok",
            files_seen=1,
            files_indexed=0,
            parse_errors=[ParseError(path=rel, message=parsed.error)],
            duration_s=time.perf_counter() - started,
        )
        log.warning("index.file_parse_error", path=rel, error=parsed.error)
        return report

    if previous is not None:
        query, rows = delete_file_subtree(rel)
        client.run_write_batch(query, rows)

    nodes_written, rels_written, unresolved_calls = _upsert_parsed_files(client, [parsed])
    _update_index_version(client)
    report = IndexReport(
        status="ok",
        files_seen=1,
        files_indexed=1,
        files_skipped=0,
        nodes_written=nodes_written,
        rels_written=rels_written,
        unresolved_calls=unresolved_calls,
        duration_s=time.perf_counter() - started,
    )
    log.info(
        "index.file_done",
        path=rel,
        nodes_written=nodes_written,
        rels_written=rels_written,
    )
    return report


def clone_and_index(
    repo_url: str | None = None,
    repo_root: str | Path | None = None,
) -> IndexReport:
    """Clone (or update) the target repo and index it into Neo4j."""
    settings = IndexingSettings.from_env()
    url = repo_url or settings.repo_url
    dest = Path(repo_root) if repo_root is not None else Path(settings.repo_root)
    started = time.perf_counter()
    try:
        clone_repo(url, dest)
        with GraphClient() as client:
            ensure_schema(client)
            report = index_repository(client, dest)
        report = report.model_copy(update={"duration_s": time.perf_counter() - started})
        save_report(report)
        return report
    except Exception as exc:
        log.exception("index.failed", error=str(exc))
        report = IndexReport(
            status="error",
            detail=str(exc),
            duration_s=time.perf_counter() - started,
        )
        save_report(report)
        return report


def run_index_file(path: str, repo_root: str | Path | None = None) -> IndexReport:
    """Open Neo4j and reindex a single file under the configured repo root."""
    settings = IndexingSettings.from_env()
    dest = Path(repo_root) if repo_root is not None else Path(settings.repo_root)
    started = time.perf_counter()
    try:
        with GraphClient() as client:
            ensure_schema(client)
            report = index_file(client, dest, path)
        save_report(report)
        return report
    except Exception as exc:
        log.exception("index.file_failed", path=path, error=str(exc))
        report = IndexReport(
            status="error",
            detail=str(exc),
            duration_s=time.perf_counter() - started,
        )
        save_report(report)
        return report


def save_report(report: IndexReport, path: str | Path | None = None) -> None:
    """Persist ``report`` as JSON so status can be read across processes."""
    report_path = Path(path) if path is not None else Path(IndexingSettings.from_env().report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json())


def read_saved_report(path: str | Path | None = None) -> IndexReport | None:
    """Load the last saved ``IndexReport``, if present."""
    report_path = Path(path) if path is not None else Path(IndexingSettings.from_env().report_path)
    if not report_path.is_file():
        return None
    try:
        return IndexReport.model_validate_json(report_path.read_text())
    except Exception as exc:
        log.warning("index.report_invalid", path=str(report_path), error=str(exc))
        return None


def load_index_status(
    *,
    last_report: IndexReport | None = None,
    running: bool = False,
) -> IndexStatus:
    """Return last report (memory or disk) plus live node/relationship counts."""
    report = last_report or read_saved_report()
    try:
        with GraphClient() as client:
            node_count, rel_count = query_graph_counts(client)
            counts = query_label_counts(client)
        return IndexStatus(
            running=running,
            last_report=report,
            node_count=node_count,
            rel_count=rel_count,
            counts=counts,
        )
    except Exception as exc:
        log.warning("index.status_failed", error=str(exc))
        return IndexStatus(
            running=running,
            last_report=report,
            node_count=0,
            rel_count=0,
            counts={label: 0 for label in NODE_LABELS},
            detail=str(exc),
        )


def _upsert_parsed_files(
    client: IndexGraphClient,
    parsed_files: Sequence[ParsedFile],
) -> tuple[int, int, int]:
    file_records: list[FileRecord] = []
    module_records: list[ModuleRecord] = []
    class_records: list[ClassRecord] = []
    function_records: list[FunctionRecord] = []
    method_records: list[MethodRecord] = []
    parameter_records: list[ParameterRecord] = []
    decorator_rows: list[DecoratorRelRow] = []
    import_records: list[ImportRecord] = []
    docstring_records: list[DocstringRecord] = []
    contains_rows: list[ContainsRow] = []
    contains_classes: list[ContainsEntityRow] = []
    contains_functions: list[ContainsEntityRow] = []
    contains_methods: list[ContainsEntityRow] = []
    inherits_rows: list[InheritsRow] = []
    depends_on_rows: list[DependsOnRow] = []
    import_depends_rows: list[ImportDependsOnRow] = []

    qn_index: dict[str, str] = {}
    name_index: dict[str, list[str]] = {}
    class_qn_index: dict[str, str] = {}
    class_name_index: dict[str, list[str]] = {}
    module_qn_index: dict[str, str] = {}
    module_name_index: dict[str, list[str]] = {}

    for parsed in parsed_files:
        file_records.append(FileRecord(path=parsed.path, content_hash=parsed.content_hash))
        module_name = parsed.module.rsplit(".", 1)[-1] if parsed.module else parsed.path
        module_records.append(
            ModuleRecord(
                qualified_name=parsed.module,
                name=module_name,
                file_path=parsed.path,
                line_start=parsed.line_start,
                line_end=parsed.line_end,
            )
        )
        contains_rows.append(
            ContainsRow(path=parsed.path, module_qualified_name=parsed.module)
        )
        _add_docstring(docstring_records, parsed.module, parsed.docstring, parsed.path)
        _add_name(module_qn_index, module_name_index, parsed.module, module_name)
        for cls in parsed.classes:
            class_records.append(
                ClassRecord(
                    qualified_name=cls.qualified_name,
                    name=cls.name,
                    file_path=parsed.path,
                    line_start=cls.line_start,
                    line_end=cls.line_end,
                    bases=list(cls.bases),
                )
            )
            contains_classes.append(
                ContainsEntityRow(
                    parent_qualified_name=parsed.module,
                    child_qualified_name=cls.qualified_name,
                )
            )
            _add_docstring(docstring_records, cls.qualified_name, cls.docstring, parsed.path)
            for decorator in cls.decorators:
                decorator_rows.append(
                    DecoratorRelRow(owner_qualified_name=cls.qualified_name, name=decorator)
                )
            _add_name(class_qn_index, class_name_index, cls.qualified_name, cls.name)
        for fn in parsed.functions:
            function_records.append(_function_record(fn, parsed.path))
            contains_functions.append(
                ContainsEntityRow(
                    parent_qualified_name=parsed.module,
                    child_qualified_name=fn.qualified_name,
                )
            )
            _add_callable_details(
                fn, parsed.path, parameter_records, decorator_rows, docstring_records
            )
            _add_name(qn_index, name_index, fn.qualified_name, fn.name)
        for method in parsed.methods:
            method_records.append(_method_record(method, parsed.path))
            parent_qn = method.qualified_name.rsplit(".", 1)[0]
            contains_methods.append(
                ContainsEntityRow(
                    parent_qualified_name=parent_qn,
                    child_qualified_name=method.qualified_name,
                )
            )
            _add_callable_details(
                method, parsed.path, parameter_records, decorator_rows, docstring_records
            )
            _add_name(qn_index, name_index, method.qualified_name, method.name)
        for position, imported in enumerate(parsed.imports):
            module = imported.module or ""
            import_records.append(
                ImportRecord(
                    owner_qualified_name=parsed.module,
                    file_path=parsed.path,
                    position=position,
                    module=module,
                    names=list(imported.names),
                    alias=imported.alias,
                )
            )

    _write(client, *upsert_files(file_records))
    _write(client, *upsert_modules(module_records))
    _write(client, *upsert_classes(class_records))
    _write(client, *upsert_functions(function_records))
    _write(client, *upsert_methods(method_records))
    rels = 0
    rels += _write(client, *upsert_contains(contains_rows))
    rels += _write(client, *upsert_contains_classes(contains_classes))
    rels += _write(client, *upsert_contains_functions(contains_functions))
    rels += _write(client, *upsert_contains_methods(contains_methods))
    rels += _write(client, *upsert_parameters(parameter_records))
    rels += _write(client, *upsert_decorators(decorator_rows))
    rels += _write(client, *upsert_docstrings(docstring_records))
    rels += _write(client, *upsert_imports(import_records))

    for row in client.run_read(LOAD_CALLABLES_QUERY):
        qn = row.get("qualified_name")
        name = row.get("name")
        if isinstance(qn, str) and isinstance(name, str):
            _add_name(qn_index, name_index, qn, name)
    for row in client.run_read(LOAD_CLASSES_QUERY):
        qn = row.get("qualified_name")
        name = row.get("name")
        if isinstance(qn, str) and isinstance(name, str):
            _add_name(class_qn_index, class_name_index, qn, name)
    for row in client.run_read(LOAD_MODULES_QUERY):
        qn = row.get("qualified_name")
        name = row.get("name")
        if isinstance(qn, str) and isinstance(name, str):
            _add_name(module_qn_index, module_name_index, qn, name)

    for parsed in parsed_files:
        for cls in parsed.classes:
            for base in cls.bases:
                parent = _resolve_name(
                    base,
                    module=parsed.module,
                    caller_qn=cls.qualified_name,
                    qn_index=class_qn_index,
                    name_index=class_name_index,
                )
                if parent and parent != cls.qualified_name:
                    inherits_rows.append(
                        InheritsRow(
                            child_qualified_name=cls.qualified_name,
                            parent_qualified_name=parent,
                        )
                    )
        for position, imported in enumerate(parsed.imports):
            if not imported.module:
                continue
            target = imported.module.lstrip(".")
            if not target:
                continue
            resolved = _resolve_name(
                target,
                module=parsed.module,
                caller_qn=parsed.module,
                qn_index=module_qn_index,
                name_index=module_name_index,
            )
            if resolved is None:
                continue
            import_depends_rows.append(
                ImportDependsOnRow(
                    file_path=parsed.path,
                    position=position,
                    module_qualified_name=resolved,
                )
            )
            if resolved != parsed.module:
                depends_on_rows.append(
                    DependsOnRow(
                        from_qualified_name=parsed.module,
                        to_qualified_name=resolved,
                    )
                )
    rels += _write(client, *upsert_inherits(inherits_rows))
    rels += _write(client, *upsert_import_depends_on(import_depends_rows))
    rels += _write(client, *upsert_depends_on(depends_on_rows))

    call_rows: list[CallsRow] = []
    unresolved = 0
    for parsed in parsed_files:
        for call in parsed.calls:
            callee = _resolve_name(
                call.callee,
                module=parsed.module,
                caller_qn=call.caller_qualified_name,
                qn_index=qn_index,
                name_index=name_index,
            )
            if callee is None:
                unresolved += 1
                continue
            call_rows.append(
                CallsRow(
                    caller_qualified_name=call.caller_qualified_name,
                    callee_qualified_name=callee,
                )
            )
    rels += _write(client, *upsert_calls(call_rows))

    nodes = (
        len(file_records)
        + len(module_records)
        + len(class_records)
        + len(function_records)
        + len(method_records)
        + len(parameter_records)
        + len({row.name for row in decorator_rows})
        + len(import_records)
        + len(docstring_records)
    )
    return nodes, rels, unresolved


def _function_record(fn: ParsedCallable, file_path: str) -> FunctionRecord:
    return FunctionRecord(
        qualified_name=fn.qualified_name,
        name=fn.name,
        file_path=file_path,
        line_start=fn.line_start,
        line_end=fn.line_end,
    )


def _method_record(fn: ParsedCallable, file_path: str) -> MethodRecord:
    return MethodRecord(
        qualified_name=fn.qualified_name,
        name=fn.name,
        file_path=file_path,
        line_start=fn.line_start,
        line_end=fn.line_end,
    )


def _add_callable_details(
    item: ParsedCallable,
    file_path: str,
    parameter_records: list[ParameterRecord],
    decorator_rows: list[DecoratorRelRow],
    docstring_records: list[DocstringRecord],
) -> None:
    for param in item.parameters:
        parameter_records.append(
            ParameterRecord(
                owner_qualified_name=item.qualified_name,
                name=param.name,
                annotation=param.annotation,
                default=param.default,
                position=param.position,
                file_path=file_path,
            )
        )
    for decorator in item.decorators:
        decorator_rows.append(
            DecoratorRelRow(owner_qualified_name=item.qualified_name, name=decorator)
        )
    _add_docstring(docstring_records, item.qualified_name, item.docstring, file_path)


def _add_docstring(
    records: list[DocstringRecord],
    owner_qualified_name: str,
    docstring: ParsedDocstring | None,
    file_path: str,
) -> None:
    if docstring is None:
        return
    records.append(
        DocstringRecord(
            owner_qualified_name=owner_qualified_name,
            text=docstring.text,
            summary=docstring.summary,
            file_path=file_path,
        )
    )


def _add_name(
    qn_index: dict[str, str],
    name_index: dict[str, list[str]],
    qualified_name: str,
    name: str,
) -> None:
    qn_index[qualified_name] = qualified_name
    bucket = name_index.setdefault(name, [])
    if qualified_name not in bucket:
        bucket.append(qualified_name)


def _resolve_name(
    callee: str,
    *,
    module: str,
    caller_qn: str,
    qn_index: dict[str, str],
    name_index: dict[str, list[str]],
) -> str | None:
    candidates: list[str] = []
    if callee.startswith("self."):
        parent = caller_qn.rsplit(".", 1)[0]
        candidates.append(f"{parent}.{callee.removeprefix('self.')}")
    if module:
        candidates.append(f"{module}.{callee}")
    candidates.append(callee)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in qn_index:
            return qn_index[candidate]
    bare = callee.rsplit(".", 1)[-1]
    matches = name_index.get(bare, [])
    if len(matches) == 1:
        return matches[0]
    return None


def _write(
    client: IndexGraphClient,
    query: str,
    rows: Sequence[Mapping[str, Any]],
) -> int:
    if not rows:
        return 0
    client.run_write_batch(query, rows)
    return len(rows)


def compute_index_version(file_hashes: Mapping[str, str]) -> str:
    """Hash sorted ``path:content_hash`` entries into one stable index version."""
    digest = hashlib.sha256()
    for path, content_hash in sorted(file_hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_index_meta(client: IndexGraphClient) -> tuple[str | None, str | None]:
    """Return stored ``(index_version, last_indexed_at)`` if present."""
    rows = client.run_read(LOAD_INDEX_META_QUERY)
    if not rows:
        return None, None
    row = rows[0]
    index_version = row.get("index_version")
    last_indexed_at = row.get("last_indexed_at")
    return _as_opt_str(index_version), _as_opt_str(last_indexed_at)


def _update_index_version(client: IndexGraphClient) -> None:
    file_hashes = load_file_hashes(client)
    version = compute_index_version(file_hashes)
    updated_at = datetime.now(tz=UTC).isoformat()
    _write(
        client,
        *upsert_meta(
            [
                MetaRecord(
                    key="index_version",
                    value=version,
                    updated_at=updated_at,
                )
            ]
        ),
    )


def _as_opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = [
    "IndexReport",
    "IndexStatus",
    "already_running_report",
    "clone_and_index",
    "compute_index_version",
    "index_file",
    "index_repository",
    "load_index_meta",
    "load_index_status",
    "query_graph_counts",
    "query_label_counts",
    "run_index_file",
    "save_report",
]
