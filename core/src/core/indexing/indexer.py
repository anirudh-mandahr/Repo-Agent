"""Walk a repository, parse Python files, and upsert the knowledge graph."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator, Mapping, Sequence, Set
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
    EmbeddingRecord,
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
    upsert_embeddings,
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
    ParsedClass,
    ParsedDocstring,
    ParsedFile,
    ParsedImport,
    ParseError,
    hash_file,
    parse_file,
)
from core.logging import bind_correlation_id, get_correlation_id, get_logger
from core.querying.embeddings import (
    EMBEDDING_FINGERPRINT_KEY,
    EmbeddingProvider,
    build_embedding_text,
    default_embedding_provider,
    provider_fingerprint,
)
from core.settings import IndexingSettings

log = get_logger(__name__)

# Vectors are embedded and written this many at a time so that a mid-run
# failure of a remote backend leaves the earlier chunks persisted.
EMBEDDING_CHUNK_SIZE = 500

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
LOAD_IMPORTS_QUERY = (
    "MATCH (m:Module)-[:IMPORTS]->(i:Import) "
    "RETURN m.qualified_name AS module, m.file_path AS file_path, "
    "i.module AS import_module, i.names AS names, i.alias AS alias, "
    "i.position AS position "
    "ORDER BY module, position"
)
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
    ) -> list[dict[str, Any]]:
        """Run a read-only Cypher query.

        Args:
            query: Cypher text.
            params: Query parameters.
            timeout_s: Per-query timeout in seconds.

        Returns:
            Result rows as dictionaries.
        """
        ...

    def run_write_batch(self, query: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Execute a batched UNWIND write.

        Args:
            query: Cypher using ``UNWIND $rows``.
            rows: Parameter rows to write.
        """
        ...


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
    files_purged: int = 0
    mode: Literal["full", "incremental"] = "incremental"


class IndexStatus(BaseModel):
    """Last index report plus live Neo4j counts."""

    running: bool = False
    last_report: IndexReport | None = None
    node_count: int = 0
    rel_count: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    detail: str | None = None


def already_running_report() -> IndexReport:
    """Return the payload used when an index pass is already in progress.
    
    Returns:
        IndexReport.
    """
    return IndexReport(status="already_running", detail="index already running")


def walk_python_files(
    repo_root: str | Path,
    *,
    skip_tests: bool = False,
    skip_docs: bool = False,
) -> Iterator[Path]:
    """Yield ``*.py`` files under ``repo_root``, optionally skipping tests/ and docs/.
    
    Args:
        repo_root: str | Path.
        skip_tests: bool.
        skip_docs: bool.

    Returns:
        Iterator[Path].
    """
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
    """Load existing ``{path: content_hash}`` from the graph in one query.
    
    Args:
        client: IndexGraphClient.

    Returns:
        dict[str, str].
    """
    rows = client.run_read(LOAD_FILE_HASHES_QUERY)
    hashes: dict[str, str] = {}
    for row in rows:
        path = row.get("path")
        content_hash = row.get("content_hash")
        if isinstance(path, str) and isinstance(content_hash, str):
            hashes[path] = content_hash
    return hashes


def purge_stale_files(client: IndexGraphClient, stale_paths: Sequence[str]) -> int:
    """DETACH DELETE file subtrees whose paths are no longer on disk.
    
    Args:
        client: IndexGraphClient.
        stale_paths: Sequence[str].

    Returns:
        int.
    """
    if not stale_paths:
        return 0
    query, _rows = delete_file_subtree(stale_paths[0])
    client.run_write_batch(query, [{"path": path} for path in stale_paths])
    log.info("index.purge_stale", files_purged=len(stale_paths))
    return len(stale_paths)


def query_graph_counts(client: IndexGraphClient) -> tuple[int, int]:
    """Return ``(node_count, rel_count)`` from a live read-only query.
    
    Args:
        client: IndexGraphClient.

    Returns:
        tuple[int, int].
    """
    rows = client.run_read(GRAPH_COUNTS_QUERY)
    if not rows:
        return 0, 0
    row = rows[0]
    return int(row.get("nodes") or 0), int(row.get("rels") or 0)


def query_label_counts(client: IndexGraphClient) -> dict[str, int]:
    """Return live node counts keyed by label.
    
    Args:
        client: IndexGraphClient.

    Returns:
        dict[str, int].
    """
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
    embedding_provider: EmbeddingProvider | None = None,
    mode: Literal["full", "incremental"] = "incremental",
) -> IndexReport:
    """Parse ``*.py`` files and upsert new/changed files into the graph.

    ``incremental`` (default) skips files whose content hash is unchanged.
    ``full`` re-parses every file regardless of hash.

    Args:
        client: IndexGraphClient.
        repo_root: str | Path.
        skip_tests: bool | None.
        skip_docs: bool | None.
        embedding_provider: EmbeddingProvider | None.
        mode: ``full`` or ``incremental``.

    Returns:
        IndexReport.
    """
    bind_correlation_id(get_correlation_id() if get_correlation_id() != "-" else None)
    settings = IndexingSettings.from_env()
    skip_tests = settings.skip_tests if skip_tests is None else skip_tests
    skip_docs = settings.skip_docs if skip_docs is None else skip_docs
    index_mode: Literal["full", "incremental"] = (
        mode if mode in {"full", "incremental"} else "incremental"
    )
    root = Path(repo_root)
    started = time.perf_counter()

    paths = list(walk_python_files(root, skip_tests=skip_tests, skip_docs=skip_docs))
    log.info(
        "index.start",
        repo_root=str(root),
        file_count=len(paths),
        skip_tests=skip_tests,
        skip_docs=skip_docs,
        mode=index_mode,
    )

    existing = load_file_hashes(client)
    parse_errors: list[ParseError] = []
    to_upsert: list[ParsedFile] = []
    changed_paths: list[str] = []
    seen_rel: set[str] = set()
    files_skipped = 0
    files_seen = 0

    for path in paths:
        files_seen += 1
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        seen_rel.add(rel)
        content_hash = hash_file(path)
        previous = existing.get(rel)
        if index_mode != "full" and previous == content_hash:
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

    stale_paths = [path for path in existing if path not in seen_rel]
    files_purged = purge_stale_files(client, stale_paths)

    nodes_written = 0
    rels_written = 0
    unresolved_calls = 0
    if to_upsert:
        if changed_paths:
            query, _rows = delete_file_subtree(changed_paths[0])
            client.run_write_batch(query, [{"path": p} for p in changed_paths])
        nodes_written, rels_written, unresolved_calls = _upsert_parsed_files(
            client,
            to_upsert,
            embedding_provider=embedding_provider,
        )
    _update_index_version(client)

    duration_s = time.perf_counter() - started
    report = IndexReport(
        status="ok",
        files_seen=files_seen,
        files_indexed=len(to_upsert),
        files_skipped=files_skipped,
        files_purged=files_purged,
        nodes_written=nodes_written,
        rels_written=rels_written,
        parse_errors=parse_errors,
        unresolved_calls=unresolved_calls,
        duration_s=duration_s,
        mode=index_mode,
    )
    log.info(
        "index.done",
        files_seen=report.files_seen,
        files_indexed=report.files_indexed,
        files_skipped=report.files_skipped,
        files_purged=report.files_purged,
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
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> IndexReport:
    """Hash-check and reindex a single file, reusing delete_file_subtree + upsert.
    
    Args:
        client: IndexGraphClient.
        repo_root: str | Path.
        path: str.
        embedding_provider: EmbeddingProvider | None.

    Returns:
        IndexReport.
    """
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

    nodes_written, rels_written, unresolved_calls = _upsert_parsed_files(
        client,
        [parsed],
        embedding_provider=embedding_provider,
    )
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
    *,
    mode: Literal["full", "incremental"] = "incremental",
) -> IndexReport:
    """Clone (or update) the target repo and index it into Neo4j.

    Args:
        repo_url: str | None.
        repo_root: str | Path | None.
        mode: ``incremental`` skips unchanged hashes; ``full`` re-parses all files.

    Returns:
        IndexReport.
    """
    settings = IndexingSettings.from_env()
    url = repo_url or settings.repo_url
    dest = Path(repo_root) if repo_root is not None else Path(settings.repo_root)
    started = time.perf_counter()
    try:
        clone_repo(url, dest)
        with GraphClient() as client:
            ensure_schema(client)
            report = index_repository(client, dest, mode=mode)
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
    """Open Neo4j and reindex a single file under the configured repo root.
    
    Args:
        path: str.
        repo_root: str | Path | None.

    Returns:
        IndexReport.
    """
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
    """Persist ``report`` as JSON so status can be read across processes.
    
    Args:
        report: IndexReport.
        path: str | Path | None.
    """
    report_path = Path(path) if path is not None else Path(IndexingSettings.from_env().report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json())


def read_saved_report(path: str | Path | None = None) -> IndexReport | None:
    """Load the last saved ``IndexReport``, if present.
    
    Args:
        path: str | Path | None.

    Returns:
        IndexReport | None.
    """
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
    """Return last report (memory or disk) plus live node/relationship counts.
    
    Args:
        last_report: IndexReport | None.
        running: bool.

    Returns:
        IndexStatus.
    """
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
    *,
    embedding_provider: EmbeddingProvider | None = None,
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
                    parent_qualified_name=cls.parent_qualified_name or parsed.module,
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
                    parent_qualified_name=fn.parent_qualified_name or parsed.module,
                    child_qualified_name=fn.qualified_name,
                )
            )
            _add_callable_details(
                fn, parsed.path, parameter_records, decorator_rows, docstring_records
            )
            _add_name(qn_index, name_index, fn.qualified_name, fn.name)
        for method in parsed.methods:
            method_records.append(_method_record(method, parsed.path))
            parent_qn = (
                method.parent_qualified_name or method.qualified_name.rsplit(".", 1)[0]
            )
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

    module_qns: set[str] = set(module_qn_index)
    file_bindings: dict[str, dict[str, str]] = {}
    file_visible: dict[str, set[str]] = {}
    # Every indexed module's import bindings, so a name imported from a package
    # (``from fastapi import Depends``) can be followed through the package's own
    # ``from .param_functions import Depends`` to the defining module. Modules
    # outside this batch come from the graph; batch files override them below.
    module_bindings = load_module_bindings(client)

    for parsed in parsed_files:
        is_package = parsed.path.endswith("__init__.py")
        bindings = import_bindings(parsed, is_package=is_package)
        file_bindings[parsed.path] = bindings
        if parsed.module:
            module_bindings[parsed.module] = bindings
        visible = {parsed.module} if parsed.module else set()
        for position, imported in enumerate(parsed.imports):
            if not imported.module:
                continue
            target = resolve_relative_module(
                imported.module, current_module=parsed.module, is_package=is_package
            )
            if not target:
                continue
            # An import only becomes an edge when it names a module we actually
            # indexed. Third-party targets are left unresolved on purpose.
            resolved = target if target in module_qns else None
            if resolved is None:
                continue
            visible.add(resolved)
            if resolved != parsed.module:
                import_depends_rows.append(
                    ImportDependsOnRow(
                        file_path=parsed.path,
                        position=position,
                        module_qualified_name=resolved,
                    )
                )
                depends_on_rows.append(
                    DependsOnRow(
                        from_qualified_name=parsed.module,
                        to_qualified_name=resolved,
                    )
                )
        file_visible[parsed.path] = visible

    for parsed in parsed_files:
        for cls in parsed.classes:
            for base in cls.bases:
                parent = _resolve_name(
                    base,
                    module=parsed.module,
                    caller_qn=cls.qualified_name,
                    qn_index=class_qn_index,
                    name_index=class_name_index,
                    bindings=file_bindings.get(parsed.path),
                    visible_modules=file_visible.get(parsed.path),
                    module_qns=module_qns,
                    class_qns=class_qn_index.keys(),
                    reexports=module_bindings,
                )
                if parent and parent != cls.qualified_name:
                    inherits_rows.append(
                        InheritsRow(
                            child_qualified_name=cls.qualified_name,
                            parent_qualified_name=parent,
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
                bindings=file_bindings.get(parsed.path),
                visible_modules=file_visible.get(parsed.path),
                module_qns=module_qns,
                class_qns=class_qn_index.keys(),
                reexports=module_bindings,
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
    _write_embeddings(client, parsed_files, embedding_provider)

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


def _write_embeddings(
    client: IndexGraphClient,
    parsed_files: Sequence[ParsedFile],
    embedding_provider: EmbeddingProvider | None,
) -> int:
    """Embed the parsed entities and persist the vectors chunk by chunk.

    A configured backend may be a remote model, so embedding is the one step
    here that can fail for reasons unrelated to the repository. The structural
    graph is already written and valid at this point; losing it to a 429 would
    be a worse outcome than a graph whose vectors are stale. Chunks that made
    it are therefore kept, the failure is logged, and the pass continues.

    Args:
        client: IndexGraphClient.
        parsed_files: Files whose classes, functions, and methods get vectors.
        embedding_provider: Backend override. Env default when omitted.

    Returns:
        Number of vectors written.
    """
    provider = embedding_provider or default_embedding_provider()
    documents = _embedding_documents(parsed_files)
    if not documents:
        return 0
    written = 0
    for start in range(0, len(documents), EMBEDDING_CHUNK_SIZE):
        chunk = documents[start : start + EMBEDDING_CHUNK_SIZE]
        try:
            records = _embedding_records(chunk, provider)
        except Exception as exc:
            log.error(
                "index.embeddings_failed",
                written=written,
                skipped=len(documents) - written,
                error=str(exc),
            )
            break
        if not records:
            continue
        _write(client, *upsert_embeddings(records))
        written += len(records)
    if written:
        _record_embedding_fingerprint(client, provider)
        log.info("index.embeddings_written", count=written)
    return written


def _record_embedding_fingerprint(client: IndexGraphClient, provider: EmbeddingProvider) -> None:
    """Store which vector space the stored embeddings belong to."""
    fingerprint = provider_fingerprint(provider)
    if fingerprint is None:
        return
    _write(
        client,
        *upsert_meta(
            [
                MetaRecord(
                    key=EMBEDDING_FINGERPRINT_KEY,
                    value=fingerprint,
                    updated_at=datetime.now(tz=UTC).isoformat(),
                )
            ]
        ),
    )


def _embedding_documents(parsed_files: Sequence[ParsedFile]) -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    for parsed in parsed_files:
        for cls in parsed.classes:
            documents.append((cls.qualified_name, _class_embedding_text(cls)))
        for fn in parsed.functions:
            documents.append((fn.qualified_name, _callable_embedding_text(fn)))
        for method in parsed.methods:
            documents.append((method.qualified_name, _callable_embedding_text(method)))
    return documents


def _embedding_records(
    documents: Sequence[tuple[str, str]],
    provider: EmbeddingProvider,
) -> list[EmbeddingRecord]:
    if not documents:
        return []
    vectors = list(provider.embed([text for _qn, text in documents]))
    if len(vectors) != len(documents):
        log.warning(
            "index.embedding_length_mismatch",
            expected=len(documents),
            actual=len(vectors),
        )
        return []
    records: list[EmbeddingRecord] = []
    for (qualified_name, text), vector in zip(documents, vectors, strict=True):
        records.append(
            EmbeddingRecord(
                qualified_name=qualified_name,
                embedding=[float(value) for value in vector],
                embedding_text=text[:4000],
            )
        )
    return records


def _class_embedding_text(cls: ParsedClass) -> str:
    docstring = cls.docstring.text if cls.docstring is not None else ""
    return build_embedding_text(
        qualified_name=cls.qualified_name,
        name=cls.name,
        docstring=docstring,
        source_summary=cls.source_summary,
    )


def _callable_embedding_text(item: ParsedCallable) -> str:
    docstring = item.docstring.text if item.docstring is not None else ""
    return build_embedding_text(
        qualified_name=item.qualified_name,
        name=item.name,
        docstring=docstring,
        source_summary=item.source_summary,
        param_names=[param.name for param in item.parameters],
    )


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


def resolve_relative_module(module_ref: str, *, current_module: str, is_package: bool) -> str:
    """Resolve a PEP 328 relative import to an absolute dotted module name.

    ``from .b import x`` inside package ``a`` resolves against ``a`` itself;
    inside module ``a.c`` it resolves against ``a``. Each extra leading dot
    strips one further package level.

    Args:
        module_ref: The ``module`` text of the import, e.g. ``".."`` or ``".b"``.
        current_module: Dotted name of the module containing the import.
        is_package: True when the containing file is an ``__init__.py``.

    Returns:
        The absolute dotted name, or ``""`` when it walks above the repo root.
    """
    if not module_ref.startswith("."):
        return module_ref
    stripped = module_ref.lstrip(".")
    level = len(module_ref) - len(stripped)
    base = current_module if is_package else current_module.rpartition(".")[0]
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    if not base:
        return stripped
    return f"{base}.{stripped}" if stripped else base


def import_bindings(
    parsed: ParsedFile,
    *,
    is_package: bool,
) -> dict[str, str]:
    """Map each name the file binds via import to the dotted target it names.

    ``import a.b as c`` binds ``c`` -> ``a.b``; ``from a.b import c`` binds
    ``c`` -> ``a.b.c``. Relative imports are made absolute first. This is what
    lets call and inheritance resolution prefer targets the file can actually
    see, instead of guessing from a bare name.

    Args:
        parsed: The parsed file whose imports should be read.
        is_package: True when the file is an ``__init__.py``.

    Returns:
        Mapping of local binding name to absolute dotted target.
    """
    return bindings_from_imports(parsed.imports, module=parsed.module, is_package=is_package)


def bindings_from_imports(
    imports: Sequence[ParsedImport],
    *,
    module: str,
    is_package: bool,
) -> dict[str, str]:
    """Map imported names to dotted targets for one module's import statements.

    Same rules as :func:`import_bindings`, but fed by any import list -- parsed
    from source or reconstructed from ``:Import`` nodes already in the graph.

    Args:
        imports: The module's import statements in source order.
        module: Dotted name of the module containing the imports.
        is_package: True when the module is an ``__init__.py``.

    Returns:
        Mapping of local binding name to absolute dotted target.
    """
    bindings: dict[str, str] = {}
    for imported in imports:
        raw = imported.module or ""
        target = resolve_relative_module(raw, current_module=module, is_package=is_package)
        if raw.startswith("."):
            # ``from .pkg import name`` binds each name under the resolved package.
            for name in imported.names:
                bindings.setdefault(imported.alias or name, f"{target}.{name}" if target else name)
            continue
        if imported.names == [raw]:
            # Plain ``import a.b`` (optionally ``as c``): binds the dotted path itself.
            bindings.setdefault(imported.alias or raw, raw)
            head = raw.partition(".")[0]
            bindings.setdefault(head, head)
            continue
        for name in imported.names:
            bindings.setdefault(
                imported.alias or name, f"{target}.{name}" if target else name
            )
    return bindings


def load_module_bindings(client: IndexGraphClient) -> dict[str, dict[str, str]]:
    """Rebuild every indexed module's import bindings from its ``:Import`` nodes.

    This is the re-export table: for package ``fastapi`` it maps ``Depends`` to
    ``fastapi.param_functions.Depends``, so a file that only sees the package can
    still be linked to the defining module.

    Args:
        client: IndexGraphClient.

    Returns:
        Module qualified name to its binding map.
    """
    grouped: dict[str, tuple[bool, list[ParsedImport]]] = {}
    for row in client.run_read(LOAD_IMPORTS_QUERY):
        module = row.get("module")
        if not isinstance(module, str) or not module:
            continue
        file_path = row.get("file_path")
        is_package = isinstance(file_path, str) and file_path.endswith("__init__.py")
        raw_names = row.get("names") or []
        names = [str(name) for name in raw_names] if isinstance(raw_names, list) else []
        imported = ParsedImport(
            module=_as_opt_str(row.get("import_module")),
            names=names,
            alias=_as_opt_str(row.get("alias")),
        )
        grouped.setdefault(module, (is_package, []))[1].append(imported)
    return {
        module: bindings_from_imports(imports, module=module, is_package=is_package)
        for module, (is_package, imports) in grouped.items()
    }


def _owning_module(qualified_name: str, module_qns: Set[str]) -> str | None:
    prefix = qualified_name
    while prefix:
        if prefix in module_qns:
            return prefix
        prefix = prefix.rpartition(".")[0]
    return None


_INSTANCE_NAMES = frozenset({"self", "cls"})
# ``a`` re-exports from ``b`` which re-exports from ``c``: enough hops for any
# real package layout, and a hard stop for accidental import cycles.
_REEXPORT_MAX_HOPS = 4


def _enclosing_class(qualified_name: str, class_qns: Set[str]) -> str | None:
    """Return the nearest class that ``qualified_name`` is defined inside of."""
    prefix = qualified_name.rpartition(".")[0]
    while prefix:
        if prefix in class_qns:
            return prefix
        prefix = prefix.rpartition(".")[0]
    return None


def _follow_reexports(
    candidate: str,
    qn_index: Mapping[str, str],
    module_qns: Set[str] | None,
    reexports: Mapping[str, Mapping[str, str]] | None,
) -> str | None:
    """Look ``candidate`` up, following ``from .x import name`` re-exports.

    ``fastapi.Depends`` is not a node, but module ``fastapi`` binds ``Depends``
    to ``fastapi.param_functions.Depends``, which is. Each hop rewrites the
    first segment after the owning module and tries again.
    """
    current = candidate
    visited: set[str] = set()
    for _ in range(_REEXPORT_MAX_HOPS + 1):
        if current in qn_index:
            return qn_index[current]
        if not reexports or module_qns is None or current in visited:
            return None
        visited.add(current)
        owner = _owning_module(current, module_qns)
        if owner is None or owner == current:
            return None
        remainder = current[len(owner) + 1 :]
        first, dot, tail = remainder.partition(".")
        target = reexports.get(owner, {}).get(first)
        if not target:
            return None
        current = f"{target}.{tail}" if dot else target
    return None


def _first_indexed(
    candidates: Sequence[str],
    qn_index: Mapping[str, str],
    module_qns: Set[str] | None,
    reexports: Mapping[str, Mapping[str, str]] | None,
) -> str | None:
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved = _follow_reexports(candidate, qn_index, module_qns, reexports)
        if resolved is not None:
            return resolved
    return None


def _resolve_name(
    callee: str,
    *,
    module: str,
    caller_qn: str,
    qn_index: dict[str, str],
    name_index: dict[str, list[str]],
    bindings: Mapping[str, str] | None = None,
    visible_modules: Set[str] | None = None,
    module_qns: Set[str] | None = None,
    class_qns: Set[str] | None = None,
    reexports: Mapping[str, Mapping[str, str]] | None = None,
) -> str | None:
    """Resolve a textual callee/base name to an indexed qualified name.

    Resolution is by receiver, never by trailing name alone:

    - ``self.run()`` / ``cls.run()`` resolve against the nearest enclosing class
      (so a closure inside a method still finds the class). ``self.router.get()``
      is a call on an attribute of unknown type and stays unresolved.
    - A dotted callee whose head is an import binding follows that binding, then
      any re-exports (``fastapi.Depends`` -> ``fastapi.param_functions.Depends``).
    - Same-module and literal spellings are tried next.
    - Only an undotted name that is still unresolved falls back to a unique
      bare-name match, limited to modules the file actually imports.

    The old bare-name fallback also fired for dotted callees, which turned
    ``req.scope.get(...)`` and ``client.get("/")`` into ``CALLS`` edges onto
    ``FastAPI.get`` / ``APIRouter.get`` and produced self-loops. Unresolved is
    the correct answer for those.

    Args:
        callee: Callee or base-class text exactly as written in the source.
        module: Dotted name of the module containing the reference.
        caller_qn: Qualified name of the referring entity.
        qn_index: Known qualified names.
        name_index: Bare name to qualified names.
        bindings: Local binding name to dotted target, from ``import_bindings``.
        visible_modules: Modules the file may resolve into (its own plus imports).
        module_qns: Every known module qualified name, for ownership lookup.
        class_qns: Every known class qualified name, for ``self.`` lookup.
        reexports: Per-module import bindings, for following re-exports.

    Returns:
        The resolved qualified name, or ``None`` when it cannot be resolved.
    """
    head, dot, rest = callee.partition(".")
    if head in _INSTANCE_NAMES and dot:
        if "." in rest:
            return None
        if class_qns is not None:
            owner = _enclosing_class(caller_qn, class_qns)
        else:
            owner = caller_qn.rpartition(".")[0] or None
        if owner is None:
            return None
        return _first_indexed([f"{owner}.{rest}"], qn_index, module_qns, reexports)

    candidates: list[str] = []
    if bindings:
        target = bindings.get(head)
        if target:
            candidates.append(f"{target}.{rest}" if dot else target)
    if module:
        candidates.append(f"{module}.{callee}")
    candidates.append(callee)
    resolved = _first_indexed(candidates, qn_index, module_qns, reexports)
    if resolved is not None or dot:
        return resolved

    matches = name_index.get(callee, [])
    if visible_modules is not None and module_qns is not None:
        matches = [
            match
            for match in matches
            if _owning_module(match, module_qns) in visible_modules
        ]
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
    """Hash sorted ``path:content_hash`` entries into one stable index version.
    
    Args:
        file_hashes: Mapping[str, str].

    Returns:
        str.
    """
    digest = hashlib.sha256()
    for path, content_hash in sorted(file_hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_index_meta(client: IndexGraphClient) -> tuple[str | None, str | None]:
    """Return stored ``(index_version, last_indexed_at)`` if present.
    
    Args:
        client: IndexGraphClient.

    Returns:
        tuple[str | None, str | None].
    """
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
    "bindings_from_imports",
    "clone_and_index",
    "compute_index_version",
    "import_bindings",
    "index_file",
    "index_repository",
    "load_index_meta",
    "load_index_status",
    "load_module_bindings",
    "purge_stale_files",
    "resolve_relative_module",
    "query_graph_counts",
    "query_label_counts",
    "run_index_file",
    "save_report",
]
