"""Indexer hash skip/upsert logic with a recording fake graph client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.indexing.indexer import (
    IndexReport,
    already_running_report,
    compute_index_version,
    index_file,
    index_repository,
    load_index_meta,
    query_graph_counts,
    query_label_counts,
    read_saved_report,
    save_report,
)
from core.indexing.parser import hash_file


def _rows_for(client: RecordingClient, fragment: str) -> list[dict[str, Any]]:
    """Rows of the write whose Cypher contains ``fragment``; empty if never written."""
    for query, rows in client.writes:
        if fragment in query:
            return rows
    return []


SAMPLE = Path(__file__).parent / "fixtures" / "sample_module.py"
BROKEN = Path(__file__).parent / "fixtures" / "broken.py"


class RecordingClient:
    """Records Cypher reads/writes. Returns configured file hashes on hash queries."""

    def __init__(
        self,
        hashes: dict[str, str] | None = None,
        *,
        node_count: int = 0,
        rel_count: int = 0,
        label_counts: dict[str, int] | None = None,
        callables: list[dict[str, Any]] | None = None,
        modules: list[dict[str, Any]] | None = None,
        imports: list[dict[str, Any]] | None = None,
    ) -> None:
        self.hashes = hashes or {}
        self.node_count = node_count
        self.rel_count = rel_count
        self.label_counts = label_counts or {}
        self.callables = callables or []
        self.modules = modules or []
        self.imports = imports or []
        self.reads: list[str] = []
        self.writes: list[tuple[str, list[dict[str, Any]]]] = []

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        _ = params
        _ = timeout_s
        self.reads.append(query)
        if "content_hash" in query:
            return [{"path": path, "content_hash": digest} for path, digest in self.hashes.items()]
        if "AS nodes" in query:
            return [{"nodes": self.node_count, "rels": self.rel_count}]
        if "UNWIND labels(n)" in query:
            return [
                {"label": label, "count": count} for label, count in self.label_counts.items()
            ]
        if "index_version" in query and "last_indexed_at" in query:
            return [
                {
                    "index_version": "meta-version",
                    "last_indexed_at": "2026-08-18T00:00:00+00:00",
                }
            ]
        if "i.module AS import_module" in query:
            return list(self.imports)
        if "n:Function OR n:Method" in query:
            return list(self.callables)
        if "MATCH (n:Module) RETURN" in query:
            return list(self.modules)
        return []

    def run_write_batch(self, query: str, rows: Sequence[Mapping[str, Any]]) -> None:
        self.writes.append((query, [dict(row) for row in rows]))


def _write_sample(repo: Path, name: str = "sample_module.py") -> Path:
    dest = repo / name
    dest.write_bytes(SAMPLE.read_bytes())
    return dest


def _queries(client: RecordingClient) -> list[str]:
    return [query for query, _rows in client.writes]


def test_new_file_is_upserted_without_delete(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    client = RecordingClient()
    report = index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    assert report.files_seen == 1
    assert report.files_indexed == 1
    assert report.files_skipped == 0
    assert report.nodes_written > 0
    queries = _queries(client)
    assert any("DETACH DELETE" in query for query in queries) is False
    assert any("MERGE (f:File {path: row.path})" in query for query in queries)
    assert any("MERGE (m:Meta {key: row.key})" in query for query in queries)
    file_rows = next(rows for query, rows in client.writes if "MERGE (f:File" in query)
    assert file_rows[0]["path"] == "sample_module.py"
    assert file_rows[0]["content_hash"] == hash_file(tmp_path / "sample_module.py")
    embedding_rows = next(
        rows for query, rows in client.writes if "n.embedding = row.embedding" in query
    )
    assert embedding_rows
    assert len(embedding_rows[0]["embedding"]) == 256
    names = {row["qualified_name"] for row in embedding_rows}
    assert "sample_module.helper" in names
    assert "sample_module.Worker" in names
    assert any(row["embedding_text"] for row in embedding_rows)


class _FlakyEmbedder:
    """Embeds ``fail_after`` chunks, then raises the way a remote backend would."""

    def __init__(self, fail_after: int) -> None:
        self.fail_after = fail_after
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls > self.fail_after:
            raise ConnectionError("embedding backend unreachable")
        return [[1.0] + [0.0] * 255 for _ in texts]


def test_the_embedding_backend_is_recorded_alongside_the_vectors(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    client = RecordingClient()

    index_repository(client, tmp_path, skip_tests=True, skip_docs=True)

    meta_rows = [
        row
        for query, rows in client.writes
        if "MERGE (m:Meta {key: row.key})" in query
        for row in rows
    ]
    fingerprints = [row for row in meta_rows if row["key"] == "embedding_fingerprint"]
    assert fingerprints
    assert fingerprints[0]["value"] == "hash:256"


def test_embedding_failure_keeps_the_structural_index(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _write_sample(tmp_path)
    monkeypatch.setattr("core.indexing.indexer.EMBEDDING_CHUNK_SIZE", 1)
    provider = _FlakyEmbedder(fail_after=1)
    client = RecordingClient()

    report = index_repository(
        client,
        tmp_path,
        skip_tests=True,
        skip_docs=True,
        embedding_provider=provider,
    )

    assert report.status == "ok"
    assert report.files_indexed == 1
    assert report.nodes_written > 0
    assert any("MERGE (f:File {path: row.path})" in query for query in _queries(client))
    # The one chunk embedded before the failure is still persisted.
    written = [rows for query, rows in client.writes if "n.embedding = row.embedding" in query]
    assert len(written) == 1
    assert len(written[0]) == 1


def test_unchanged_hash_is_skipped(tmp_path: Path) -> None:
    path = _write_sample(tmp_path)
    client = RecordingClient(hashes={"sample_module.py": hash_file(path)})
    report = index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    assert report.files_seen == 1
    assert report.files_indexed == 0
    assert report.files_skipped == 1
    assert report.nodes_written == 0
    assert report.mode == "incremental"
    assert len(client.writes) == 1
    assert "MERGE (m:Meta {key: row.key})" in client.writes[0][0]


def test_full_mode_reindexes_unchanged_hash(tmp_path: Path) -> None:
    path = _write_sample(tmp_path)
    client = RecordingClient(hashes={"sample_module.py": hash_file(path)})
    report = index_repository(
        client, tmp_path, skip_tests=True, skip_docs=True, mode="full"
    )
    assert report.mode == "full"
    assert report.files_seen == 1
    assert report.files_indexed == 1
    assert report.files_skipped == 0
    assert report.nodes_written > 0


def test_changed_hash_deletes_subtree_then_upserts(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    client = RecordingClient(hashes={"sample_module.py": "outdated-hash"})
    report = index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    assert report.files_seen == 1
    assert report.files_indexed == 1
    assert report.files_skipped == 0
    queries = _queries(client)
    delete_at = next(i for i, query in enumerate(queries) if "DETACH DELETE" in query)
    upsert_at = next(i for i, query in enumerate(queries) if "MERGE (f:File" in query)
    assert delete_at < upsert_at
    delete_rows = client.writes[delete_at][1]
    assert delete_rows == [{"path": "sample_module.py"}]
    assert "NOT n:Decorator" in queries[delete_at]


def test_same_module_calls_and_inherits_are_resolved(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    client = RecordingClient()
    report = index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    assert report.unresolved_calls == 0
    calls_rows = next(rows for query, rows in client.writes if "CALLS" in query)
    callees = {row["callee_qualified_name"] for row in calls_rows}
    callers = {row["caller_qualified_name"] for row in calls_rows}
    assert "sample_module.helper" in callees
    assert "sample_module.Worker.run" in callers
    assert "sample_module.fetch_all" in callers
    assert "sample_module.ping" in callers
    inherits_rows = next(rows for query, rows in client.writes if "INHERITS_FROM" in query)
    assert inherits_rows == [
        {
            "child_qualified_name": "sample_module.Worker",
            "parent_qualified_name": "sample_module.Base",
        }
    ]


def test_upsert_emits_parameter_decorator_docstring_and_contains(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    client = RecordingClient()
    index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    queries = _queries(client)
    assert any("HAS_PARAMETER" in query for query in queries)
    assert any("DECORATED_BY" in query for query in queries)
    assert any("DOCUMENTED_BY" in query for query in queries)
    assert any("MERGE (i:Import" in query for query in queries)
    assert any("DEFINES" in query for query in queries) is False
    param_rows = next(rows for query, rows in client.writes if "HAS_PARAMETER" in query)
    assert any(row["name"] == "value" and row["annotation"] == "int" for row in param_rows)
    dec_rows = next(rows for query, rows in client.writes if "DECORATED_BY" in query)
    names = {row["name"] for row in dec_rows}
    assert {"app.get", "staticmethod", "property"} <= names
    doc_rows = next(rows for query, rows in client.writes if "DOCUMENTED_BY" in query)
    assert any(row["summary"] == "Load records." for row in doc_rows)
    contains_queries = [query for query in queries if "[:CONTAINS]" in query]
    assert len(contains_queries) >= 4


def test_index_file_hash_check_and_single_file_reindex(tmp_path: Path) -> None:
    path = _write_sample(tmp_path)
    client = RecordingClient()
    report = index_file(client, tmp_path, "sample_module.py")
    assert report.files_seen == 1
    assert report.files_indexed == 1
    assert any("DETACH DELETE" in query for query in _queries(client)) is False

    client_skip = RecordingClient(hashes={"sample_module.py": hash_file(path)})
    skipped = index_file(client_skip, tmp_path, "sample_module.py")
    assert skipped.files_skipped == 1
    assert skipped.files_indexed == 0
    assert client_skip.writes == []

    client_changed = RecordingClient(hashes={"sample_module.py": "outdated-hash"})
    changed = index_file(client_changed, tmp_path, "sample_module.py")
    assert changed.files_indexed == 1
    queries = _queries(client_changed)
    delete_at = next(i for i, query in enumerate(queries) if "DETACH DELETE" in query)
    upsert_at = next(i for i, query in enumerate(queries) if "MERGE (f:File" in query)
    assert delete_at < upsert_at


def test_index_file_missing_returns_error(tmp_path: Path) -> None:
    client = RecordingClient()
    report = index_file(client, tmp_path, "missing.py")
    assert report.status == "error"
    assert report.detail is not None
    assert client.writes == []


def test_parse_errors_are_collected_and_other_files_indexed(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    (tmp_path / "broken.py").write_bytes(BROKEN.read_bytes())
    client = RecordingClient()
    report = index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    assert report.files_seen == 2
    assert report.files_indexed == 1
    assert len(report.parse_errors) == 1
    assert report.parse_errors[0].path == "broken.py"
    assert report.parse_errors[0].message


def test_tests_and_docs_dirs_are_skipped_in_fast_profile(tmp_path: Path) -> None:
    _write_sample(tmp_path, "app.py")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok() -> None:\n    return None\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "example.py").write_text("VALUE = 1\n")
    client = RecordingClient()
    report = index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    assert report.files_seen == 1
    file_rows = next(rows for query, rows in client.writes if "MERGE (f:File" in query)
    assert file_rows[0]["path"] == "app.py"


def test_default_index_includes_tests_and_docs(tmp_path: Path) -> None:
    _write_sample(tmp_path, "app.py")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok() -> None:\n    return None\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "example.py").write_text("VALUE = 1\n")
    client = RecordingClient()
    report = index_repository(client, tmp_path, skip_tests=False, skip_docs=False)
    assert report.files_seen == 3


def test_stale_file_nodes_are_purged(tmp_path: Path) -> None:
    path = _write_sample(tmp_path)
    client = RecordingClient(
        hashes={
            "sample_module.py": hash_file(path),
            "removed.py": "stale-hash",
        }
    )
    report = index_repository(client, tmp_path, skip_tests=True, skip_docs=True)
    assert report.files_skipped == 1
    assert report.files_indexed == 0
    assert report.files_purged == 1
    delete_writes = [
        (query, rows) for query, rows in client.writes if "DETACH DELETE" in query
    ]
    assert delete_writes
    assert delete_writes[0][1] == [{"path": "removed.py"}]


def test_already_running_report_status() -> None:
    report = already_running_report()
    assert report.status == "already_running"
    assert isinstance(report, IndexReport)
    assert report.files_indexed == 0


def test_query_graph_counts_reads_live_totals() -> None:
    client = RecordingClient(node_count=10, rel_count=4)
    assert query_graph_counts(client) == (10, 4)
    assert any("AS nodes" in query for query in client.reads)


def test_query_label_counts_includes_spec_labels() -> None:
    client = RecordingClient(
        label_counts={"Parameter": 3, "Decorator": 2, "Docstring": 4, "Module": 1}
    )
    counts = query_label_counts(client)
    assert counts["Parameter"] == 3
    assert counts["Decorator"] == 2
    assert counts["Docstring"] == 4
    assert counts["Function"] == 0


def test_compute_index_version_is_stable_for_sorted_paths() -> None:
    a = compute_index_version({"b.py": "hash-b", "a.py": "hash-a"})
    b = compute_index_version({"a.py": "hash-a", "b.py": "hash-b"})
    c = compute_index_version({"a.py": "hash-a", "b.py": "hash-c"})
    assert a == b
    assert a != c


def test_load_index_meta_reads_version_and_timestamp() -> None:
    client = RecordingClient()
    version, last_indexed_at = load_index_meta(client)
    assert version == "meta-version"
    assert last_indexed_at == "2026-08-18T00:00:00+00:00"


def test_save_and_read_report_roundtrip(tmp_path: Path) -> None:
    report = IndexReport(files_seen=3, files_indexed=2, duration_s=1.5)
    path = tmp_path / "index_report.json"
    save_report(report, path)
    loaded = read_saved_report(path)
    assert loaded == report
    assert read_saved_report(tmp_path / "missing.json") is None


def test_class_nested_in_function_is_indexed_and_parented(tmp_path: Path) -> None:
    """A subclass declared inside a test function is still a node with its base edge."""
    (tmp_path / "routers.py").write_text(
        "class APIRouter:\n"
        "    pass\n"
        "\n"
        "\n"
        "def test_subclass() -> None:\n"
        "    class HeaderRouter(APIRouter):\n"
        "        def matches(self):\n"
        "            return True\n"
    )
    client = RecordingClient()
    index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    classes = _rows_for(client, "MERGE (n:Class")
    assert {row["qualified_name"] for row in classes} == {
        "routers.APIRouter",
        "routers.test_subclass.HeaderRouter",
    }
    inherits = _rows_for(client, "INHERITS_FROM")
    assert inherits == [
        {
            "child_qualified_name": "routers.test_subclass.HeaderRouter",
            "parent_qualified_name": "routers.APIRouter",
        }
    ]
    contains = _rows_for(client, "MATCH (child:Class")
    assert {
        "parent_qualified_name": "routers.test_subclass",
        "child_qualified_name": "routers.test_subclass.HeaderRouter",
    } in contains
    methods = _rows_for(client, "MERGE (n:Method")
    assert {row["qualified_name"] for row in methods} == {
        "routers.test_subclass.HeaderRouter.matches"
    }


def test_decorators_on_nested_functions_are_recorded(tmp_path: Path) -> None:
    (tmp_path / "wrap.py").write_text(
        "import functools\n"
        "\n"
        "\n"
        "def outer(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(*args):\n"
        "        return fn(*args)\n"
        "    return wrapper\n"
    )
    client = RecordingClient()
    index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    decorators = _rows_for(client, "MERGE (d:Decorator")
    assert {
        "owner_qualified_name": "wrap.outer.wrapper",
        "name": "functools.wraps",
    } in decorators


def test_third_party_import_is_not_rewired_to_a_same_named_local_module(
    tmp_path: Path,
) -> None:
    """``from starlette.responses import X`` must not link to a local ``responses``."""
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "responses.py").write_text("class JSONResponse:\n    pass\n")
    (pkg / "main.py").write_text(
        "from starlette.responses import JSONResponse\n"
        "\n"
        "\n"
        "def send():\n"
        "    return JSONResponse()\n"
    )
    client = RecordingClient()
    index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    depends = _rows_for(client, "MATCH (src:Module")
    assert not any(row["to_qualified_name"] == "app.responses" for row in depends)
    calls = _rows_for(client, "MERGE (caller)-[:CALLS]")
    assert calls == []


def test_relative_import_resolves_against_its_package(tmp_path: Path) -> None:
    """``from .utils import x`` in ``tests/`` must not need a globally unique name."""
    for package in ("alpha", "beta"):
        pkg = tmp_path / package
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "utils.py").write_text("def helper():\n    return 1\n")
        (pkg / "runner.py").write_text(
            "from .utils import helper\n\n\ndef run():\n    return helper()\n"
        )
    client = RecordingClient()
    index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    depends = _rows_for(client, "MATCH (src:Module")
    pairs = {(row["from_qualified_name"], row["to_qualified_name"]) for row in depends}
    assert ("alpha.runner", "alpha.utils") in pairs
    assert ("beta.runner", "beta.utils") in pairs
    assert ("alpha.runner", "beta.utils") not in pairs

    calls = _rows_for(client, "MERGE (caller)-[:CALLS]")
    call_pairs = {
        (row["caller_qualified_name"], row["callee_qualified_name"]) for row in calls
    }
    assert ("alpha.runner.run", "alpha.utils.helper") in call_pairs
    assert ("beta.runner.run", "beta.utils.helper") in call_pairs


def _call_pairs(client: RecordingClient) -> set[tuple[str, str]]:
    return {
        (row["caller_qualified_name"], row["callee_qualified_name"])
        for row in _rows_for(client, "MERGE (caller)-[:CALLS]")
    }


def test_call_on_unknown_receiver_does_not_match_a_same_named_method(
    tmp_path: Path,
) -> None:
    """``self.router.get()`` / ``req.scope.get()`` must not become ``CALLS -> App.get``.

    This reproduces the FastAPI graph: ``FastAPI.get`` delegates to
    ``self.router.get`` and the old bare-name fallback linked it to itself.
    """
    (tmp_path / "app.py").write_text(
        "class App:\n"
        "    def __init__(self):\n"
        "        self.router = None\n"
        "        self.setup()\n"
        "\n"
        "    def setup(self):\n"
        "        def redoc(req):\n"
        "            root = req.scope.get('root_path')\n"
        "            return self.openapi()\n"
        "        return redoc\n"
        "\n"
        "    def openapi(self):\n"
        "        return {}\n"
        "\n"
        "    def get(self, path):\n"
        "        return self.router.get(path)\n"
        "\n"
        "    def include(self, other):\n"
        "        return super().include(other)\n"
    )
    client = RecordingClient()
    report = index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    pairs = _call_pairs(client)
    assert ("app.App.__init__", "app.App.setup") in pairs
    # ``self.openapi()`` inside a closure resolves against the enclosing class.
    assert ("app.App.setup.redoc", "app.App.openapi") in pairs
    # No self-loops, and nothing points at ``App.get`` from an unknown receiver.
    assert not any(caller == callee for caller, callee in pairs)
    assert not any(callee == "app.App.get" for _caller, callee in pairs)
    assert not any(callee == "app.App.include" for _caller, callee in pairs)
    # self.router.get, req.scope.get and the builtin super() are reported as
    # unresolved rather than guessed.
    assert report.unresolved_calls == 3


def test_dotted_call_with_unbound_head_stays_unresolved(tmp_path: Path) -> None:
    """``client.get('/')`` in a test must not link to an imported ``Router.get``."""
    pkg = tmp_path / "web"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "routing.py").write_text(
        "class Router:\n    def get(self, path):\n        return path\n"
    )
    (tmp_path / "test_routes.py").write_text(
        "from web.routing import Router\n"
        "\n"
        "\n"
        "def test_root(client):\n"
        "    return client.get('/')\n"
    )
    client = RecordingClient()
    index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    assert _call_pairs(client) == set()


def test_reexported_name_resolves_to_its_defining_module(tmp_path: Path) -> None:
    """``from pkg import Depends`` follows ``pkg/__init__.py`` to ``pkg.params.Depends``."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .params import Depends as Depends\n")
    (pkg / "params.py").write_text("def Depends(dep):\n    return dep\n")
    (tmp_path / "main.py").write_text(
        "import pkg\n"
        "from pkg import Depends\n"
        "\n"
        "\n"
        "def by_name():\n"
        "    return Depends(None)\n"
        "\n"
        "\n"
        "def by_module():\n"
        "    return pkg.Depends(None)\n"
    )
    client = RecordingClient()
    report = index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    pairs = _call_pairs(client)
    assert ("main.by_name", "pkg.params.Depends") in pairs
    assert ("main.by_module", "pkg.params.Depends") in pairs
    assert report.unresolved_calls == 0


def test_reexported_base_class_resolves_inheritance(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .exceptions import HTTPError\n")
    (pkg / "exceptions.py").write_text("class HTTPError(Exception):\n    pass\n")
    (tmp_path / "main.py").write_text(
        "from pkg import HTTPError\n\n\nclass NotFound(HTTPError):\n    pass\n"
    )
    client = RecordingClient()
    index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    inherits = _rows_for(client, "INHERITS_FROM")
    assert {
        "child_qualified_name": "main.NotFound",
        "parent_qualified_name": "pkg.exceptions.HTTPError",
    } in inherits


def test_incremental_pass_follows_reexports_stored_in_the_graph(tmp_path: Path) -> None:
    """Only ``main.py`` is re-parsed; ``pkg``'s re-export table comes from :Import nodes."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .params import Depends as Depends\n")
    (pkg / "params.py").write_text("def Depends(dep):\n    return dep\n")
    (tmp_path / "main.py").write_text(
        "from pkg import Depends\n\n\ndef by_name():\n    return Depends(None)\n"
    )
    client = RecordingClient(
        hashes={
            "pkg/__init__.py": hash_file(pkg / "__init__.py"),
            "pkg/params.py": hash_file(pkg / "params.py"),
        },
        callables=[{"qualified_name": "pkg.params.Depends", "name": "Depends"}],
        modules=[
            {"qualified_name": "pkg", "name": "pkg"},
            {"qualified_name": "pkg.params", "name": "params"},
        ],
        imports=[
            {
                "module": "pkg",
                "file_path": "pkg/__init__.py",
                "import_module": ".params",
                "names": ["Depends"],
                "alias": "Depends",
                "position": 0,
            }
        ],
    )
    report = index_repository(client, tmp_path, skip_tests=False, skip_docs=False)

    assert report.files_indexed == 1
    assert ("main.by_name", "pkg.params.Depends") in _call_pairs(client)
