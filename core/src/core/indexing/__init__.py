"""Repository clone, AST parse, and knowledge-graph indexing."""

from __future__ import annotations

from pathlib import Path

from core.indexing.cloner import clone_repo, default_repo_url
from core.indexing.indexer import (
    IndexReport,
    IndexStatus,
    already_running_report,
    clone_and_index,
    compute_index_version,
    index_file,
    index_repository,
    load_index_meta,
    load_index_status,
    query_graph_counts,
    query_label_counts,
    run_index_file,
    save_report,
)
from core.indexing.parser import (
    ExtractedGraph,
    ParsedFile,
    ParseError,
    extract_entities,
    parse_file,
    parse_python_ast,
)
from core.logging import bind_correlation_id, configure_logging, get_logger


def trigger_index(*, repo_root: str = "/repo") -> IndexReport:
    """Clone the configured repo and index it. Used by ``make index``."""
    configure_logging()
    bind_correlation_id()
    log = get_logger(__name__)
    log.info("index.requested", repo_root=repo_root)
    report = clone_and_index(repo_root=Path(repo_root))
    if report.status == "error":
        raise RuntimeError(report.detail or "index failed")
    return report


__all__ = [
    "ExtractedGraph",
    "IndexReport",
    "IndexStatus",
    "ParseError",
    "ParsedFile",
    "already_running_report",
    "clone_and_index",
    "compute_index_version",
    "clone_repo",
    "default_repo_url",
    "extract_entities",
    "index_file",
    "index_repository",
    "load_index_meta",
    "load_index_status",
    "parse_file",
    "parse_python_ast",
    "query_graph_counts",
    "query_label_counts",
    "run_index_file",
    "save_report",
    "trigger_index",
]
