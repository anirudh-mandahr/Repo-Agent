"""Read numbered source snippets from the read-only repository volume."""

from __future__ import annotations

import os
from pathlib import Path

from core.logging import get_logger
from core.settings import DEFAULT_REPO_ROOT

log = get_logger(__name__)

DEFAULT_CONTEXT = 5


class PathTraversalError(ValueError):
    """Raised when ``file_path`` resolves outside ``REPO_ROOT``."""


def repo_root_from_env() -> Path:
    """Return the repository root from ``REPO_ROOT`` (default ``/repo``)."""
    return Path(os.environ.get("REPO_ROOT", DEFAULT_REPO_ROOT))


def resolve_repo_path(file_path: str, repo_root: Path | None = None) -> Path:
    """Resolve ``file_path`` inside ``repo_root``. Reject path traversal."""
    root = (repo_root if repo_root is not None else repo_root_from_env()).resolve()
    raw = Path(file_path)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        log.warning("snippet.path_traversal", file_path=file_path, repo_root=str(root))
        raise PathTraversalError(
            f"path {file_path!r} resolves outside repository root {str(root)!r}"
        )
    return resolved


def get_snippet(
    file_path: str,
    line_start: int,
    line_end: int,
    context: int = DEFAULT_CONTEXT,
    *,
    repo_root: Path | str | None = None,
) -> str:
    """Return numbered source text for ``[line_start, line_end]`` plus ``context`` lines.

    Paths are resolved against ``REPO_ROOT``. Traversal outside the root is rejected.
    """
    root = Path(repo_root) if repo_root is not None else repo_root_from_env()
    path = resolve_repo_path(file_path, root)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    start = max(int(line_start), 1)
    end = max(int(line_end), start)
    pad = max(int(context), 0)
    window_start = max(1, start - pad)
    window_end = min(len(raw_lines), end + pad)
    if window_start > len(raw_lines):
        return ""
    chunk = raw_lines[window_start - 1 : window_end]
    width = max(4, len(str(window_end)))
    numbered = (
        f"{number:>{width}} {line}" for number, line in enumerate(chunk, start=window_start)
    )
    return "\n".join(numbered)
