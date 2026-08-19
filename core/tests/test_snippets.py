"""Numbered snippet retrieval from REPO_ROOT with traversal checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.analysis.snippets import (
    PathTraversalError,
    get_snippet,
    get_snippet_async,
    resolve_repo_path,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_get_snippet_returns_numbered_source_with_context() -> None:
    text = get_snippet("sample_module.py", 13, 14, context=2, repo_root=FIXTURES)
    assert "13" in text
    assert "def helper" in text
    assert "return value" in text
    lines = text.splitlines()
    assert lines[0].strip().startswith("11")
    assert any(line.lstrip().startswith("14") for line in lines)


def test_resolve_repo_path_rejects_absolute_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ok.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(PathTraversalError):
        resolve_repo_path("/etc/passwd", repo)
    resolved = resolve_repo_path("ok.py", repo)
    assert resolved == (repo / "ok.py").resolve()


def test_resolve_repo_path_normalizes_nested_relative_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "pkg"
    nested.mkdir(parents=True)
    target = nested / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")

    resolved = resolve_repo_path("pkg/../pkg/module.py", repo)

    assert resolved == target.resolve()


def test_get_snippet_returns_empty_when_start_is_past_eof(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    sample = repo / "sample.py"
    sample.write_text("a = 1\n", encoding="utf-8")

    text = get_snippet("sample.py", 20, 25, repo_root=repo)

    assert text == ""


@pytest.mark.asyncio
async def test_get_snippet_async_offloads_read_and_matches_sync() -> None:
    sync_text = get_snippet("sample_module.py", 13, 14, context=2, repo_root=FIXTURES)
    async_text = await get_snippet_async(
        "sample_module.py", 13, 14, context=2, repo_root=FIXTURES
    )
    assert async_text == sync_text
    assert "def helper" in async_text
