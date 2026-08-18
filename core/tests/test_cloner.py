"""clone_repo uses git clone for a new dest and fetch+reset when .git exists."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from core.indexing.cloner import clone_repo
from core.settings import IndexingSettings


def test_clone_repo_shallow_clones_when_dest_is_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    dest = tmp_path / "repo"
    result = clone_repo("https://github.com/fastapi/fastapi", dest, depth=1)
    assert result == dest
    assert dest.is_dir()
    assert calls == [
        ["git", "clone", "--depth", "1", "https://github.com/fastapi/fastapi", str(dest)]
    ]


def test_clone_repo_fetches_and_resets_when_git_dir_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "repo"
    dest.mkdir()
    (dest / ".git").mkdir()
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clone_repo("https://github.com/fastapi/fastapi", dest, depth=1)
    assert calls == [
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin"],
        ["git", "-C", str(dest), "reset", "--hard", "FETCH_HEAD"],
    ]


def test_clone_repo_raises_on_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, args, stderr="clone failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        clone_repo("https://github.com/fastapi/fastapi", tmp_path / "repo")


def test_indexing_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPO_URL", "https://example.com/repo.git")
    monkeypatch.setenv("REPO_ROOT", "/tmp/repo")
    monkeypatch.setenv("INDEX_SKIP_TESTS", "0")
    monkeypatch.setenv("INDEX_SKIP_DOCS", "false")
    settings = IndexingSettings.from_env()
    assert settings.repo_url == "https://example.com/repo.git"
    assert settings.repo_root == "/tmp/repo"
    assert settings.skip_tests is False
    assert settings.skip_docs is False

