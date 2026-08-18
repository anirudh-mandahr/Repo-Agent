"""Shallow-clone or update a git repository via subprocess."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from core.logging import get_logger
from core.settings import DEFAULT_REPO_URL, IndexingSettings

log = get_logger(__name__)


def clone_repo(url: str, dest: str | Path, depth: int = 1) -> Path:
    """Shallow-clone ``url`` into ``dest``. If ``dest`` is already a git repo, fetch + reset."""
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if _is_git_repo(destination):
        log.info("git.fetch", url=url, dest=str(destination), depth=depth)
        _run_git(
            ["git", "-C", str(destination), "fetch", "--depth", str(depth), "origin"],
            env=env,
        )
        _run_git(
            ["git", "-C", str(destination), "reset", "--hard", "FETCH_HEAD"],
            env=env,
        )
    else:
        log.info("git.clone", url=url, dest=str(destination), depth=depth)
        _run_git(
            ["git", "clone", "--depth", str(depth), url, str(destination)],
            env=env,
        )
    return destination


def default_repo_url() -> str:
    """Return ``REPO_URL`` or the FastAPI default."""
    return IndexingSettings.from_env().repo_url or DEFAULT_REPO_URL


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _run_git(args: list[str], *, env: dict[str, str]) -> None:
    log.info("git.run", args=args)
    try:
        subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        log.error("git.failed", args=args, returncode=exc.returncode, stderr=stderr)
        raise
