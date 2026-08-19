"""Shallow-clone or update a git repository via subprocess."""

from __future__ import annotations

import ipaddress
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from core.exceptions import UnsafeCloneUrlError
from core.logging import get_logger
from core.settings import DEFAULT_REPO_URL, IndexingSettings

log = get_logger(__name__)

DEFAULT_CLONE_ALLOWED_SCHEMES = frozenset({"https"})
DEFAULT_CLONE_ALLOWED_HOSTS = frozenset({"github.com"})
_SHELL_METACHARACTERS = frozenset(';&|`$(){}<>\n\r\x00\\')


def clone_repo(
    url: str,
    dest: str | Path,
    depth: int = 1,
    *,
    settings: IndexingSettings | None = None,
) -> Path:
    """Shallow-clone ``url`` into ``dest``. If ``dest`` is already a git repo, fetch + reset.

    The URL is allowlisted in-process before any subprocess is spawned.

    Args:
        url: Git remote URL. Must be ``https`` to an allowlisted host.
        dest: Destination directory.
        depth: Clone/fetch depth.
        settings: Optional indexing settings (host/scheme allowlists).

    Returns:
        Path of ``dest``.

    Raises:
        UnsafeCloneUrlError: ``url`` fails the scheme or host allowlist.
    """
    policy = settings or IndexingSettings.from_env()
    validate_clone_url(
        url,
        allowed_hosts=_csv_set(policy.clone_allowed_hosts) or DEFAULT_CLONE_ALLOWED_HOSTS,
        allowed_schemes=_csv_set(policy.clone_allowed_schemes) or DEFAULT_CLONE_ALLOWED_SCHEMES,
    )
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


def validate_clone_url(
    url: str,
    *,
    allowed_hosts: frozenset[str] | None = None,
    allowed_schemes: frozenset[str] | None = None,
) -> str:
    """Reject clone URLs that are not an allowlisted HTTPS host.

    SSH and SCP-style ``git@host:path`` remotes are rejected. They can target
    internal SSH endpoints and are a common vehicle for shell-metacharacter
    injection; HTTPS to an allowlisted host (default ``github.com``) is enough
    for the FastAPI clone and other public GitHub repositories.

    Args:
        url: Caller-supplied git remote.
        allowed_hosts: Hostnames permitted after lowercasing.
        allowed_schemes: URI schemes permitted. Defaults to ``https`` only.

    Returns:
        The original URL when it passes the allowlist.

    Raises:
        UnsafeCloneUrlError: The URL is not safe to pass to git.
    """
    hosts = allowed_hosts if allowed_hosts is not None else DEFAULT_CLONE_ALLOWED_HOSTS
    schemes = allowed_schemes if allowed_schemes is not None else DEFAULT_CLONE_ALLOWED_SCHEMES
    if not url or url.strip() != url:
        raise UnsafeCloneUrlError(
            agent="indexer",
            message="clone URL must be a non-empty https remote with no surrounding whitespace",
            data={"url": url},
        )
    if any(char in _SHELL_METACHARACTERS for char in url):
        raise UnsafeCloneUrlError(
            agent="indexer",
            message="clone URL contains shell metacharacters",
            data={"url": url},
        )
    if "://" not in url or url.startswith("git@"):
        raise UnsafeCloneUrlError(
            agent="indexer",
            message=(
                "clone URL must use an allowlisted URI scheme (https); "
                "SSH/SCP remotes are not permitted"
            ),
            data={"url": url},
        )
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in schemes:
        raise UnsafeCloneUrlError(
            agent="indexer",
            message=f"clone URL scheme {scheme!r} is not allowlisted",
            data={"url": url, "scheme": scheme},
        )
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeCloneUrlError(
            agent="indexer",
            message="clone URL must not include userinfo",
            data={"url": url},
        )
    if parsed.params or parsed.fragment:
        raise UnsafeCloneUrlError(
            agent="indexer",
            message="clone URL must not include params or a fragment",
            data={"url": url},
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise UnsafeCloneUrlError(
            agent="indexer",
            message="clone URL is missing a hostname",
            data={"url": url},
        )
    if _is_ip_address(host):
        raise UnsafeCloneUrlError(
            agent="indexer",
            message="clone URL must not target a raw IP address",
            data={"url": url, "host": host},
        )
    if host not in hosts:
        raise UnsafeCloneUrlError(
            agent="indexer",
            message=f"clone URL host {host!r} is not allowlisted",
            data={"url": url, "host": host},
        )
    return url


def default_repo_url() -> str:
    """Return ``REPO_URL`` or the FastAPI default.

    Returns:
        str.
    """
    return IndexingSettings.from_env().repo_url or DEFAULT_REPO_URL


def _csv_set(value: str) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in value.split(",") if part.strip())


def _is_ip_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


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
