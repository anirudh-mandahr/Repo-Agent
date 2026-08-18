"""
Prove incremental indexing behavior against the running Compose stack.

Contract:
- Trigger a full index (record IndexReport).
- Modify exactly one Python file under /repo by appending a comment.
- Reindex that specific file only.
- Assert the incremental report indicates exactly one file reindexed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

INDEXER_SERVICE = "indexer"
COMPOSE_ARGS: list[str] = ["docker", "compose"]


def _docker_exec_indexer_python(code: str) -> str:
    """Execute Python code inside the indexer container and return stdout."""
    proc = subprocess.run(
        [*COMPOSE_ARGS, "exec", "-T", INDEXER_SERVICE, "python", "-c", code],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker exec failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def _extract_prefixed_json(stdout: str, *, prefix: str) -> dict[str, Any]:
    """
    Extract JSON from a line like:
        __REPORT__= {...}
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith(prefix):
            payload = line[len(prefix) :]
            return json.loads(payload)
    raise RuntimeError(f"missing prefix {prefix!r} in output")


def _format_side_by_side(
    title_left: str,
    title_right: str,
    report1: dict[str, Any],
    report2: dict[str, Any],
) -> None:
    keys = [
        "status",
        "files_seen",
        "files_indexed",
        "files_skipped",
        "nodes_written",
        "rels_written",
        "unresolved_calls",
        "parse_errors",
        "duration_s",
        "detail",
    ]
    print(f"\n=== {title_left} ===")
    for k in keys:
        print(f"{k:16}: {report1.get(k)}")
    print(f"\n=== {title_right} ===")
    for k in keys:
        print(f"{k:16}: {report2.get(k)}")


def main() -> None:
    file_marker = "# prove-incremental\n"
    rel_path: str

    # 1) Full index
    full_code = (
        "from core.indexing import trigger_index; "
        "r=trigger_index(repo_root='/repo'); "
        "print('__REPORT__='+r.model_dump_json())"
    )
    full_stdout = _docker_exec_indexer_python(full_code)
    report_full = _extract_prefixed_json(full_stdout, prefix="__REPORT__=")

    assert report_full["status"] == "ok", f"full index failed: {report_full.get('detail')}"
    assert int(report_full.get("files_seen", 0)) > 0

    # 2) Append a comment to exactly one file under /repo and capture relative path
    modify_code = f"""
from pathlib import Path
root = Path("/repo").resolve()
marker = {file_marker!r}
skip_parts = {{
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}}

for p in root.rglob("*.py"):
    if any(part in skip_parts for part in p.parts):
        continue
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if marker in txt:
        continue
    if not txt.endswith("\\n"):
        p.write_text(txt + "\\n" + marker, encoding="utf-8", errors="ignore")
    else:
        p.write_text(txt + marker, encoding="utf-8", errors="ignore")
    rel = p.resolve().relative_to(root).as_posix()
    print('__REL__=' + rel)
    break
else:
    raise SystemExit("no suitable python file found under /repo")
"""
    modify_stdout = _docker_exec_indexer_python(modify_code)
    rel_path = None  # type: ignore[assignment]
    for line in reversed(modify_stdout.splitlines()):
        line = line.strip()
        if line.startswith("__REL__="):
            rel_path = line[len("__REL__=") :]
            break
    if not rel_path:
        raise RuntimeError("could not determine modified relative path")

    # 3) Reindex only that file
    incremental_code = (
        "from core.indexing import run_index_file; "
        f"r=run_index_file({rel_path!r}); "
        "print('__REPORT__='+r.model_dump_json())"
    )
    incremental_stdout = _docker_exec_indexer_python(incremental_code)
    report_inc = _extract_prefixed_json(incremental_stdout, prefix="__REPORT__=")

    # 4) Assertions: reindexed exactly one file
    assert report_inc["status"] == "ok", f"incremental index failed: {report_inc.get('detail')}"
    assert report_inc["files_seen"] == 1, report_inc
    assert report_inc["files_indexed"] == 1, report_inc
    assert report_inc["files_skipped"] == 0, report_inc
    assert report_inc.get("parse_errors") == [], report_inc
    assert int(report_inc.get("nodes_written", 0)) > 0, report_inc

    # 5) Print reports
    _format_side_by_side(
        title_left=f"full index (trigger_index) modified={rel_path}",
        title_right="incremental index (index_file for modified path)",
        report1=report_full,
        report2=report_inc,
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"prove_incremental assertion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"prove_incremental failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

