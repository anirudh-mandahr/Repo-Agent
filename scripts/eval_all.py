"""Run routing, QA, and trap evals with a per-tier scorecard."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalResult:
    name: str
    ok: bool
    detail: str


def _run(name: str, command: list[str]) -> EvalResult:
    env = os.environ.copy()
    env.setdefault("LOG_LEVEL", "ERROR")
    # `make eval` is the CI/offline path. Do not inherit a leftover live run.
    env["EVAL_PROVIDER"] = "offline"
    env.pop("EVAL_WRITE_README", None)
    env.pop("EVAL_SAMPLE_ONLY", None)
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    detail = proc.stdout.strip() or proc.stderr.strip() or f"exit_code={proc.returncode}"
    return EvalResult(name=name, ok=proc.returncode == 0, detail=detail)


def main() -> None:
    """Run routing + QA + trap evals and print the QA scorecard."""
    results = [
        _run("routing", [sys.executable, "scripts/eval_routing.py"]),
        _run("qa", [sys.executable, "scripts/eval_qa.py"]),
        _run(
            "traps",
            [
                sys.executable,
                "-m",
                "pytest",
                "core/tests/test_trap_evals.py",
                "-m",
                "integration",
            ],
        ),
    ]

    print("Eval summary (scorecard is per-tier; hard assertions still gate the exit code):")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"- {result.name}: {status}")
        print(result.detail)

    raise SystemExit(0 if all(result.ok for result in results) else 1)


if __name__ == "__main__":
    main()
