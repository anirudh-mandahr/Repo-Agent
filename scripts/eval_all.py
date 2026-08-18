"""Run routing and trap evals with one consolidated PASS/FAIL summary."""

from __future__ import annotations

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
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    detail = proc.stdout.strip() or proc.stderr.strip() or f"exit_code={proc.returncode}"
    return EvalResult(name=name, ok=proc.returncode == 0, detail=detail)


def main() -> None:
    results = [
        _run("routing", [sys.executable, "scripts/eval_routing.py"]),
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

    print("Eval summary:")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"- {result.name}: {status}")
        print(result.detail)

    raise SystemExit(0 if all(result.ok for result in results) else 1)


if __name__ == "__main__":
    main()
