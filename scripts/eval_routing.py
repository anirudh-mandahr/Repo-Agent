"""Executed-plan routing evals plus optional live-routing comparison."""

from __future__ import annotations

import asyncio
import json
import os
import sys

os.environ.setdefault("LOG_LEVEL", "ERROR")

from core.eval.harness import load_routing_cases, run_executed_plan_eval
from core.eval.report import format_query_table, format_scorecard_text
from core.orchestration.router import analyze_query
from core.orchestration.service import build_default_llm_provider


async def _live_report() -> list[str]:
    provider = build_default_llm_provider()
    lines = ["Live routing agreement report (non-gating):"]
    for case in load_routing_cases():
        try:
            intent = await analyze_query(case.query, None, llm_provider=provider)
            actual = intent.target_agents
            status = "AGREE" if list(actual) == list(case.expected_agents) else "DIFF"
            lines.append(
                f"{status} query={case.query!r} expected={case.expected_agents} actual={actual}"
            )
        except Exception as exc:  # pragma: no cover - reporting only
            lines.append(f"ERROR query={case.query!r} error={exc}")
    return lines


def main() -> None:
    """Score ``metadata.tools_invoked`` from ``handle_query``, not router intent."""
    scorecard = asyncio.run(run_executed_plan_eval(load_routing_cases(), tier="complex"))
    print("Routing eval results (executed plan via tools_invoked):")
    print(format_scorecard_text(scorecard))
    print()
    print(format_query_table(scorecard.turns))

    if os.environ.get("RUN_LIVE") == "1":
        for line in asyncio.run(_live_report()):
            print(line)

    raise SystemExit(0 if scorecard.hard_assertions_passed else 1)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"routing eval failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except json.JSONDecodeError as exc:
        print(f"routing eval failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
