"""Deterministic routing evals plus optional live-routing comparison."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from core.orchestration.router import analyze_query, intent_from_rule_result, rule_based_route
from core.settings import OrchestratorSettings
from orchestrator.service import build_default_llm_provider

EVAL_PATH = Path(__file__).resolve().parents[1] / "evals" / "routing.jsonl"


@dataclass(frozen=True)
class RoutingCase:
    query: str
    expected_agents: list[str]
    min_agents: int
    expected_mode: str


def _load_cases() -> list[RoutingCase]:
    cases: list[RoutingCase] = []
    for raw in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        cases.append(
            RoutingCase(
                query=str(payload["query"]),
                expected_agents=[str(agent) for agent in payload["expected_agents"]],
                min_agents=int(payload["min_agents"]),
                expected_mode=str(payload["expected_mode"]),
            )
        )
    return cases


def _check_rule_based() -> tuple[bool, list[str]]:
    ok = True
    lines: list[str] = []
    zero_llm = 0
    settings = OrchestratorSettings.from_env()
    for case in _load_cases():
        route = rule_based_route(case.query, settings=settings)
        if route.ambiguous:
            actual_mode = "llm"
            intent = None
            actual_agents = case.expected_agents
        else:
            actual_mode = "rules"
            zero_llm += 1
            intent = intent_from_rule_result(case.query, route)
            actual_agents = intent.target_agents
        if (
            actual_mode != case.expected_mode
            or actual_agents != case.expected_agents
            or len(actual_agents) < case.min_agents
        ):
            ok = False
            lines.append(
                "FAIL "
                f"query={case.query!r} expected={case.expected_agents} "
                f"expected_mode={case.expected_mode} min_agents={case.min_agents} "
                f"actual={actual_agents} actual_mode={actual_mode}"
            )
            continue
        lines.append(f"PASS query={case.query!r} agents={actual_agents} mode={actual_mode}")
    total = len(_load_cases())
    lines.append(f"Zero-LLM routing fraction: {zero_llm}/{total} ({zero_llm / total:.0%})")
    return ok, lines


async def _live_report() -> list[str]:
    provider = build_default_llm_provider()
    lines = ["Live routing agreement report (non-gating):"]
    for case in _load_cases():
        try:
            intent = await analyze_query(case.query, None, llm_provider=provider)
            actual = intent.target_agents
            status = "AGREE" if actual == case.expected_agents else "DIFF"
            lines.append(
                f"{status} query={case.query!r} expected={case.expected_agents} actual={actual}"
            )
        except Exception as exc:  # pragma: no cover - reporting only
            lines.append(f"ERROR query={case.query!r} error={exc}")
    return lines


def main() -> None:
    ok, lines = _check_rule_based()
    print("Routing eval results:")
    for line in lines:
        print(line)

    if os.environ.get("RUN_LIVE") == "1":
        for line in asyncio.run(_live_report()):
            print(line)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"routing eval failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
