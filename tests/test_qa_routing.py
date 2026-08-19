"""Offline routing checks for the labelled QA set plus the original golden nine."""

from __future__ import annotations

import json
from pathlib import Path

from core.orchestration.router import intent_from_rule_result, rule_based_route
from core.settings import OrchestratorSettings

ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "evals" / "qa.jsonl"
ROUTING_PATH = ROOT / "evals" / "routing.jsonl"


def _turns() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in (ROUTING_PATH, QA_PATH):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            payload = json.loads(raw)
            turns = payload.get("turns")
            items = turns if isinstance(turns, list) else [payload]
            for item in items:
                query = str(item["query"])
                if query in seen:
                    continue
                seen.add(query)
                rows.append(item)
    return rows


def test_qa_jsonl_has_fifty_labelled_cases() -> None:
    cases = [
        json.loads(raw)
        for raw in QA_PATH.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    assert len(cases) == 50
    traps = [case for case in cases if case.get("out_of_scope")]
    multiturn = [case for case in cases if isinstance(case.get("turns"), list)]
    assert len(traps) == 6
    assert len(multiturn) == 8
    tiers = {str(case.get("tier")) for case in cases}
    assert {"simple", "medium", "complex", "trap", "multi-turn"} <= tiers


def test_labelled_routing_matches_rule_based_router() -> None:
    settings = OrchestratorSettings.from_env()
    failures: list[str] = []
    for item in _turns():
        query = str(item["query"])
        expected_agents = [str(agent) for agent in item["expected_agents"]]
        expected_mode = str(item["expected_mode"])
        min_agents = int(item.get("min_agents") or len(expected_agents))
        route = rule_based_route(query, settings=settings)
        if route.ambiguous:
            actual_mode = "llm"
            actual_agents = expected_agents
        else:
            actual_mode = "rules"
            actual_agents = intent_from_rule_result(query, route).target_agents
        if (
            actual_mode != expected_mode
            or list(actual_agents) != expected_agents
            or len(actual_agents) < min_agents
        ):
            failures.append(
                f"{query!r} expected={expected_agents}/{expected_mode} "
                f"actual={actual_agents}/{actual_mode}"
            )
    assert not failures, "\n".join(failures)
