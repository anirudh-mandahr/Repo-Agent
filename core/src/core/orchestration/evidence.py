"""Evaluate specialist evidence and produce a heuristic follow-up plan."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.logging import get_logger

from .models import (
    AgentName,
    AgentOutput,
    EvidenceAssessment,
    ExecutionPlan,
)
from .router import _LOOKUP_FILLER, retrieval_search_terms, route_to_agents
from .scope import is_out_of_scope

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def evaluate_evidence(
    query: str,
    plan: ExecutionPlan,
    outputs: Mapping[AgentName, AgentOutput],
) -> EvidenceAssessment:
    """Decide whether collected evidence can answer ``query``.

    Args:
        query: User question.
        plan: Plan that produced ``outputs``.
        outputs: Per-agent results from ``run_plan``.

    Returns:
        Sufficiency assessment plus suggested refinements.
    """
    intent = plan.intent
    planned = set(plan.agents)
    if is_out_of_scope(query):
        return EvidenceAssessment(sufficient=True, reason="out_of_scope")
    graph_hits = _has_graph_hits(outputs)
    analyst_evidence = _has_analyst_evidence(outputs)
    graph_failed = _agent_failed(outputs.get("graph_query"))
    analyst_failed = _agent_failed(outputs.get("code_analyst"))
    tried = _tried_terms(plan, outputs)

    if intent.intent == "indexing":
        indexer = outputs.get("indexer")
        if indexer is not None and indexer.ok:
            return EvidenceAssessment(sufficient=True, reason="index_complete")
        return EvidenceAssessment(
            sufficient=False,
            reason="indexer_failed",
            suggested_agents=["indexer"],
        )

    if intent.intent == "pattern":
        if analyst_evidence or (outputs.get("code_analyst") is not None and not analyst_failed):
            return EvidenceAssessment(sufficient=True, reason="pattern_complete")
        return EvidenceAssessment(
            sufficient=False,
            reason="pattern_empty",
            suggested_agents=["code_analyst"] if "code_analyst" not in planned else [],
            broader_terms=_broader_terms(query, tried),
        )

    needs_graph = intent.intent in {
        "lookup",
        "relationship",
        "explanation",
        "comparison",
        "mixed",
    }
    needs_analyst = intent.intent in {"explanation", "comparison", "mixed"}
    if intent.intent in {"lookup", "relationship"} and "code_analyst" in planned:
        needs_analyst = True

    if needs_graph and graph_hits:
        if needs_analyst and not analyst_evidence and "code_analyst" not in planned:
            return EvidenceAssessment(
                sufficient=False,
                reason="missing_analyst",
                suggested_agents=["code_analyst"],
            )
        if needs_analyst and analyst_failed and not analyst_evidence:
            return EvidenceAssessment(
                sufficient=False,
                reason="analyst_failed",
                suggested_agents=["code_analyst"],
                broader_terms=_broader_terms(query, tried),
            )
        return EvidenceAssessment(sufficient=True, reason="graph_hits")

    if analyst_evidence:
        return EvidenceAssessment(sufficient=True, reason="analyst_evidence")

    broader = _broader_terms(query, tried)
    suggested: list[AgentName] = []
    if "code_analyst" not in planned:
        suggested.append("code_analyst")
    if "graph_query" not in planned and needs_graph:
        suggested.append("graph_query")

    reason = "no_graph_hits"
    if graph_failed:
        reason = "graph_failed"
    elif needs_analyst and not analyst_evidence:
        reason = "insufficient_evidence"

    return EvidenceAssessment(
        sufficient=False,
        reason=reason,
        broader_terms=broader,
        suggested_agents=suggested,
        suggested_tools=["find_entity"] if needs_graph else [],
    )


def refine_plan(
    query: str,
    plan: ExecutionPlan,
    assessment: EvidenceAssessment,
) -> ExecutionPlan | None:
    """Build a heuristic follow-up plan from an insufficient first round.

    This expands search terms and missing agents from the query and prior
    evidence. It is not an LLM replan.

    Args:
        query: User question.
        plan: Plan that just ran.
        assessment: Sufficiency assessment.

    Returns:
        A new :class:`ExecutionPlan`, or ``None`` when nothing new can be tried.
    """
    if assessment.sufficient:
        return None
    intent = plan.intent
    new_agents = list(intent.target_agents)
    for agent in assessment.suggested_agents:
        if agent not in new_agents:
            new_agents.append(agent)
    new_terms = [term for term in assessment.broader_terms if term.strip()]
    new_entities = list(intent.entities)
    for term in new_terms:
        if term not in new_entities:
            new_entities.append(term)

    changed_agents = new_agents != list(intent.target_agents)
    changed_terms = bool(new_terms)
    if not changed_agents and not changed_terms:
        return None

    new_kind = intent.intent
    if changed_agents and new_kind == "lookup" and "code_analyst" in new_agents:
        new_kind = "mixed"
    elif changed_agents and new_kind in {"lookup", "relationship"} and "code_analyst" in new_agents:
        new_kind = "explanation" if new_kind == "lookup" else new_kind

    new_intent = intent.model_copy(
        update={
            "entities": new_entities,
            "target_agents": new_agents,
            "intent": new_kind,
            "reasoning": f"refine: {assessment.reason}",
        }
    )
    follow_up = route_to_agents(new_intent)
    search_terms = new_terms or None
    log.info(
        "orchestrator.refine",
        reason=assessment.reason,
        agents=new_agents,
        search_terms=search_terms or new_entities,
        iteration=plan.iteration + 1,
    )
    return follow_up.model_copy(
        update={
            "search_terms": search_terms,
            "iteration": plan.iteration + 1,
            "refinement_reason": assessment.reason,
        }
    )


def merge_agent_outputs(
    base: Mapping[AgentName, AgentOutput],
    extra: Mapping[AgentName, AgentOutput],
) -> dict[AgentName, AgentOutput]:
    """Merge a follow-up round into prior outputs.

    Args:
        base: Outputs from earlier iterations.
        extra: Outputs from the latest iteration.

    Returns:
        Combined per-agent outputs.
    """
    merged = dict(base)
    for agent, incoming in extra.items():
        previous = merged.get(agent)
        if previous is None:
            merged[agent] = incoming
            continue
        merged[agent] = _merge_one(previous, incoming)
    return merged


def _merge_one(previous: AgentOutput, incoming: AgentOutput) -> AgentOutput:
    tools = list(previous.tools_invoked)
    for item in incoming.tools_invoked:
        if item not in tools:
            tools.append(item)
    if incoming.ok and not previous.ok:
        return incoming.model_copy(update={"tools_invoked": tools})
    if incoming.ok and previous.ok:
        prev_out = previous.output if isinstance(previous.output, dict) else {}
        new_out = incoming.output if isinstance(incoming.output, dict) else {}
        combined: dict[str, Any] = dict(prev_out)
        for key, value in new_out.items():
            existing = combined.get(key)
            if isinstance(value, list) and isinstance(existing, list):
                combined[key] = [*existing, *value]
            else:
                combined[key] = value
        return incoming.model_copy(
            update={
                "output": combined or incoming.output,
                "tools_invoked": tools,
                "degraded_note": incoming.degraded_note or previous.degraded_note,
            }
        )
    if previous.ok:
        return previous.model_copy(update={"tools_invoked": tools})
    return incoming.model_copy(update={"tools_invoked": tools})


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    payload = value.model_dump() if hasattr(value, "model_dump") else value
    return payload if isinstance(payload, Mapping) else None


def _agent_failed(output: AgentOutput | None) -> bool:
    if output is None:
        return False
    if output.ok is False:
        return True
    return bool(output.degraded_note or output.error)


def _is_entity_hit(value: Any) -> bool:
    payload = _as_mapping(value)
    if payload is None:
        return False
    return bool(
        payload.get("qualified_name")
        or payload.get("file_path")
        or payload.get("filePath")
        or payload.get("name")
        or payload.get("path")
    )


def _has_graph_hits(outputs: Mapping[AgentName, AgentOutput]) -> bool:
    graph = outputs.get("graph_query")
    payload = _as_mapping(graph)
    if payload is None:
        return False
    inner = payload.get("output")
    if not isinstance(inner, Mapping):
        return False
    candidates = inner.get("candidates")
    if isinstance(candidates, list) and any(_is_entity_hit(item) for item in candidates):
        return True
    entities = inner.get("entities")
    if not isinstance(entities, list):
        return False
    for item in entities:
        mapped = _as_mapping(item)
        if mapped is None:
            continue
        matches = mapped.get("matches")
        if isinstance(matches, list) and any(_is_entity_hit(match) for match in matches):
            return True
        if _is_entity_hit(mapped):
            return True
    return False


def _usable_analysis(value: Any) -> bool:
    payload = _as_mapping(value)
    if payload is None:
        return False
    if payload.get("error"):
        return False
    return bool(
        payload.get("explanation")
        or payload.get("analysis")
        or payload.get("summary")
        or payload.get("text")
        or payload.get("comparison")
        or payload.get("instances")
        or payload.get("pattern")
    )


def _has_analyst_evidence(outputs: Mapping[AgentName, AgentOutput]) -> bool:
    analyst = outputs.get("code_analyst")
    payload = _as_mapping(analyst)
    if payload is None:
        return False
    inner = payload.get("output")
    if not isinstance(inner, Mapping):
        return False
    snippets = inner.get("snippets")
    if isinstance(snippets, list):
        for snippet in snippets:
            mapped = _as_mapping(snippet)
            if mapped is not None and mapped.get("text") and not mapped.get("error"):
                return True
    for key in ("explanations", "function_analyses", "comparison", "patterns"):
        value = inner.get(key)
        if isinstance(value, list) and any(_usable_analysis(item) for item in value):
            return True
        if _usable_analysis(value):
            return True
    return False


def _tried_terms(plan: ExecutionPlan, outputs: Mapping[AgentName, AgentOutput]) -> list[str]:
    tried: list[str] = []
    if plan.search_terms:
        tried.extend(plan.search_terms)
    tried.extend(plan.intent.entities)
    graph = outputs.get("graph_query")
    payload = _as_mapping(graph)
    if payload is not None:
        inner = payload.get("output")
        if isinstance(inner, Mapping):
            queried = inner.get("queried_entities")
            if isinstance(queried, list):
                tried.extend(str(item) for item in queried)
    return tried


def _broader_terms(query: str, already_tried: Sequence[str]) -> list[str]:
    tried = {item.strip().lower() for item in already_tried if str(item).strip()}
    candidates = retrieval_search_terms(query, None)
    for token in _TOKEN_RE.findall(query):
        if len(token) <= 2 or token.lower() in _LOOKUP_FILLER:
            continue
        candidates.append(token)
    stripped = query.strip()
    if stripped:
        candidates.append(stripped)
    broader: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip()
        lowered = key.lower()
        if not key or lowered in tried or lowered in seen:
            continue
        seen.add(lowered)
        broader.append(key)
    return broader
