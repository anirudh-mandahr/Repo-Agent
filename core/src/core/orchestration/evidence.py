"""Evaluate specialist evidence and produce a heuristic follow-up plan."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.logging import get_logger

from .executor import _tool_plan
from .models import (
    AgentName,
    AgentOutput,
    EvidenceAssessment,
    ExecutionPlan,
)
from .router import _LOOKUP_FILLER, retrieval_search_terms, route_to_agents
from .scope import is_out_of_scope

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_GRAPH_TOOLS = frozenset(
    {"find_entity", "get_dependencies", "get_dependents", "trace_imports", "find_related"}
)
_ANALYST_TOOL_BUCKETS: dict[str, str] = {
    "explain_implementation": "explanations",
    "analyze_function": "function_analyses",
    "analyze_class": "class_analyses",
    "get_code_snippet": "snippets",
    "compare_implementations": "comparison",
    "find_patterns": "patterns",
}
_REFINEMENT_STOPWORDS = frozenset(
    {
        "about",
        "across",
        "all",
        "also",
        "any",
        "been",
        "being",
        "both",
        "but",
        "can",
        "code",
        "codebase",
        "could",
        "did",
        "do",
        "done",
        "each",
        "example",
        "examples",
        "few",
        "get",
        "give",
        "got",
        "had",
        "handle",
        "handles",
        "handling",
        "has",
        "have",
        "implementation",
        "implemented",
        "in",
        "injection",
        "into",
        "it",
        "just",
        "like",
        "made",
        "make",
        "me",
        "more",
        "most",
        "need",
        "not",
        "off",
        "only",
        "or",
        "other",
        "out",
        "over",
        "request",
        "same",
        "should",
        "snippet",
        "some",
        "source",
        "such",
        "tell",
        "than",
        "that",
        "their",
        "them",
        "these",
        "those",
        "to",
        "under",
        "use",
        "used",
        "using",
        "validation",
        "via",
        "want",
        "was",
        "we",
        "were",
        "with",
        "work",
        "working",
        "works",
        "would",
        "you",
        "your",
    }
)


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
    keep_terms = _successful_search_terms(plan, outputs)
    failed_analyst = _failed_analyst_tools(query, plan, outputs)

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
            broader_terms=_broader_terms(query, tried, keep_terms=keep_terms),
            suggested_tools=failed_analyst,
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
                broader_terms=_broader_terms(query, tried, keep_terms=keep_terms),
                suggested_tools=failed_analyst,
            )
        return EvidenceAssessment(sufficient=True, reason="graph_hits")

    if analyst_evidence:
        return EvidenceAssessment(sufficient=True, reason="analyst_evidence")

    broader = _broader_terms(query, tried, keep_terms=keep_terms)
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

    suggested_tools: list[str] = []
    if needs_graph:
        suggested_tools.append("find_entity")
    suggested_tools.extend(tool for tool in failed_analyst if tool not in suggested_tools)

    return EvidenceAssessment(
        sufficient=False,
        reason=reason,
        broader_terms=broader,
        suggested_agents=suggested,
        suggested_tools=suggested_tools,
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

    retry_tools: list[str] | None = None
    if "code_analyst" in plan.agents:
        retry_tools = [
            tool for tool in assessment.suggested_tools if tool not in _GRAPH_TOOLS
        ]

    changed_agents = new_agents != list(intent.target_agents)
    changed_terms = bool(new_terms)
    if not changed_agents and not changed_terms and not retry_tools:
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
        retry_tools=retry_tools,
        iteration=plan.iteration + 1,
    )
    return follow_up.model_copy(
        update={
            "search_terms": search_terms,
            "iteration": plan.iteration + 1,
            "refinement_reason": assessment.reason,
            "retry_tools": retry_tools,
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


def _term_produced_hits(value: Any) -> bool:
    payload = _as_mapping(value)
    if payload is None:
        return False
    matches = payload.get("matches")
    if isinstance(matches, list):
        return any(_is_entity_hit(match) for match in matches)
    return _is_entity_hit(payload)


def _successful_search_terms(
    plan: ExecutionPlan,
    outputs: Mapping[AgentName, AgentOutput],
) -> list[str]:
    """Prior search terms that produced graph hits and should be carried forward."""
    if not _has_graph_hits(outputs):
        return []
    graph = outputs.get("graph_query")
    payload = _as_mapping(graph)
    queried: list[str] = []
    entities: list[Any] = []
    if payload is not None:
        inner = payload.get("output")
        if isinstance(inner, Mapping):
            raw_queried = inner.get("queried_entities")
            if isinstance(raw_queried, list):
                queried = [str(item) for item in raw_queried if str(item).strip()]
            raw_entities = inner.get("entities")
            if isinstance(raw_entities, list):
                entities = list(raw_entities)
    if queried and entities and len(queried) == len(entities):
        kept = [
            term
            for term, result in zip(queried, entities, strict=True)
            if _term_produced_hits(result)
        ]
        if kept:
            return kept
    if queried:
        return queried
    if plan.search_terms:
        return [term for term in plan.search_terms if term.strip()]
    return [term for term in plan.intent.entities if term.strip()]


def _invoked_tool_names(output: AgentOutput | None) -> set[str]:
    if output is None:
        return set()
    names: set[str] = set()
    for item in output.tools_invoked:
        if "." in item:
            names.add(item.split(".", 1)[1])
        elif item:
            names.add(item)
    return names


def _analyst_tool_succeeded(tool: str, inner: Mapping[str, Any] | None) -> bool:
    if inner is None:
        return False
    bucket = _ANALYST_TOOL_BUCKETS.get(tool)
    if bucket is None:
        return False
    value = inner.get(bucket)
    if tool == "get_code_snippet":
        if not isinstance(value, list):
            return False
        for item in value:
            mapped = _as_mapping(item)
            if mapped is not None and mapped.get("text") and not mapped.get("error"):
                return True
        return False
    if isinstance(value, list):
        return any(_usable_analysis(item) for item in value)
    return _usable_analysis(value)


def _failed_analyst_tools(
    query: str,
    plan: ExecutionPlan,
    outputs: Mapping[AgentName, AgentOutput],
) -> list[str]:
    """Analyst tools that should be retried against refined candidates."""
    if "code_analyst" not in plan.agents:
        return []
    planned = _tool_plan(plan.intent.intent, "code_analyst", query).get("parallel", ())
    if not planned:
        return []
    analyst = outputs.get("code_analyst")
    inner: Mapping[str, Any] | None = None
    if analyst is not None:
        payload = _as_mapping(analyst)
        if payload is not None:
            output = payload.get("output")
            inner = output if isinstance(output, Mapping) else None
    invoked = _invoked_tool_names(analyst)
    agent_failed = _agent_failed(analyst)
    had_candidates = _has_graph_hits(outputs)
    failed: list[str] = []
    for tool in planned:
        if _analyst_tool_succeeded(tool, inner):
            continue
        if tool in invoked or agent_failed or not had_candidates:
            failed.append(tool)
    return failed


def _looks_like_identifier(token: str) -> bool:
    stripped = token.strip()
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    if "." in stripped or "_" in stripped:
        return True
    return any(ch.isupper() for ch in stripped) and any(ch.islower() for ch in stripped)


def _is_refinement_token(token: str) -> bool:
    stripped = token.strip()
    if len(stripped) <= 2:
        return False
    lowered = stripped.lower()
    if lowered in _LOOKUP_FILLER or lowered in _REFINEMENT_STOPWORDS:
        return False
    return _looks_like_identifier(stripped)


def _broader_terms(
    query: str,
    already_tried: Sequence[str],
    *,
    keep_terms: Sequence[str] = (),
) -> list[str]:
    """Carry successful terms and add identifier-like refinements, not generic English."""
    tried = {item.strip().lower() for item in already_tried if str(item).strip()}
    broader: list[str] = []
    seen: set[str] = set()

    def _add(item: str) -> None:
        key = item.strip()
        lowered = key.lower()
        if not key or lowered in seen:
            return
        seen.add(lowered)
        broader.append(key)

    for item in keep_terms:
        _add(str(item))

    candidates = retrieval_search_terms(query, None)
    for token in _TOKEN_RE.findall(query):
        if _is_refinement_token(token):
            candidates.append(token)
    for item in candidates:
        key = item.strip()
        lowered = key.lower()
        if not key or lowered in tried or lowered in seen:
            continue
        if not _is_refinement_token(key):
            continue
        seen.add(lowered)
        broader.append(key)

    stripped = query.strip()
    if stripped and stripped.lower() not in tried and stripped.lower() not in seen:
        _add(stripped)
    return broader
