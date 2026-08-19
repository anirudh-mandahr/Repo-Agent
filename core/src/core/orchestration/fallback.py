"""Deterministic, LLM-free synthesis fallback from retrieved specialist evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import AgentName
from .synthesis import (
    _analyst_items,
    _as_mapping,
    _citation_line,
    _evidence_citations,
    _graph_output,
    _retrieval_incomplete,
    _usable_analysis,
    _usable_snippet,
)

EVIDENCE_ONLY_HEADER = "synthesis unavailable, showing retrieved evidence"


def render_evidence_answer(
    query: str,
    agent_outputs: Mapping[AgentName, Any],
    *,
    degraded_reason: str | None = None,
) -> str:
    """Render retrieved agent outputs as structured markdown.

    Used when the synthesis LLM times out or errors so successful retrieval is
    never discarded.

    Args:
        query: User question.
        agent_outputs: Specialist payloads already collected for this turn.
        degraded_reason: Exception class name that triggered the fallback.

    Returns:
        Markdown answer beginning with :data:`EVIDENCE_ONLY_HEADER`.
    """
    lines: list[str] = [EVIDENCE_ONLY_HEADER, ""]
    if degraded_reason:
        lines.append(
            f"The synthesizer failed ({degraded_reason}). "
            "Retrieved specialist evidence is shown below."
        )
    else:
        lines.append("Retrieved specialist evidence is shown below.")
    lines.append("")
    lines.append(f"**Query:** {query}")
    if _retrieval_incomplete(agent_outputs):
        lines.append("")
        lines.append(
            "Retrieval was incomplete because an agent errored, timed out, or was degraded."
        )

    entity_lines = _entity_section(agent_outputs)
    if entity_lines:
        lines.extend(["", "### Entities", *entity_lines])

    dependent_lines = _neighbor_section(agent_outputs, "dependents", "Dependents")
    if dependent_lines:
        lines.extend(["", *dependent_lines])

    dependency_lines = _neighbor_section(agent_outputs, "dependencies", "Dependencies")
    if dependency_lines:
        lines.extend(["", *dependency_lines])

    snippet_lines = _snippet_section(agent_outputs)
    if snippet_lines:
        lines.extend(["", "### Snippets", *snippet_lines])

    analysis_lines = _analysis_section(agent_outputs)
    if analysis_lines:
        lines.extend(["", "### Analysis", *analysis_lines])

    citations = _evidence_citations(agent_outputs)
    if citations:
        lines.extend(["", "### Sources", *[f"- {item}" for item in citations]])

    if not any(
        (entity_lines, dependent_lines, dependency_lines, snippet_lines, analysis_lines)
    ):
        lines.extend(["", "No structured evidence was retrieved."])

    return "\n".join(lines).rstrip() + "\n"


def _entity_section(agent_outputs: Mapping[AgentName, Any]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    graph = _graph_output(agent_outputs)
    if graph is None:
        return lines

    def _add(item: Any) -> None:
        payload = _as_mapping(item)
        if payload is None:
            return
        citation = _citation_line(payload)
        qualified = str(payload.get("qualified_name") or payload.get("name") or "").strip()
        label = citation or qualified
        if not label or label in seen:
            return
        seen.add(label)
        lines.append(f"- {label}")

    for key in ("analysis_candidates", "candidates"):
        rows = graph.get(key)
        if isinstance(rows, list):
            for row in rows:
                _add(row)
    entities = graph.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            payload = _as_mapping(entity)
            if payload is None:
                continue
            matches = payload.get("matches")
            if isinstance(matches, list):
                for match in matches:
                    _add(match)
            else:
                _add(payload)
    return lines


def _neighbor_section(
    agent_outputs: Mapping[AgentName, Any],
    key: str,
    title: str,
) -> list[str]:
    graph = _graph_output(agent_outputs)
    if graph is None:
        return []
    rows = graph.get(key)
    if not isinstance(rows, list) or not rows:
        return []
    lines = [f"### {title}"]
    seen: set[str] = set()
    for row in rows:
        payload = _as_mapping(row)
        if payload is None:
            continue
        neighbors = payload.get("neighbors")
        items = neighbors if isinstance(neighbors, list) else [payload]
        for item in items:
            neighbor = _as_mapping(item)
            if neighbor is None:
                continue
            citation = _citation_line(neighbor)
            qualified = str(
                neighbor.get("qualified_name") or neighbor.get("name") or ""
            ).strip()
            rel = str(neighbor.get("relationship_type") or "").strip()
            label = citation or qualified
            if not label:
                continue
            if rel:
                label = f"{label} ({rel})"
            if label in seen:
                continue
            seen.add(label)
            lines.append(f"- {label}")
    if len(lines) == 1:
        return []
    return lines


def _snippet_section(agent_outputs: Mapping[AgentName, Any]) -> list[str]:
    lines: list[str] = []
    for item in _analyst_items(agent_outputs, "snippets"):
        payload = _as_mapping(item)
        if payload is None or not _usable_snippet(payload):
            continue
        heading = _citation_line(payload) or str(payload.get("qualified_name") or "snippet")
        text = str(payload.get("text") or "")
        lines.append(f"#### {heading}")
        lines.append("")
        lines.append("```")
        lines.append(text.rstrip())
        lines.append("```")
        lines.append("")
    return lines


def _analysis_section(agent_outputs: Mapping[AgentName, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("explanations", "function_analyses", "comparison"):
        for item in _analyst_items(agent_outputs, key):
            payload = _as_mapping(item)
            if payload is None or not _usable_analysis(payload):
                continue
            heading = _citation_line(payload) or str(
                payload.get("qualified_name")
                or payload.get("name_a")
                or key
            )
            body = str(
                payload.get("explanation")
                or payload.get("analysis")
                or payload.get("summary")
                or payload.get("text")
                or payload.get("comparison")
                or ""
            )
            lines.append(f"#### {heading}")
            lines.append("")
            lines.append(body.rstrip())
            lines.append("")
    return lines
