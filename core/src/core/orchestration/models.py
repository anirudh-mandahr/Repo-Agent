"""Orchestration models used by the orchestrator core loop."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["indexer", "graph_query", "code_analyst", "memory"]
RoutingMode = Literal["llm", "rules", "rules_fallback"]


QueryIntentIntent = Literal[
    "lookup",
    "relationship",
    "explanation",
    "pattern",
    "comparison",
    "indexing",
    "mixed",
]


class QueryIntent(BaseModel):
    """LLM-routed intent for a user query."""

    routing_mode: RoutingMode = "llm"
    intent: QueryIntentIntent
    entities: list[str] = Field(default_factory=list)
    target_agents: list[AgentName] = Field(default_factory=list)
    reasoning: str = ""


class ExecutionPlan(BaseModel):
    """Agent execution plan with per-phase parallelism."""

    routing_mode: RoutingMode = "llm"
    intent: QueryIntent
    phases: list[list[AgentName]] = Field(default_factory=list)


class RuleRouteResult(BaseModel):
    """Cheap keyword routing result before escalating to the LLM."""

    target_agents: list[AgentName] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
    ambiguous: bool = False


class AgentOutput(BaseModel):
    """One agent result for later synthesis."""

    agent: AgentName
    ok: bool = True
    output: Any | None = None
    error: str | None = None
    degraded_note: str | None = None


class SourceAttribution(BaseModel):
    """Best-effort citation of a source snippet."""

    agent: AgentName
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    quote: str | None = None

