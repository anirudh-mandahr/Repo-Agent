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
    """One specialist set for a retrieval/analysis round.

    Agents run concurrently. The executor waits internally when ``code_analyst``
    needs graph coordinates it does not already have. Sequencing is not modeled
    as multiple plan phases.
    """

    routing_mode: RoutingMode = "llm"
    intent: QueryIntent
    agents: list[AgentName] = Field(default_factory=list)
    search_terms: list[str] | None = None
    iteration: int = 1
    refinement_reason: str = ""


class EvidenceAssessment(BaseModel):
    """Whether collected specialist evidence can answer the query."""

    sufficient: bool
    reason: str
    broader_terms: list[str] = Field(default_factory=list)
    suggested_agents: list[AgentName] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)


class PlanIteration(BaseModel):
    """One planning/execution round, emitted as a routing SSE/WebSocket event."""

    iteration: int
    routing_mode: RoutingMode = "llm"
    agents: list[AgentName] = Field(default_factory=list)
    tools_invoked: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    sufficient: bool = False
    reason: str = ""
    refinement: str | None = None


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
    tools_invoked: list[str] = Field(default_factory=list)


class SourceAttribution(BaseModel):
    """Best-effort citation of a source snippet."""

    agent: AgentName
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    quote: str | None = None


class PromptTruncation(BaseModel):
    """Record of synthesis-prompt fields dropped to fit the token budget."""

    original_estimated_tokens: int
    final_estimated_tokens: int
    budget: int
    dropped: list[dict[str, Any]] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """Final synthesizer output plus degradation / budget metadata."""

    answer: str
    evidence_only: bool = False
    degraded_reason: str | None = None
    prompt_truncated: PromptTruncation | None = None
    estimated_tokens: int = 0
    prompt_chars: int = 0

