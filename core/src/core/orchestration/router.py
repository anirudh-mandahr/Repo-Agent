"""Analyze and route queries to specialist agents."""

from __future__ import annotations

import re

from core.exceptions import SchemaValidationError
from core.llm.provider import LLMProvider, Message
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger
from core.settings import OrchestratorSettings

from .models import (
    AgentName,
    ExecutionPlan,
    QueryIntent,
    QueryIntentIntent,
    RoutingMode,
    RuleRouteResult,
)
from .prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT


def _rule_based_entities(query: str) -> list[str]:
    """Extract a few likely codebase entities for fallback routing.

    This is intentionally heuristic: when structured LLM routing is unavailable,
    we still want downstream agents to receive enough context to execute real
    lookups/snippet fetches instead of short-circuiting on an empty entity list.
    """

    candidates: list[str] = []

    for match in re.findall(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", query):
        token = next((part.strip() for part in match if part.strip()), "")
        if token:
            candidates.append(token)

    for token in re.findall(r"\b(?:[A-Z][A-Za-z0-9_]+|[a-z_][a-z0-9_]*\.[a-z0-9_\.]+)\b", query):
        if token.lower() not in {"what", "how"}:
            candidates.append(token)

    lowered = query.lower()
    if "dependency injection" in lowered:
        candidates.append("fastapi.dependencies.utils")
    if "apirouter" in lowered:
        candidates.append("APIRouter")
    if "fastapi" in lowered:
        candidates.append("FastAPI")

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _estimated_query_tokens(query: str) -> int:
    return max(1, (len(query) + 3) // 4) if query else 0


def rule_based_route(
    query: str,
    *,
    settings: OrchestratorSettings | None = None,
) -> RuleRouteResult:
    """Cheap keyword router that can decline ambiguous or long queries."""

    settings = settings or OrchestratorSettings.from_env()
    lowered = query.lower()
    wants_explanation = bool(re.search(r"explain|how does|why", lowered))
    wants_relationships = bool(
        re.search(r"who calls|depends|import|inherit|extends|subclass|uses", lowered)
    )
    wants_code_examples = bool(
        re.search(r"example|examples|implementation|implemented|codebase|source|snippet", lowered)
    )
    wants_comparison = "compare" in lowered
    wants_indexing = bool(re.search(r"\bindex\b|\breindex\b", lowered))
    wants_lookup = bool(
        re.search(r"^(what is|where is|find|show|list)\b", lowered)
        or "what classes" in lowered
        or "what functions" in lowered
        or "what modules" in lowered
    )
    token_estimate = _estimated_query_tokens(query)
    if token_estimate > settings.rules_max_query_tokens:
        return RuleRouteResult(
            target_agents=[],
            matched_rules=["too_long"],
            ambiguous=True,
        )

    matched_rules: list[str] = []
    categories: set[str] = set()

    if wants_indexing:
        matched_rules.append("index")
        categories.add("index")
    if wants_relationships:
        matched_rules.append("relationship")
        categories.add("relationship")
    if wants_lookup:
        matched_rules.append("lookup")
        categories.add("lookup")
    if wants_comparison:
        matched_rules.append("compare")
        categories.add("compare")
    if wants_explanation:
        matched_rules.append("explain")
        categories.add("explain")
    if wants_code_examples:
        matched_rules.append("examples")
        categories.add("examples")

    compatible_categories = categories - {"lookup"}
    if not matched_rules or len(compatible_categories) > 1:
        return RuleRouteResult(
            target_agents=[],
            matched_rules=matched_rules,
            ambiguous=True,
        )

    intent = next(iter(compatible_categories)) if compatible_categories else "lookup"
    target_agents: list[AgentName]
    if intent == "indexing":
        target_agents = ["indexer"]
    elif intent in {"lookup", "relationship"}:
        target_agents = ["graph_query"]
    elif intent == "comparison":
        target_agents = ["graph_query", "code_analyst"]
    else:
        target_agents = ["graph_query", "code_analyst"]

    return RuleRouteResult(
        target_agents=target_agents,
        matched_rules=matched_rules,
        ambiguous=False,
    )


def intent_from_rule_result(
    query: str,
    route: RuleRouteResult,
    *,
    routing_mode: RoutingMode = "rules",
) -> QueryIntent:
    """Build the existing `QueryIntent` shape from an unambiguous rule result."""

    entities = _rule_based_entities(query)
    intent: QueryIntentIntent
    if route.target_agents == ["indexer"]:
        intent = "indexing"
    elif route.target_agents == ["graph_query"]:
        intent = "relationship" if "relationship" in route.matched_rules else "lookup"
    elif "compare" in route.matched_rules:
        intent = "comparison"
    else:
        intent = "explanation"
    return QueryIntent(
        routing_mode=routing_mode,
        intent=intent,
        entities=entities,
        target_agents=route.target_agents,
        reasoning=f"rules: matched {', '.join(route.matched_rules)}",
    )


async def analyze_query(
    query: str,
    context: ConversationContext | None,
    *,
    llm_provider: LLMProvider,
    token_ledger: TokenLedger | None = None,
    correlation_id: str | None = None,
) -> QueryIntent:
    """Use one structured-output LLM call to produce a QueryIntent.

    If schema validation fails (the provider already retried once), fall back
    to ``rule_based_route``.
    """

    _ = context  # context intentionally not included in the routing prompt
    messages = [
        Message(role="system", content=ROUTER_SYSTEM_PROMPT),
        Message(
            role="user",
            content=ROUTER_USER_PROMPT.format(query=query),
        ),
    ]
    try:
        result = await llm_provider.complete(
            messages,
            response_model=QueryIntent,
            purpose="routing",
            agent="orchestrator",
        )
        if token_ledger is not None and correlation_id is not None:
            token_ledger.record(correlation_id, "routing", result.usage)
    except SchemaValidationError:
        fallback = rule_based_route(query)
        if fallback.ambiguous:
            return QueryIntent(
                routing_mode="rules_fallback",
                intent="mixed",
                entities=_rule_based_entities(query),
                target_agents=["graph_query", "code_analyst"],
                reasoning="rules_fallback: ambiguous fallback defaulted to mixed",
            )
        return intent_from_rule_result(query, fallback, routing_mode="rules_fallback")

    if isinstance(result.parsed, QueryIntent):
        return result.parsed
    # Extremely defensive: if provider returned non-structured result.
    fallback = rule_based_route(query)
    if fallback.ambiguous:
        return QueryIntent(
            routing_mode="rules_fallback",
            intent="mixed",
            entities=_rule_based_entities(query),
            target_agents=["graph_query", "code_analyst"],
            reasoning="rules_fallback: non-structured result defaulted to mixed",
        )
    return intent_from_rule_result(query, fallback, routing_mode="rules_fallback")


def route_to_agents(intent: QueryIntent) -> ExecutionPlan:
    """Convert a QueryIntent into a simple phased execution plan."""

    targets: list[AgentName] = list(intent.target_agents)
    phases: list[list[AgentName]] = []

    if "graph_query" in targets and "code_analyst" in targets:
        # The analyst needs locations/snippets; run graph_query first.
        phases.append([agent for agent in targets if agent != "code_analyst"])
        phases.append(["code_analyst"])
        return ExecutionPlan(
            routing_mode=intent.routing_mode,
            intent=intent,
            phases=phases,
        )

    # Independent lookups in parallel.
    phases.append(targets)
    return ExecutionPlan(
        routing_mode=intent.routing_mode,
        intent=intent,
        phases=phases,
    )

