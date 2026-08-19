"""Analyze and route queries to specialist agents."""

from __future__ import annotations

import re

from core.exceptions import RoutingError
from core.llm.provider import LLMProvider, Message
from core.logging import get_logger
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

log = get_logger(__name__)

_REFERRING_RE = re.compile(
    r"\b(?:it|its|this|that|these|those|them|they|their)\b",
    re.IGNORECASE,
)
_CODEBASE_GROUNDING_RE = re.compile(
    r"example|examples|implementation|implemented|codebase|source|snippet",
    re.IGNORECASE,
)
_AGENT_ORDER: tuple[AgentName, ...] = ("indexer", "graph_query", "code_analyst", "memory")


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

    return _dedupe(candidates)


_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOOKUP_FILLER = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "class",
        "classes",
        "compare",
        "depend",
        "dependent",
        "dependents",
        "depends",
        "does",
        "explain",
        "file",
        "files",
        "find",
        "for",
        "from",
        "function",
        "functions",
        "how",
        "inherit",
        "inherits",
        "is",
        "its",
        "list",
        "method",
        "methods",
        "module",
        "modules",
        "of",
        "on",
        "please",
        "show",
        "the",
        "then",
        "this",
        "what",
        "where",
        "who",
        "why",
    }
)


def retrieval_search_terms(query: str, entities: list[str] | None = None) -> list[str]:
    """Names to look up: identifiers first, plus the full query for conceptual questions.
    
    Args:
        query: str.
        entities: list[str] | None.

    Returns:
        list[str].
    """
    extracted = [item.strip() for item in (entities or []) if item and str(item).strip()]
    stripped = query.strip()
    identifiers = _dedupe([*extracted, *_rule_based_entities(query)])
    if stripped and (
        not identifiers
        or wants_codebase_grounding(query)
        or _has_conceptual_residue(stripped, identifiers)
    ):
        return _dedupe([*identifiers, stripped])
    if identifiers:
        return identifiers
    return [stripped] if stripped else []


def _has_conceptual_residue(query: str, identifiers: list[str]) -> bool:
    known = {item.lower() for item in identifiers}
    for token in _QUERY_TOKEN_RE.findall(query):
        lowered = token.lower()
        if len(token) <= 1 or lowered in _LOOKUP_FILLER or lowered in known:
            continue
        return True
    return False


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in items:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _estimated_query_tokens(query: str) -> int:
    return max(1, (len(query) + 3) // 4) if query else 0


def _has_referring_expression(query: str) -> bool:
    return bool(_REFERRING_RE.search(query))


def wants_codebase_grounding(query: str) -> bool:
    """True when the user asked for repository examples, source, or implementations.
    
    Args:
        query: str.

    Returns:
        bool.
    """
    return bool(_CODEBASE_GROUNDING_RE.search(query))


def _ordered_agents(agents: list[AgentName]) -> list[AgentName]:
    ordered: list[AgentName] = []
    for agent in _AGENT_ORDER:
        if agent in agents and agent not in ordered:
            ordered.append(agent)
    for agent in agents:
        if agent not in ordered:
            ordered.append(agent)
    return ordered


def fallback_target_agents(query: str) -> list[AgentName]:
    """Agents used when LLM routing fails and rules are ambiguous.

    Args:
        query: User question.

    Returns:
        Ordered specialist names. Indexer is included when the query asks to
        (re)index so the executed plan still matches mixed indexing labels.
    """
    agents: list[AgentName] = ["graph_query", "code_analyst"]
    if re.search(r"\bindex\b|\breindex\b", query.lower()):
        agents = ["indexer", *agents]
    return _ordered_agents(agents)


def normalize_grounded_intent(intent: QueryIntent, query: str) -> QueryIntent:
    """Require graph retrieval (and analysis) when the user asks for codebase evidence.
    
    The LLM router may omit ``graph_query``; this step is deterministic and runs
    after every routing path so conceptual queries still hit the cascade.
    
    Args:
        intent: QueryIntent.
        query: str.

    Returns:
        QueryIntent.
    """
    if not wants_codebase_grounding(query):
        return intent
    agents = _ordered_agents(list(intent.target_agents))
    if "graph_query" not in agents:
        agents = _ordered_agents([*agents, "graph_query"])
    if "code_analyst" not in agents:
        agents = _ordered_agents([*agents, "code_analyst"])
    new_kind: QueryIntentIntent = intent.intent
    if new_kind == "indexing" and ("graph_query" in agents or "code_analyst" in agents):
        new_kind = "mixed"
    elif new_kind in {"lookup", "pattern"}:
        new_kind = "explanation"
    elif new_kind == "relationship" and "code_analyst" in agents:
        new_kind = "mixed"
    if agents == list(intent.target_agents) and new_kind == intent.intent:
        return intent
    return intent.model_copy(update={"target_agents": agents, "intent": new_kind})


def entities_from_context(context: ConversationContext | None) -> list[str]:
    """Pull codebase entities from recent user turns and the folded summary.

    Coreference is heuristic: regex pronoun detection plus entity carry from
    prior user turns (newest first) and the session summary once older turns
    have been folded. It is not model-based resolution.

    Args:
        context: ConversationContext | None.

    Returns:
        list[str].
    """
    if context is None:
        return []
    collected: list[str] = []
    for turn in reversed(context.recent_turns):
        if turn.role != "user":
            continue
        collected.extend(_rule_based_entities(turn.content))
    summary = (context.summary or "").strip()
    if summary:
        collected.extend(_rule_based_entities(summary))
    return _dedupe(collected)


def resolve_entities(
    query: str,
    extracted: list[str],
    context: ConversationContext | None,
) -> list[str]:
    """Fill referring expressions from prior turns/summary when the query has none.

    Pronouns such as ``it`` / ``this`` / ``that`` trigger a merge with entities
    carried from context. This is regex + entity carry, not coreference parsing.
    
    Args:
        query: str.
        extracted: list[str].
        context: ConversationContext | None.

    Returns:
        list[str].
    """
    prior = entities_from_context(context)
    if not extracted:
        return prior
    if _has_referring_expression(query) and prior:
        return _dedupe([*prior, *extracted])
    return extracted


def _prior_entities_block(context: ConversationContext | None) -> str:
    prior = entities_from_context(context)
    if not prior:
        return ""
    return (
        "\nPreviously mentioned entities (resolve referring expressions against these): "
        + ", ".join(prior)
        + "\n"
    )


def rule_based_route(
    query: str,
    *,
    settings: OrchestratorSettings | None = None,
) -> RuleRouteResult:
    """Cheap keyword router that can decline ambiguous or long queries.
    
    Args:
        query: str.
        settings: OrchestratorSettings | None.

    Returns:
        RuleRouteResult.
    """
    settings = settings or OrchestratorSettings.from_env()
    lowered = query.lower()
    wants_explanation = bool(re.search(r"explain|how does|why", lowered))
    wants_relationships = bool(
        re.search(
            r"who calls|depends on|who depends|dependents|"
            r"\bimports?\b|inherit|extends|subclass|\buses\b",
            lowered,
        )
    )
    wants_code_examples = bool(
        re.search(r"example|examples|implementation|implemented|codebase|source|snippet", lowered)
    )
    wants_comparison = "compare" in lowered
    wants_indexing = bool(re.search(r"\bindex\b|\breindex\b", lowered))
    wants_pattern = bool(re.search(r"\bpatterns?\b", lowered))
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
        categories.add("indexing")
    if wants_relationships:
        matched_rules.append("relationship")
        categories.add("relationship")
    if wants_lookup:
        matched_rules.append("lookup")
        categories.add("lookup")
    if wants_comparison:
        matched_rules.append("compare")
        categories.add("comparison")
    if wants_explanation:
        matched_rules.append("explain")
        categories.add("explain")
    if wants_code_examples:
        matched_rules.append("examples")
        categories.add("examples")
    if wants_pattern:
        matched_rules.append("pattern")
        categories.add("pattern")
    if not matched_rules and _has_referring_expression(query):
        matched_rules.append("lookup")
        categories.add("lookup")

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
    elif intent == "pattern":
        target_agents = ["code_analyst"]
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
    context: ConversationContext | None = None,
) -> QueryIntent:
    """Build the existing `QueryIntent` shape from an unambiguous rule result.
    
    Args:
        query: str.
        route: RuleRouteResult.
        routing_mode: RoutingMode.
        context: ConversationContext | None.

    Returns:
        QueryIntent.
    """
    entities = resolve_entities(query, _rule_based_entities(query), context)
    intent: QueryIntentIntent
    if route.target_agents == ["indexer"]:
        intent = "indexing"
    elif route.target_agents == ["code_analyst"]:
        intent = "pattern"
    elif route.target_agents == ["graph_query"]:
        intent = "relationship" if "relationship" in route.matched_rules else "lookup"
    elif "compare" in route.matched_rules:
        intent = "comparison"
    else:
        intent = "explanation"
    built = QueryIntent(
        routing_mode=routing_mode,
        intent=intent,
        entities=entities,
        target_agents=route.target_agents,
        reasoning=f"rules: matched {', '.join(route.matched_rules)}",
    )
    return normalize_grounded_intent(built, query)


def apply_conversation_entities(
    intent: QueryIntent,
    query: str,
    context: ConversationContext | None,
) -> QueryIntent:
    """Carry prior-turn entities onto an intent that has none of its own.
    
    Args:
        intent: QueryIntent.
        query: str.
        context: ConversationContext | None.

    Returns:
        QueryIntent.
    """
    resolved = resolve_entities(query, list(intent.entities), context)
    if resolved == intent.entities:
        return intent
    return intent.model_copy(update={"entities": resolved})


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
    to ``rule_based_route``. Conversation context is used for heuristic
    coreference (pronoun regex + entity carry from recent user turns and the
    folded summary), not as a model-based session rewrite.
    
    Args:
        query: str.
        context: ConversationContext | None.
        llm_provider: LLMProvider.
        token_ledger: TokenLedger | None.
        correlation_id: str | None.

    Returns:
        QueryIntent.

    Raises:
        RoutingError: See exception message.
    """
    messages = [
        Message(role="system", content=ROUTER_SYSTEM_PROMPT),
        Message(
            role="user",
            content=ROUTER_USER_PROMPT.format(
                query=query,
                prior_entities_block=_prior_entities_block(context),
            ),
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
    except Exception as exc:
        if isinstance(exc, RoutingError):
            raise
        log.warning(
            "orchestrator.routing_llm_failed",
            error=str(exc),
            exception_type=type(exc).__name__,
            correlation_id=correlation_id or "-",
        )
        return _intent_from_rules_fallback(query, context)

    if isinstance(result.parsed, QueryIntent):
        return normalize_grounded_intent(
            apply_conversation_entities(result.parsed, query, context),
            query,
        )
    return _intent_from_rules_fallback(query, context)


def _intent_from_rules_fallback(
    query: str,
    context: ConversationContext | None,
) -> QueryIntent:
    """Build a rules-based intent when the routing LLM is unavailable."""
    fallback = rule_based_route(query)
    if fallback.ambiguous:
        return normalize_grounded_intent(
            QueryIntent(
                routing_mode="rules_fallback",
                intent="mixed",
                entities=resolve_entities(query, _rule_based_entities(query), context),
                target_agents=fallback_target_agents(query),
                reasoning="rules_fallback: ambiguous fallback defaulted to mixed",
            ),
            query,
        )
    return intent_from_rule_result(
        query, fallback, routing_mode="rules_fallback", context=context
    )


def route_to_agents(intent: QueryIntent) -> ExecutionPlan:
    """Convert a QueryIntent into a flat specialist set.

    Args:
        intent: Routed query intent including ``target_agents``.

    Returns:
        An execution plan whose ``agents`` run concurrently. The executor
        waits internally when ``code_analyst`` needs graph coordinates.

    Raises:
        RoutingError: If ``intent.target_agents`` is empty.
    """
    targets: list[AgentName] = list(intent.target_agents)
    if not targets:
        raise RoutingError(
            agent="orchestrator",
            message="cannot route query: no target agents selected",
        )
    return ExecutionPlan(
        routing_mode=intent.routing_mode,
        intent=intent,
        agents=targets,
    )
