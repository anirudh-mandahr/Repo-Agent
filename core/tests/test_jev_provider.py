"""Tests for the TypeSafe Jev routing provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from core.eval.model_bakeoff import routing_messages
from core.eval.routing_providers import is_typesafe_model, routing_provider_factory
from core.exceptions import ConfigurationError
from core.llm.jev_provider import (
    AGENT_ORDER,
    JEV_MODEL_ID,
    JevRoutingError,
    JevRoutingProvider,
    build_questions,
    intent_from_answers,
    parse_routing_prompt,
)
from core.llm.pricing import CATALOG
from core.llm.provider import Message
from core.memory import ConversationContext, ConversationTurn
from core.orchestration.models import QueryIntent
from core.orchestration.prompts import ROUTER_USER_PROMPT
from core.orchestration.router import _AGENT_ORDER, _prior_entities_block
from core.settings import _SECRET_ENV_KEYS, TypeSafeSettings


def _answers(intent: str = "lookup", **nouls: float) -> dict[str, Any]:
    payload: dict[str, Any] = {"intent": {"type": "choice", "choice": intent}}
    for agent in AGENT_ORDER:
        payload[f"needs_{agent}"] = {"type": "noul", "noul": nouls.get(agent, 0.0)}
    return payload


# --- drift guards -----------------------------------------------------------


def test_agent_order_matches_router() -> None:
    """The provider's ordering must not drift from the router's."""
    assert AGENT_ORDER == _AGENT_ORDER


def test_parse_round_trips_bakeoff_prompt() -> None:
    """The query must survive the real bake-off prompt rendering."""
    query = "Explain how get_openapi is implemented and who calls it"
    parsed, prior = parse_routing_prompt(routing_messages(query))
    assert parsed == query
    assert prior == []


def test_parse_round_trips_prompt_with_prior_entities() -> None:
    """`analyze_query` renders a prior-entities block; the parser must read it."""
    context = ConversationContext(
        recent_turns=[
            ConversationTurn(
                id=1,
                role="user",
                content="What is the FastAPI class?",
                created_at="2026-09-21T00:00:00Z",
                token_estimate=8,
            )
        ],
        summary="",
    )
    block = _prior_entities_block(context)
    assert block, "fixture should produce a non-empty prior-entities block"
    messages = [
        Message(
            role="user",
            content=ROUTER_USER_PROMPT.format(query="Who calls it?", prior_entities_block=block),
        )
    ]
    parsed, prior = parse_routing_prompt(messages)
    assert parsed == "Who calls it?"
    assert "FastAPI" in prior


def test_parse_rejects_unrecognised_prompt() -> None:
    with pytest.raises(JevRoutingError):
        parse_routing_prompt([Message(role="user", content="not a router prompt")])


# --- question construction --------------------------------------------------


def test_questions_are_one_choice_and_one_noul_per_agent() -> None:
    questions = build_questions()
    assert questions["intent"]["type"] == "choice"
    assert set(questions["intent"]["criteria"]) == {
        "lookup",
        "relationship",
        "explanation",
        "pattern",
        "comparison",
        "indexing",
        "mixed",
    }
    for agent in AGENT_ORDER:
        assert questions[f"needs_{agent}"]["type"] == "noul"
    assert len(questions) == len(AGENT_ORDER) + 1


# --- composition ------------------------------------------------------------


def test_agents_selected_above_threshold_in_router_order() -> None:
    intent = intent_from_answers(
        _answers("explanation", code_analyst=0.91, graph_query=0.88, memory=0.10),
        "Explain how dependency injection works in the codebase",
        [],
    )
    assert intent.target_agents == ["graph_query", "code_analyst"]
    assert intent.intent == "explanation"
    assert intent.routing_mode == "llm"


def test_threshold_boundary_is_inclusive() -> None:
    intent = intent_from_answers(
        _answers("lookup", graph_query=0.5), "What is the FastAPI class?", [], threshold=0.5
    )
    assert intent.target_agents == ["graph_query"]


def test_all_below_threshold_falls_back_rather_than_routing_nowhere() -> None:
    intent = intent_from_answers(
        _answers("mixed"), "Reindex the repository please", []
    )
    # `fallback_target_agents` adds the indexer for index/reindex wording.
    assert intent.target_agents == ["indexer", "graph_query", "code_analyst"]


def test_entities_come_from_the_rule_extractor_not_the_model() -> None:
    intent = intent_from_answers(
        _answers("lookup", graph_query=0.9), "What is the APIRouter class?", []
    )
    assert "APIRouter" in intent.entities


def test_prior_entities_used_when_query_has_none() -> None:
    intent = intent_from_answers(
        _answers("relationship", graph_query=0.9), "who calls it?", ["APIRouter"]
    )
    assert intent.entities == ["APIRouter"]


def test_unknown_intent_value_is_rejected() -> None:
    answers = _answers(graph_query=0.9)
    answers["intent"] = {"type": "choice", "choice": "not_a_real_intent"}
    with pytest.raises(JevRoutingError):
        intent_from_answers(answers, "anything", [])


def test_missing_noul_is_treated_as_zero_not_an_error() -> None:
    answers = _answers("lookup", graph_query=0.9)
    del answers["needs_memory"]
    intent = intent_from_answers(answers, "What is the FastAPI class?", [])
    assert "memory" not in intent.target_agents


# --- provider ---------------------------------------------------------------


def _provider(handler: Any, **kwargs: Any) -> JevRoutingProvider:
    return JevRoutingProvider(
        api_key="test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


async def test_complete_sends_one_batched_request_and_parses_it() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": _answers("relationship", graph_query=0.97),
                "usage": {"input_tokens": 392, "output_tokens": 65},
            },
        )

    provider = _provider(handler)
    result = await provider.complete(
        routing_messages("What classes inherit from APIRouter?"),
        QueryIntent,
        purpose="routing",
        agent="orchestrator",
    )

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer test-key"
    # The vendor prefix is internal; the wire wants the bare model name.
    assert seen["body"]["model"] == "jev-latest"
    assert seen["body"]["state"]["user_query"] == "What classes inherit from APIRouter?"
    assert len(seen["body"]["questions"]) == len(AGENT_ORDER) + 1

    assert isinstance(result.parsed, QueryIntent)
    assert result.parsed.target_agents == ["graph_query"]
    assert result.usage.prompt_tokens == 392
    assert result.usage.completion_tokens == 65
    assert result.usage.model == JEV_MODEL_ID
    assert result.usage.estimated is False


async def test_non_routing_purpose_is_refused() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    with pytest.raises(JevRoutingError, match="only purpose='routing'"):
        await provider.complete(
            routing_messages("anything"), QueryIntent, purpose="synthesis"
        )


async def test_http_error_is_wrapped() -> None:
    provider = _provider(lambda request: httpx.Response(429, text="rate limited"))
    with pytest.raises(JevRoutingError, match="429"):
        await provider.complete(routing_messages("anything"), QueryIntent, purpose="routing")


async def test_missing_answers_map_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"model": "jev-1.13.0"}))
    with pytest.raises(JevRoutingError, match="no answers"):
        await provider.complete(routing_messages("anything"), QueryIntent, purpose="routing")


# --- factory and pricing ----------------------------------------------------


def test_typesafe_ids_are_recognised() -> None:
    assert is_typesafe_model(JEV_MODEL_ID)
    assert not is_typesafe_model("anthropic/claude-sonnet-4.5")


def test_factory_requires_a_key() -> None:
    with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
        routing_provider_factory(JEV_MODEL_ID, settings=TypeSafeSettings(api_key=None))


def test_factory_builds_jev_when_a_key_is_present() -> None:
    provider = routing_provider_factory(
        JEV_MODEL_ID,
        settings=TypeSafeSettings(api_key=SecretStr("k"), agent_threshold=0.7),
    )
    assert isinstance(provider, JevRoutingProvider)


@pytest.mark.parametrize("name", ["TYPESAFE_API_KEY", "JEV_API_KEY", "JEV-API-KEY"])
def test_every_accepted_key_spelling_resolves(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    for candidate in ("TYPESAFE_API_KEY", "JEV_API_KEY", "JEV-API-KEY"):
        monkeypatch.delenv(candidate, raising=False)
    monkeypatch.setenv(name, "exported-secret")
    settings = TypeSafeSettings()
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "exported-secret"


def test_key_in_dotenv_is_ignored_like_every_other_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Secrets are export-only; .env is interpolated into containers."""
    for candidate in ("TYPESAFE_API_KEY", "JEV_API_KEY", "JEV-API-KEY"):
        monkeypatch.delenv(candidate, raising=False)
    assert "TYPESAFE_API_KEY" in _SECRET_ENV_KEYS
    assert "JEV_API_KEY" in _SECRET_ENV_KEYS
    assert TypeSafeSettings().api_key is None


def test_threshold_is_validated_as_a_probability() -> None:
    with pytest.raises(ValidationError):
        TypeSafeSettings(agent_threshold=1.5)
    assert TypeSafeSettings(agent_threshold=0.7).agent_threshold == 0.7


def test_jev_is_priced_so_cost_is_not_the_sonnet_default() -> None:
    """An unpriced model silently inherits $3/$15 and would be reported wrong."""
    rates = CATALOG[JEV_MODEL_ID]
    assert rates.prompt_usd_per_million == 0.042
    assert rates.completion_usd_per_million == 0.0


async def test_transient_503_is_retried_then_surfaced() -> None:
    """Observed live: the edge proxy 503s when no backend is healthy."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="no healthy upstream")

    provider = _provider(handler, max_retries=1, retry_backoff_s=0.0)
    with pytest.raises(JevRoutingError, match="503"):
        await provider.complete(routing_messages("anything"), QueryIntent, purpose="routing")
    assert calls["n"] == 2


async def test_retry_recovers_when_the_second_attempt_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(529, text="overloaded")
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": _answers("lookup", graph_query=0.9),
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    provider = _provider(handler, max_retries=1, retry_backoff_s=0.0)
    result = await provider.complete(
        routing_messages("What is the FastAPI class?"), QueryIntent, purpose="routing"
    )
    assert calls["n"] == 2
    assert isinstance(result.parsed, QueryIntent)
    assert result.parsed.target_agents == ["graph_query"]


async def test_auth_failure_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    provider = _provider(handler, max_retries=1, retry_backoff_s=0.0)
    with pytest.raises(JevRoutingError, match="401"):
        await provider.complete(routing_messages("anything"), QueryIntent, purpose="routing")
    assert calls["n"] == 1
