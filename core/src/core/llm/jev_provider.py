"""TypeSafe Jev implementation of LLMProvider. Routing purpose only.

Jev is a System One model: it returns typed judgments (Choice / Noul / Score)
with calibrated probabilities rather than generated text. That makes it a fit
for the two *decision* fields of :class:`QueryIntent` (``intent`` and
``target_agents``) and a non-fit for its two *generated* fields (``entities``
and ``reasoning``), which this adapter fills from the existing deterministic
rule extractor instead.

The adapter deliberately performs no post-processing beyond that mapping.
``normalize_grounded_intent`` is applied by ``analyze_query`` to every routing
backend equally, so applying it here would give Jev an advantage the OpenRouter
arm does not get in the bake-off.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Final

import httpx
from pydantic import BaseModel

from core.exceptions import AgentError
from core.llm.provider import LLMPurpose, LLMResult, Message, TokenUsage
from core.logging import get_logger
from core.orchestration.models import AgentName, QueryIntent, QueryIntentIntent
from core.orchestration.router import extract_entities, fallback_target_agents

log = get_logger(__name__)

# Internal id used by the bake-off, the pricing catalog, and logs. The wire
# protocol wants the bare model name, so the vendor prefix is stripped in
# `_wire_model`.
JEV_MODEL_ID: Final = "typesafe/jev-latest"
_VENDOR_PREFIX: Final = "typesafe/"

DEFAULT_BASE_URL: Final = "https://api.typesafe.ai/v1"
SYSTEM_ONE_PATH: Final = "/systemone"

# Documented SDK default. Routing sits on the request critical path, so this is
# deliberately tight.
DEFAULT_TIMEOUT_S: Final = 10.0

# A Noul returns the calibrated probability of "yes" and carries no separate
# confidence field, so the raw value is thresholded directly. 0.5 is the
# neutral starting point, NOT a tuned value -- see docs/evaluation.md.
DEFAULT_AGENT_THRESHOLD: Final = 0.5

# 429 and 529 are documented; 503 is what the edge proxy returns when no
# backend is healthy, observed live on 2026-09-21. All three are transient, so
# one bounded retry is worth the latency on a request this short.
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 503, 529})
DEFAULT_MAX_RETRIES: Final = 1
DEFAULT_RETRY_BACKOFF_S: Final = 0.5

# Mirrors `_AGENT_ORDER` in core.orchestration.router. Asserted equal in
# core/tests/test_jev_provider.py so the two cannot drift apart silently.
AGENT_ORDER: Final[tuple[AgentName, ...]] = (
    "indexer",
    "graph_query",
    "code_analyst",
    "memory",
)

_INTENT_QUESTION_ID: Final = "intent"
_AGENT_QUESTION_PREFIX: Final = "needs_"

# Literal fragments of ROUTER_USER_PROMPT / `_prior_entities_block`. The
# bake-off hands providers the rendered prompt, so the query has to be
# recovered from it; test_jev_provider asserts this round-trips the real
# `routing_messages()` output.
_PRIOR_ENTITIES_PREFIX: Final = (
    "Previously mentioned entities (resolve referring expressions against these): "
)
_PROMPT_RE: Final = re.compile(
    r"User query:\n(?P<query>.*?)\n(?P<block>.*?)\nDecide:\n",
    re.DOTALL,
)

_INTENT_CRITERIA: Final[dict[str, str]] = {
    "lookup": (
        "The user wants to locate a specific named symbol, file, or definition in the "
        "repository. Example: 'What is the FastAPI class?'"
    ),
    "relationship": (
        "The user asks how code entities connect to one another -- who calls, imports, "
        "depends on, inherits from, or subclasses what. Example: 'Who calls get_openapi?'"
    ),
    "explanation": (
        "The user asks how or why something works and wants prose grounded in the "
        "repository source. Example: 'Explain how dependency injection works here.'"
    ),
    "pattern": (
        "The user asks about recurring design patterns in the code, such as decorators, "
        "dependency injection, or factories, rather than about one named symbol."
    ),
    "comparison": (
        "The user asks for two or more entities to be contrasted with each other. "
        "Example: 'Compare FastAPI and APIRouter.'"
    ),
    "indexing": (
        "The user asks for the repository to be indexed or reindexed. This is a request "
        "to do work on the graph, not a question about code."
    ),
    "mixed": (
        "The request combines two or more of the other kinds and cannot be served by one "
        "of them alone. Example: 'Reindex the repository and then explain dependency "
        "injection with examples.'"
    ),
}

_AGENT_INSTRUCTIONS: Final[dict[AgentName, str]] = {
    "indexer": (
        "Should the indexer agent run for this request? The indexer clones the repository "
        "and parses its source into the code graph. It is the only writer. Answer yes only "
        "when the user explicitly asks for the repository to be indexed or reindexed; "
        "answer no when the user is asking a question about code that is already indexed."
    ),
    "graph_query": (
        "Should the graph query agent run for this request? It is the only reader of the "
        "code graph: it resolves entity names to nodes and traverses import, call, and "
        "inheritance edges. Answer yes whenever answering requires locating a symbol in the "
        "repository or relating symbols to each other, including when the user asks for "
        "codebase examples, source, or implementations."
    ),
    "code_analyst": (
        "Should the code analyst agent run for this request? It reads source text at graph "
        "coordinates and produces explanations, comparisons, and design-pattern analysis. "
        "Answer yes whenever answering requires interpreting or explaining code rather than "
        "only locating it. Answer no when a bare lookup or a graph traversal fully answers "
        "the question."
    ),
    "memory": (
        "Should the memory agent run for this request? It stores and retrieves this "
        "conversation's own history. Answer yes only when answering requires recalling what "
        "was said in earlier turns of this conversation; answer no for questions about the "
        "repository that stand on their own."
    ),
}


class JevRoutingError(AgentError):
    """The TypeSafe API call failed or returned an unusable payload."""


def _wire_model(model_id: str) -> str:
    return model_id[len(_VENDOR_PREFIX) :] if model_id.startswith(_VENDOR_PREFIX) else model_id


def parse_routing_prompt(messages: list[Message]) -> tuple[str, list[str]]:
    """Recover ``(query, prior_entities)`` from a rendered router prompt.

    Args:
        messages: The messages built by ``core.eval.model_bakeoff.routing_messages``
            or by ``analyze_query``.

    Returns:
        The user query and any previously mentioned entities carried in the prompt.

    Raises:
        JevRoutingError: When no user message matches the router prompt template.
    """
    for message in reversed(messages):
        if message.role != "user":
            continue
        match = _PROMPT_RE.search(message.content)
        if match is None:
            continue
        query = match.group("query").strip()
        block = match.group("block").strip()
        prior: list[str] = []
        if block.startswith(_PRIOR_ENTITIES_PREFIX):
            raw = block[len(_PRIOR_ENTITIES_PREFIX) :]
            prior = [item.strip() for item in raw.split(",") if item.strip()]
        return query, prior
    raise JevRoutingError(
        agent="orchestrator",
        message="no user message matched the router prompt template",
    )


def build_questions() -> dict[str, dict[str, Any]]:
    """Return the batched System One question set for one routing decision.

    All questions share one request and are evaluated in parallel, so the
    per-agent Nouls cost latency only once.

    Returns:
        Question id to question body, ready to send as ``questions``.
    """
    questions: dict[str, dict[str, Any]] = {
        _INTENT_QUESTION_ID: {
            "type": "choice",
            "instructions": (
                "What kind of task is the user's request? Choose the single best fit. "
                "Choose 'mixed' only when the request genuinely combines two or more of "
                "the other kinds."
            ),
            "criteria": dict(_INTENT_CRITERIA),
        }
    }
    for agent in AGENT_ORDER:
        questions[f"{_AGENT_QUESTION_PREFIX}{agent}"] = {
            "type": "noul",
            "instructions": _AGENT_INSTRUCTIONS[agent],
        }
    return questions


def intent_from_answers(
    answers: dict[str, Any],
    query: str,
    prior_entities: list[str],
    *,
    threshold: float = DEFAULT_AGENT_THRESHOLD,
) -> QueryIntent:
    """Compose a :class:`QueryIntent` from one System One answer set.

    Args:
        answers: The ``answers`` map from the API response.
        query: The user query the judgments were made about.
        prior_entities: Entities carried in from earlier turns, if any.
        threshold: Noul probability at or above which an agent is selected.

    Returns:
        A routed intent. ``entities`` and ``reasoning`` come from deterministic
        code, since Jev returns judgments rather than generated text.

    Raises:
        JevRoutingError: When the intent answer is missing or not a known value.
    """
    intent_answer = answers.get(_INTENT_QUESTION_ID) or {}
    raw_intent = intent_answer.get("choice")
    if raw_intent not in _INTENT_CRITERIA:
        raise JevRoutingError(
            agent="orchestrator",
            message=f"unusable intent answer: {raw_intent!r}",
        )
    intent_kind: QueryIntentIntent = raw_intent

    probabilities: dict[AgentName, float] = {}
    for agent in AGENT_ORDER:
        answer = answers.get(f"{_AGENT_QUESTION_PREFIX}{agent}") or {}
        value = answer.get("noul")
        probabilities[agent] = float(value) if isinstance(value, int | float) else 0.0

    selected: list[AgentName] = [
        agent for agent in AGENT_ORDER if probabilities[agent] >= threshold
    ]
    if not selected:
        # Every Noul came in below threshold. Rather than route nowhere, reuse
        # the same conservative default the LLM-failure path already uses.
        selected = fallback_target_agents(query)

    entities = extract_entities(query)
    if not entities:
        entities = list(prior_entities)

    detail = " ".join(f"{agent}={probabilities[agent]:.2f}" for agent in AGENT_ORDER)
    confidence = intent_answer.get("confidence")
    confidence_note = f" confidence={float(confidence):.2f}" if confidence is not None else ""
    return QueryIntent(
        routing_mode="llm",
        intent=intent_kind,
        entities=entities,
        target_agents=selected,
        reasoning=f"jev: intent={intent_kind}{confidence_note}; p(agent) {detail}",
    )


class JevRoutingProvider:
    """LLMProvider backed by TypeSafe's System One endpoint. Routing only.

    Only ``purpose="routing"`` with ``response_model=QueryIntent`` is supported;
    every other call raises. Jev does not generate text, so it cannot stand in
    for the synthesis or analysis purposes.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = JEV_MODEL_ID,
        base_url: str = DEFAULT_BASE_URL,
        threshold: float = DEFAULT_AGENT_THRESHOLD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a Jev routing client.

        Args:
            api_key: TypeSafe API key.
            model: Internal model id; the vendor prefix is stripped on the wire.
            base_url: API base, without the endpoint path.
            threshold: Noul probability at or above which an agent is selected.
            timeout_s: Per-request HTTP timeout.
            max_retries: Extra attempts after a transient failure.
            retry_backoff_s: Base linear backoff between attempts.
            client: Optional pre-built client, injected by tests.
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._threshold = threshold
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._client = client
        self._owns_client = client is None

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        try:
            response = await self._post_with_retry(client, payload)
        finally:
            if self._owns_client:
                await client.aclose()
        decoded = response.json()
        if not isinstance(decoded, dict):
            raise JevRoutingError(
                agent="orchestrator",
                message=f"typesafe returned a non-object body: {type(decoded).__name__}",
            )
        return decoded

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> httpx.Response:
        last: JevRoutingError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    f"{self._base_url}{SYSTEM_ONE_PATH}",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
            except httpx.HTTPError as exc:
                last = JevRoutingError(
                    agent="orchestrator",
                    message=f"typesafe request failed: {exc}",
                )
                retryable = True
            else:
                if response.status_code < 400:
                    return response
                last = JevRoutingError(
                    agent="orchestrator",
                    message=(
                        f"typesafe returned {response.status_code}: {response.text[:300]}"
                    ),
                )
                retryable = response.status_code in RETRYABLE_STATUS
            if not retryable or attempt == self._max_retries:
                break
            log.warning("jev.retrying", attempt=attempt, reason=str(last))
            await asyncio.sleep(self._retry_backoff_s * (attempt + 1))
        assert last is not None
        raise last

    async def judge(
        self,
        query: str,
        prior_entities: list[str] | None = None,
        *,
        agent: str = "orchestrator",
    ) -> tuple[dict[str, Any], TokenUsage]:
        """Ask the batched routing questions and return the raw answers.

        Exposed separately from :meth:`complete` because the agent threshold is
        applied to these probabilities after the fact: one pass over a case set
        can be replayed at any threshold without re-calling the API.

        Args:
            query: User question.
            prior_entities: Entities carried in from earlier turns, if any.
            agent: Calling agent name, for error reporting.

        Returns:
            The ``answers`` map and the call's token usage.

        Raises:
            JevRoutingError: On a failed request or a payload with no answers.
        """
        state: dict[str, Any] = {"user_query": query}
        if prior_entities:
            state["previously_mentioned_entities"] = list(prior_entities)

        payload = await self._post(
            {
                "state": state,
                "model": _wire_model(self._model),
                "questions": build_questions(),
            }
        )
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise JevRoutingError(
                agent=agent,
                message="typesafe response contained no answers map",
            )
        raw_usage = payload.get("usage") or {}
        prompt_tokens = int(raw_usage.get("input_tokens") or 0)
        completion_tokens = int(raw_usage.get("output_tokens") or 0)
        return answers, TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated=False,
            model=self._model,
            cached_prompt_tokens=0,
        )

    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResult:
        """Route one query with a single batched System One request.

        Args:
            messages: Rendered router prompt; the query is recovered from it.
            response_model: Must be ``QueryIntent``.
            purpose: Must be ``"routing"``.
            agent: Calling agent name, for error reporting.
            max_tokens: Ignored; Jev does not generate text.
            temperature: Ignored; Jev returns calibrated probabilities.

        Returns:
            An ``LLMResult`` whose ``parsed`` is a :class:`QueryIntent`.

        Raises:
            JevRoutingError: For an unsupported purpose/model, a failed request,
                or an unusable payload.
        """
        if purpose != "routing" or response_model is not QueryIntent:
            raise JevRoutingError(
                agent=agent,
                message=(
                    "JevRoutingProvider supports only purpose='routing' with "
                    f"response_model=QueryIntent; got purpose={purpose!r} "
                    f"response_model={getattr(response_model, '__name__', response_model)!r}"
                ),
            )

        query, prior_entities = parse_routing_prompt(messages)
        answers, usage = await self.judge(query, prior_entities, agent=agent)
        intent = intent_from_answers(
            answers, query, prior_entities, threshold=self._threshold
        )
        log.info(
            "jev.routed",
            intent=intent.intent,
            agents=intent.target_agents,
            input_tokens=usage.prompt_tokens,
        )
        return LLMResult(
            text=intent.model_dump_json(),
            parsed=intent,
            usage=usage,
            schema_attempts=1,
            first_attempt_valid=True,
        )
