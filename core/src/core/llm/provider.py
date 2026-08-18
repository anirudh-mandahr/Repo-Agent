"""LLM provider interface. All model calls go through LLMProvider."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from core.exceptions import SchemaValidationError
from core.logging import get_correlation_id, get_logger

log = get_logger(__name__)

MAX_SCHEMA_ATTEMPTS = 2
VALIDATION_RETRY_PROMPT = (
    "The previous response failed schema validation:\n{error}\n"
    "Reply with valid JSON matching the required schema."
)



class Message(BaseModel):
    """One chat message sent to an LLM provider."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    """Completion text plus an optional validated structured payload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    parsed: BaseModel | None = None
    usage: TokenUsage


class TokenUsage(BaseModel):
    """Token accounting attached to every provider response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated: bool = False


LLMPurpose = Literal["routing", "synthesis", "analysis", "summarization"]


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for chat-completion backends."""

    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
    ) -> LLMResult:
        """Return a completion. Validate against ``response_model`` when given."""
        ...


def parse_structured[TModel: BaseModel](
    raw: str | Mapping[str, Any] | BaseModel,
    response_model: type[TModel],
    *,
    agent: str,
    correlation_id: str | None = None,
) -> TModel:
    """Validate ``raw`` as ``response_model``, raising ``SchemaValidationError``."""
    if isinstance(raw, response_model):
        return raw
    data: Any
    if isinstance(raw, BaseModel):
        data = raw.model_dump()
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                agent=agent,
                correlation_id=correlation_id,
                message=f"invalid JSON: {exc}",
            ) from exc
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        raise SchemaValidationError(
            agent=agent,
            correlation_id=correlation_id,
            message=f"unsupported response type: {type(raw).__name__}",
        )
    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(
            agent=agent,
            correlation_id=correlation_id,
            message=str(exc),
        ) from exc


def as_text(raw: str | Mapping[str, Any] | BaseModel) -> str:
    """Normalize a provider payload to text for ``LLMResult.text``."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, BaseModel):
        return raw.model_dump_json()
    return json.dumps(raw)


async def complete_with_schema_retry(
    invoke: Callable[
        [list[Message]],
        Awaitable[tuple[str | Mapping[str, Any] | BaseModel, TokenUsage | None]],
    ],
    messages: Sequence[Message],
    response_model: type[BaseModel] | None,
    *,
    purpose: LLMPurpose,
    agent: str,
) -> LLMResult:
    """Invoke ``invoke`` and, when ``response_model`` is set, retry once on failure."""
    current = list(messages)
    last_error: SchemaValidationError | None = None
    attempts = MAX_SCHEMA_ATTEMPTS if response_model is not None else 1
    for attempt in range(attempts):
        raw, usage = await invoke(current)
        text = as_text(raw)
        resolved_usage = usage or estimate_usage(current, text)
        _log_llm_call(purpose=purpose, usage=resolved_usage)
        if response_model is None:
            return LLMResult(text=text, usage=resolved_usage)
        try:
            parsed = parse_structured(raw, response_model, agent=agent)
            log.info("llm.complete", attempt=attempt, structured=True)
            return LLMResult(text=text, parsed=parsed, usage=resolved_usage)
        except SchemaValidationError as exc:
            last_error = exc
            log.warning(
                "llm.schema_validation_failed",
                attempt=attempt,
                error=str(exc),
            )
            current = [
                *current,
                Message(role="user", content=VALIDATION_RETRY_PROMPT.format(error=exc)),
            ]
    if last_error is not None:
        raise last_error
    raise SchemaValidationError(
        agent=agent,
        message="schema validation failed",
    )


def estimate_usage(messages: Sequence[Message], text: str) -> TokenUsage:
    """Return a cheap fallback usage estimate when providers omit it."""

    prompt_chars = sum(len(message.content) for message in messages)
    prompt_tokens = max(1, (prompt_chars + 3) // 4) if prompt_chars else 0
    completion_tokens = max(1, (len(text) + 3) // 4) if text else 0
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated=True,
    )


def _log_llm_call(*, purpose: LLMPurpose, usage: TokenUsage) -> None:
    log.info(
        "llm_call",
        purpose=purpose,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        estimated=usage.estimated,
        correlation_id=get_correlation_id(),
    )
