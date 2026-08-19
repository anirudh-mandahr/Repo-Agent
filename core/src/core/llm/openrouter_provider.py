"""OpenRouter implementation of LLMProvider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from core.llm.provider import (
    LLMPurpose,
    LLMResult,
    Message,
    TokenUsage,
    complete_with_schema_retry,
    estimate_usage,
)
from core.logging import get_logger
from core.settings import LLMSettings

log = get_logger(__name__)

STRUCTURED_TOOL_NAME = "emit_result"


class OpenRouterProvider:
    """LLMProvider backed by OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        *,
        settings: LLMSettings | None = None,
        temperature: float | None = None,
    ) -> None:
        """Create an OpenRouter chat client.

        Args:
            client: Optional pre-built AsyncOpenAI client.
            model: Model id override applied to every purpose when set.
            settings: Optional settings snapshot. Loaded from env when omitted.
            temperature: Optional sampling temperature applied to every call.
        """
        self._settings = settings or LLMSettings.from_env()
        self._model_override = model
        self._model = model or self._settings.model
        self._temperature = temperature
        self._api_key = (
            self._settings.api_key.get_secret_value() if self._settings.api_key else None
        )
        self._base_url = self._settings.base_url
        self._client = client

    @classmethod
    def from_env(cls) -> OpenRouterProvider:
        """Build a provider from the OpenRouter environment settings.

        Returns:
            OpenRouterProvider.
        """
        return cls()

    def model_for(self, purpose: LLMPurpose) -> str:
        """Return the model id used for ``purpose``.

        Args:
            purpose: Routing, synthesis, analysis, or summarization.

        Returns:
            Constructor override when set, otherwise the purpose-specific setting
            falling back to ``OPENROUTER_MODEL``.
        """
        if self._model_override:
            return self._model_override
        return self._settings.resolve_model(purpose)

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            import httpx2

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=httpx2.Timeout(timeout=60.0, connect=20.0),
                max_retries=0,
            )
        return self._client

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
        """Call OpenRouter, forcing a function call for structured responses.

        Args:
            messages: list[Message].
            response_model: type[BaseModel] | None.
            purpose: LLMPurpose.
            agent: str.
            max_tokens: int.
            temperature: Optional per-call sampling temperature.

        Returns:
            LLMResult.
        """
        resolved_temperature = self._temperature if temperature is None else temperature

        async def invoke(
            attempt_messages: list[Message],
        ) -> tuple[str | Mapping[str, Any], TokenUsage | None]:
            return await self._invoke(
                attempt_messages,
                response_model,
                max_tokens,
                purpose=purpose,
                temperature=resolved_temperature,
            )

        return await complete_with_schema_retry(
            invoke,
            messages,
            response_model,
            purpose=purpose,
            agent=agent,
        )

    async def stream(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[tuple[str, TokenUsage | None]]:
        """Stream an OpenRouter completion. Structured calls fall back to complete.

        Args:
            messages: Chat messages.
            response_model: When set, uses the non-streaming complete path.
            purpose: Ledger purpose.
            agent: Calling agent.
            max_tokens: Completion cap.
            temperature: Optional sampling temperature override.

        Yields:
            ``(delta, usage_or_none)`` pairs. Usage is attached to the last chunk.
        """
        _ = agent
        if response_model is not None:
            result = await self.complete(
                messages,
                response_model,
                purpose=purpose,
                agent=agent,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            yield result.text, result.usage
            return
        resolved_temperature = self._temperature if temperature is None else temperature
        model = self.model_for(purpose)
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _serialize_messages(
                messages,
                cache_system=_should_cache_system(model, self._settings),
            ),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not messages:
            payload["messages"] = [{"role": "user", "content": "(empty)"}]
        if resolved_temperature is not None:
            payload["temperature"] = resolved_temperature
        log.info(
            "llm.openrouter.stream",
            model=model,
            purpose=purpose,
            max_tokens=max_tokens,
        )
        response = await self._get_client().chat.completions.create(**payload)
        usage: TokenUsage | None = None
        async for chunk in response:
            delta = _stream_delta_text(chunk)
            chunk_usage = _stream_usage(chunk, messages, model=model)
            if chunk_usage is not None:
                usage = chunk_usage
            if delta:
                yield delta, None
        if usage is None:
            usage = estimate_usage(messages, "")
        yield "", usage

    async def _invoke(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        max_tokens: int,
        *,
        purpose: LLMPurpose,
        temperature: float | None = None,
    ) -> tuple[str | Mapping[str, Any], TokenUsage]:
        model = self.model_for(purpose)
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _serialize_messages(
                messages,
                cache_system=_should_cache_system(model, self._settings),
            ),
        }
        if not messages:
            payload["messages"] = [{"role": "user", "content": "(empty)"}]
        if temperature is not None:
            payload["temperature"] = temperature
        if response_model is not None:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": STRUCTURED_TOOL_NAME,
                        "description": (
                            f"Return a result matching the {response_model.__name__} schema."
                        ),
                        "parameters": _tool_schema(response_model),
                    },
                }
            ]
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": STRUCTURED_TOOL_NAME},
            }
        log.info(
            "llm.openrouter.complete",
            model=model,
            purpose=purpose,
            structured=response_model is not None,
            max_tokens=max_tokens,
            prompt_cache=_should_cache_system(model, self._settings),
        )
        response = await self._get_client().chat.completions.create(**payload)
        raw = _extract_payload(response)
        usage = _extract_usage(response, messages, raw, model=model)
        return raw, usage


def supports_explicit_prompt_cache(model: str) -> bool:
    """True when the provider accepts Anthropic-style ``cache_control`` blocks.

    Args:
        model: OpenRouter model id.

    Returns:
        True for Anthropic model ids.
    """
    return model.startswith("anthropic/")


def _should_cache_system(model: str, settings: LLMSettings) -> bool:
    return bool(settings.prompt_cache) and supports_explicit_prompt_cache(model)


def _serialize_messages(
    messages: list[Message],
    *,
    cache_system: bool,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    cached_system = False
    for message in messages:
        if cache_system and message.role == "system" and not cached_system:
            serialized.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": message.content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
            cached_system = True
            continue
        serialized.append({"role": message.role, "content": message.content})
    return serialized


def _tool_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    schema = response_model.model_json_schema()
    schema.pop("$schema", None)
    if schema.get("type") != "object":
        schema["type"] = "object"
    return schema


def _extract_payload(response: Any) -> str | Mapping[str, Any]:
    choices = getattr(response, "choices", [])
    if not choices:
        return ""
    message = choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        arguments = getattr(tool_calls[0].function, "arguments", "")
        return str(arguments)
    content = getattr(message, "content", None)
    return "" if content is None else str(content)


def _extract_usage(
    response: Any,
    messages: list[Message],
    raw: str | Mapping[str, Any],
    *,
    model: str,
) -> TokenUsage:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    cached_prompt_tokens = _cached_prompt_tokens(usage)
    if (
        isinstance(prompt_tokens, int)
        and isinstance(completion_tokens, int)
        and isinstance(total_tokens, int)
    ):
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated=False,
            model=model,
            cached_prompt_tokens=cached_prompt_tokens,
        )
    estimated = estimate_usage(messages, _payload_text(raw))
    return TokenUsage(
        prompt_tokens=estimated.prompt_tokens,
        completion_tokens=estimated.completion_tokens,
        total_tokens=estimated.total_tokens,
        estimated=True,
        model=model,
        cached_prompt_tokens=0,
    )


def _cached_prompt_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _int_attr(
        details,
        "cached_tokens",
        "cache_read_input_tokens",
        "cached_prompt_tokens",
    )
    if cached:
        return cached
    return _int_attr(
        usage,
        "cache_read_input_tokens",
        "cached_tokens",
        "prompt_cache_hit_tokens",
    )


def _int_attr(payload: Any, *names: str) -> int:
    if payload is None:
        return 0
    if isinstance(payload, Mapping):
        for name in names:
            value = payload.get(name)
            if isinstance(value, int) and value >= 0:
                return value
        return 0
    for name in names:
        value = getattr(payload, name, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _payload_text(raw: str | Mapping[str, Any]) -> str:
    if isinstance(raw, str):
        return raw
    return str(raw)


def _stream_delta_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None) if delta is not None else None
    return "" if content is None else str(content)


def _stream_usage(chunk: Any, messages: list[Message], *, model: str) -> TokenUsage | None:
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if not (
        isinstance(prompt_tokens, int)
        and isinstance(completion_tokens, int)
        and isinstance(total_tokens, int)
    ):
        return None
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated=False,
        model=model,
        cached_prompt_tokens=_cached_prompt_tokens(usage),
    )
