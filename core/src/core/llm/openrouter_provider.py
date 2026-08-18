"""OpenRouter implementation of LLMProvider."""

from __future__ import annotations

from collections.abc import Mapping
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
    ) -> None:
        settings = LLMSettings.from_env()
        self._model = model or settings.model
        self._api_key = settings.api_key.get_secret_value() if settings.api_key else None
        self._base_url = settings.base_url
        self._client = client

    @classmethod
    def from_env(cls) -> OpenRouterProvider:
        """Build a provider from the OpenRouter environment settings."""
        return cls()

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
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
    ) -> LLMResult:
        """Call OpenRouter, forcing a function call for structured responses."""

        async def invoke(
            attempt_messages: list[Message],
        ) -> tuple[str | Mapping[str, Any], TokenUsage | None]:
            return await self._invoke(attempt_messages, response_model, max_tokens)

        return await complete_with_schema_retry(
            invoke,
            messages,
            response_model,
            purpose=purpose,
            agent=agent,
        )

    async def _invoke(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        max_tokens: int,
    ) -> tuple[str | Mapping[str, Any], TokenUsage]:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
        }
        if not messages:
            payload["messages"] = [{"role": "user", "content": "(empty)"}]
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
            model=self._model,
            structured=response_model is not None,
            max_tokens=max_tokens,
        )
        response = await self._get_client().chat.completions.create(**payload)
        raw = _extract_payload(response)
        usage = _extract_usage(response, messages, raw)
        return raw, usage


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
) -> TokenUsage:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
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
        )
    return estimate_usage(messages, _payload_text(raw))


def _payload_text(raw: str | Mapping[str, Any]) -> str:
    if isinstance(raw, str):
        return raw
    return str(raw)
