"""Deterministic LLM stand-in. Records calls and returns queued canned responses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from core.exceptions import SchemaValidationError
from core.llm.provider import (
    LLMPurpose,
    LLMResult,
    Message,
    TokenUsage,
    complete_with_schema_retry,
)
from core.logging import get_logger

log = get_logger(__name__)


@dataclass
class RecordedCall:
    """One ``complete`` attempt captured by ``StubProvider``."""

    messages: list[Message]
    response_model: type[BaseModel] | None
    purpose: LLMPurpose


class StubProvider:
    """Test double. Never calls a real API. Used by all tests."""

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        *,
        stream_chunk_delay_s: float = 0.0,
        delay_per_prompt_token_s: float = 0.0,
    ) -> None:
        """Queue canned completions for tests.

        Args:
            responses: FIFO of text, mappings, or models to return.
            stream_chunk_delay_s: Optional delay between streamed chunks.
            delay_per_prompt_token_s: When ``purpose`` is ``synthesis``, sleep
                this many seconds per estimated prompt token before completing.
                Models synthesis latency that scales with evidence volume.
        """
        self.calls: list[RecordedCall] = []
        self._queue: list[Any] = list(responses or [])
        self._stream_chunk_delay_s = stream_chunk_delay_s
        self._delay_per_prompt_token_s = delay_per_prompt_token_s

    def enqueue(self, *responses: Any) -> None:
        """Append canned responses to the FIFO queue.
        
        Args:
            responses: Any.
        """
        self._queue.extend(responses)

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
        """Pop the next canned response. Validate and retry once when modeled.
        
        Args:
            messages: list[Message].
            response_model: type[BaseModel] | None.
            purpose: LLMPurpose.
            agent: str.
            max_tokens: int.
            temperature: Unused; present to match ``LLMProvider``.

        Returns:
            LLMResult.

        Raises:
            SchemaValidationError: See exception message.
        """
        _ = max_tokens
        _ = agent
        _ = temperature

        if purpose == "synthesis" and self._delay_per_prompt_token_s > 0:
            prompt = "\n".join(message.content for message in messages)
            tokens = max(1, (len(prompt) + 3) // 4)
            await asyncio.sleep(tokens * self._delay_per_prompt_token_s)

        async def invoke(attempt_messages: list[Message]) -> tuple[Any, TokenUsage]:
            self.calls.append(
                RecordedCall(
                    messages=list(attempt_messages),
                    response_model=response_model,
                    purpose=purpose,
                )
            )
            log.info(
                "llm.stub.complete",
                call_count=len(self.calls),
                purpose=purpose,
                response_model=None if response_model is None else response_model.__name__,
                queue_remaining=max(len(self._queue) - 1, 0),
            )
            if not self._queue:
                raise SchemaValidationError(
                    agent=agent,
                    message="StubProvider has no canned responses",
                )
            return self._queue.pop(0), TokenUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                estimated=False,
                model="stub",
                cached_prompt_tokens=0,
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
        """Yield canned text in small chunks, then the usage on the last delta.

        Args:
            messages: Chat messages.
            response_model: Optional structured schema (collected via complete).
            purpose: Ledger purpose.
            agent: Calling agent.
            max_tokens: Unused; matches ``LLMProvider``.
            temperature: Unused; matches ``LLMProvider``.

        Yields:
            ``(delta, usage_or_none)`` pairs.
        """
        result = await self.complete(
            messages,
            response_model,
            purpose=purpose,
            agent=agent,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = result.text
        if not text:
            yield "", result.usage
            return
        step = max(1, min(16, max(len(text) // 4, 1)))
        for index in range(0, len(text), step):
            if self._stream_chunk_delay_s > 0:
                await asyncio.sleep(self._stream_chunk_delay_s)
            chunk = text[index : index + step]
            is_last = index + step >= len(text)
            yield chunk, (result.usage if is_last else None)
