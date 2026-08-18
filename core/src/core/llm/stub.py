"""Deterministic LLM stand-in. Records calls and returns queued canned responses."""

from __future__ import annotations

from collections.abc import Sequence
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

    def __init__(self, responses: Sequence[Any] | None = None) -> None:
        self.calls: list[RecordedCall] = []
        self._queue: list[Any] = list(responses or [])

    def enqueue(self, *responses: Any) -> None:
        """Append canned responses to the FIFO queue."""
        self._queue.extend(responses)

    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
    ) -> LLMResult:
        """Pop the next canned response. Validate and retry once when modeled."""
        _ = max_tokens
        _ = agent

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
            )

        return await complete_with_schema_retry(
            invoke,
            messages,
            response_model,
            purpose=purpose,
            agent=agent,
        )
