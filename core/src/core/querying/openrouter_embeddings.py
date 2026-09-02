"""Real embedding model behind the ``EmbeddingProvider`` protocol.

Calls OpenRouter's OpenAI-compatible ``/embeddings`` endpoint, reusing the
credentials the LLM stack already has. ``embed`` is synchronous because both
call sites -- the indexer's parse thread and graph_query's ``asyncio.to_thread``
worker -- are already off the event loop; calling it from a running loop would
block it.

Requesting ``dimensions`` keeps output the same width as the Neo4j vector
indexes, so a real model can replace the hash fallback without recreating
them. The dimension is validated up front rather than left to fail as a Neo4j
write rejection thousands of nodes later.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from core.graph.schema import VECTOR_INDEX_DIMENSIONS
from core.logging import get_logger
from core.querying.embeddings import l2_normalize
from core.settings import EmbeddingSettings

if TYPE_CHECKING:
    from openai.types import CreateEmbeddingResponse

log = get_logger(__name__)

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0


class EmbeddingDimensionError(ValueError):
    """Configured or returned vectors do not match the Neo4j vector indexes."""


class OpenRouterEmbeddingProvider:
    """EmbeddingProvider backed by OpenRouter's embeddings endpoint."""

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        settings: EmbeddingSettings | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        """Create a provider.

        Args:
            client: Optional pre-built OpenAI-compatible client.
            settings: Optional settings snapshot. Loaded from env when omitted.
            sleep: Injected for tests so backoff does not slow the suite.

        Raises:
            EmbeddingDimensionError: Configured dimension does not match the
                dimension the Neo4j vector indexes were created with.
        """
        self._settings = settings or EmbeddingSettings.from_env()
        self.dimensions = self._settings.dimensions
        if self.dimensions != VECTOR_INDEX_DIMENSIONS:
            raise EmbeddingDimensionError(
                f"EMBEDDING_DIMENSIONS={self.dimensions} does not match the Neo4j "
                f"vector indexes ({VECTOR_INDEX_DIMENSIONS}). Drop and recreate "
                "them before changing the dimension; CREATE ... IF NOT EXISTS "
                "will not resize an existing index."
            )
        self.fingerprint = f"openrouter:{self._settings.model}:{self.dimensions}"
        self._client = client
        self._sleep = sleep

    @classmethod
    def from_env(cls) -> OpenRouterEmbeddingProvider:
        """Build a provider from the embedding environment settings.

        Returns:
            OpenRouterEmbeddingProvider.
        """
        return cls()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, in batches, preserving input order.

        Blank inputs get a zero vector without a network call, matching the hash
        provider and avoiding an API rejection for empty input.

        Args:
            texts: Input strings (query or docstring-plus-summary documents).

        Returns:
            One unit-length vector per text, each of length ``self.dimensions``.

        Raises:
            EmbeddingDimensionError: The API returned an unexpected width.
            Exception: The last transient error after exhausting attempts, or
                any non-transient API error immediately.
        """
        vectors: list[list[float]] = [[0.0] * self.dimensions for _ in texts]
        pending = [(index, text) for index, text in enumerate(texts) if text.strip()]
        if not pending:
            return vectors

        batch_size = self._settings.batch_size
        started = time.perf_counter()
        total_tokens = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            response = self._embed_batch([text for _index, text in batch])
            total_tokens += getattr(response.usage, "total_tokens", 0) or 0
            for item in response.data:
                if item.index >= len(batch):
                    raise EmbeddingDimensionError(
                        f"embedding response index {item.index} outside batch of {len(batch)}"
                    )
                vector = [float(value) for value in item.embedding]
                if len(vector) != self.dimensions:
                    raise EmbeddingDimensionError(
                        f"model returned {len(vector)}-dim vectors, expected {self.dimensions}"
                    )
                vectors[batch[item.index][0]] = l2_normalize(vector)

        log.info(
            "embeddings.batch_done",
            model=self._settings.model,
            texts=len(pending),
            batches=-(-len(pending) // batch_size),
            tokens=total_tokens,
            usd=round(total_tokens / 1_000_000 * self._settings.usd_per_million, 6),
            elapsed_s=round(time.perf_counter() - started, 3),
        )
        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> CreateEmbeddingResponse:
        """Call the endpoint once, retrying transient failures with backoff."""
        attempts = self._settings.max_attempts
        for attempt in range(1, attempts + 1):
            try:
                return self._get_client().embeddings.create(
                    model=self._settings.model,
                    input=list(batch),
                    dimensions=self.dimensions,
                )
            except TRANSIENT_ERRORS as exc:
                if attempt >= attempts:
                    log.error(
                        "embeddings.batch_failed",
                        model=self._settings.model,
                        size=len(batch),
                        attempts=attempts,
                        error=str(exc),
                    )
                    raise
                delay = min(_BACKOFF_BASE_S * 2 ** (attempt - 1), _BACKOFF_CAP_S)
                log.warning(
                    "embeddings.retry",
                    attempt=attempt,
                    attempts=attempts,
                    delay_s=delay,
                    error=str(exc),
                )
                self._sleep(delay)
        raise AssertionError("unreachable: loop either returns or raises")

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = self._settings.api_key
            self._client = OpenAI(
                api_key=api_key.get_secret_value() if api_key else None,
                base_url=self._settings.base_url,
                timeout=self._settings.timeout_s,
                max_retries=0,
            )
        return self._client


__all__ = ["EmbeddingDimensionError", "OpenRouterEmbeddingProvider"]
