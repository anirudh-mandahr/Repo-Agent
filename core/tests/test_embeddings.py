"""Embedding backend selection, batching, retry, and dimension enforcement.

No network: the OpenAI-compatible client is a recording fake throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from openai import APIConnectionError, RateLimitError

from core.graph.schema import VECTOR_INDEX_DIMENSIONS
from core.querying.embeddings import (
    HashingEmbeddingProvider,
    cosine_similarity,
    default_embedding_provider,
)
from core.querying.openrouter_embeddings import (
    EmbeddingDimensionError,
    OpenRouterEmbeddingProvider,
)
from core.settings import EmbeddingSettings

DIMS = VECTOR_INDEX_DIMENSIONS


class _Item:
    def __init__(self, index: int, embedding: Sequence[float]) -> None:
        self.index = index
        self.embedding = list(embedding)


class _Response:
    def __init__(self, items: Sequence[_Item], total_tokens: int = 10) -> None:
        self.data = list(items)
        self.usage = type("Usage", (), {"total_tokens": total_tokens})()


class _FakeEmbeddings:
    """Records every create() call and replays a queue of scripted outcomes."""

    def __init__(self, outcomes: list[Any] | None = None, *, dimensions: int = DIMS) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outcomes = outcomes or []
        self.dimensions = dimensions

    def create(self, *, model: str, input: list[str], dimensions: int) -> _Response:
        self.calls.append({"model": model, "input": list(input), "dimensions": dimensions})
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome is not None:
                return outcome
        return _Response(
            [_Item(i, [float(i + 1)] + [0.0] * (self.dimensions - 1)) for i in range(len(input))]
        )


class _FakeClient:
    def __init__(self, embeddings: _FakeEmbeddings) -> None:
        self.embeddings = embeddings


def _provider(
    embeddings: _FakeEmbeddings,
    **overrides: Any,
) -> tuple[OpenRouterEmbeddingProvider, list[float]]:
    """Build a provider over the fake client, capturing backoff sleeps."""
    slept: list[float] = []
    settings = EmbeddingSettings(backend="openrouter", api_key="k", **overrides)
    provider = OpenRouterEmbeddingProvider(
        _FakeClient(embeddings),
        settings=settings,
        sleep=slept.append,
    )
    return provider, slept


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=None)  # type: ignore[arg-type]


def _rate_limit_error() -> RateLimitError:
    response = type("R", (), {"status_code": 429, "headers": {}, "request": None})()
    return RateLimitError("slow down", response=response, body=None)  # type: ignore[arg-type]


def test_backend_defaults_to_hash_even_when_an_api_key_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    assert isinstance(default_embedding_provider(), HashingEmbeddingProvider)


def _use_settings(monkeypatch: pytest.MonkeyPatch, settings: EmbeddingSettings) -> None:
    """Pin the settings the factory reads.

    The repo's own .env supplies OPENROUTER_API_KEY, so deleting the process
    env var is not enough to simulate an unconfigured deployment.
    """
    monkeypatch.setattr(
        "core.querying.embeddings.EmbeddingSettings.from_env",
        classmethod(lambda cls: settings),
    )


def test_openrouter_backend_without_a_key_degrades_to_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_settings(monkeypatch, EmbeddingSettings(backend="openrouter", api_key=None))
    assert isinstance(default_embedding_provider(), HashingEmbeddingProvider)


def test_openrouter_backend_with_a_key_selects_the_model_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_settings(monkeypatch, EmbeddingSettings(backend="openrouter", api_key="sk-test"))
    assert isinstance(default_embedding_provider(), OpenRouterEmbeddingProvider)


def test_dimension_mismatch_fails_at_construction_not_at_write_time() -> None:
    settings = EmbeddingSettings(backend="openrouter", api_key="k", dimensions=DIMS + 1)
    with pytest.raises(EmbeddingDimensionError, match="Drop and recreate"):
        OpenRouterEmbeddingProvider(settings=settings)


def test_requests_the_index_dimension_so_indexes_need_no_migration() -> None:
    embeddings = _FakeEmbeddings()
    provider, _slept = _provider(embeddings)
    provider.embed(["fastapi routing"])
    assert embeddings.calls[0]["dimensions"] == VECTOR_INDEX_DIMENSIONS
    assert embeddings.calls[0]["model"] == "openai/text-embedding-3-small"


def test_texts_are_split_into_batches() -> None:
    embeddings = _FakeEmbeddings()
    provider, _slept = _provider(embeddings, batch_size=10)
    vectors = provider.embed([f"entity {i}" for i in range(25)])
    assert len(vectors) == 25
    assert [len(call["input"]) for call in embeddings.calls] == [10, 10, 5]


def test_out_of_order_responses_are_realigned_with_their_inputs() -> None:
    shuffled = _Response(
        [
            _Item(2, [3.0] + [0.0] * (DIMS - 1)),
            _Item(0, [1.0] + [0.0] * (DIMS - 1)),
            _Item(1, [2.0] + [0.0] * (DIMS - 1)),
        ]
    )
    embeddings = _FakeEmbeddings([shuffled])
    provider, _slept = _provider(embeddings)
    vectors = provider.embed(["a", "b", "c"])
    # Every scripted vector is a distinct positive multiple of e0, so after
    # normalization each is e0 -- position is what proves the realignment.
    assert [vector[0] for vector in vectors] == [1.0, 1.0, 1.0]
    assert embeddings.calls[0]["input"] == ["a", "b", "c"]


def test_blank_texts_get_a_zero_vector_and_are_never_sent() -> None:
    embeddings = _FakeEmbeddings()
    provider, _slept = _provider(embeddings)
    vectors = provider.embed(["", "   ", "real text"])
    assert vectors[0] == [0.0] * DIMS
    assert vectors[1] == [0.0] * DIMS
    assert any(vectors[2])
    assert embeddings.calls[0]["input"] == ["real text"]


def test_all_blank_input_skips_the_api_entirely() -> None:
    embeddings = _FakeEmbeddings()
    provider, _slept = _provider(embeddings)
    assert provider.embed(["", " "]) == [[0.0] * DIMS, [0.0] * DIMS]
    assert embeddings.calls == []


def test_returned_vectors_are_unit_length() -> None:
    embeddings = _FakeEmbeddings([_Response([_Item(0, [3.0, 4.0] + [0.0] * (DIMS - 2))])])
    provider, _slept = _provider(embeddings)
    vector = provider.embed(["scale me"])[0]
    assert vector[0] == pytest.approx(0.6)
    assert vector[1] == pytest.approx(0.8)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_a_transient_failure_is_retried_with_backoff() -> None:
    embeddings = _FakeEmbeddings([_rate_limit_error(), _connection_error(), None])
    provider, slept = _provider(embeddings)
    vectors = provider.embed(["retry me"])
    assert len(vectors) == 1
    assert len(embeddings.calls) == 3
    assert slept == [0.5, 1.0]


def test_retries_are_bounded_and_the_last_error_propagates() -> None:
    embeddings = _FakeEmbeddings([_connection_error() for _ in range(3)])
    provider, slept = _provider(embeddings, max_attempts=3)
    with pytest.raises(APIConnectionError):
        provider.embed(["never works"])
    assert len(embeddings.calls) == 3
    assert slept == [0.5, 1.0]


def test_the_score_floor_is_backend_specific() -> None:
    """Neo4j reports cosine as (1 + cos) / 2, so the two backends need different floors.

    Hash vectors of unrelated text are exactly orthogonal, landing on 0.5; a
    real model puts unrelated code text near 0.62 normalized, which would clear
    a 0.6 floor and make the tier match everything.
    """
    assert EmbeddingSettings(backend="hash").resolve_min_score() == 0.6
    assert EmbeddingSettings(backend="openrouter").resolve_min_score() == 0.7


def test_an_explicit_score_floor_overrides_the_backend_default() -> None:
    settings = EmbeddingSettings(backend="openrouter", min_score=0.55)
    assert settings.resolve_min_score() == 0.55


def test_a_blank_score_floor_falls_back_to_the_backend_default() -> None:
    """Compose passes an unset optional through as an empty string."""
    settings = EmbeddingSettings(backend="openrouter", min_score="")  # type: ignore[arg-type]
    assert settings.min_score is None
    assert settings.resolve_min_score() == 0.7


def test_a_model_returning_the_wrong_width_is_rejected() -> None:
    embeddings = _FakeEmbeddings([_Response([_Item(0, [0.1] * (DIMS + 8))])])
    provider, _slept = _provider(embeddings)
    with pytest.raises(EmbeddingDimensionError, match="expected"):
        provider.embed(["wrong width"])
