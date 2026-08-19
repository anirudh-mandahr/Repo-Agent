"""Pluggable vector protocol and a local hashed bag-of-words fallback.

``HashingEmbeddingProvider`` is lexical token overlap (256-dim signed hash),
not a semantic embedding model. Graph Query's third retrieval tier uses this
when ``GQ_EMBEDDINGS_ENABLED`` is on. Tests inject a stub provider. Indexer
and graph_query share the same provider so index-time vectors match query-time
vectors. A real embedding model is future work; keep this protocol as the seam.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from core.graph.schema import VECTOR_INDEX_DIMENSIONS

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]+|[a-z]+|\d+")
_EMBED_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "where",
        "who",
        "why",
        "with",
    }
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Maps texts to dense vectors. Implementations must be deterministic in tests."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per input text, all of equal dimension.
        
        Args:
            texts: Sequence[str].

        Returns:
            Sequence[Sequence[float]].
        """
        ...


class HashingEmbeddingProvider:
    """Deterministic signed hashing embedder (lexical fallback, not semantics).

    Same text always yields the same vector. Token identity is hashed into a
    fixed-width bag-of-words; similar wording can match, synonyms will not.
    """

    def __init__(self, dimensions: int = VECTOR_INDEX_DIMENSIONS) -> None:
        """Create a provider with a fixed output dimension.

        Args:
            dimensions: Vector length. Must match the Neo4j vector index.
        """
        if dimensions < 8:
            raise ValueError("embedding dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text as an L2-normalized hashed bag-of-words vector.

        Args:
            texts: Input strings (query or docstring-plus-summary documents).

        Returns:
            One vector per text, each of length ``self.dimensions``.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize_for_embedding(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)


def tokenize_for_embedding(text: str) -> list[str]:
    """Split identifiers and stem tokens so 'dependency' matches 'dependencies'.
    
    Args:
        text: str.

    Returns:
        list[str].
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        pieces = _CAMEL_RE.findall(raw) if any(ch.isupper() for ch in raw[1:]) else [raw]
        if not pieces:
            pieces = [raw]
        for piece in pieces:
            stemmed = _stem_token(piece.lower())
            if len(stemmed) < 2 or stemmed in _EMBED_STOPWORDS or stemmed in seen:
                continue
            seen.add(stemmed)
            tokens.append(stemmed)
    return tokens


def build_embedding_text(
    *,
    qualified_name: str,
    name: str,
    docstring: str = "",
    source_summary: str = "",
    param_names: Sequence[str] | None = None,
) -> str:
    """Build the document stored/embedded for a Class, Function, or Method.
    
    Args:
        qualified_name: str.
        name: str.
        docstring: str.
        source_summary: str.
        param_names: Sequence[str] | None.

    Returns:
        str.
    """
    params = [item for item in (param_names or []) if item]
    entity_name = qualified_name or name
    signature = f"{entity_name}({', '.join(params)})" if params else entity_name
    parts = [signature, name]
    doc = docstring.strip()
    if doc:
        parts.append(doc)
    summary = source_summary.strip()
    if summary:
        parts.append(summary)
    return "\n".join(part for part in parts if part).strip()


def default_embedding_provider() -> HashingEmbeddingProvider:
    """Return the shared local embedding backend used by indexer and graph_query.
    
    Returns:
        HashingEmbeddingProvider.
    """
    return HashingEmbeddingProvider()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity in ``[-1, 1]``. Zero-length vectors score 0.
    
    Args:
        left: Sequence[float].
        right: Sequence[float].

    Returns:
        float.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _stem_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 5:
        token = token[:-3] + "y"
    elif token.endswith("es") and len(token) > 5:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]
    if token.endswith("ion") and len(token) > 6:
        token = token[:-3]
    if token.endswith("ing") and len(token) > 6:
        token = token[:-3]
    return token


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "build_embedding_text",
    "cosine_similarity",
    "default_embedding_provider",
    "tokenize_for_embedding",
]
