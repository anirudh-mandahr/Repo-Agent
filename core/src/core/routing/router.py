"""Route a user query to a downstream agent. Stub implementation."""

from __future__ import annotations

from typing import Literal

AgentName = Literal["indexer", "graph_query", "code_analyst", "memory"]


def route_query(query: str) -> AgentName:
    """Return which specialist agent should handle `query`."""
    lowered = query.lower()
    if "index" in lowered:
        return "indexer"
    if "memory" in lowered or "remember" in lowered:
        return "memory"
    if "file" in lowered or "source" in lowered or "read" in lowered:
        return "code_analyst"
    return "graph_query"
