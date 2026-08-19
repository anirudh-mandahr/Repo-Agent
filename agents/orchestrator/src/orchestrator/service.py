"""Thin adapter re-exporting the orchestrator loop from core.

Business logic lives in ``core.orchestration.service``. This package stays a
FastMCP-facing import path for tests and scripts that still import
``orchestrator.service``.
"""

from core.orchestration.service import (
    CodeAnalystClient,
    GraphQueryClient,
    HandleQueryResult,
    IndexerClient,
    MemoryClient,
    OrchestratorClients,
    OrchestratorService,
    build_default_llm_provider,
)

__all__ = [
    "CodeAnalystClient",
    "GraphQueryClient",
    "HandleQueryResult",
    "IndexerClient",
    "MemoryClient",
    "OrchestratorClients",
    "OrchestratorService",
    "build_default_llm_provider",
]
