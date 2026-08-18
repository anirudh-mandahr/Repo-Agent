"""Smoke check: ``find_entity('get_openapi')`` against the indexed graph."""

from __future__ import annotations

from core.graph.client import GraphClient
from core.logging import bind_correlation_id, configure_logging, get_logger
from core.querying.service import GraphQueryService

log = get_logger(__name__)


def run_smoke() -> None:
    """Fail if ``find_entity('get_openapi')`` returns no matches."""
    configure_logging()
    bind_correlation_id()
    with GraphClient() as client:
        client.verify_connectivity()
        result = GraphQueryService(client).find_entity("get_openapi")
    if result.error:
        raise RuntimeError(result.error)
    if result.result_count < 1:
        raise RuntimeError("find_entity('get_openapi') returned no matches")
    hit = next(
        (
            item
            for item in result.matches
            if item.qualified_name.endswith("get_openapi")
        ),
        result.matches[0],
    )
    log.info(
        "smoke.find_entity",
        qualified_name=hit.qualified_name,
        result_count=result.result_count,
    )
