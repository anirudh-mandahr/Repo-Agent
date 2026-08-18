"""MCP client graph_lookup. Calls the Graph Query agent's execute_query tool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from core.logging import get_correlation_id, get_logger
from core.querying.service import QueryResult
from core.settings import AnalysisSettings

log = get_logger(__name__)


def _payload(result: CallToolResult) -> dict[str, Any]:
    structured = result.structuredContent
    if isinstance(structured, dict):
        inner = structured.get("result", structured)
        if isinstance(inner, dict):
            return inner
    if result.content:
        block = result.content[0]
        text = block.text if isinstance(block, TextContent) else str(block)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


class GraphQueryLookup:
    """Async callable matching ``GraphLookup``: Cypher in, row dicts out."""

    def __init__(self, url: str | None = None) -> None:
        settings = AnalysisSettings.from_env()
        self._url = url or settings.graph_query_url

    async def __call__(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        correlation_id = get_correlation_id()
        arguments = {"cypher": cypher, "params": dict(params or {})}
        log.info(
            "analysis.graph_lookup",
            url=self._url,
            param_keys=sorted(arguments["params"]),
        )
        async with streamable_http_client(self._url) as (read, write, _session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "execute_query",
                    arguments=arguments,
                    meta={"correlation_id": correlation_id},
                )
        if result.isError:
            log.error("analysis.graph_lookup_failed", url=self._url)
            return []
        payload = _payload(result)
        try:
            parsed = QueryResult.model_validate(payload)
        except ValueError:
            rows = payload.get("rows")
            return list(rows) if isinstance(rows, list) else []
        return parsed.rows
