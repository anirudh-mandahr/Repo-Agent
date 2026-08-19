"""Day-2 MCP smoke: find_entity → get_dependents → explain_implementation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from mcp.types import CallToolResult, TextContent

from core.mcp.client import open_streamable_http_session, parse_call_tool_result

DEFAULT_GRAPH_QUERY_URL = "http://graph_query:8003/mcp"
DEFAULT_CODE_ANALYST_URL = "http://code_analyst:8004/mcp"
READ_TIMEOUT = timedelta(seconds=180)
FIND_NAME = "FastAPI"


class SmokeError(RuntimeError):
    """Raised when a smoke step fails."""


def _payload(result: CallToolResult) -> dict[str, Any]:
    structured = result.structuredContent
    if isinstance(structured, dict):
        inner = structured.get("result", structured)
        if isinstance(inner, dict):
            return dict(inner)
    if result.content:
        block = result.content[0]
        text = block.text if isinstance(block, TextContent) else str(block)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"detail": text}
        if isinstance(parsed, dict):
            return parsed
    return {}


async def call_tool(url: str, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Open a streamable-HTTP MCP session, call ``tool``, and return its payload.

    Args:
        url: MCP endpoint.
        tool: Tool name.
        arguments: Tool arguments.

    Returns:
        Parsed JSON-ish payload.

    Raises:
        SmokeError: The tool returned ``isError`` or an error field.
    """
    session = await open_streamable_http_session(url)
    try:
        raw = await session.call_tool(tool, arguments=dict(arguments))
    finally:
        await session.aclose()
    payload = _payload(raw) if hasattr(raw, "structuredContent") else {}
    if hasattr(raw, "isError") and raw.isError:
        detail = payload.get("error") or payload.get("detail") or str(getattr(raw, "content", raw))
        raise SmokeError(f"{tool} failed: {detail}")
    try:
        parsed = parse_call_tool_result(
            raw,
            agent="smoke",
            tool=tool,
            correlation_id="smoke",
        )
    except Exception as exc:
        raise SmokeError(f"{tool} failed: {exc}") from exc
    if isinstance(parsed, dict):
        payload = parsed
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, str) and error.strip():
        raise SmokeError(f"{tool} returned error: {error}")
    if not isinstance(payload, dict):
        return {"result": payload}
    return payload


def _print_step(title: str, payload: Mapping[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(dict(payload), indent=2, default=str))


def _pick_entity(matches: list[Any]) -> dict[str, Any]:
    typed = [item for item in matches if isinstance(item, dict)]
    for match in typed:
        if match.get("entity_type") == "Class" and match.get("qualified_name"):
            return match
    for match in typed:
        if match.get("qualified_name") or match.get("name"):
            return match
    raise SmokeError(f"find_entity({FIND_NAME!r}) returned no usable match")


async def run() -> None:
    """Run the three-step MCP smoke against graph_query and code_analyst."""
    graph_url = os.environ.get("SMOKE_GRAPH_QUERY_URL", DEFAULT_GRAPH_QUERY_URL)
    analyst_url = os.environ.get("SMOKE_CODE_ANALYST_URL", DEFAULT_CODE_ANALYST_URL)
    started = time.perf_counter()

    entities = await call_tool(graph_url, "find_entity", {"name": FIND_NAME})
    _print_step(f'find_entity("{FIND_NAME}")', entities)
    matches = entities.get("matches")
    if not isinstance(matches, list) or not matches:
        raise SmokeError(f"find_entity({FIND_NAME!r}) returned no matches")
    hit = _pick_entity(matches)
    target = str(hit.get("qualified_name") or hit.get("name") or "")
    if not target:
        raise SmokeError("selected find_entity match has no name")

    dependents = await call_tool(graph_url, "get_dependents", {"name": target})
    _print_step(f'get_dependents("{target}")', dependents)

    explained = await call_tool(analyst_url, "explain_implementation", {"qualified_name": target})
    _print_step(f'explain_implementation("{target}")', explained)
    if not str(explained.get("explanation") or "").strip():
        raise SmokeError("explain_implementation returned an empty explanation")

    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"\n=== total latency ===\n{elapsed_ms:.0f} ms")
    _ = READ_TIMEOUT


def main() -> None:
    """Entrypoint for ``python -m core.mcp.smoke``."""
    try:
        asyncio.run(run())
    except SmokeError as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
