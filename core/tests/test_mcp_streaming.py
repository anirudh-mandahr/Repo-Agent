"""MCP progress codec used to stream orchestrator events to the gateway."""

from __future__ import annotations

from typing import Any

import pytest

from core.mcp.streaming import (
    decode_stream_event,
    encode_stream_event,
    progress_callback_for_stream,
    report_stream_event,
    stream_callbacks_from_mcp_context,
)


def test_encode_decode_roundtrip_token_and_routing() -> None:
    token_msg = encode_stream_event("token", chunk="Hel")
    decoded = decode_stream_event(token_msg)
    assert decoded is not None
    kind, payload = decoded
    assert kind == "token"
    assert payload["chunk"] == "Hel"

    routing_msg = encode_stream_event("routing", data={"mode": "rules"})
    decoded_routing = decode_stream_event(routing_msg)
    assert decoded_routing is not None
    assert decoded_routing[0] == "routing"
    assert decoded_routing[1]["data"] == {"mode": "rules"}


def test_decode_stream_event_rejects_invalid_payloads() -> None:
    assert decode_stream_event(None) is None
    assert decode_stream_event("") is None
    assert decode_stream_event("not-json") is None
    assert decode_stream_event("[]") is None
    assert decode_stream_event('{"event":""}') is None
    assert decode_stream_event('{"event":1}') is None


@pytest.mark.asyncio
async def test_stream_callbacks_noop_without_ctx() -> None:
    on_token, on_event = stream_callbacks_from_mcp_context(None)
    await on_token("x")
    await on_event("routing", {"mode": "rules"})
    await report_stream_event(None, index=1, event="token", chunk="x")
    await report_stream_event(object(), index=1, event="token", chunk="x")
    await report_stream_event(object(), index=1, event="token", chunk="")


@pytest.mark.asyncio
async def test_report_stream_event_swallows_progress_errors() -> None:
    class _Boom:
        async def report_progress(
            self, progress: float, total: float | None = None, message: str | None = None
        ) -> None:
            _ = progress, total, message
            raise RuntimeError("no session")

    await report_stream_event(_Boom(), index=1, event="token", chunk="x")


@pytest.mark.asyncio
async def test_stream_callbacks_emit_progress_messages() -> None:
    reported: list[tuple[float, str | None]] = []

    class _Ctx:
        async def report_progress(
            self, progress: float, total: float | None = None, message: str | None = None
        ) -> None:
            _ = total
            reported.append((progress, message))

    on_token, on_event = stream_callbacks_from_mcp_context(_Ctx())
    await on_token("")
    await on_token("Hel")
    await on_event("routing", {"mode": "rules"})
    assert len(reported) == 2
    assert reported[0][0] == 1.0
    first = decode_stream_event(reported[0][1])
    second = decode_stream_event(reported[1][1])
    assert first is not None and first[0] == "token" and first[1]["chunk"] == "Hel"
    assert second is not None and second[0] == "routing"


@pytest.mark.asyncio
async def test_progress_callback_dispatches_token_and_event() -> None:
    tokens: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_token(chunk: str) -> None:
        tokens.append(chunk)

    async def on_event(event_type: str, data: dict[str, Any]) -> None:
        events.append((event_type, data))

    callback = progress_callback_for_stream(on_token=on_token, on_event=on_event)
    await callback(1.0, None, encode_stream_event("token", chunk="lo"))
    await callback(2.0, None, encode_stream_event("agent_result", data={"ok": True}))
    await callback(3.0, None, "not-a-stream-event")
    await callback(4.0, None, encode_stream_event("token", chunk=""))
    assert tokens == ["lo"]
    assert events == [("agent_result", {"ok": True})]
