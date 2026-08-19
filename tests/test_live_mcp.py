"""Live stack tests: gateway MCP path and correlation-id log propagation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from typing import Any

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("LIVE_STACK") != "1", reason="LIVE_STACK not set"),
]

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")


def _post_chat(message: str, *, correlation_id: str, session_id: str) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    payload = json.dumps({"message": message, "session_id": session_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{GATEWAY_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-correlation-id": correlation_id,
            "x-api-key": os.environ.get("GATEWAY_API_KEY", "dev-gateway-key"),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
            body["_status"] = response.status
            return body
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise AssertionError(f"chat failed HTTP {exc.code}: {detail}") from exc


def _compose_logs(service: str) -> str:
    proc = subprocess.run(
        ["docker", "compose", "logs", "--no-color", "--no-log-prefix", service],
        capture_output=True,
        text=True,
        check=False,
    )
    return f"{proc.stdout}\n{proc.stderr}"


def test_gateway_chat_mcp_path_returns_grounded_tools() -> None:
    correlation_id = f"live-e2e-{uuid.uuid4()}"
    body = _post_chat(
        "Show me the FastAPI class implementation in the codebase",
        correlation_id=correlation_id,
        session_id=f"live-e2e-{uuid.uuid4()}",
    )
    assert body.get("answer")
    tools = body.get("tools_invoked") or []
    assert tools, f"expected tools_invoked, got {body!r}"
    answer = str(body["answer"])
    assert ".py" in answer or "fastapi/" in answer.lower()
    citations = re.findall(r"[\w./-]+\.py", answer)
    assert citations, f"expected file citations in answer: {answer!r}"


def test_correlation_id_appears_in_gateway_and_agent_logs() -> None:
    correlation_id = f"live-corr-{uuid.uuid4()}"
    body = _post_chat(
        "What is the FastAPI class?",
        correlation_id=correlation_id,
        session_id=f"live-corr-{uuid.uuid4()}",
    )
    assert body.get("answer")
    gateway_logs = _compose_logs("gateway")
    agent_logs = _compose_logs("graph_query") + "\n" + _compose_logs("orchestrator")
    assert correlation_id in gateway_logs, gateway_logs[-4000:]
    assert correlation_id in agent_logs, agent_logs[-4000:]
