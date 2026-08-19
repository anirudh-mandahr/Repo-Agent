"""Day-3 smoke: end-to-end cache + degraded behavior via gateway SSE.

Contract (run against full Docker Compose stack):
1. Ask: "What is the FastAPI class?"
2. Ask: "What classes inherit from APIRouter?"
3. Ask: "How does dependency injection work and show me examples from the codebase"
   - with `stream=true` and print SSE events as they arrive.
4. Repeat query 1 with same `session_id`
   - assert `done.cached == true` and low `done.latency_ms`.
5. Stop `code_analyst` container
   - rerun query 3 with same `session_id`
   - assert `done.degraded == true` and non-empty answer
   - start `code_analyst` again

Finally, print a pass/fail table and exit non-zero on failures.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx


class SmokeError(RuntimeError):
    """Raised when a smoke step fails."""


@dataclass(frozen=True)
class StepResult:
    step: str
    ok: bool
    assertions: str
    details: str


def _docker_compose(*args: str) -> None:
    proc = subprocess.run(
        ["docker", "compose", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "docker compose command failed: "
            f"args={args!r}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


async def _wait_gateway_ready(*, url: str, timeout_s: float) -> None:
    start = time.perf_counter()
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.perf_counter() - start < timeout_s:
            try:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    return
            except Exception as exc:  # pragma: no cover - defensive
                last_exc = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"gateway not ready within {timeout_s}s: last_exc={last_exc}")


def _parse_sse_done(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SmokeError("SSE data payload is not a dict")
    done = payload.get("type") == "done"
    # Our gateway SSE `data:` payload contains {type, correlation_id, ...event.data}
    if not done:
        return {}
    return payload


async def _post_chat_non_stream(
    *,
    client: httpx.AsyncClient,
    url: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    r = await client.post(
        f"{url}/api/chat",
        json={"message": message, "session_id": session_id, "stream": False},
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict):
        raise SmokeError("chat response is not a JSON object")
    return payload


async def _post_chat_stream_print(
    *,
    client: httpx.AsyncClient,
    url: str,
    session_id: str,
    message: str,
    print_prefix: str,
) -> tuple[str, dict[str, Any]]:
    done_payload: dict[str, Any] | None = None
    answer_parts: list[str] = []

    current_event: str | None = None
    async with client.stream(
        "POST",
        f"{url}/api/chat",
        json={"message": message, "session_id": session_id, "stream": True},
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
                continue
            if not line.startswith("data:"):
                continue

            raw = line.split(":", 1)[1].strip()
            data = json.loads(raw)
            if not isinstance(data, dict):
                continue

            if current_event is None:
                continue

            # Print events as they arrive.
            if current_event == "answer":
                chunk = data.get("chunk", "")
                print(f"{print_prefix} event=answer chunk={chunk!r}")
                if isinstance(chunk, str):
                    answer_parts.append(chunk)
            else:
                print(f"{print_prefix} event={current_event} data_keys={list(data.keys())}")

            if current_event == "done":
                done_payload = data

    if done_payload is None:
        raise SmokeError("stream response missing done event")
    return "".join(answer_parts), done_payload


def _format_table(results: list[StepResult]) -> str:
    def row(cols: list[str]) -> str:
        return "| " + " | ".join(cols) + " |"

    lines: list[str] = []
    lines.append(row(["Step", "Status", "Assertions", "Details"]))
    lines.append(row(["---", "---", "---", "---"]))
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        lines.append(row([r.step, status, r.assertions, r.details]))
    return "\n".join(lines)


async def run() -> None:
    gateway_url = os.environ.get("SMOKE_GATEWAY_URL", "http://127.0.0.1:8000")
    cached_latency_ms_max = int(os.environ.get("SMOKE_CACHED_MAX_MS", "250"))

    await _wait_gateway_ready(url=gateway_url, timeout_s=90)

    session_id = os.environ.get("SMOKE_SESSION_ID") or str(uuid.uuid4())

    cached_latency_factor = float(os.environ.get("SMOKE_CACHED_FACTOR", "0.35"))

    async with httpx.AsyncClient(
        headers={"X-API-Key": os.environ.get("GATEWAY_API_KEY", "dev-gateway-key")}
    ) as client:
        results: list[StepResult] = []

        def record(step: str, ok: bool, assertions: str, details: str) -> None:
            results.append(StepResult(step=step, ok=ok, assertions=assertions, details=details))

        q1 = "What is the FastAPI class?"
        q2 = "What classes inherit from APIRouter?"
        q3 = (
            "How does dependency injection work and show me examples from the codebase"
        )

        # Step 1: query q1
        latency1: int | None = None
        step1_was_cached = False
        try:
            resp1 = await _post_chat_non_stream(
                client=client, url=gateway_url, session_id=session_id, message=q1
            )
            answer1 = str(resp1.get("answer", ""))
            done1 = resp1.get("done") or {}
            assert answer1.strip(), "answer empty"
            latency1 = int(done1.get("latency_ms", -1))
            step1_was_cached = bool(done1.get("cached", False))
            assert latency1 >= 0, f"missing latency_ms: done={done1!r}"
            record(
                "1",
                True,
                "answer non-empty",
                f"latency_ms={latency1} cached={done1.get('cached')}",
            )
        except Exception as exc:
            record("1", False, "answer non-empty", str(exc))

        # Step 2: query q2
        try:
            resp2 = await _post_chat_non_stream(
                client=client, url=gateway_url, session_id=session_id, message=q2
            )
            answer2 = str(resp2.get("answer", ""))
            done2 = resp2.get("done") or {}
            assert answer2.strip(), "answer empty"
            record("2", True, "answer non-empty", f"latency_ms={done2.get('latency_ms')}")
        except Exception as exc:
            record("2", False, "answer non-empty", str(exc))

        # Step 3: query q3 (streamed)
        try:
            answer3, done3 = await _post_chat_stream_print(
                client=client,
                url=gateway_url,
                session_id=session_id,
                message=q3,
                print_prefix="[step3]",
            )
            assert answer3.strip(), "answer empty"
            done3_latency = int(done3.get("latency_ms", -1))
            assert done3_latency >= 0, f"missing latency_ms: done={done3!r}"
            record(
                "3",
                True,
                "degraded/cached sanity (answer non-empty)",
                " ".join(
                    [
                        f"latency_ms={done3_latency}",
                        f"degraded={done3.get('degraded')}",
                        f"cached={done3.get('cached')}",
                    ]
                ),
            )
        except Exception as exc:
            record("3", False, "streamed answer non-empty", str(exc))

        # Step 4: repeat q1 => cached should be true, latency low
        done4: dict[str, Any] = {}
        try:
            resp4 = await _post_chat_non_stream(
                client=client, url=gateway_url, session_id=session_id, message=q1
            )
            answer4 = str(resp4.get("answer", ""))
            done4 = resp4.get("done") or {}
            assert answer4.strip(), "answer empty"
            cached4 = bool(done4.get("cached", False))
            latency4 = int(done4.get("latency_ms", -1))
            assert cached4 is True, f"expected cached=true, got {cached4} (done={done4!r})"
            assert latency4 >= 0, f"missing latency_ms: done={done4!r}"
            assert latency4 <= cached_latency_ms_max, (
                f"latency_ms too high: {latency4} > {cached_latency_ms_max} (done={done4!r})"
            )
            # Only compare relative latency when step 1 was a fresh (uncached)
            # response; if step 1 already hit the cache both are similarly fast.
            if (
                not step1_was_cached
                and latency1 is not None
                and isinstance(latency1, int)
                and latency1 > 0
            ):
                threshold = int(latency1 * cached_latency_factor) + 5
                assert latency4 <= threshold, (
                    " ".join(
                        [
                            "cached latency not lower enough:",
                            f"{latency4} > {threshold}",
                            f"(latency1={latency1})",
                        ]
                    )
                )
            record(
                "4",
                True,
                "cached=true + low latency",
                f"latency_ms={latency4} cached={cached4}",
            )
        except Exception as exc:
            record("4", False, "cached=true + low latency", str(exc))

        # Step 5: stop code_analyst => rerun q3 variant => degraded=true
        # Use a unique query so the orchestrator cache won't match.
        q3_degraded = f"{q3} (run {session_id[:8]})"
        try:
            _docker_compose("stop", "code_analyst")
            # Give TCP connections a moment to fail over.
            time.sleep(2.0)

            answer5, done5 = await _post_chat_stream_print(
                client=client,
                url=gateway_url,
                session_id=session_id,
                message=q3_degraded,
                print_prefix="[step5]",
            )
            assert answer5.strip(), "answer empty"
            degraded5 = bool(done5.get("degraded", False))
            assert degraded5 is True, f"expected degraded=true, got {degraded5} (done={done5!r})"

            record("5", True, "degraded=true + non-empty answer", f"degraded={degraded5}")
        except Exception as exc:
            record("5", False, "degraded=true + non-empty answer", str(exc))
        finally:
            # Best-effort restore for subsequent dev work.
            try:
                _docker_compose("start", "code_analyst")
            except Exception:
                pass

        print("\n" + _format_table(results))
        if any(not r.ok for r in results):
            raise SystemExit(1)


def main() -> None:
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

