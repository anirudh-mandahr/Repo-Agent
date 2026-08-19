"""Synthesize an orchestrator final answer from partial agent outputs."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from core.llm.provider import LLMProvider
from core.logging import get_logger
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger
from core.querying.service import proper_noun_tokens
from core.settings import OrchestratorSettings

from .models import AgentName, SynthesisResult
from .prompt_budget import (
    TruncationOrder,
    apply_prompt_budget,
    compact_neighbor_payloads,
    measure_synthesis_prompt,
    render_synthesis_user_prompt,
)
from .prompts import SYNTHESIS_SYSTEM_PROMPT
from .router import wants_codebase_grounding
from .scope import is_out_of_scope

log = get_logger(__name__)

TokenCallback = Callable[[str], Awaitable[None]]

_INCOMPLETE_RETRIEVAL_PREFIX = (
    "Retrieval was incomplete because an agent errored, timed out, or was degraded. "
    "The answer below uses only the partial evidence that was retrieved."
)
_INCOMPLETE_NO_EVIDENCE = (
    "Retrieval was incomplete because an agent errored, timed out, or was degraded. "
    "I don't have enough indexed FastAPI evidence to answer confidently."
)
_FILE_LINE_CITE_RE = re.compile(r"\b[\w./-]+\.py:\d+")


def _format_session_context(context: ConversationContext) -> str:
    summary = context.summary or "(none)"
    recent = context.recent_turns
    recent_lines = []
    for turn in recent:
        recent_lines.append(f"- [{turn.role}] {turn.content}")
    recent_block = "\n".join(recent_lines) if recent_lines else "- (none)"
    return f"Summary:\n{summary}\n\nRecent turns:\n{recent_block}"


def _render_session_context_block(context: ConversationContext) -> str:
    if not context.summary and not context.recent_turns:
        return ""
    return f"\nConversation context (summary + recent turns):\n{_format_session_context(context)}\n"


def _graph_output(agent_outputs: Mapping[AgentName, Any]) -> Mapping[str, Any] | None:
    graph_output = agent_outputs.get("graph_query")
    payload = (
        graph_output.model_dump()
        if graph_output is not None and hasattr(graph_output, "model_dump")
        else graph_output
    )
    if not isinstance(payload, Mapping):
        return None
    output = payload.get("output")
    if not isinstance(output, Mapping):
        return None
    return output


def _snippet_evidence(agent_outputs: Mapping[AgentName, Any]) -> bool:
    for item in _analyst_items(agent_outputs, "snippets"):
        if _usable_snippet(item):
            return True
    return False


def _analyst_explanation_evidence(agent_outputs: Mapping[AgentName, Any]) -> bool:
    for key in ("explanations", "function_analyses", "comparison"):
        for item in _analyst_items(agent_outputs, key):
            if _usable_analysis(item):
                return True
    return False


def _analyst_evidence(agent_outputs: Mapping[AgentName, Any]) -> bool:
    return _snippet_evidence(agent_outputs) or _analyst_explanation_evidence(agent_outputs)


def _analyst_items(agent_outputs: Mapping[AgentName, Any], key: str) -> list[Any]:
    code_output = agent_outputs.get("code_analyst")
    code_payload = _as_mapping(code_output)
    if code_payload is None:
        return []
    inner = code_payload.get("output")
    if not isinstance(inner, Mapping):
        return []
    value = inner.get(key)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    return []


def _usable_snippet(value: Any) -> bool:
    payload = _as_mapping(value)
    if payload is None:
        return False
    if payload.get("error"):
        return False
    return bool(payload.get("text"))


def _usable_analysis(value: Any) -> bool:
    payload = _as_mapping(value)
    if payload is None:
        return False
    if payload.get("error"):
        return False
    return bool(
        payload.get("explanation")
        or payload.get("analysis")
        or payload.get("summary")
        or payload.get("text")
        or payload.get("comparison")
    )


def _agent_failed(payload: Mapping[str, Any] | None) -> bool:
    if payload is None:
        return False
    if payload.get("ok") is False:
        return True
    return bool(payload.get("degraded_note") or payload.get("error"))


def _retrieval_incomplete(agent_outputs: Mapping[AgentName, Any]) -> bool:
    return _agent_failed(_as_mapping(agent_outputs.get("graph_query"))) or _agent_failed(
        _as_mapping(agent_outputs.get("code_analyst"))
    )


def _missing_entity_answer(
    query: str,
    agent_outputs: Mapping[AgentName, Any],
) -> str | None:
    if is_out_of_scope(query):
        proper = proper_noun_tokens(query)
        missing = ", ".join(proper[:3]) if proper else "the query"
        return (
            "This topic is not in the indexed FastAPI codebase. "
            f"The query appears to refer to entities outside this repository ({missing}), "
            "so I can't explain it from indexed FastAPI sources."
        )
    output = _graph_output(agent_outputs)
    hits = _entity_hit_payloads(agent_outputs)
    has_hits = bool(hits)
    has_evidence = _analyst_evidence(agent_outputs)
    incomplete = _retrieval_incomplete(agent_outputs)

    if has_hits:
        if _incidental_fulltext_only(query, hits):
            proper = proper_noun_tokens(query)
            missing = ", ".join(proper[:3]) if proper else "the query"
            return (
                "This topic is not in the indexed FastAPI codebase. "
                f"The query appears to refer to entities outside this repository ({missing}), "
                "so I can't explain it from indexed FastAPI sources."
            )
        return None
    if has_evidence:
        return None
    if incomplete:
        return _INCOMPLETE_NO_EVIDENCE

    queried_entities: list[Any] = []
    if output is not None:
        maybe_queried = output.get("queried_entities")
        if isinstance(maybe_queried, list):
            queried_entities = maybe_queried

    if queried_entities:
        missing = ", ".join(str(entity) for entity in queried_entities[:3])
        return (
            "This topic is not in the indexed FastAPI codebase. "
            f"The query appears to refer to entities outside this repository ({missing}), "
            "so I can't explain it from indexed FastAPI sources."
        )

    if wants_codebase_grounding(query):
        return (
            "I could not find indexed FastAPI source for this request. "
            "I won't invent tutorial examples or paths that were not retrieved."
        )
    return None


def _retrieval_hits(output: Mapping[str, Any]) -> bool:
    """True when any cascade tier produced a usable entity hit."""
    return bool(_entity_hits_from_graph_output(output))


def _entity_hit_payloads(agent_outputs: Mapping[AgentName, Any]) -> list[Mapping[str, Any]]:
    graph = _graph_output(agent_outputs)
    if graph is None:
        return []
    return _entity_hits_from_graph_output(graph)


def _entity_hits_from_graph_output(output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    hits: list[Mapping[str, Any]] = []
    seen: set[int] = set()

    def _add(value: Any) -> None:
        payload = _as_mapping(value)
        if payload is None or not _is_entity_hit(payload):
            return
        key = id(payload)
        if key in seen:
            return
        seen.add(key)
        hits.append(payload)

    for key in ("analysis_candidates", "candidates"):
        rows = output.get(key)
        if isinstance(rows, list):
            for row in rows:
                _add(row)
    entities = output.get("entities")
    if isinstance(entities, list):
        for item in entities:
            payload = _as_mapping(item)
            if payload is None:
                continue
            matches = payload.get("matches")
            if isinstance(matches, list):
                for match in matches:
                    _add(match)
            else:
                _add(payload)
    return hits


def _incidental_fulltext_only(query: str, hits: Sequence[Mapping[str, Any]]) -> bool:
    """True when the only hits are weak fulltext/lexical names unrelated to the query."""
    if not hits:
        return False
    tiers = {str(hit.get("tier") or "exact") for hit in hits}
    if "exact" in tiers:
        return False
    if not (tiers & {"fulltext", "lexical"}):
        return False
    proper = [token.lower() for token in proper_noun_tokens(query)]
    if not proper:
        return False
    names = _retrieved_entity_names(hits)
    return not any(token in names for token in proper)


def _retrieved_entity_names(hits: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for hit in hits:
        for raw in (hit.get("name"), hit.get("qualified_name")):
            if not raw:
                continue
            text = str(raw).strip()
            if not text:
                continue
            names.add(text.lower())
            if "." in text:
                names.add(text.rsplit(".", 1)[-1].lower())
    return names


def _is_entity_hit(value: Any) -> bool:
    payload = value.model_dump() if hasattr(value, "model_dump") else value
    if not isinstance(payload, Mapping):
        return False
    return bool(
        payload.get("qualified_name")
        or payload.get("file_path")
        or payload.get("filePath")
        or payload.get("name")
        or payload.get("path")
    )


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    payload = value.model_dump() if hasattr(value, "model_dump") else value
    return payload if isinstance(payload, Mapping) else None


def _is_module_hit(item: Mapping[str, Any]) -> bool:
    entity_type = str(item.get("entity_type") or item.get("type") or "")
    if entity_type == "Module":
        return True
    labels = item.get("labels")
    return isinstance(labels, list) and any(str(label) == "Module" for label in labels)


def _citation_line(item: Mapping[str, Any]) -> str | None:
    path = str(item.get("file_path") or item.get("filePath") or item.get("path") or "").strip()
    qualified = str(item.get("qualified_name") or item.get("name") or "").strip()
    if not path and not qualified:
        return None
    line_start = item.get("line_start") or item.get("lineStart")
    line_end = item.get("line_end") or item.get("lineEnd")
    location = path or qualified
    if path and line_start and line_end and not _is_module_hit(item):
        location = f"{path}:{line_start}-{line_end}"
    if path and qualified and qualified not in path:
        return f"{location} ({qualified})"
    return location


def _evidence_citations(agent_outputs: Mapping[AgentName, Any]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()

    def _add(item: Any) -> None:
        payload = _as_mapping(item)
        if payload is None:
            return
        line = _citation_line(payload)
        if not line or line in seen:
            return
        seen.add(line)
        citations.append(line)

    graph = _graph_output(agent_outputs)
    if graph is not None:
        for key in ("analysis_candidates", "candidates"):
            rows = graph.get(key)
            if isinstance(rows, list):
                for row in rows:
                    _add(row)
        entities = graph.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                payload = _as_mapping(entity)
                if payload is None:
                    continue
                matches = payload.get("matches")
                if isinstance(matches, list):
                    for match in matches:
                        _add(match)
                else:
                    _add(payload)

    code = agent_outputs.get("code_analyst")
    code_payload = _as_mapping(code)
    if code_payload is not None:
        inner = code_payload.get("output")
        if isinstance(inner, Mapping):
            snippets = inner.get("snippets")
            if isinstance(snippets, list):
                for snippet in snippets:
                    payload = _as_mapping(snippet)
                    if payload is not None and payload.get("text") and not payload.get("error"):
                        _add(payload)
            for key in ("explanations", "function_analyses"):
                rows = inner.get(key)
                if isinstance(rows, list):
                    for row in rows:
                        payload = _as_mapping(row)
                        if payload is not None and _usable_analysis(payload):
                            _add(payload)
    return citations


def _with_grounded_sources(
    query: str,
    answer: str,
    agent_outputs: Mapping[AgentName, Any],
) -> str:
    if not wants_codebase_grounding(query):
        return answer
    citations = _evidence_citations(agent_outputs)
    if not citations:
        return answer
    # Bare paths in prose are not citations. Only skip the Sources footer when
    # the answer already has resolvable ``file:line`` coordinates; otherwise
    # every Sources path used to fail validation or be omitted from precision.
    if _FILE_LINE_CITE_RE.search(answer):
        return answer
    rendered = "\n".join(f"- {citation}" for citation in citations)
    return f"{answer.rstrip()}\n\nSources:\n{rendered}"


async def synthesize_response(
    query: str,
    agent_outputs: Mapping[AgentName, Any],
    context: ConversationContext | None,
    *,
    llm_provider: LLMProvider,
    settings: OrchestratorSettings,
    token_ledger: TokenLedger | None = None,
    correlation_id: str | None = None,
    timeout_s: float | None = None,
    on_token: TokenCallback | None = None,
) -> SynthesisResult:
    """Merge agent outputs into a final answer, falling back to retrieved evidence.

    One LLM call synthesizes the answer. When that call times out or errors, the
    already-retrieved specialist outputs are rendered as markdown instead of
    raising, so successful retrieval is never discarded. Tokens are streamed to
    ``on_token`` when the provider supports ``stream``; a mid-stream error still
    falls back to evidence.

    Args:
        query: User question.
        agent_outputs: Specialist payloads collected for this turn.
        context: Optional conversation context included in the synthesis prompt.
        llm_provider: LLM backend used for the synthesis call.
        settings: Orchestrator timeouts and the synthesis prompt token budget.
        token_ledger: Optional per-request token accumulator.
        correlation_id: Request id for logging and ledger keys.
        timeout_s: Optional derived wall-clock timeout. Defaults to
            ``settings.synthesis_timeout_s``.
        on_token: Optional callback invoked with each streamed synthesis chunk.

    Returns:
        Answer text plus budget / degradation metadata.
    """
    context = context or ConversationContext()
    incomplete = _retrieval_incomplete(agent_outputs)
    missing_entity = _missing_entity_answer(query, agent_outputs)
    if missing_entity is not None:
        log.info(
            "orchestrator.out_of_scope_refusal",
            correlation_id=correlation_id or "-",
        )
        return SynthesisResult(answer=missing_entity)

    agent_payload: dict[str, Any] = {
        agent: (output.model_dump() if hasattr(output, "model_dump") else output)
        for agent, output in agent_outputs.items()
    }
    agent_payload = compact_neighbor_payloads(agent_payload)
    session_block = _render_session_context_block(context)
    order: TruncationOrder = settings.prompt_truncation_order
    budgeted_payload, truncation = apply_prompt_budget(
        agent_payload,
        query=query,
        session_context_block=session_block,
        budget=settings.synthesis_prompt_token_budget,
        order=order,
    )
    user_prompt = render_synthesis_user_prompt(query, session_block, budgeted_payload)
    prompt_chars, estimated_tokens = measure_synthesis_prompt(
        query, session_block, budgeted_payload
    )

    from core.llm.provider import Message
    from core.observability.metrics import (
        record_synthesis_latency,
        record_ttft,
    )

    llm_messages = [
        Message(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]
    timeout = settings.synthesis_timeout_s if timeout_s is None else timeout_s
    started = time.perf_counter()
    first_token_at: float | None = None

    async def _emit(chunk: str) -> None:
        nonlocal first_token_at
        if not chunk:
            return
        if first_token_at is None:
            first_token_at = time.perf_counter()
            record_ttft(first_token_at - started)
        if on_token is not None:
            await on_token(chunk)

    if timeout <= 0:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.warning(
            "orchestrator.synthesis_skipped_no_time",
            correlation_id=correlation_id or "-",
            prompt_chars=prompt_chars,
            estimated_tokens=estimated_tokens,
            elapsed_ms=elapsed_ms,
            agents_invoked=list(agent_outputs.keys()),
        )
        return _evidence_fallback(
            query,
            agent_outputs,
            truncation=truncation,
            estimated_tokens=estimated_tokens,
            prompt_chars=prompt_chars,
            reason="TimeoutError",
            duration_s=time.perf_counter() - started,
        )

    try:
        result = await asyncio.wait_for(
            _complete_synthesis(
                llm_provider,
                llm_messages,
                on_delta=_emit,
            ),
            timeout=timeout,
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        exception_type = type(exc).__name__
        log.warning(
            "orchestrator.synthesis_failed",
            correlation_id=correlation_id or "-",
            prompt_chars=prompt_chars,
            estimated_tokens=estimated_tokens,
            elapsed_ms=elapsed_ms,
            agents_invoked=list(agent_outputs.keys()),
            exception_type=exception_type,
        )
        return _evidence_fallback(
            query,
            agent_outputs,
            truncation=truncation,
            estimated_tokens=estimated_tokens,
            prompt_chars=prompt_chars,
            reason=exception_type,
            duration_s=time.perf_counter() - started,
        )
    duration_s = time.perf_counter() - started
    record_synthesis_latency(duration_s)
    if first_token_at is None:
        record_ttft(duration_s)
        if on_token is not None and result.text:
            await on_token(result.text)
    if token_ledger is not None and correlation_id is not None:
        token_ledger.record(correlation_id, "synthesis", result.usage)
    answer = result.text
    if incomplete:
        answer = f"{_INCOMPLETE_RETRIEVAL_PREFIX}\n\n{answer.lstrip()}"
    return SynthesisResult(
        answer=_with_grounded_sources(query, answer, agent_outputs),
        prompt_truncated=truncation,
        estimated_tokens=estimated_tokens,
        prompt_chars=prompt_chars,
    )


async def _complete_synthesis(
    llm_provider: LLMProvider,
    llm_messages: list[Any],
    *,
    on_delta: TokenCallback,
) -> Any:
    from core.llm.provider import LLMResult, estimate_usage

    stream = getattr(llm_provider, "stream", None)
    if callable(stream):
        parts: list[str] = []
        usage = None
        iterator = stream(
            llm_messages,
            response_model=None,
            purpose="synthesis",
            agent="orchestrator",
            max_tokens=1024,
        )
        if inspect.isawaitable(iterator):
            iterator = await iterator
        async for item in iterator:
            delta, chunk_usage = _split_stream_item(item)
            if delta:
                parts.append(delta)
                await on_delta(delta)
            if chunk_usage is not None:
                usage = chunk_usage
        text = "".join(parts)
        resolved = usage or estimate_usage(llm_messages, text)
        return LLMResult(text=text, usage=resolved)
    result = await llm_provider.complete(
        llm_messages,
        response_model=None,
        purpose="synthesis",
        agent="orchestrator",
        max_tokens=1024,
    )
    return result


def _split_stream_item(item: Any) -> tuple[str, Any]:
    if isinstance(item, tuple) and len(item) == 2:
        delta, usage = item
        return str(delta or ""), usage
    return str(item or ""), None


def _evidence_fallback(
    query: str,
    agent_outputs: Mapping[AgentName, Any],
    *,
    truncation: Any,
    estimated_tokens: int,
    prompt_chars: int,
    reason: str,
    duration_s: float,
) -> SynthesisResult:
    from core.observability.metrics import record_evidence_only, record_synthesis_latency

    record_synthesis_latency(duration_s)
    record_evidence_only()
    from .fallback import render_evidence_answer

    answer = render_evidence_answer(query, agent_outputs, degraded_reason=reason)
    return SynthesisResult(
        answer=answer,
        evidence_only=True,
        degraded_reason=reason,
        prompt_truncated=truncation,
        estimated_tokens=estimated_tokens,
        prompt_chars=prompt_chars,
    )
