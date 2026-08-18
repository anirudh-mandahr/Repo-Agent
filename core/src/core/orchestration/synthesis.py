"""Synthesize an orchestrator final answer from partial agent outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, cast

from core.llm.provider import LLMProvider
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger
from core.settings import OrchestratorSettings

from .models import AgentName
from .prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_PROMPT


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


def _missing_entity_answer(
    query: str,
    agent_outputs: Mapping[AgentName, Any],
) -> str | None:
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

    queried_entities = output.get("queried_entities")
    entities = output.get("entities")
    if not isinstance(queried_entities, list) or not queried_entities:
        return None
    if isinstance(entities, list) and entities:
        return None

    code_output = agent_outputs.get("code_analyst")
    code_payload = (
        code_output.model_dump()
        if code_output is not None and hasattr(code_output, "model_dump")
        else code_output
    )
    snippets: list[Any] = []
    if isinstance(code_payload, Mapping):
        inner = code_payload.get("output")
        if isinstance(inner, Mapping):
            maybe_snippets = inner.get("snippets")
            if isinstance(maybe_snippets, list):
                snippets = maybe_snippets
    if any(isinstance(item, Mapping) and item.get("text") for item in snippets):
        return None

    missing = ", ".join(str(entity) for entity in queried_entities[:3])
    return (
        "This topic is not in the indexed FastAPI codebase. "
        f"The query appears to refer to entities outside this repository ({missing}), "
        "so I can't explain it from indexed FastAPI sources."
    )


async def synthesize_response(
    query: str,
    agent_outputs: Mapping[AgentName, Any],
    context: ConversationContext | None,
    *,
    llm_provider: LLMProvider,
    settings: OrchestratorSettings,
    token_ledger: TokenLedger | None = None,
    correlation_id: str | None = None,
) -> str:
    """One LLM call that merges agent outputs into a final answer."""

    context = context or ConversationContext()
    missing_entity = _missing_entity_answer(query, agent_outputs)
    if missing_entity is not None:
        return missing_entity

    # Keep this deterministic and LLM-ingestible.
    agent_payload = {
        agent: (output.model_dump() if hasattr(output, "model_dump") else output)
        for agent, output in agent_outputs.items()
    }
    user_prompt = SYNTHESIS_USER_PROMPT.format(
        query=query,
        session_context_block=_render_session_context_block(context),
        agent_outputs=json.dumps(agent_payload, default=str, indent=2),
    )

    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # The core LLM provider expects `core.llm.provider.Message`; keep this small
    # and avoid importing circular modules by calling through the raw shape.
    # (The StubProvider only records message content.)
    from core.llm.provider import Message  # local import to avoid circulars

    llm_messages = [
        Message(
            role=cast(Literal["system", "user", "assistant"], m["role"]),
            content=m["content"],
        )
        for m in messages
    ]
    result = await llm_provider.complete(
        llm_messages,
        response_model=None,
        purpose="synthesis",
        agent="orchestrator",
        max_tokens=1024,
    )
    if token_ledger is not None and correlation_id is not None:
        token_ledger.record(correlation_id, "synthesis", result.usage)
    _ = settings  # reserved for future structured synthesis/timeboxing
    return result.text

