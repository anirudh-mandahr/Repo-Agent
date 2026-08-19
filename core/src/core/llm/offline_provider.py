"""Deterministic LLM stand-in when ``OPENROUTER_API_KEY`` is unset (local smoke)."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from pydantic import BaseModel

from core.analysis.models import (
    ClassAnalysis,
    FunctionAnalysis,
    ImplementationComparison,
    ImplementationExplanation,
    PatternAnalysis,
    PatternInstance,
)
from core.llm.provider import (
    LLMPurpose,
    LLMResult,
    Message,
    TokenUsage,
    as_text,
    estimate_usage,
    parse_structured,
)
from core.logging import get_logger

log = get_logger(__name__)

_OFFLINE_NOTE = (
    "Offline analysis stub (set OPENROUTER_API_KEY for full LLM-backed output)."
)


class OfflineProvider:
    """Return minimal structured payloads without calling an external API."""

    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResult:
        """Complete.
        
        Args:
            messages: list[Message].
            response_model: type[BaseModel] | None.
            purpose: LLMPurpose.
            agent: str.
            max_tokens: int.
            temperature: Unused; present to match ``LLMProvider``.

        Returns:
            LLMResult.
        """
        _ = max_tokens
        _ = agent, purpose, temperature
        prompt = _last_user_content(messages)
        payload_text = (
            _OFFLINE_NOTE
            if response_model is None
            else as_text(_offline_payload(prompt, response_model))
        )
        usage = estimate_usage(messages, payload_text).model_copy(update={"model": "offline"})
        if response_model is None:
            return LLMResult(text=_OFFLINE_NOTE, usage=usage)
        payload = _offline_payload(prompt, response_model)
        parsed = parse_structured(payload, response_model, agent=agent)
        log.info(
            "llm.offline.complete",
            response_model=response_model.__name__,
        )
        return LLMResult(
            text=as_text(payload),
            parsed=parsed,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated=True,
                model="offline",
                cached_prompt_tokens=0,
            ),
        )

    async def stream(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[tuple[str, TokenUsage | None]]:
        """Yield the offline completion in small chunks.

        Args:
            messages: Chat messages.
            response_model: Optional structured schema.
            purpose: Ledger purpose.
            agent: Calling agent.
            max_tokens: Unused; matches ``LLMProvider``.
            temperature: Unused; matches ``LLMProvider``.

        Yields:
            ``(delta, usage_or_none)`` pairs.
        """
        result = await self.complete(
            messages,
            response_model,
            purpose=purpose,
            agent=agent,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = result.text
        if not text:
            yield "", result.usage
            return
        step = max(1, min(16, max(len(text) // 4, 1)))
        for index in range(0, len(text), step):
            chunk = text[index : index + step]
            is_last = index + step >= len(text)
            yield chunk, (result.usage if is_last else None)


def _last_user_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content if messages else ""


def _extract_field(text: str, label: str) -> str:
    prefix = f"{label}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_section_names(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    items: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == heading:
            in_section = True
            continue
        if in_section and line.strip().startswith("==="):
            break
        if in_section and line.startswith("- "):
            items.append(line[2:].strip())
        elif in_section and line.strip() and not line.startswith("- ") and ":" in line:
            break
    return items


def _offline_payload(prompt: str, response_model: type[BaseModel]) -> dict[str, object]:
    qualified_name = _extract_field(prompt, "Qualified name") or "unknown"
    fields = set(response_model.model_fields)
    if response_model is FunctionAnalysis:
        return FunctionAnalysis(
            qualified_name=qualified_name,
            summary=f"Offline summary for {qualified_name}.",
            purpose=_OFFLINE_NOTE,
            parameters=_extract_section_names(prompt, "Parameters:"),
            dependents=_extract_section_names(prompt, "Dependents:"),
            decorators=_extract_section_names(prompt, "Decorators:"),
            module=_extract_field(prompt, "Module") or None,
            class_name=_extract_field(prompt, "Class") or None,
        ).model_dump()
    if response_model is ClassAnalysis:
        return ClassAnalysis(
            qualified_name=qualified_name,
            summary=f"Offline summary for {qualified_name}.",
            purpose=_OFFLINE_NOTE,
            methods=_extract_section_names(prompt, "Methods:"),
            bases=_extract_section_names(prompt, "Bases:"),
            decorators=_extract_section_names(prompt, "Decorators:"),
            module=_extract_field(prompt, "Module") or None,
        ).model_dump()
    if response_model is ImplementationExplanation:
        return ImplementationExplanation(
            qualified_name=qualified_name,
            explanation=(
                f"Offline explanation for {qualified_name}. "
                f"{_OFFLINE_NOTE}"
            ),
        ).model_dump()
    if response_model is PatternAnalysis:
        pattern = _extract_field(prompt, "Explain these instances of the '") or "pattern"
        match = re.search(r"instances of the '([^']+)' pattern", prompt)
        resolved_pattern = match.group(1) if match else pattern
        instances = [
            PatternInstance(qualified_name=name, explanation=_OFFLINE_NOTE)
            for name in _extract_section_names(prompt, "Instances:")
        ]
        return PatternAnalysis(
            pattern=resolved_pattern,
            instances=instances,
        ).model_dump()
    if response_model is ImplementationComparison:
        name_a, name_b = _extract_compare_names(prompt)
        return ImplementationComparison(
            name_a=name_a,
            name_b=name_b,
            summary=_OFFLINE_NOTE,
            similarities=["Offline comparison stub."],
            differences=["Set OPENROUTER_API_KEY for real analysis."],
        ).model_dump()
    if "summary" in fields and len(fields) == 1:
        return {"summary": _OFFLINE_NOTE}
    return {"detail": _OFFLINE_NOTE}


def _extract_compare_names(prompt: str) -> tuple[str, str]:
    names = re.findall(r"^=== (.+?) ===$", prompt, flags=re.MULTILINE)
    if len(names) >= 2:
        return names[0], names[1]
    return "unknown", "unknown"
