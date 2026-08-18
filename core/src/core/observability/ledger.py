"""In-process token ledger keyed by correlation_id."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from core.llm.provider import LLMPurpose, TokenUsage


@dataclass
class _LedgerEntry:
    prompt: int = 0
    completion: int = 0
    total: int = 0
    llm_calls: int = 0
    by_purpose: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
    )


class TokenLedger:
    """Accumulate token usage during one `handle_query` request."""

    def __init__(self) -> None:
        self._entries: dict[str, _LedgerEntry] = {}

    def open(self, correlation_id: str) -> None:
        self._entries[correlation_id] = _LedgerEntry()

    def record(self, correlation_id: str, purpose: LLMPurpose, usage: TokenUsage) -> None:
        entry = self._entries.setdefault(correlation_id, _LedgerEntry())
        entry.prompt += usage.prompt_tokens
        entry.completion += usage.completion_tokens
        entry.total += usage.total_tokens
        entry.llm_calls += 1
        purpose_bucket = entry.by_purpose[purpose]
        purpose_bucket["prompt"] += usage.prompt_tokens
        purpose_bucket["completion"] += usage.completion_tokens
        purpose_bucket["total"] += usage.total_tokens

    def close(self, correlation_id: str) -> dict[str, object]:
        entry = self._entries.pop(correlation_id, _LedgerEntry())
        return {
            "total": entry.total,
            "prompt": entry.prompt,
            "completion": entry.completion,
            "llm_calls": entry.llm_calls,
            "by_purpose": dict(entry.by_purpose),
        }
