"""Cap synthesis prompt size by dropping lowest-value evidence fields first."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from .budget import (
    REFERENCE_SYNTHESIS_PROMPT_TOKENS as REFERENCE_SYNTHESIS_PROMPT_TOKENS,
)
from .budget import (
    SYNTHESIS_EXTRA_SECONDS_PER_TOKEN as SYNTHESIS_EXTRA_SECONDS_PER_TOKEN,
)
from .budget import scaled_synthesis_reserve_s as scaled_synthesis_reserve_s
from .models import PromptTruncation
from .prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_PROMPT

Path = tuple[str | int, ...]
TruncationOrder = Literal["snippets_then_lists", "lists_then_snippets"]
DEFAULT_TRUNCATION_ORDER: TruncationOrder = "snippets_then_lists"
NEIGHBOR_PROMPT_KEEP = 8
SYNTHESIS_TURN_TOKEN_CEILING = 8000
_GRAPH_NEIGHBOR_LIST_KEYS = frozenset({"dependents", "dependencies", "related"})

SNIPPET_BODY_KEYS = frozenset(
    {"text", "snippet", "snippet_a", "snippet_b", "code", "source", "body"}
)
_MIN_SNIPPET_CHARS = 80
_MIN_LIST_LEN = 2
_MAX_TRIM_ROUNDS = 64
_OMISSION_SUFFIX = "\n... (truncated)"


def estimate_tokens(text: str) -> int:
    """Cheap character-based token estimate (same 4-chars-per-token heuristic as routing).

    Args:
        text: Prompt or field text.

    Returns:
        Estimated token count, or ``0`` for empty input.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def dump_agent_payload(payload: Mapping[str, Any] | dict[str, Any]) -> str:
    """Serialize agent outputs the way the synthesis prompt does.

    Args:
        payload: JSON-serializable agent output mapping.

    Returns:
        Indented JSON.
    """
    return json.dumps(payload, default=str, indent=2)


def render_synthesis_user_prompt(
    query: str,
    session_context_block: str,
    agent_payload: Mapping[str, Any],
) -> str:
    """Build the synthesis user prompt for ``query`` and ``agent_payload``.

    Args:
        query: User question.
        session_context_block: Optional conversation-context section.
        agent_payload: Serialized specialist outputs.

    Returns:
        Formatted user prompt.
    """
    return SYNTHESIS_USER_PROMPT.format(
        query=query,
        session_context_block=session_context_block,
        agent_outputs=dump_agent_payload(agent_payload),
    )


def agent_payload_dict(agent_outputs: Mapping[Any, Any]) -> dict[str, Any]:
    """JSON-ish mapping of specialist outputs for prompt measurement.

    Args:
        agent_outputs: Per-agent results (models or mappings).

    Returns:
        A dict keyed by agent name.
    """
    payload: dict[str, Any] = {}
    for agent, output in agent_outputs.items():
        payload[str(agent)] = output.model_dump() if hasattr(output, "model_dump") else output
    return payload


def estimated_synthesis_prompt_tokens(
    query: str,
    agent_outputs: Mapping[Any, Any],
    *,
    session_context_block: str = "",
    token_budget: int,
    order: TruncationOrder = DEFAULT_TRUNCATION_ORDER,
) -> int:
    """Estimate synthesis prompt tokens after neighbor compact and the token cap.

    Args:
        query: User question.
        agent_outputs: Specialist outputs collected so far.
        session_context_block: Optional conversation-context section.
        token_budget: ``ORCH_SYNTHESIS_PROMPT_TOKEN_BUDGET``.
        order: Truncation order used by synthesis.

    Returns:
        Estimated tokens of the prompt that synthesis will actually send.
    """
    payload = compact_neighbor_payloads(agent_payload_dict(agent_outputs))
    budgeted, _truncation = apply_prompt_budget(
        payload,
        query=query,
        session_context_block=session_context_block,
        budget=token_budget,
        order=order,
    )
    return measure_synthesis_prompt(query, session_context_block, budgeted)[1]


def measure_synthesis_prompt(
    query: str,
    session_context_block: str,
    agent_payload: Mapping[str, Any],
) -> tuple[int, int]:
    """Return ``(prompt_chars, estimated_tokens)`` for the full synthesis prompt.

    Args:
        query: User question.
        session_context_block: Optional conversation-context section.
        agent_payload: Serialized specialist outputs.

    Returns:
        Character count and estimated tokens of system + user messages.
    """
    user = render_synthesis_user_prompt(query, session_context_block, agent_payload)
    prompt = f"{SYNTHESIS_SYSTEM_PROMPT}\n{user}"
    return len(prompt), estimate_tokens(prompt)


def compact_neighbor_payloads(
    payload: Mapping[str, Any],
    *,
    keep: int = NEIGHBOR_PROMPT_KEEP,
) -> dict[str, Any]:
    """Replace long neighbor lists with counts plus a short sample.

    Args:
        payload: Serialized specialist outputs.
        keep: Maximum neighbors to keep per list for the synthesis prompt.

    Returns:
        A deep copy whose graph neighbor lists are summarised.
    """
    compacted = copy.deepcopy(dict(payload))
    graph = compacted.get("graph_query")
    inner: Any = None
    if isinstance(graph, Mapping):
        output = graph.get("output")
        if isinstance(output, dict):
            inner = output
        elif "neighbors" in graph or any(key in graph for key in _GRAPH_NEIGHBOR_LIST_KEYS):
            inner = graph
    if not isinstance(inner, dict):
        return compacted
    for key in _GRAPH_NEIGHBOR_LIST_KEYS:
        rows = inner.get(key)
        if isinstance(rows, list):
            inner[key] = [_summarize_neighbor_result(item, keep=keep) for item in rows]
        elif isinstance(rows, Mapping):
            inner[key] = _summarize_neighbor_result(rows, keep=keep)
    return compacted


def _summarize_neighbor_result(item: Any, *, keep: int) -> Any:
    if not isinstance(item, Mapping):
        return item
    neighbors = item.get("neighbors")
    if not isinstance(neighbors, list) or len(neighbors) <= keep:
        return dict(item)
    total = int(item.get("total_count") or item.get("result_count") or len(neighbors))
    summarized = dict(item)
    summarized["neighbors"] = neighbors[:keep]
    summarized["result_count"] = len(neighbors[:keep])
    summarized["total_count"] = total
    summarized["truncated"] = True
    summarized["omitted"] = max(0, total - keep)
    summarized["summary"] = (
        f"{total} neighbors; showing {keep}, omitted {max(0, total - keep)}"
    )
    return summarized


def apply_prompt_budget(
    agent_payload: Mapping[str, Any],
    *,
    query: str,
    session_context_block: str,
    budget: int,
    order: TruncationOrder = DEFAULT_TRUNCATION_ORDER,
) -> tuple[dict[str, Any], PromptTruncation | None]:
    """Copy ``agent_payload`` and truncate until the synthesis prompt fits ``budget``.

    Truncation order is chosen from eval quality under budget pressure (see
    ``core.eval.budget_compare``), not assumed a priori.

    Args:
        agent_payload: Specialist outputs to include in the prompt.
        query: User question (counted in the prompt overhead).
        session_context_block: Optional conversation context.
        budget: Maximum estimated tokens for system + user prompt.
        order: Field drop order. Default is the eval-selected order.

    Returns:
        A possibly-truncated payload copy and a truncation record when anything
        was dropped, otherwise ``(copy, None)``.
    """
    payload = copy.deepcopy(dict(agent_payload))
    original_chars, original_tokens = measure_synthesis_prompt(
        query, session_context_block, payload
    )
    _ = original_chars
    if budget <= 0 or original_tokens <= budget:
        return payload, None

    dropped: list[dict[str, Any]] = []
    for _ in range(_MAX_TRIM_ROUNDS):
        _, tokens = measure_synthesis_prompt(query, session_context_block, payload)
        if tokens <= budget:
            break
        excess_chars = max(1, (tokens - budget) * 4)
        if order == "lists_then_snippets":
            if _trim_longest_list(payload, dropped):
                continue
            if _truncate_longest_snippet(payload, dropped, excess_chars):
                continue
        else:
            if _truncate_longest_snippet(payload, dropped, excess_chars):
                continue
            if _trim_longest_list(payload, dropped):
                continue
        break

    _, final_tokens = measure_synthesis_prompt(query, session_context_block, payload)
    if not dropped:
        return payload, None
    return payload, PromptTruncation(
        original_estimated_tokens=original_tokens,
        final_estimated_tokens=final_tokens,
        budget=budget,
        dropped=dropped,
    )


def format_path(path: Path) -> str:
    """Render a nested path tuple as a dotted / indexed string.

    Args:
        path: Sequence of mapping keys and list indices.

    Returns:
        Human-readable path such as ``code_analyst.output.snippets[0].text``.
    """
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif not rendered:
            rendered = str(part)
        else:
            rendered += f".{part}"
    return rendered


def _walk(obj: Any, prefix: Path = ()) -> Iterator[tuple[Path, Any]]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            path = prefix + (str(key),)
            yield path, value
            yield from _walk(value, path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            path = prefix + (index,)
            yield path, value
            yield from _walk(value, path)


def _set(root: dict[str, Any], path: Path, value: Any) -> None:
    current: Any = root
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _parent_key(path: Path) -> str | None:
    for part in reversed(path):
        if isinstance(part, str):
            return part
    return None


def _truncate_longest_snippet(
    payload: dict[str, Any],
    dropped: list[dict[str, Any]],
    excess_chars: int,
) -> bool:
    longest: tuple[Path, str] | None = None
    longest_len = _MIN_SNIPPET_CHARS
    for path, value in _walk(payload):
        key = _parent_key(path)
        if key not in SNIPPET_BODY_KEYS or not isinstance(value, str):
            continue
        if len(value) <= longest_len:
            continue
        longest = (path, value)
        longest_len = len(value)
    if longest is None:
        return False
    path, value = longest
    keep = max(0, len(value) - excess_chars)
    if keep >= len(value):
        keep = max(0, len(value) // 2)
    clipped = value[:keep] + _OMISSION_SUFFIX if keep else ""
    _set(payload, path, clipped)
    dropped.append(
        {
            "path": format_path(path),
            "kind": "snippet",
            "original_chars": len(value),
            "kept_chars": len(clipped),
        }
    )
    return True


def _trim_longest_list(payload: dict[str, Any], dropped: list[dict[str, Any]]) -> bool:
    longest: tuple[Path, list[Any]] | None = None
    longest_len = _MIN_LIST_LEN - 1
    for path, value in _walk(payload):
        if not isinstance(value, list) or len(value) <= longest_len:
            continue
        longest = (path, value)
        longest_len = len(value)
    if longest is None:
        return False
    path, items = longest
    if len(items) < _MIN_LIST_LEN:
        return False
    keep_len = max(1, len(items) // 2)
    _set(payload, path, items[:keep_len])
    dropped.append(
        {
            "path": format_path(path),
            "kind": "list",
            "original_len": len(items),
            "kept_len": keep_len,
        }
    )
    return True
