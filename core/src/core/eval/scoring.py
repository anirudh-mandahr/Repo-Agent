"""Score executed plans and answer quality for eval JSONL cases."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from core.eval.models import (
    THRESHOLD_CITATION_PRECISION,
    THRESHOLD_GROUNDEDNESS,
    THRESHOLD_RECALL,
    TurnScore,
    TurnSpec,
)
from core.observability.metrics import estimate_cost_usd
from core.orchestration.fallback import EVIDENCE_ONLY_HEADER
from core.orchestration.router import intent_from_rule_result, rule_based_route
from core.settings import OrchestratorSettings

CITATION_RE = re.compile(r"\b([\w./-]+\.py):(\d+)(?:-(\d+))?\b")
CITATION_PROSE_RE = re.compile(
    r"([\w./-]+\.py)(?:[`*)]*)?(?:\s*\(|,|\s)+lines?\s+(\d+)\s*[–—-]\s*(\d+)",
    re.IGNORECASE,
)
CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CAMEL_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9_]*\b")
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(_[a-z0-9]+)+\b")
QUALIFIED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
KNOWN_AGENTS = ("indexer", "graph_query", "code_analyst", "memory")
REFUSAL_NEEDLES = ("not in the indexed fastapi codebase", "could not find indexed fastapi")
_SKIP_CLAIM_PREFIXES = ("sources:", "###", "####", "- ", "* ")
_CLAIM_SKIP_TOKENS = frozenset({"the", "this", "that", "sources", "offline"})
_QUALIFIED_SKIP = frozenset({"e.g", "i.e", "vs"})
_PATH_KEYS = frozenset({"file_path", "path", "file"})
_PY_PATH_RE = re.compile(r"(?:^|[\s\"'`=(])([\w./-]+\.py)\b")


class GraphReader(Protocol):
    """Minimal read interface used to resolve citation coordinates."""

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        """Execute a read-only Cypher query."""
        ...


def agents_from_tools_invoked(tools_invoked: Sequence[str]) -> list[str]:
    """Return unique specialist names recorded in ``metadata.tools_invoked``.

    Args:
        tools_invoked: ``agent.tool`` strings from ``handle_query`` metadata.

    Returns:
        Deduplicated agent names in first-seen order.
    """
    agents: list[str] = []
    for tool in tools_invoked:
        raw = str(tool).strip()
        if not raw:
            continue
        agent = raw.split(".", 1)[0] if "." in raw else raw
        if agent in KNOWN_AGENTS and agent not in agents:
            agents.append(agent)
    return agents


def router_only_agents(
    turn: TurnSpec,
    settings: OrchestratorSettings | None = None,
) -> tuple[list[str], str]:
    """Reproduce the old router-intent check that ignored execution.

    Ambiguous rules results are treated as a free pass: the labelled
    ``expected_agents`` are returned without running a plan.

    Args:
        turn: Labelled eval turn.
        settings: Orchestrator settings for the token-length ambiguity guard.

    Returns:
        ``(agents, mode)`` as the old eval would have scored them.
    """
    route = rule_based_route(turn.query, settings=settings or OrchestratorSettings.from_env())
    if route.ambiguous:
        return list(turn.expected_agents), "llm"
    intent = intent_from_rule_result(turn.query, route)
    return list(intent.target_agents), "rules"


def routing_mode_ok(expected: str, actual: str) -> bool:
    """Return whether ``actual`` satisfies the labelled routing mode.

    Args:
        expected: Labelled mode (``rules`` or ``llm``).
        actual: ``metadata.routing_mode`` from ``handle_query``.

    Returns:
        True when rules queries stayed on rules, or when llm-labelled queries
        escalated (``llm`` or ``rules_fallback``).
    """
    if expected == "rules":
        return actual == "rules"
    return actual in {"llm", "rules_fallback"}


def executed_agents_ok(
    turn: TurnSpec,
    tools_invoked: Sequence[str],
    *,
    routing_mode: str | None = None,
) -> bool:
    """True when specialists that actually ran match ``expected_agents``.

    Args:
        turn: Labelled eval turn.
        tools_invoked: ``metadata.tools_invoked`` from ``handle_query``.
        routing_mode: Optional ``metadata.routing_mode``; when omitted, mode is
            not checked.

    Returns:
        True when the executed agent set matches the label.
    """
    actual = agents_from_tools_invoked(tools_invoked)
    if set(actual) != set(turn.expected_agents):
        return False
    if len(actual) < turn.min_agents:
        return False
    if routing_mode is not None and not routing_mode_ok(turn.expected_mode, routing_mode):
        return False
    return True


def extract_citations(answer: str) -> list[tuple[str, int]]:
    """Return unique ``(path, line)`` citations mentioned in ``answer``.

    Args:
        answer: Synthesized assistant text.

    Returns:
        Deduplicated file/line pairs, using the start line of a range.
    """
    seen: set[tuple[str, int]] = set()
    citations: list[tuple[str, int]] = []

    def _add(path: str, line: int) -> None:
        key = (path, line)
        if key in seen:
            return
        seen.add(key)
        citations.append(key)

    for match in CITATION_RE.finditer(answer):
        _add(match.group(1), int(match.group(2)))
    for match in CITATION_PROSE_RE.finditer(answer):
        _add(match.group(1), int(match.group(2)))
    return citations


def citation_exists_on_disk(path: str, line: int, repo_root: Path) -> bool:
    """True when ``path:line`` is a real line inside ``repo_root``.

    Args:
        path: Repo-relative file path cited in the answer.
        line: 1-based line number.
        repo_root: Indexed repository root.

    Returns:
        True when the file exists, is inside the root, and contains ``line``.
    """
    if line < 1:
        return False
    try:
        root = repo_root.resolve()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            return False
        with resolved.open(encoding="utf-8") as handle:
            for index, _row in enumerate(handle, start=1):
                if index >= line:
                    return True
        return False
    except OSError:
        return False


def citation_exists_in_graph(client: GraphReader, path: str, line: int) -> bool:
    """True when a graph node covers ``path`` at ``line``.

    Args:
        client: Read-only graph client.
        path: Repo-relative file path.
        line: 1-based line number.

    Returns:
        True when at least one node matches the coordinate.
    """
    rows = client.run_read(
        (
            "MATCH (n) "
            "WHERE (n.file_path = $path OR (n:File AND n.path = $path)) "
            "AND ("
            "  n.line_start IS NULL "
            "  OR (n.line_start <= $line AND (n.line_end IS NULL OR n.line_end >= $line))"
            ") "
            "RETURN count(n) AS count"
        ),
        {"path": path, "line": line},
    )
    count = int(rows[0]["count"]) if rows else 0
    return count > 0


def citation_exists(
    path: str,
    line: int,
    *,
    graph_client: GraphReader | None = None,
    repo_root: Path | None = None,
) -> bool:
    """True when a citation exists in the graph or as a real source line.

    Args:
        path: Repo-relative file path.
        line: 1-based line number.
        graph_client: Optional Neo4j reader.
        repo_root: Optional indexed repository root.

    Returns:
        True when either backend confirms the citation.
    """
    if graph_client is not None and citation_exists_in_graph(graph_client, path, line):
        return True
    if repo_root is not None and citation_exists_on_disk(path, line, repo_root):
        return True
    return False


def score_citation_precision(
    answer: str,
    *,
    graph_client: GraphReader | None = None,
    repo_root: Path | None = None,
) -> tuple[float | None, list[str]]:
    """Precision of ``file:line`` citations in ``answer``.

    Args:
        answer: Synthesized assistant text.
        graph_client: Optional Neo4j reader.
        repo_root: Optional indexed repository root.

    Returns:
        ``(precision, invalid_citation_labels)``. No citations returns
        ``None`` so the turn is excluded from the aggregate mean — absence
        is not perfect precision.
    """
    citations = extract_citations(answer)
    if not citations:
        return None, []
    invalid: list[str] = []
    hits = 0
    for path, line in citations:
        if citation_exists(path, line, graph_client=graph_client, repo_root=repo_root):
            hits += 1
        else:
            invalid.append(f"{path}:{line}")
    return hits / len(citations), invalid


def evidence_text(agent_outputs: Mapping[str, Any] | None) -> str:
    """Serialize specialist outputs into a searchable evidence blob.

    Args:
        agent_outputs: ``handle_query`` specialist payloads.

    Returns:
        JSON text used to ground claims.
    """
    if not agent_outputs:
        return ""
    return json.dumps(agent_outputs, default=str).lower()


def extract_claims(answer: str) -> list[str]:
    """Split an answer into sentence-sized claims.

    Args:
        answer: Synthesized assistant text.

    Returns:
        Stripped non-empty claim strings.
    """
    claims: list[str] = []
    for raw in CLAIM_SPLIT_RE.split(answer):
        text = raw.strip().strip("-*").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(lowered.startswith(prefix) for prefix in _SKIP_CLAIM_PREFIXES):
            continue
        claims.append(text)
    return claims


def _checkable_symbols(claim: str) -> list[str]:
    """Return CamelCase, snake_case, and dotted names that can be grounded."""
    symbols: list[str] = []
    for token in (*CAMEL_RE.findall(claim), *SNAKE_RE.findall(claim)):
        if token.lower() not in _CLAIM_SKIP_TOKENS:
            symbols.append(token)
    for token in QUALIFIED_RE.findall(claim):
        lowered = token.lower().rstrip(".")
        if lowered in _QUALIFIED_SKIP or lowered.endswith(".py"):
            continue
        symbols.append(token)
    return symbols


def is_checkable_claim(claim: str) -> bool:
    """True when a claim cites a path or a distinctive code symbol.

    Args:
        claim: One sentence from the answer.

    Returns:
        True when the claim can be checked against agent evidence. CamelCase,
        snake_case (``get_openapi``), and dotted qualified names all count;
        stopwords such as ``Offline`` do not.
    """
    if extract_citations(claim) or ".py" in claim:
        return True
    return bool(_checkable_symbols(claim))


def _citation_supported(path: str, line: int, evidence: str) -> bool:
    path_l = path.lower()
    return path_l in evidence or f"{path_l}:{line}" in evidence


def _symbol_supported(token: str, evidence: str) -> bool:
    lowered = token.lower()
    last = lowered.rsplit(".", 1)[-1]
    return lowered in evidence or last in evidence


def claim_token_hits(claim: str, evidence: str) -> tuple[int, int]:
    """Return ``(supported, total)`` distinctive tokens in ``claim``.

    Citations count as one unit (path *or* ``path:line``). Symbols are scored
    independently so one unmatched paraphrase token cannot zero the claim.

    Args:
        claim: One checkable sentence.
        evidence: Lowercased specialist-output blob.

    Returns:
        Supported-token count and the number of checkable units.
    """
    units: list[bool] = []
    if not evidence:
        total = len(extract_citations(claim)) + len(_checkable_symbols(claim))
        return 0, total
    for path, line in extract_citations(claim):
        units.append(_citation_supported(path, line, evidence))
    for token in _checkable_symbols(claim):
        units.append(_symbol_supported(token, evidence))
    return sum(1 for hit in units if hit), len(units)


def claim_supported(claim: str, evidence: str) -> bool:
    """True when cited paths and code symbols appear in evidence.

    Args:
        claim: One checkable sentence.
        evidence: Lowercased specialist-output blob.

    Returns:
        True when every distinctive token is supported.
    """
    hits, total = claim_token_hits(claim, evidence)
    return total > 0 and hits == total


def score_groundedness(
    answer: str,
    agent_outputs: Mapping[str, Any] | None,
) -> tuple[float | None, list[str]]:
    """Fraction of checkable tokens supported by ``agent_outputs``.

    Args:
        answer: Synthesized assistant text.
        agent_outputs: Specialist payloads from ``handle_query``.

    Returns:
        ``(score, ungrounded_claim_snippets)``. Score is token-level, not
        claim-level all-or-nothing. No checkable claims returns ``None`` so
        the turn is excluded from the aggregate mean — an empty or stub
        answer is not perfect groundedness.
    """
    evidence = evidence_text(agent_outputs)
    checkable = [claim for claim in extract_claims(answer) if is_checkable_claim(claim)]
    if not checkable:
        return None, []
    ungrounded: list[str] = []
    hit_total = 0
    unit_total = 0
    for claim in checkable:
        hits, total = claim_token_hits(claim, evidence)
        if total == 0:
            continue
        hit_total += hits
        unit_total += total
        if hits < total:
            ungrounded.append(claim[:160])
    if unit_total == 0:
        return None, []
    return hit_total / unit_total, ungrounded


def entity_aliases(name: str) -> set[str]:
    """Return lowercase aliases used for entity matching.

    Args:
        name: Labelled entity name or qualified name.

    Returns:
        Alias set including the final path component.
    """
    text = name.strip()
    if not text:
        return set()
    aliases = {text.lower()}
    if "." in text:
        aliases.add(text.rsplit(".", 1)[-1].lower())
    return aliases


def score_entity_recall(
    answer: str,
    expected_entities: Sequence[str],
    agent_outputs: Mapping[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Recall of labelled entities in the answer or retrieved evidence.

    Args:
        answer: Synthesized assistant text.
        expected_entities: Entities that should appear in the turn.
        agent_outputs: Specialist payloads from ``handle_query``.

    Returns:
        ``(recall, missing_entities)``. No expected entities scores ``1.0``.
    """
    if not expected_entities:
        return 1.0, []
    haystack = f"{answer.lower()}\n{evidence_text(agent_outputs)}"
    missing: list[str] = []
    hits = 0
    for entity in expected_entities:
        aliases = entity_aliases(entity)
        found = any(re.search(rf"\b{re.escape(alias)}\b", haystack) for alias in aliases if alias)
        if found:
            hits += 1
        else:
            missing.append(entity)
    return hits / len(expected_entities), missing


def _looks_like_repo_path(text: str) -> bool:
    cleaned = text.strip().replace("\\", "/")
    if not cleaned or "://" in cleaned:
        return False
    if cleaned.endswith(".py"):
        return True
    return "/" in cleaned and not cleaned.startswith("/")


def extract_retrieved_paths(agent_outputs: Mapping[str, Any] | None) -> list[str]:
    """Collect repo-relative paths from specialist payloads.

    Args:
        agent_outputs: ``handle_query`` specialist payloads.

    Returns:
        Deduplicated paths in first-seen order. Does not inspect the answer.
    """
    if not agent_outputs:
        return []
    seen: set[str] = set()
    paths: list[str] = []

    def _add(raw: str) -> None:
        text = raw.strip().replace("\\", "/")
        if not _looks_like_repo_path(text) or text in seen:
            return
        seen.add(text)
        paths.append(text)

    def _walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) in _PATH_KEYS and isinstance(value, str):
                    _add(value)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            for match in _PY_PATH_RE.finditer(node):
                _add(match.group(1))

    _walk(agent_outputs)
    return paths


def expected_file_retrieved(expected: str, retrieved: Sequence[str]) -> bool:
    """True when ``expected`` is an exact retrieved path or a directory prefix.

    Args:
        expected: Labelled file or directory from the eval case.
        retrieved: Paths extracted from specialist payloads.

    Returns:
        True when retrieval included the labelled path.
    """
    want = expected.strip().replace("\\", "/").rstrip("/")
    if not want:
        return True
    for path in retrieved:
        got = path.replace("\\", "/").rstrip("/")
        if got == want or got.startswith(want + "/"):
            return True
    return False


def score_retrieval_correctness(
    expected_files: Sequence[str],
    agent_outputs: Mapping[str, Any] | None,
) -> tuple[float, list[str], list[str]]:
    """Recall of labelled files in retrieved specialist payloads.

    Independent of whether synthesized claims match the evidence blob.
    Entity recall can still be 1.00 when a test fixture mentions the symbol.

    Args:
        expected_files: Files (or directory prefixes) that should be retrieved.
        agent_outputs: Specialist payloads from ``handle_query``.

    Returns:
        ``(recall, missing_files, retrieved_paths)``. No expected files
        scores ``1.0``.
    """
    retrieved = extract_retrieved_paths(agent_outputs)
    if not expected_files:
        return 1.0, [], retrieved
    missing = [path for path in expected_files if not expected_file_retrieved(path, retrieved)]
    hits = len(expected_files) - len(missing)
    return hits / len(expected_files), missing, retrieved


def is_refusal(answer: str) -> bool:
    """True when the answer refuses an out-of-repo question.

    Args:
        answer: Synthesized assistant text.

    Returns:
        True when a known refusal needle is present.
    """
    lowered = answer.lower()
    return any(needle in lowered for needle in REFUSAL_NEEDLES)


def is_evidence_only(answer: str, metadata: Mapping[str, Any]) -> bool:
    """True when synthesis fell back to an evidence-only dump.

    Args:
        answer: Synthesized assistant text.
        metadata: ``handle_query`` metadata.

    Returns:
        True when the evidence-only flag or header is present.
    """
    if bool(metadata.get("evidence_only")):
        return True
    return EVIDENCE_ONLY_HEADER in answer


def is_degraded(metadata: Mapping[str, Any]) -> bool:
    """True when ``handle_query`` marked the turn as degraded.

    Args:
        metadata: ``handle_query`` metadata.

    Returns:
        True when ``degraded`` is set.
    """
    return bool(metadata.get("degraded"))


def tokens_from_metadata(metadata: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Extract token totals from orchestrator metadata.

    Args:
        metadata: ``handle_query`` metadata.

    Returns:
        ``(total, prompt, completion, llm_calls)``.
    """
    raw = metadata.get("tokens")
    tokens = raw if isinstance(raw, Mapping) else {}
    return (
        int(tokens.get("total") or 0),
        int(tokens.get("prompt") or 0),
        int(tokens.get("completion") or 0),
        int(tokens.get("llm_calls") or 0),
    )


def cost_from_tokens(
    prompt: int,
    completion: int,
    *,
    cached_prompt: int = 0,
    model: str | None = None,
) -> float:
    """Estimate USD cost from prompt and completion tokens.

    Args:
        prompt: Prompt-side tokens.
        completion: Completion-side tokens.
        cached_prompt: Prompt tokens served from the provider cache.
        model: Optional model id for per-1M rates.

    Returns:
        Estimated USD.
    """
    return estimate_cost_usd(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_prompt_tokens=cached_prompt,
        model=model,
    )


def score_turn(
    case_id: str,
    tier: str,
    turn: TurnSpec,
    *,
    answer: str,
    metadata: Mapping[str, Any],
    agent_outputs: Mapping[str, Any] | None = None,
    graph_client: GraphReader | None = None,
    repo_root: Path | None = None,
    latency_ms: int = 0,
    quality: bool = True,
) -> TurnScore:
    """Score one executed ``handle_query`` turn.

    Args:
        case_id: Parent case id.
        tier: Eval tier.
        turn: Labelled turn.
        answer: Assistant text.
        metadata: ``handle_query`` metadata (must include ``tools_invoked``).
        agent_outputs: Specialist payloads used for groundedness.
        graph_client: Optional graph reader for citation checks.
        repo_root: Optional repo root for on-disk citation checks.
        latency_ms: Wall time for the turn.
        quality: When False, skip citation/groundedness/entity gates.

    Returns:
        Populated :class:`TurnScore`.
    """
    tools = [str(item) for item in list(metadata.get("tools_invoked") or [])]
    executed = agents_from_tools_invoked(tools)
    routing_mode = str(metadata.get("routing_mode") or "")
    agents_passed = executed_agents_ok(turn, tools, routing_mode=routing_mode)
    degraded = is_degraded(metadata)
    evidence_only = is_evidence_only(answer, metadata)
    total, prompt, completion, llm_calls = tokens_from_metadata(metadata)
    raw_tokens = metadata.get("tokens")
    token_map = raw_tokens if isinstance(raw_tokens, Mapping) else {}
    cached_prompt = int(token_map.get("cached_prompt") or 0)
    ledger_cost = token_map.get("cost_usd")
    if isinstance(ledger_cost, int | float):
        turn_cost = float(ledger_cost)
    else:
        turn_cost = cost_from_tokens(prompt, completion, cached_prompt=cached_prompt)

    citation_precision: float | None = None
    invalid_citations: list[str] = []
    groundedness: float | None = None
    ungrounded: list[str] = []
    entity_recall = 1.0
    missing_entities: list[str] = []
    retrieval_correctness = 1.0
    missing_files: list[str] = []
    retrieved_paths: list[str] = []
    if quality:
        citation_precision, invalid_citations = score_citation_precision(
            answer, graph_client=graph_client, repo_root=repo_root
        )
        groundedness, ungrounded = score_groundedness(answer, agent_outputs)
        entity_recall, missing_entities = score_entity_recall(
            answer, turn.expected_entities, agent_outputs
        )
        retrieval_correctness, missing_files, retrieved_paths = score_retrieval_correctness(
            turn.expected_files, agent_outputs
        )

    refusal: bool | None = None
    if turn.out_of_scope or tier == "trap":
        refusal = is_refusal(answer)

    citation_passed = citation_precision is None or (
        citation_precision >= THRESHOLD_CITATION_PRECISION
    )
    grounded_passed = (
        (not quality) or groundedness is None or (groundedness >= THRESHOLD_GROUNDEDNESS)
    )
    entity_passed = (not quality) or entity_recall >= THRESHOLD_RECALL
    retrieval_passed = (not quality) or retrieval_correctness >= THRESHOLD_RECALL
    if turn.out_of_scope or tier == "trap":
        grounded_passed = True
        entity_passed = True
        retrieval_passed = True
        citation_passed = citation_passed or not extract_citations(answer)

    return TurnScore(
        case_id=case_id,
        tier=tier,
        query=turn.query,
        expected_agents=list(turn.expected_agents),
        executed_agents=executed,
        routing_mode=routing_mode,
        agents_passed=agents_passed,
        citation_precision=citation_precision,
        citation_passed=citation_passed,
        groundedness=groundedness,
        grounded_passed=grounded_passed,
        entity_recall=entity_recall,
        entity_passed=entity_passed,
        degraded=degraded,
        evidence_only=evidence_only,
        refusal_ok=refusal,
        retrieval_correctness=retrieval_correctness,
        retrieval_passed=retrieval_passed,
        ungrounded_claims=ungrounded,
        invalid_citations=invalid_citations,
        missing_entities=missing_entities,
        missing_files=missing_files,
        retrieved_paths=retrieved_paths,
        answer=answer,
        latency_ms=latency_ms,
        tokens_total=total,
        tokens_prompt=prompt,
        tokens_completion=completion,
        llm_calls=llm_calls,
        cost_usd=turn_cost,
        tools_invoked=tools,
    )


def non_trap_degraded_violation(score: TurnScore) -> bool:
    """True when a non-trap turn returned an evidence-only or degraded answer.

    Args:
        score: Per-turn score.

    Returns:
        True when the hard assertion is violated.
    """
    return score.tier != "trap" and (score.degraded or score.evidence_only)
