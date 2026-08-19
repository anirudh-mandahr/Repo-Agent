"""Render eval scorecards as markdown and patch the docs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from core.eval.models import (
    LIVE_PROVIDER_LABEL,
    OFFLINE_PROVIDER_LABEL,
    THRESHOLD_GROUNDEDNESS,
    TIERS,
    BudgetComparison,
    EvalScorecard,
    MetricScore,
    TierScorecard,
    TurnScore,
    provider_label,
)
from core.eval.scoring import extract_citations

BEGIN_MARKER = "<!-- BEGIN_EVAL_REPORT -->"
END_MARKER = "<!-- END_EVAL_REPORT -->"
SUMMARY_BEGIN_MARKER = "<!-- BEGIN_EVAL_SUMMARY -->"
SUMMARY_END_MARKER = "<!-- END_EVAL_SUMMARY -->"
_ROOT = Path(__file__).resolve().parents[4]
# Full report lives in docs/; README carries only the hoisted summary.
DEFAULT_README = _ROOT / "docs" / "evaluation.md"
DEFAULT_SUMMARY_DOC = _ROOT / "README.md"
LAYER_FAIL_BLURB = (
    "Retrieval FAIL means the executed agents missed the label or the specialist "
    "payloads did not contain the labelled files (entity recall can still be 1.00 "
    "when a test fixture mentions the same symbol). Synthesis FAIL means the LLM "
    "timed out, errored, or returned an evidence-only dump; grounding FAIL means "
    "citation precision or token-level groundedness missed the calibrated gate "
    "even when retrieval and synthesis succeeded."
)


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0%}"


def _quality_cell(value: float | None, excluded: int) -> str:
    """Format a mean that may have excluded turns.

    Args:
        value: Aggregate score, or ``None`` when every turn was excluded.
        excluded: Count of turns omitted from the mean.

    Returns:
        Percentage plus an excluded-count suffix when needed.
    """
    if value is None:
        return f"— ({excluded} excl.)" if excluded else "—"
    if excluded:
        return f"{_pct(value)} ({excluded} excl.)"
    return _pct(value)


def _usd(value: float) -> str:
    return f"${value:.4f}"


def _trim_answer(answer: str, limit: int = 280) -> str:
    collapsed = re.sub(r"\s+", " ", answer).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _agents(score: TurnScore) -> str:
    return ", ".join(f"`{agent}`" for agent in score.executed_agents) or "(none)"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_tier_scorecard(tiers: Sequence[TierScorecard]) -> str:
    """Render the per-tier pass-rate table.

    Args:
        tiers: Aggregated tier rows.

    Returns:
        Markdown table.
    """
    lines = [
        "| Tier | n | Pass rate | Executed agents | Retrieval | Citation precision |"
        " Groundedness | Entity recall | Degraded fails |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_name = {item.tier: item for item in tiers}
    for name in TIERS:
        item = by_name.get(name)
        if item is None:
            continue
        refusal = ""
        if item.refusal is not None:
            refusal = f" (refusal {_pct(item.refusal)})"
        lines.append(
            f"| {item.tier} | {item.n} | {_pct(item.pass_rate)}{refusal} | "
            f"{_pct(item.executed_agents)} | "
            f"{_pct(item.retrieval_correctness)} | "
            f"{_quality_cell(item.citation_precision, item.citation_excluded)} | "
            f"{_quality_cell(item.groundedness, item.groundedness_excluded)} | "
            f"{_pct(item.entity_recall)} | {item.degraded_fails} |"
        )
    return "\n".join(lines)


def format_metrics(metrics: Sequence[MetricScore]) -> str:
    """Render gated metrics as a table.

    Args:
        metrics: Named metric scores.

    Returns:
        Markdown table.
    """
    if not metrics:
        return ""
    lines = [
        "| Metric | Score | n | Status | Detail |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in metrics:
        if item.n == 0 and item.name in {"citation_precision", "groundedness"}:
            status = "n/a"
            score = "n/a"
        else:
            status = "PASS" if item.passed else "FAIL"
            score = f"{item.value:.2f}"
        lines.append(f"| `{item.name}` | {score} | {item.n} | {status} | {item.detail} |")
    return "\n".join(lines)


def format_query_table(turns: Sequence[TurnScore]) -> str:
    """Render per-query executed-plan stats with layer verdicts.

    Args:
        turns: Scored turns.

    Returns:
        Markdown table.
    """
    lines = [
        "| Case | Query | Expected agents | Agents invoked | Mode | Latency ms | "
        "Tokens | Cost | Retrieval | Synthesis | Grounding |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in turns:
        expected = ", ".join(f"`{agent}`" for agent in item.expected_agents)
        executed = ", ".join(f"`{agent}`" for agent in item.executed_agents) or "(none)"
        query = item.query.replace("|", "\\|")
        lines.append(
            f"| `{item.case_id}` | {query} | {expected} | {executed} | `{item.routing_mode}` | "
            f"{item.latency_ms} | {item.tokens_total} | {_usd(item.cost_usd)} | "
            f"{_verdict(item.retrieval_verdict)} | {_verdict(item.synthesis_verdict)} | "
            f"{_verdict(item.grounding_verdict)} |"
        )
    return "\n".join(lines)


def format_transcripts(turns: Sequence[TurnScore]) -> str:
    """Render per-query transcripts from eval output.

    Args:
        turns: Scored turns (typically the routing golden set).

    Returns:
        Markdown sections.
    """
    blocks: list[str] = []
    for index, item in enumerate(turns, start=1):
        citations = extract_citations(item.answer)
        cite_md = ", ".join(f"`{path}:{line}`" for path, line in citations) or "(none)"
        blocks.append(
            "\n".join(
                [
                    f"### {index}. {item.query}",
                    "",
                    f"- **Result:** retrieval {_verdict(item.retrieval_verdict)} · "
                    f"synthesis {_verdict(item.synthesis_verdict)} · "
                    f"grounding {_verdict(item.grounding_verdict)}",
                    f"- **Agents actually invoked:** {_agents(item)}",
                    f"- **Expected agents:** "
                    f"{', '.join(f'`{agent}`' for agent in item.expected_agents)}",
                    f"- **Routing mode:** `{item.routing_mode}`",
                    f"- **Latency:** {item.latency_ms} ms",
                    f"- **Tokens:** {item.tokens_total} "
                    f"(prompt {item.tokens_prompt}, completion {item.tokens_completion}, "
                    f"llm_calls {item.llm_calls})",
                    f"- **Cost:** {_usd(item.cost_usd)}",
                    f"- **Tools:** "
                    f"{', '.join(f'`{tool}`' for tool in item.tools_invoked) or '(none)'}",
                    f"- **Citations:** {cite_md}",
                    f"- **Missing files:** "
                    f"{', '.join(f'`{path}`' for path in item.missing_files) or '(none)'}",
                    f"- **Answer:** {_trim_answer(item.answer, limit=2500)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_scorecard_text(scorecard: EvalScorecard) -> str:
    """Plain-text scorecard for `make eval` / CI logs.

    Args:
        scorecard: Aggregated eval results.

    Returns:
        Human-readable scorecard.
    """
    lines = [
        "Eval scorecard (per-tier pass rates, not a single PASS/FAIL):",
        provider_label(
            scorecard.provider if scorecard.provider in {"offline", "live"} else "offline"
        ),
        "",
    ]
    if scorecard.provider != "live":
        lines.append(
            "Offline synthesis is a stub. Citation precision and groundedness exclude "
            "turns with no citations / no checkable claims; those turns are not scored as 1.00."
        )
        lines.append("")
    lines.append(
        f"{'tier':<12} {'n':>4} {'pass':>8} {'agents':>8} {'retr':>8} {'cite':>8} "
        f"{'ground':>8} {'entity':>8} {'degraded':>8}"
    )
    for tier in scorecard.tiers:
        lines.append(
            f"{tier.tier:<12} {tier.n:>4} {_pct(tier.pass_rate):>8} "
            f"{_pct(tier.executed_agents):>8} "
            f"{_pct(tier.retrieval_correctness):>8} "
            f"{_quality_cell(tier.citation_precision, tier.citation_excluded):>8} "
            f"{_quality_cell(tier.groundedness, tier.groundedness_excluded):>8} "
            f"{_pct(tier.entity_recall):>8} "
            f"{tier.degraded_fails:>8}"
        )
    lines.append("")
    for metric in scorecard.metrics:
        if metric.n == 0 and metric.name in {"citation_precision", "groundedness"}:
            status = "n/a"
            score = "n/a"
        else:
            status = "PASS" if metric.passed else "FAIL"
            score = f"{metric.value:.2f}"
        lines.append(f"- {metric.name}: {status} {score} ({metric.detail})")
    hard = "PASS" if scorecard.hard_assertions_passed else "FAIL"
    lines.append(f"- hard_assertions: {hard}")
    for extra in scorecard.lines:
        lines.append(extra)
    failures = [row for row in scorecard.turns if not row.passed]
    if failures:
        lines.append("")
        lines.append("Failed turns:")
        for row in failures[:20]:
            reasons: list[str] = []
            layers = (
                f"retr={_verdict(row.retrieval_verdict)} "
                f"synth={_verdict(row.synthesis_verdict)} "
                f"ground={_verdict(row.grounding_verdict)}"
            )
            reasons.append(layers)
            if not row.agents_passed:
                reasons.append(
                    f"agents expected={row.expected_agents} executed={row.executed_agents}"
                )
            if row.tier != "trap" and (row.degraded or row.evidence_only):
                reasons.append("degraded/evidence_only")
            if not row.retrieval_passed and row.missing_files:
                reasons.append("missing_files=" + ",".join(row.missing_files))
            if not row.citation_passed:
                reasons.append("citations=" + (",".join(row.invalid_citations) or "invalid"))
            if not row.grounded_passed:
                reasons.append("ungrounded")
            if not row.entity_passed:
                reasons.append("missing=" + ",".join(row.missing_entities))
            if row.refusal_ok is False:
                reasons.append("refusal")
            lines.append(f"  FAIL {row.case_id}: {row.query!r} ({'; '.join(reasons)})")
        if len(failures) > 20:
            lines.append(f"  ... +{len(failures) - 20} more")
    return "\n".join(lines)


def format_budget_comparison(comparison: BudgetComparison) -> str:
    """Render request-budget on vs off quality plus truncation-order choice.

    Args:
        comparison: Paired QA scorecards and truncation-order scores.

    Returns:
        Markdown section.
    """
    lines = [
        "### Request budgets (quality comparison)",
        "",
        "Outer request budget (`ORCH_REQUEST_DEADLINE_S`, `ORCH_REQUEST_TOKEN_BUDGET`, "
        "`ORCH_REQUEST_COST_USD_MAX`) is distinct from the per-prompt synthesis cap. "
        "Latency percentiles use `core.eval.model_bakeoff.percentile`. This comparison "
        "is run with the offline stub provider (routing/retrieval); it is not a live "
        "LLM quality score.",
        "",
        f"- Budgets enabled vs disabled: citation-precision and groundedness "
        f"{'did not drop' if comparison.no_quality_drop else 'DROPPED'} per tier.",
        f"- Enabled-run latency p50/p95: {comparison.latency_p50_ms:.0f} / "
        f"{comparison.latency_p95_ms:.0f} ms.",
        f"- Truncation order under budget pressure (measured against the indexed "
        f"tree via `repo_root`/`graph_client`, not assumed): "
        f"`{comparison.chosen_truncation_order}`. Citation precision is `—` when "
        f"no backend was available to resolve `file:line` coordinates.",
        "",
        "| Setting | simple cite | simple ground | medium cite | medium ground | "
        "complex cite | complex ground |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, card in (("enabled", comparison.enabled), ("disabled", comparison.disabled)):
        by_tier = {item.tier: item for item in card.tiers}
        cells = [label]
        for name in ("simple", "medium", "complex"):
            row = by_tier.get(name)
            cite = _pct(row.citation_precision) if row else "—"
            ground = _pct(row.groundedness) if row else "—"
            cells.extend([cite, ground])
        lines.append("| " + " | ".join(cells) + " |")
    if comparison.truncation_orders:
        lines.extend(
            [
                "",
                "| Truncation order | Citation precision | Groundedness |",
                "| --- | ---: | ---: |",
            ]
        )
        for item in comparison.truncation_orders:
            marker = " (chosen)" if item.order == comparison.chosen_truncation_order else ""
            lines.append(
                f"| `{item.order}`{marker} | {_pct(item.citation_precision)} | "
                f"{_pct(item.groundedness)} |"
            )
    return "\n".join(lines)


def _provider_heading(scorecard: EvalScorecard) -> str:
    kind = scorecard.provider if scorecard.provider in {"offline", "live"} else "offline"
    return f"**{provider_label(kind)}**"


def render_readme_section(
    qa: EvalScorecard,
    routing: EvalScorecard,
    *,
    generated_on: date | None = None,
    budget_comparison: BudgetComparison | None = None,
    companion: EvalScorecard | None = None,
) -> str:
    """Markdown block that replaces the hand-captured README eval section.

    Args:
        qa: Primary QA scorecard (live when measuring answer quality).
        routing: Executed-plan scores for the nine golden queries.
        generated_on: Report date.
        budget_comparison: Optional budgets-on vs budgets-off scorecards.
        companion: Optional scorecard from the other provider so offline and
            live results sit next to each other and cannot be confused.

    Returns:
        Markdown including markers.
    """
    day = (generated_on or date.today()).isoformat()
    live_card = (
        qa
        if qa.provider == "live"
        else companion
        if companion is not None and companion.provider == "live"
        else None
    )
    offline_card = (
        qa
        if qa.provider != "live"
        else companion
        if companion is not None and companion.provider != "live"
        else None
    )
    command = "`make eval-live`" if live_card is not None else "`make eval`"
    parts = [
        BEGIN_MARKER,
        f"Generated by {command} on {day} from `handle_query` metadata "
        f"(`tools_invoked`, tokens, latency), not from the router’s intent.",
        "",
    ]
    if live_card is not None:
        live_routing = routing if routing.provider == "live" or qa.provider == "live" else live_card
        parts.extend(
            [
                "### Live quality measurement",
                "",
                _provider_heading(live_card),
                "",
                "LLM-backed synthesis over every labelled case in `evals/qa.jsonl` "
                "(simple, medium, complex, trap, and multi-turn). Citation precision "
                "and groundedness exclude turns with no citations / no checkable claims "
                "rather than scoring empty output as 1.00. Groundedness is a token-level "
                f"fraction; the pass gate is {THRESHOLD_GROUNDEDNESS:.2f}, one decimal "
                "below the 2026-08-19 live claim-level mean of 0.74.",
                "",
                format_tier_scorecard(live_card.tiers),
                "",
                format_metrics(live_card.metrics),
                "",
            ]
        )
        parts.extend(
            [
                "### Per-turn layer verdicts (live)",
                "",
                LAYER_FAIL_BLURB,
                "",
                "Agents are taken from `metadata.tools_invoked` after the plan ran.",
                "",
                format_query_table(live_card.turns),
                "",
                "### Sample query transcripts",
                "",
                f"{LIVE_PROVIDER_LABEL}. Nine routing golden-set queries shown as samples; "
                "the scorecard above scores every case in `evals/qa.jsonl`. Each row is one "
                "`handle_query` turn with real synthesis and `file:line` citations resolved "
                "against the graph and indexed tree.",
                "",
                format_transcripts(live_routing.turns),
                "",
            ]
        )
    else:
        parts.extend(
            [
                "### Per-tier scorecard",
                "",
                _provider_heading(qa),
                "",
                "Pass rates are reported per tier. Hard assertions still fail `make eval` when "
                "executed agents diverge from `expected_agents` or when a non-trap answer is "
                "evidence-only / degraded. This offline run is routing/retrieval only — "
                "synthesis is a stub and is not a quality measurement.",
                "",
                format_tier_scorecard(qa.tiers),
                "",
                format_metrics(qa.metrics),
                "",
            ]
        )
        if budget_comparison is not None:
            parts.extend([format_budget_comparison(budget_comparison), ""])
        parts.extend(
            [
                "### Per-turn layer verdicts",
                "",
                LAYER_FAIL_BLURB,
                "",
                "Agents are taken from `metadata.tools_invoked` after the plan ran. "
                "This table is the nine routing golden-set queries; the scorecard above "
                "includes every labelled case.",
                "",
                format_query_table(routing.turns),
                "",
                "### Sample query transcripts",
                "",
                f"{OFFLINE_PROVIDER_LABEL}. Indexed graph when `NEO4J_URI` is set.",
                "",
                format_transcripts(routing.turns),
                "",
            ]
        )
    if offline_card is not None and live_card is not None:
        parts.extend(
            [
                "### Offline routing/retrieval (not a quality measurement)",
                "",
                _provider_heading(offline_card),
                "",
                "Same queries through `OfflineProvider` (all labelled cases in "
                "`evals/qa.jsonl`). Citation precision and "
                "groundedness exclude turns with no citations / no checkable claims; "
                "a stub answer is never reported as 1.00 quality.",
                "",
                format_tier_scorecard(offline_card.tiers),
                "",
                format_metrics(offline_card.metrics),
                "",
            ]
        )
        if budget_comparison is not None:
            parts.extend([format_budget_comparison(budget_comparison), ""])
    parts.append(END_MARKER)
    return "\n".join(parts).rstrip() + "\n"


def write_readme_section(
    qa: EvalScorecard,
    routing: EvalScorecard,
    *,
    readme_path: Path | None = None,
    generated_on: date | None = None,
    budget_comparison: BudgetComparison | None = None,
    companion: EvalScorecard | None = None,
) -> Path:
    """Replace the marked eval section in the evaluation doc.

    Args:
        qa: Full QA scorecard.
        routing: Golden-set executed-plan scores.
        readme_path: Document to patch (defaults to ``docs/evaluation.md``).
        generated_on: Report date.
        budget_comparison: Optional budgets-on vs budgets-off scorecards.
        companion: Optional scorecard from the other provider.

    Returns:
        Path written.

    Raises:
        ValueError: When the report markers are missing.
    """
    path = readme_path or DEFAULT_README
    text = path.read_text(encoding="utf-8")
    section = render_readme_section(
        qa,
        routing,
        generated_on=generated_on,
        budget_comparison=budget_comparison,
        companion=companion,
    )
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"{path} is missing {BEGIN_MARKER} / {END_MARKER} markers")
    path.write_text(pattern.sub(section.rstrip(), text), encoding="utf-8")
    return path


def format_summary_tiers(tiers: Sequence[TierScorecard]) -> str:
    """Render the compact per-tier table hoisted into the README.

    Args:
        tiers: Aggregated tier rows.

    Returns:
        Markdown table without the executed-agent and degraded columns.
    """
    lines = [
        "| Tier | n | Pass rate | Citation precision | Groundedness | Entity recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_name = {item.tier: item for item in tiers}
    for name in TIERS:
        item = by_name.get(name)
        if item is None:
            continue
        refusal = ""
        if item.refusal is not None:
            refusal = f" (refusal {_pct(item.refusal)})"
        lines.append(
            f"| {item.tier} | {item.n} | {_pct(item.pass_rate)}{refusal} | "
            f"{_quality_cell(item.citation_precision, item.citation_excluded)} | "
            f"{_quality_cell(item.groundedness, item.groundedness_excluded)} | "
            f"{_pct(item.entity_recall)} |"
        )
    return "\n".join(lines)


def format_summary_metrics(metrics: Sequence[MetricScore]) -> str:
    """Render gated metrics without the detail column, for the README summary.

    Args:
        metrics: Named metric scores.

    Returns:
        Markdown table, or an empty string when there are no metrics.
    """
    if not metrics:
        return ""
    lines = [
        "| Gated metric | Score | n | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in metrics:
        if item.n == 0 and item.name in {"citation_precision", "groundedness"}:
            status = "n/a"
            score = "n/a"
        else:
            status = "PASS" if item.passed else "FAIL"
            score = f"{item.value:.2f}"
        lines.append(f"| `{item.name}` | {score} | {item.n} | {status} |")
    return "\n".join(lines)


def render_summary_section(
    scorecard: EvalScorecard,
    *,
    generated_on: date | None = None,
    report_link: str = "docs/evaluation.md",
) -> str:
    """Markdown summary block for the README, linking to the full report.

    Args:
        scorecard: Scorecard to summarize (live when one is available).
        generated_on: Report date.
        report_link: Relative path to the full evaluation document.

    Returns:
        Markdown including the summary markers.
    """
    day = (generated_on or date.today()).isoformat()
    kind = "Live LLM run" if scorecard.provider == "live" else "Offline stub run"
    turns = sum(item.n for item in scorecard.tiers)
    parts = [
        SUMMARY_BEGIN_MARKER,
        f"{kind} over all {turns} labelled turns in `evals/qa.jsonl` ({day}). "
        f"Full scorecard, per-turn verdicts, and the model bake-off: "
        f"**[{report_link}]({report_link})**.",
        "",
        format_summary_tiers(scorecard.tiers),
        "",
        format_summary_metrics(scorecard.metrics),
        SUMMARY_END_MARKER,
    ]
    return "\n".join(parts).rstrip() + "\n"


def write_summary_section(
    scorecard: EvalScorecard,
    *,
    summary_path: Path | None = None,
    generated_on: date | None = None,
) -> Path:
    """Replace the marked summary block in the README.

    Args:
        scorecard: Scorecard to summarize.
        summary_path: Document to patch (defaults to the repo ``README.md``).
        generated_on: Report date.

    Returns:
        Path written.

    Raises:
        ValueError: When the summary markers are missing.
    """
    path = summary_path or DEFAULT_SUMMARY_DOC
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(SUMMARY_BEGIN_MARKER) + r".*?" + re.escape(SUMMARY_END_MARKER),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(
            f"{path} is missing {SUMMARY_BEGIN_MARKER} / {SUMMARY_END_MARKER} markers"
        )
    section = render_summary_section(scorecard, generated_on=generated_on)
    path.write_text(pattern.sub(section.rstrip(), text), encoding="utf-8")
    return path
