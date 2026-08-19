"""QA eval harness: executed plan, citation precision, groundedness, retrieval, entity recall."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")

from core.eval.budget_compare import run_budget_comparison
from core.eval.harness import (
    load_routing_cases,
    parse_eval_provider,
    routing_qa_cases,
    run_qa_eval,
)
from core.eval.models import EvalScorecard
from core.eval.report import (
    format_budget_comparison,
    format_scorecard_text,
    write_readme_section,
    write_summary_section,
)
from core.settings import AnalysisSettings


def _routing_rows(report: EvalScorecard) -> EvalScorecard:
    wanted = [turn.query for turn in load_routing_cases()]
    by_query = {item.query: item for item in report.turns}
    turns = [by_query[query] for query in wanted if query in by_query]
    return EvalScorecard(
        turns=turns,
        tiers=report.tiers,
        metrics=report.metrics,
        provider=report.provider,
    )


def main() -> None:
    """Run the QA eval suite and exit non-zero when a hard assertion fails."""
    provider = parse_eval_provider()
    sample_only = os.environ.get("EVAL_SAMPLE_ONLY") == "1"
    # `make eval-live` scores every labelled case. SAMPLE_ONLY is a debug slice
    # of the nine routing golden queries; it is not the published scorecard.
    cases = routing_qa_cases() if sample_only else None
    try:
        report = asyncio.run(run_qa_eval(cases, provider=provider))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"qa eval failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(format_scorecard_text(report))
    repo_root = Path(AnalysisSettings.from_env().repo_root)
    comparison = asyncio.run(
        run_budget_comparison(
            cases,
            enabled=report if provider == "offline" else None,
            repo_root=repo_root if repo_root.exists() else None,
        )
    )
    print(format_budget_comparison(comparison))
    if not comparison.no_quality_drop:
        print("budget comparison: citation-precision or groundedness dropped", file=sys.stderr)
        raise SystemExit(1)
    if os.environ.get("EVAL_WRITE_README") == "1":
        companion = None
        if provider == "live":
            try:
                companion = asyncio.run(run_qa_eval(cases, provider="offline"))
                print(format_scorecard_text(companion))
            except Exception as exc:
                print(f"offline companion failed: {exc}", file=sys.stderr)
        write_readme_section(
            report,
            _routing_rows(report),
            budget_comparison=comparison,
            companion=companion,
        )
        write_summary_section(report)
        print("Wrote eval scorecard into docs/evaluation.md and the README summary")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
