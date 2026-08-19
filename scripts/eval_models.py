"""Purpose-aware model bake-off over evals/qa.jsonl."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")

from core.eval.harness import load_qa_cases
from core.eval.model_bakeoff import (
    DEFAULT_BAKEOFF_MODELS,
    DEFAULT_FIXTURES,
    DEFAULT_REPEATS,
    PurposeBakeoff,
    capture_synthesis_payloads,
    format_purpose_table,
    live_clients_factory,
    load_frozen_payloads,
    parse_models_arg,
    run_router_bakeoff,
    run_synthesis_bakeoff,
    write_model_bakeoff_section,
)
from core.settings import AnalysisSettings, LLMSettings


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bake off routing and synthesis models on evals/qa.jsonl. "
            "Reports mean ± spread across repeats; rows are not ranked."
        )
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_BAKEOFF_MODELS),
        help="Comma-separated OpenRouter model ids.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Independent temperature-0 passes (default 3).",
    )
    parser.add_argument(
        "--purpose",
        choices=("routing", "synthesis", "both"),
        default="both",
        help="Which bake-off to run.",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES,
        help="Frozen synthesis payload JSONL.",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Re-capture agent_outputs into --fixtures before synthesis replay.",
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Patch README.md model-bakeoff section.",
    )
    return parser.parse_args(argv)


async def _maybe_capture(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        return
    neo4j_uri = os.environ.get("NEO4J_URI")
    if not neo4j_uri:
        await capture_synthesis_payloads(path=path)
        print(f"captured stub payloads -> {path}", file=sys.stderr)
        return
    from core.graph.client import GraphClient

    repo_root = Path(AnalysisSettings.from_env().repo_root)
    client = GraphClient()
    try:
        client.verify_connectivity()
        await capture_synthesis_payloads(
            path=path,
            clients_factory=live_clients_factory(client, repo_root),
        )
        print(f"captured live graph payloads -> {path}", file=sys.stderr)
    finally:
        client.close()


async def _run(args: argparse.Namespace) -> list[PurposeBakeoff]:
    models = parse_models_arg(args.models)
    cases = load_qa_cases()
    reports: list[PurposeBakeoff] = []
    if args.purpose in {"routing", "both"}:
        reports.append(
            await run_router_bakeoff(models, cases, repeats=args.repeats)
        )
    if args.purpose in {"synthesis", "both"}:
        await _maybe_capture(args.fixtures, force=args.capture)
        if not args.fixtures.exists():
            raise FileNotFoundError(f"synthesis fixtures missing: {args.fixtures}")
        payloads = load_frozen_payloads(args.fixtures)
        graph_client = None
        repo_root = Path(AnalysisSettings.from_env().repo_root)
        if os.environ.get("NEO4J_URI"):
            from core.graph.client import GraphClient

            graph_client = GraphClient()
            graph_client.verify_connectivity()
        try:
            reports.append(
                await run_synthesis_bakeoff(
                    models,
                    payloads,
                    repeats=args.repeats,
                    graph_client=graph_client,
                    repo_root=repo_root if repo_root.exists() else None,
                )
            )
        finally:
            if graph_client is not None:
                graph_client.close()
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    """Run the bake-off and print markdown tables. Exit 0 even without a winner."""
    args = _parse_args(argv)
    if args.repeats < 1:
        print("repeats must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    settings = LLMSettings.from_env()
    if not settings.api_key:
        print(
            "OPENROUTER_API_KEY is unset; bake-off requires a live provider "
            "(unit tests inject StubProvider).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        reports = asyncio.run(_run(args))
    except FileNotFoundError as exc:
        print(f"model bake-off failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for report in reports:
        print(format_purpose_table(report), flush=True)
    if args.write_readme:
        write_model_bakeoff_section(reports)
        print("Wrote model bake-off tables into README.md", flush=True)


if __name__ == "__main__":
    main()
