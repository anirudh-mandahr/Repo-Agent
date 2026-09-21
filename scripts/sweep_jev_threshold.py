"""Sweep the Jev agent-selection threshold offline against cached judge answers.

``JevRoutingProvider`` asks TypeSafe's Jev model one batched question per query
(one Choice for ``intent`` plus one Noul per agent) and only *then* thresholds
the Noul probabilities to pick ``target_agents``. That post-hoc step means a
single pass over the case set is enough: cache every raw ``answers`` map once,
then replay ``intent_from_answers`` at any threshold with zero extra API calls.

Usage::

    set -a && . ./.env >/dev/null 2>&1 && set +a && \\
        uv run python scripts/sweep_jev_threshold.py

The first run hits the TypeSafe API once per turn (~58 calls) and writes the
raw answers to ``evals/bakeoffs/jev-threshold-sweep-raw.json``. Every later run
reads that cache and makes no API calls unless ``--refresh`` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("LOG_LEVEL", "ERROR")

from core.eval.harness import ROOT, load_qa_cases
from core.eval.model_bakeoff import flatten_turns
from core.eval.models import TurnSpec
from core.eval.routing_providers import routing_provider_factory
from core.llm.jev_provider import JEV_MODEL_ID, intent_from_answers

DEFAULT_RAW_PATH: Path = ROOT / "evals" / "bakeoffs" / "jev-threshold-sweep-raw.json"

# 0.05 .. 0.95 inclusive, in steps of 0.05. Built from integers to avoid
# floating-point step drift.
THRESHOLDS: tuple[float, ...] = tuple(round(step * 0.05, 2) for step in range(1, 20))


class RawTurn:
    """One cached judge pass over a single eval turn.

    Attributes:
        case_id: Turn identifier, as produced by ``flatten_turns``.
        tier: The parent case's tier (``"trap"`` for out-of-scope cases).
        query: The user query that was judged.
        expected_agents: Labelled target agents for this turn.
        out_of_scope: Whether this turn is a trap case.
        answers: The raw Jev ``answers`` map returned by ``judge``.
    """

    def __init__(
        self,
        *,
        case_id: str,
        tier: str,
        query: str,
        expected_agents: list[str],
        out_of_scope: bool,
        answers: dict[str, Any],
    ) -> None:
        self.case_id = case_id
        self.tier = tier
        self.query = query
        self.expected_agents = expected_agents
        self.out_of_scope = out_of_scope
        self.answers = answers

    def to_json(self) -> dict[str, Any]:
        """Serialize this row for the raw-answers cache file.

        Returns:
            A JSON-safe dict with one key per attribute.
        """
        return {
            "case_id": self.case_id,
            "tier": self.tier,
            "query": self.query,
            "expected_agents": self.expected_agents,
            "out_of_scope": self.out_of_scope,
            "answers": self.answers,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RawTurn:
        """Deserialize one row from the raw-answers cache file.

        Args:
            payload: One JSON object, as written by :meth:`to_json`.

        Returns:
            The reconstructed row.
        """
        return cls(
            case_id=str(payload["case_id"]),
            tier=str(payload["tier"]),
            query=str(payload["query"]),
            expected_agents=[str(item) for item in payload.get("expected_agents", [])],
            out_of_scope=bool(payload.get("out_of_scope", False)),
            answers=dict(payload.get("answers") or {}),
        )


class ThresholdRow:
    """Scored results for one candidate threshold.

    Attributes:
        threshold: The Noul probability cutoff evaluated.
        quality: Exact-match rate against ``expected_agents`` over non-trap turns.
        trap_pass_rate: Exact-match rate over out-of-scope (trap) turns.
        mean_agents: Mean number of agents selected across all turns.
    """

    def __init__(
        self,
        *,
        threshold: float,
        quality: float,
        trap_pass_rate: float,
        mean_agents: float,
    ) -> None:
        self.threshold = threshold
        self.quality = quality
        self.trap_pass_rate = trap_pass_rate
        self.mean_agents = mean_agents

    def to_json(self) -> dict[str, float]:
        """Serialize this row for ``--out``.

        Returns:
            A JSON-safe dict with one key per attribute.
        """
        return {
            "threshold": self.threshold,
            "quality": self.quality,
            "trap_pass_rate": self.trap_pass_rate,
            "mean_agents": self.mean_agents,
        }


def _agents_match(actual: Sequence[str], expected: Sequence[str]) -> bool:
    """Score set-equality the same way the bake-off's ``_agents_match`` does.

    Args:
        actual: Agents the provider selected.
        expected: Labelled target agents.

    Returns:
        True when the stripped, non-empty agent sets are identical.
    """
    return {item.strip() for item in actual if item.strip()} == {
        item.strip() for item in expected if item.strip()
    }


async def collect_raw_answers(turns: Sequence[tuple[str, str, TurnSpec]]) -> list[RawTurn]:
    """Judge every turn once against the live Jev backend.

    Args:
        turns: ``(case_id, tier, turn)`` rows from ``flatten_turns``.

    Returns:
        One cached row per turn, in the same order as ``turns``.
    """
    provider = routing_provider_factory(JEV_MODEL_ID)
    judge = getattr(provider, "judge", None)
    if judge is None:  # pragma: no cover - defensive; factory always returns JevRoutingProvider
        raise TypeError(f"{type(provider).__name__} has no judge() method")

    rows: list[RawTurn] = []
    for case_id, tier, turn in turns:
        answers, _usage = await judge(turn.query)
        rows.append(
            RawTurn(
                case_id=case_id,
                tier=tier,
                query=turn.query,
                expected_agents=list(turn.expected_agents),
                out_of_scope=turn.out_of_scope,
                answers=answers,
            )
        )
        print(f"judged {case_id} ({len(rows)}/{len(turns)})", file=sys.stderr)
    return rows


def load_raw_cache(path: Path) -> list[RawTurn]:
    """Load a previously written raw-answers cache.

    Args:
        path: Cache file written by :func:`save_raw_cache`.

    Returns:
        The cached rows, in file order.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RawTurn.from_json(row) for row in payload["turns"]]


def save_raw_cache(path: Path, rows: Sequence[RawTurn]) -> None:
    """Write the raw-answers cache so the sweep can be replayed offline.

    Args:
        path: Destination file. Parent directories are created as needed.
        rows: One cached judge result per turn.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": JEV_MODEL_ID, "turns": [row.to_json() for row in rows]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def score_threshold(rows: Sequence[RawTurn], threshold: float) -> ThresholdRow:
    """Recompute routed agents at ``threshold`` and score the case set.

    Args:
        rows: Cached judge results for every turn.
        threshold: Noul probability at or above which an agent is selected.

    Returns:
        Quality, trap pass rate, and mean agent count at this threshold.
    """
    non_trap_hits = 0
    non_trap_total = 0
    trap_hits = 0
    trap_total = 0
    agent_counts: list[int] = []

    for row in rows:
        intent = intent_from_answers(row.answers, row.query, [], threshold=threshold)
        matched = _agents_match(list(intent.target_agents), row.expected_agents)
        agent_counts.append(len(intent.target_agents))
        if row.out_of_scope:
            trap_total += 1
            trap_hits += int(matched)
        else:
            non_trap_total += 1
            non_trap_hits += int(matched)

    quality = non_trap_hits / non_trap_total if non_trap_total else 0.0
    trap_pass_rate = trap_hits / trap_total if trap_total else 0.0
    mean_agents = sum(agent_counts) / len(agent_counts) if agent_counts else 0.0
    return ThresholdRow(
        threshold=threshold,
        quality=quality,
        trap_pass_rate=trap_pass_rate,
        mean_agents=mean_agents,
    )


def format_table(curve: Sequence[ThresholdRow]) -> str:
    """Render the threshold/quality/trap curve as a fixed-width table.

    Args:
        curve: One row per swept threshold, in ascending threshold order.

    Returns:
        The table as a printable string, with the best-quality row marked.
    """
    best_quality = max((row.quality for row in curve), default=0.0)
    lines = [
        f"{'threshold':>9}  {'quality':>7}  {'trap pass':>9}  {'mean agents':>11}",
        f"{'-' * 9}  {'-' * 7}  {'-' * 9}  {'-' * 11}",
    ]
    for row in curve:
        marker = "  <- best" if row.quality == best_quality else ""
        lines.append(
            f"{row.threshold:>9.2f}  {row.quality:>7.3f}  {row.trap_pass_rate:>9.3f}  "
            f"{row.mean_agents:>11.2f}{marker}"
        )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Parsed namespace with ``raw``, ``refresh``, and ``out``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the Jev agent-selection threshold offline from one cached "
            "judge() pass over the bake-off's 58-turn case set."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Raw-answers cache path (default {DEFAULT_RAW_PATH}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-judge every turn against the live API even if the cache exists.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the threshold/quality/trap curve as JSON.",
    )
    return parser.parse_args(argv)


async def _amain(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.raw.exists() and not args.refresh:
        rows = load_raw_cache(args.raw)
        print(f"loaded {len(rows)} cached turns from {args.raw}", file=sys.stderr)
    else:
        turns = flatten_turns(load_qa_cases())
        rows = await collect_raw_answers(turns)
        save_raw_cache(args.raw, rows)
        print(f"wrote {len(rows)} judged turns to {args.raw}", file=sys.stderr)

    curve = [score_threshold(rows, threshold) for threshold in THRESHOLDS]
    print(format_table(curve))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([row.to_json() for row in curve], indent=2), encoding="utf-8"
        )
        print(f"wrote curve to {args.out}", file=sys.stderr)


def main() -> None:
    """Entry point: run the sweep and print the results table."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
