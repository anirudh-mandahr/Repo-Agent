"""Persist bake-off reports so arms can be run separately and compared later.

``run_router_bakeoff`` scores every model in one process. When one provider is
unavailable -- an expired key, a vendor outage -- the arms have to be captured
at different times instead. Quality, schema-failure rate and cost survive that
split because they are computed from a fixed case set at temperature 0.
Wall-clock latency does not, so :func:`merge_bakeoffs` labels it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from core.eval.model_bakeoff import (
    MeanSpread,
    ModelBakeoffRow,
    PurposeBakeoff,
)
from core.llm.pricing import ModelRates

CROSS_RUN_NOTE = (
    "Rows were captured in separate runs. Quality, schema-failure rate and cost "
    "are computed from a fixed case set at temperature 0 and remain comparable; "
    "p50/p95 latency was measured at different times under different network and "
    "vendor load, so latency is NOT a like-for-like comparison."
)


def _spread_payload(value: MeanSpread) -> dict[str, Any]:
    return {"mean": value.mean, "spread": value.spread, "values": list(value.values)}


def _spread_from_payload(payload: dict[str, Any]) -> MeanSpread:
    return MeanSpread(
        mean=float(payload["mean"]),
        spread=float(payload["spread"]),
        values=tuple(float(item) for item in payload.get("values", ())),
    )


_SPREAD_FIELDS = (
    "quality",
    "trap_pass",
    "schema_failure",
    "p50_latency_ms",
    "p95_latency_ms",
    "cost_per_query",
    "cost_per_1000",
)


def to_payload(report: PurposeBakeoff) -> dict[str, Any]:
    """Render ``report`` as JSON-safe data.

    Args:
        report: One purpose's bake-off result.

    Returns:
        A dict suitable for :func:`json.dump`.
    """
    return {
        "purpose": report.purpose,
        "captured_at": report.captured_at,
        "price_as_of": report.price_as_of,
        "repeats": report.repeats,
        "n_queries": report.n_queries,
        "notes": list(report.notes),
        "rows": [
            {
                "model": row.model,
                **{name: _spread_payload(getattr(row, name)) for name in _SPREAD_FIELDS},
                "rates": row.rates.model_dump(),
                "price_label": row.price_label,
                "repeats": row.repeats,
            }
            for row in report.rows
        ],
    }


def from_payload(payload: dict[str, Any]) -> PurposeBakeoff:
    """Rebuild a report saved by :func:`to_payload`.

    Args:
        payload: Parsed JSON produced by :func:`to_payload`.

    Returns:
        PurposeBakeoff.
    """
    return PurposeBakeoff(
        purpose=payload["purpose"],
        rows=[
            ModelBakeoffRow(
                model=row["model"],
                **{name: _spread_from_payload(row[name]) for name in _SPREAD_FIELDS},
                rates=ModelRates.model_validate(row["rates"]),
                price_label=row["price_label"],
                repeats=int(row["repeats"]),
            )
            for row in payload.get("rows", [])
        ],
        captured_at=payload.get("captured_at"),
        price_as_of=payload["price_as_of"],
        repeats=int(payload["repeats"]),
        n_queries=int(payload["n_queries"]),
        notes=list(payload.get("notes", [])),
    )


def save_bakeoff(report: PurposeBakeoff, path: Path) -> Path:
    """Write ``report`` to ``path`` as JSON.

    Args:
        report: Result to persist.
        path: Destination file; parent directories are created.

    Returns:
        The path written.
    """
    payload = to_payload(report)
    # The routing bake-off leaves captured_at unset, which renders as "unknown"
    # in a merged table. Stamp the save date so provenance survives the split.
    if not payload.get("captured_at"):
        payload["captured_at"] = date.today().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_bakeoff(path: Path) -> PurposeBakeoff:
    """Read a report written by :func:`save_bakeoff`.

    Args:
        path: Source file.

    Returns:
        PurposeBakeoff.
    """
    return from_payload(json.loads(path.read_text()))


def merge_bakeoffs(reports: Sequence[PurposeBakeoff]) -> PurposeBakeoff:
    """Combine separately captured arms of the same comparison into one report.

    Args:
        reports: Reports to merge, in the order their rows should appear.

    Returns:
        One report carrying every row, with a cross-run latency caveat in
        ``notes`` whenever more than one run contributed.

    Raises:
        ValueError: When the reports are empty, cover different purposes,
            scored different numbers of queries, or repeat a model.
    """
    if not reports:
        raise ValueError("merge_bakeoffs needs at least one report")
    purposes = {report.purpose for report in reports}
    if len(purposes) > 1:
        raise ValueError(f"cannot merge different purposes: {sorted(purposes)}")
    sizes = {report.n_queries for report in reports}
    if len(sizes) > 1:
        raise ValueError(
            f"cannot merge runs scored over different case counts: {sorted(sizes)}"
        )

    rows: list[ModelBakeoffRow] = []
    seen: set[str] = set()
    for report in reports:
        for row in report.rows:
            if row.model in seen:
                raise ValueError(f"model appears in more than one report: {row.model}")
            seen.add(row.model)
            rows.append(row)

    notes: list[str] = []
    for report in reports:
        for note in report.notes:
            if note not in notes:
                notes.append(note)
    captured = sorted(
        (report.captured_at for report in reports if report.captured_at), reverse=True
    )
    if len(reports) > 1:
        notes.append(CROSS_RUN_NOTE)
        captured_list = ", ".join(
            f"{row.model} ({report.captured_at or 'unknown'})"
            for report in reports
            for row in report.rows
        )
        notes.append(f"Captured: {captured_list}.")

    return PurposeBakeoff(
        purpose=reports[0].purpose,
        rows=rows,
        captured_at=captured[0] if captured else None,
        price_as_of=reports[0].price_as_of,
        repeats=max(report.repeats for report in reports),
        n_queries=reports[0].n_queries,
        notes=notes,
    )
