"""Tests for persisting and merging separately captured bake-off arms."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.eval.bakeoff_store import (
    CROSS_RUN_NOTE,
    load_bakeoff,
    merge_bakeoffs,
    save_bakeoff,
    to_payload,
)
from core.eval.model_bakeoff import (
    MeanSpread,
    ModelBakeoffRow,
    PurposeBakeoff,
    format_purpose_table,
)
from core.llm.pricing import CATALOG


def _row(model: str, quality: float) -> ModelBakeoffRow:
    spread = MeanSpread.from_values([quality, quality])
    zero = MeanSpread.from_values([0.0])
    return ModelBakeoffRow(
        model=model,
        quality=spread,
        trap_pass=zero,
        schema_failure=zero,
        p50_latency_ms=MeanSpread.from_values([100.0]),
        p95_latency_ms=MeanSpread.from_values([200.0]),
        cost_per_query=zero,
        cost_per_1000=zero,
        rates=CATALOG.get(model, CATALOG["anthropic/claude-sonnet-4.5"]),
        price_label=f"{model}: test",
        repeats=2,
    )


def _report(model: str, quality: float, *, captured_at: str, n_queries: int = 58) -> PurposeBakeoff:
    return PurposeBakeoff(
        purpose="routing",
        rows=[_row(model, quality)],
        captured_at=captured_at,
        price_as_of="2026-08-18",
        repeats=2,
        n_queries=n_queries,
        notes=[],
    )


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    report = _report("typesafe/jev-latest", 0.87, captured_at="2026-09-21")
    path = save_bakeoff(report, tmp_path / "nested" / "jev.json")
    restored = load_bakeoff(path)

    assert restored.purpose == report.purpose
    assert restored.n_queries == report.n_queries
    assert restored.captured_at == report.captured_at
    assert [row.model for row in restored.rows] == ["typesafe/jev-latest"]
    assert restored.rows[0].quality.mean == pytest.approx(0.87)
    assert restored.rows[0].quality.values == report.rows[0].quality.values
    # `rates` is a pydantic model nested in a dataclass; it must survive too.
    assert restored.rows[0].rates.prompt_usd_per_million == pytest.approx(0.042)


def test_payload_is_json_safe() -> None:
    import json

    json.dumps(to_payload(_report("typesafe/jev-latest", 0.5, captured_at="2026-09-21")))


def test_merge_combines_arms_and_flags_latency() -> None:
    merged = merge_bakeoffs(
        [
            _report("typesafe/jev-latest", 0.87, captured_at="2026-09-21"),
            _report("anthropic/claude-sonnet-4.5", 0.91, captured_at="2026-09-28"),
        ]
    )
    assert [row.model for row in merged.rows] == [
        "typesafe/jev-latest",
        "anthropic/claude-sonnet-4.5",
    ]
    assert CROSS_RUN_NOTE in merged.notes
    assert any("2026-09-28" in note for note in merged.notes)
    # The caveat has to reach the rendered table, not just the data.
    assert "NOT a like-for-like" in format_purpose_table(merged)


def test_single_report_merge_adds_no_caveat() -> None:
    merged = merge_bakeoffs([_report("typesafe/jev-latest", 0.87, captured_at="2026-09-21")])
    assert CROSS_RUN_NOTE not in merged.notes


def test_refuses_to_merge_different_case_counts() -> None:
    with pytest.raises(ValueError, match="different case counts"):
        merge_bakeoffs(
            [
                _report("typesafe/jev-latest", 0.87, captured_at="2026-09-21", n_queries=58),
                _report("anthropic/claude-sonnet-4.5", 0.91, captured_at="2026-09-28", n_queries=9),
            ]
        )


def test_refuses_to_merge_a_duplicated_model() -> None:
    with pytest.raises(ValueError, match="more than one report"):
        merge_bakeoffs(
            [
                _report("typesafe/jev-latest", 0.87, captured_at="2026-09-21"),
                _report("typesafe/jev-latest", 0.88, captured_at="2026-09-28"),
            ]
        )


def test_refuses_an_empty_merge() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_bakeoffs([])


def test_save_stamps_captured_at_when_the_bakeoff_left_it_unset(tmp_path: Path) -> None:
    """The routing bake-off sets captured_at=None; a merged table needs a date."""
    import json as _json

    report = _report("typesafe/jev-latest", 0.68, captured_at="2026-09-21")
    report.captured_at = None
    path = save_bakeoff(report, tmp_path / "jev.json")
    assert _json.loads(path.read_text())["captured_at"]
    assert load_bakeoff(path).captured_at is not None
