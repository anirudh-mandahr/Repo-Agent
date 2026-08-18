"""Offline tests for the SQLite-backed memory service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from core.llm.stub import StubProvider
from core.memory.service import DEFAULT_SUMMARY, MemoryService


async def test_budget_breach_folds_older_turns_and_keeps_last_six(tmp_path: Path) -> None:
    provider = StubProvider([{"summary": "Folded summary of the early turns."}])
    service = MemoryService(
        provider,
        db_path=str(tmp_path / "memory.db"),
        token_budget=80,
        recent_turns_to_keep=6,
    )

    early_user = "early-user " * 4
    early_assistant = "early-assistant " * 4
    retained_contents = [f"recent-{index} " * 4 for index in range(1, 7)]

    await service.append_turn("session-1", "user", early_user)
    await service.append_turn("session-1", "assistant", early_assistant)
    for index, content in enumerate(retained_contents):
        role = "user" if index % 2 == 0 else "assistant"
        await service.append_turn("session-1", role, content)

    context = await service.get_context("session-1", token_budget=3000)
    assert context.summary == "Folded summary of the early turns."
    assert [turn.content for turn in context.recent_turns] == retained_contents

    prompt = provider.calls[0].messages[-1].content
    assert early_user in prompt
    assert early_assistant in prompt
    for content in retained_contents:
        assert content not in prompt

    async with aiosqlite.connect(tmp_path / "memory.db") as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ? AND folded_at IS NOT NULL",
            ("session-1",),
        )
        folded_count = (await cursor.fetchone())[0]
        await cursor.close()
    assert folded_count == 2


async def test_cache_entry_expires_after_ttl(tmp_path: Path) -> None:
    service = MemoryService(
        StubProvider(),
        db_path=str(tmp_path / "memory.db"),
        cache_ttl_seconds=60,
    )
    await service.cache_put("cache-1", {"answer": 42})

    expired_at = (datetime.now(tz=UTC) - timedelta(minutes=5)).isoformat()
    async with aiosqlite.connect(tmp_path / "memory.db") as db:
        await db.execute(
            "UPDATE response_cache SET created_at = ? WHERE cache_key = ?",
            (expired_at, "cache-1"),
        )
        await db.commit()

    assert await service.cache_get("cache-1") is None

    async with aiosqlite.connect(tmp_path / "memory.db") as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM response_cache WHERE cache_key = ?",
            ("cache-1",),
        )
        remaining = (await cursor.fetchone())[0]
        await cursor.close()
    assert remaining == 0


async def test_get_context_uses_summary_budget_before_recent_turns(tmp_path: Path) -> None:
    service = MemoryService(
        StubProvider(),
        db_path=str(tmp_path / "memory.db"),
        token_budget=400,
        recent_turns_to_keep=6,
    )

    await service.append_turn("session-1", "user", "a" * 40)
    await service.append_turn("session-1", "assistant", "b" * 40)
    await service.append_turn("session-1", "user", "c" * 40)

    async with aiosqlite.connect(tmp_path / "memory.db") as db:
        await db.execute(
            """
            UPDATE sessions
            SET summary = ?, summary_token_count = ?
            WHERE session_id = ?
            """,
            ("x" * 80, 20, "session-1"),
        )
        await db.commit()

    context = await service.get_context("session-1", token_budget=30)

    assert context.summary == "x" * 80
    assert [turn.content for turn in context.recent_turns] == ["c" * 40]


async def test_summarize_session_returns_existing_summary_when_under_retention(
    tmp_path: Path,
) -> None:
    provider = StubProvider([{"summary": "should not be used"}])
    service = MemoryService(
        provider,
        db_path=str(tmp_path / "memory.db"),
        token_budget=1000,
        recent_turns_to_keep=6,
    )

    await service.append_turn("session-1", "user", "hello")
    await service.append_turn("session-1", "assistant", "world")

    async with aiosqlite.connect(tmp_path / "memory.db") as db:
        await db.execute(
            "UPDATE sessions SET summary = ?, summary_token_count = ? WHERE session_id = ?",
            ("existing summary", 4, "session-1"),
        )
        await db.commit()

    summary = await service.summarize_session("session-1")

    assert summary == "existing summary"
    assert provider.calls == []


async def test_missing_session_returns_default_summary_and_no_turns(tmp_path: Path) -> None:
    service = MemoryService(StubProvider(), db_path=str(tmp_path / "memory.db"))

    context = await service.get_context("missing-session", token_budget=10)

    assert context.summary == DEFAULT_SUMMARY
    assert context.recent_turns == []
