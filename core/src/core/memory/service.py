"""SQLite-backed conversation memory with a rolling summary and response cache.

Naive token estimation uses ``ceil(len(text) / 4)``. It is intentionally cheap and
approximate, and is only used to keep conversation context within a rough budget.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel, Field

from core.llm.provider import LLMProvider, Message
from core.logging import get_logger
from core.settings import MemorySettings

log = get_logger(__name__)

DEFAULT_SUMMARY = ""
DEFAULT_SUMMARY_MAX_TOKENS = 512
SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        summary_token_count INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        token_estimate INTEGER NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS response_cache (
        cache_key TEXT PRIMARY KEY,
        response_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
)


class ConversationTurn(BaseModel):
    """One stored conversational turn."""

    id: int
    role: str
    content: str
    created_at: str
    token_estimate: int


class ConversationContext(BaseModel):
    """Rolling summary plus the recent verbatim window."""

    summary: str = DEFAULT_SUMMARY
    recent_turns: list[ConversationTurn] = Field(default_factory=list)


class CachedResponse(BaseModel):
    """Opaque cached response payload returned when still fresh."""

    cache_key: str
    response_json: Any
    created_at: str


class SummaryPayload(BaseModel):
    """Structured summary result returned by the summarizer."""

    summary: str


class MemoryService:
    """Persist turns, summarize older context, and store cached responses."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        db_path: str | None = None,
        cache_ttl_seconds: int | None = None,
        token_budget: int | None = None,
        recent_turns_to_keep: int | None = None,
    ) -> None:
        settings = MemorySettings.from_env()
        self._llm_provider = llm_provider
        self._db_path = db_path or settings.db_path
        self._cache_ttl_seconds = (
            settings.cache_ttl_seconds if cache_ttl_seconds is None else cache_ttl_seconds
        )
        self._token_budget = settings.token_budget if token_budget is None else token_budget
        self._recent_turns_to_keep = (
            settings.recent_turns_to_keep
            if recent_turns_to_keep is None
            else recent_turns_to_keep
        )
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create the SQLite schema and any lightweight migrations once."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self._db_path) as db:
                for statement in SCHEMA_STATEMENTS:
                    await db.execute(statement)
                await self._ensure_turns_folded_column(db)
                await db.commit()
            self._initialized = True
            log.info("memory.schema_ready", db_path=self._db_path)

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Insert a new turn and trigger summarization when the budget is breached."""
        await self.initialize()
        created_at = _utc_now().isoformat()
        token_estimate = estimate_tokens(content)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_session(db, session_id, created_at)
            await db.execute(
                """
                INSERT INTO turns (
                    session_id,
                    role,
                    content,
                    created_at,
                    token_estimate,
                    folded_at
                ) VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (session_id, role, content, created_at, token_estimate),
            )
            await db.commit()
        log.info(
            "memory.turn_appended",
            session_id=session_id,
            role=role,
            token_estimate=token_estimate,
        )
        if await self._needs_summary(session_id):
            await self.summarize_session(session_id)

    async def get_context(
        self,
        session_id: str,
        token_budget: int = 3000,
    ) -> ConversationContext:
        """Return the rolling summary plus as many recent turns as fit the budget."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            session = await self._fetch_session(db, session_id)
            if session is None:
                return ConversationContext()
            summary = str(session["summary"] or "")
            summary_tokens = int(session["summary_token_count"] or 0)
            rows = await self._fetch_recent_unfolded_turn_rows(db, session_id)
        remaining = max(token_budget - summary_tokens, 0)
        chosen: list[ConversationTurn] = []
        running = 0
        for row in rows:
            turn = _turn_from_row(row)
            if running + turn.token_estimate > remaining:
                break
            chosen.append(turn)
            running += turn.token_estimate
        chosen.reverse()
        return ConversationContext(summary=summary, recent_turns=chosen)

    async def summarize_session(self, session_id: str) -> str:
        """Fold older turns into the rolling summary while keeping the last turns verbatim."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            session = await self._fetch_session(db, session_id)
            if session is None:
                return DEFAULT_SUMMARY
            all_rows = await self._fetch_unfolded_turn_rows(db, session_id)
            if len(all_rows) <= self._recent_turns_to_keep:
                return str(session["summary"] or "")
            folded_rows = all_rows[:-self._recent_turns_to_keep]
            retained_rows = all_rows[-self._recent_turns_to_keep :]
            prompt = _build_summary_prompt(
                existing_summary=str(session["summary"] or ""),
                folded_turns=[_turn_from_row(row) for row in folded_rows],
                retained_turn_count=len(retained_rows),
            )
            result = await self._llm_provider.complete(
                [
                    Message(
                        role="system",
                        content=(
                            "Summarize the conversation faithfully. Preserve decisions, "
                            "open questions, constraints, and unresolved action items."
                        ),
                    ),
                    Message(role="user", content=prompt),
                ],
                response_model=SummaryPayload,
                purpose="summarization",
                agent="memory",
                max_tokens=DEFAULT_SUMMARY_MAX_TOKENS,
            )
            parsed = result.parsed
            if not isinstance(parsed, SummaryPayload):
                raise TypeError("summarize_session expected SummaryPayload from LLM provider")
            summary = parsed.summary.strip()
            folded_at = _utc_now().isoformat()
            await db.execute(
                """
                UPDATE sessions
                SET summary = ?, summary_token_count = ?
                WHERE session_id = ?
                """,
                (summary, estimate_tokens(summary), session_id),
            )
            await db.executemany(
                "UPDATE turns SET folded_at = ? WHERE id = ?",
                [(folded_at, int(row["id"])) for row in folded_rows],
            )
            await db.commit()
        log.info(
            "memory.session_summarized",
            session_id=session_id,
            folded_turns=len(folded_rows),
            retained_turns=len(retained_rows),
        )
        return summary

    async def cache_get(self, cache_key: str) -> CachedResponse | None:
        """Return a cached response if it has not expired; otherwise drop it."""
        await self.initialize()
        cutoff = _utc_now() - timedelta(seconds=self._cache_ttl_seconds)
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT cache_key, response_json, created_at
                FROM response_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            created_at = _parse_timestamp(str(row["created_at"]))
            if created_at < cutoff:
                await db.execute("DELETE FROM response_cache WHERE cache_key = ?", (cache_key,))
                await db.commit()
                log.info("memory.cache_expired", cache_key=cache_key)
                return None
            return CachedResponse(
                cache_key=str(row["cache_key"]),
                response_json=json.loads(str(row["response_json"])),
                created_at=str(row["created_at"]),
            )

    async def cache_put(self, cache_key: str, response_json: Any) -> None:
        """Store or replace an opaque cached JSON response."""
        await self.initialize()
        created_at = _utc_now().isoformat()
        encoded = json.dumps(response_json)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO response_cache (cache_key, response_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at
                """,
                (cache_key, encoded, created_at),
            )
            await db.commit()
        log.info("memory.cache_stored", cache_key=cache_key)

    async def _needs_summary(self, session_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            session = await self._fetch_session(db, session_id)
            if session is None:
                return False
            rows = await self._fetch_unfolded_turn_rows(db, session_id)
            total_turn_tokens = sum(int(row["token_estimate"] or 0) for row in rows)
            total_tokens = int(session["summary_token_count"] or 0) + total_turn_tokens
            return total_tokens > self._token_budget and len(rows) > self._recent_turns_to_keep

    async def _ensure_session(
        self,
        db: aiosqlite.Connection,
        session_id: str,
        created_at: str,
    ) -> None:
        await db.execute(
            """
            INSERT INTO sessions (session_id, created_at, summary, summary_token_count)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, created_at, DEFAULT_SUMMARY),
        )

    async def _fetch_session(
        self,
        db: aiosqlite.Connection,
        session_id: str,
    ) -> aiosqlite.Row | None:
        cursor = await db.execute(
            """
            SELECT session_id, created_at, summary, summary_token_count
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row

    async def _fetch_unfolded_turn_rows(
        self,
        db: aiosqlite.Connection,
        session_id: str,
    ) -> list[aiosqlite.Row]:
        cursor = await db.execute(
            """
            SELECT id, role, content, created_at, token_estimate
            FROM turns
            WHERE session_id = ? AND folded_at IS NULL
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def _fetch_recent_unfolded_turn_rows(
        self,
        db: aiosqlite.Connection,
        session_id: str,
    ) -> list[aiosqlite.Row]:
        cursor = await db.execute(
            """
            SELECT id, role, content, created_at, token_estimate
            FROM turns
            WHERE session_id = ? AND folded_at IS NULL
            ORDER BY id DESC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def _ensure_turns_folded_column(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(turns)")
        rows = await cursor.fetchall()
        await cursor.close()
        columns = {str(row[1]) for row in rows}
        if "folded_at" not in columns:
            await db.execute("ALTER TABLE turns ADD COLUMN folded_at TEXT NULL")


def estimate_tokens(text: str) -> int:
    """Return a cheap, rough token estimate based on character count."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _turn_from_row(row: aiosqlite.Row) -> ConversationTurn:
    return ConversationTurn(
        id=int(row["id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
        token_estimate=int(row["token_estimate"]),
    )


def _build_summary_prompt(
    *,
    existing_summary: str,
    folded_turns: Sequence[ConversationTurn],
    retained_turn_count: int,
) -> str:
    folded_lines = "\n".join(
        f"- [{turn.role}] {turn.content}" for turn in folded_turns
    ) or "- none"
    return (
        "Update the rolling conversation summary using ONLY the folded turns below.\n"
        "Do not include or infer details from retained turns that are not shown here.\n\n"
        f"Existing summary:\n{existing_summary or '(empty)'}\n\n"
        "Folded turns to summarize:\n"
        f"{folded_lines}\n\n"
        f"Recent turns retained verbatim outside the summary: {retained_turn_count}"
    )
