"""SQLite repository for persistent chat memory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.chat.domain.models import DialogMemoryFact, DialogMessage, utc_now_iso


class SQLiteMemoryRepository:
    """Source of truth for dialog messages and manually managed facts."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dialog_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialog_key TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    rag_enabled INTEGER NOT NULL DEFAULT 0,
                    rag_query TEXT,
                    rag_total_found INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dialog_messages_dialog_created
                ON dialog_messages(dialog_key, created_at, id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dialog_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialog_key TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dialog_key, fact_key)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dialog_facts_dialog_key
                ON dialog_facts(dialog_key, fact_key)
                """
            )

    @staticmethod
    def _message_params(dialog_key: str, message: DialogMessage) -> tuple[object, ...]:
        return (
            dialog_key,
            message.role,
            message.content,
            message.created_at,
            int(message.rag_enabled),
            message.rag_query,
            message.rag_total_found,
            json.dumps(message.metadata, ensure_ascii=False),
        )

    def save_message(self, dialog_key: str, message: DialogMessage) -> None:
        self.save_turn(dialog_key, [message])

    def save_turn(self, dialog_key: str, messages: list[DialogMessage]) -> None:
        if not messages:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO dialog_messages (
                    dialog_key,
                    role,
                    content,
                    created_at,
                    rag_enabled,
                    rag_query,
                    rag_total_found,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._message_params(dialog_key, message) for message in messages],
            )

    def get_recent_messages(self, dialog_key: str, limit: int) -> list[DialogMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at, rag_enabled, rag_query, rag_total_found, metadata_json
                FROM dialog_messages
                WHERE dialog_key = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (dialog_key, limit),
            ).fetchall()

        messages: list[DialogMessage] = []
        for row in reversed(rows):
            metadata_raw = row["metadata_json"] or "{}"
            messages.append(
                DialogMessage(
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    rag_enabled=bool(row["rag_enabled"]),
                    rag_query=row["rag_query"],
                    rag_total_found=row["rag_total_found"],
                    metadata=json.loads(metadata_raw),
                )
            )
        return messages

    def upsert_fact(self, dialog_key: str, key: str, value: str, updated_at: str | None = None) -> None:
        effective_updated_at = updated_at or utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dialog_facts (dialog_key, fact_key, fact_value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(dialog_key, fact_key)
                DO UPDATE SET
                    fact_value = excluded.fact_value,
                    updated_at = excluded.updated_at
                """,
                (dialog_key, key, value, effective_updated_at),
            )

    def list_facts(self, dialog_key: str) -> list[DialogMemoryFact]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT fact_key, fact_value, updated_at
                FROM dialog_facts
                WHERE dialog_key = ?
                ORDER BY fact_key ASC
                """,
                (dialog_key,),
            ).fetchall()
        return [
            DialogMemoryFact(key=row["fact_key"], value=row["fact_value"], updated_at=row["updated_at"])
            for row in rows
        ]

    def delete_fact(self, dialog_key: str, key: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM dialog_facts WHERE dialog_key = ? AND fact_key = ?",
                (dialog_key, key),
            )
        return cursor.rowcount > 0

    def clear_dialog(self, dialog_key: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM dialog_messages WHERE dialog_key = ?", (dialog_key,))
            connection.execute("DELETE FROM dialog_facts WHERE dialog_key = ?", (dialog_key,))

    def count_messages(self, dialog_key: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM dialog_messages WHERE dialog_key = ?",
                (dialog_key,),
            ).fetchone()
        return int(row["total"]) if row is not None else 0

    def count_facts(self, dialog_key: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM dialog_facts WHERE dialog_key = ?",
                (dialog_key,),
            ).fetchone()
        return int(row["total"]) if row is not None else 0
