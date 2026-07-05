"""SQLite-backed memory store for MVP persistence."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from packages.memory.redaction import redact_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteMemoryStore:
    """A tiny key-value memory store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    scope TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(scope, item_key)
                )
                """
            )
            conn.commit()

    def write(self, scope: str, item_key: str, value: str) -> None:
        redacted = redact_text(value)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_entries(scope, item_key, value, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(scope, item_key)
                DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (scope, item_key, redacted, utc_now()),
            )
            conn.commit()

    def read(self, scope: str | None = None, limit: int = 20) -> list[dict[str, str]]:
        query = "SELECT scope, item_key, value, updated_at FROM memory_entries"
        params: list[object] = []
        if scope is not None:
            query += " WHERE scope = ?"
            params.append(scope)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "scope": str(scope_value),
                "key": str(key),
                "value": str(value),
                "updated_at": str(updated_at),
            }
            for scope_value, key, value, updated_at in rows
        ]

    def search(self, query: str, scope: str | None = None, limit: int = 20) -> list[dict[str, str]]:
        sql = """
            SELECT scope, item_key, value, updated_at
            FROM memory_entries
            WHERE value LIKE ?
        """
        params: list[object] = [f"%{query}%"]
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "scope": str(scope_value),
                "key": str(key),
                "value": str(value),
                "updated_at": str(updated_at),
            }
            for scope_value, key, value, updated_at in rows
        ]

    def update_task_summary(self, scope: str, task_key: str, summary: str) -> None:
        self.write(scope=scope, item_key=task_key, value=summary)
