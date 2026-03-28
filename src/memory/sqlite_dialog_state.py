"""Concrete dialog state adapter backed by SQLite repository."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.chat.domain.models import DialogMemoryFact, DialogMessage, build_turn_messages
from src.chat.domain.ports import DialogState

from .sqlite_repository import SQLiteMemoryRepository


class SQLiteDialogState(DialogState):
    """Per-dialog state adapter with SQLite persistence and lazy recent-history cache."""

    def __init__(
        self,
        repository: SQLiteMemoryRepository,
        *,
        dialog_key: str,
        history_window: int | None = None,
    ):
        self.repository = repository
        self.dialog_key = dialog_key
        self.history_window = history_window
        self._cached_messages: list[DialogMessage] | None = None

    def _recent_limit(self) -> int:
        if self.history_window is not None and self.history_window > 0:
            return self.history_window
        persisted = self.repository.count_messages(self.dialog_key)
        return persisted or 1

    def _ensure_cache(self) -> list[DialogMessage]:
        if self._cached_messages is None:
            self._cached_messages = self.repository.get_recent_messages(self.dialog_key, limit=self._recent_limit())
        return self._cached_messages

    def _trim_cache(self) -> None:
        cache = self._ensure_cache()
        if self.history_window is None or self.history_window <= 0:
            return
        overflow = len(cache) - self.history_window
        if overflow > 0:
            self._cached_messages = cache[overflow:]

    def recent_messages(self) -> list[DialogMessage]:
        return list(self._ensure_cache())

    def append_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        rag_enabled: bool,
        rag_query: str | None,
        rag_total_found: int,
        user_metadata: Mapping[str, Any] | None = None,
        assistant_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        turn_messages = build_turn_messages(
            user_message=user_message,
            assistant_message=assistant_message,
            rag_enabled=rag_enabled,
            rag_query=rag_query,
            rag_total_found=rag_total_found,
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
        )
        cache = self._cached_messages
        self.repository.save_turn(self.dialog_key, turn_messages)

        if cache is None:
            cache = turn_messages.copy()
            self._cached_messages = cache
        else:
            cache.extend(turn_messages)
        self._trim_cache()

    def clear(self) -> None:
        self.repository.clear_dialog(self.dialog_key)
        self._cached_messages = []

    def list_facts(self) -> list[DialogMemoryFact]:
        return self.repository.list_facts(self.dialog_key)

    def remember_fact(self, key: str, value: str) -> DialogMemoryFact:
        fact = DialogMemoryFact.create(key, value)
        self.repository.upsert_fact(
            self.dialog_key,
            key=fact.key,
            value=fact.value,
            updated_at=fact.updated_at,
        )
        return fact

    def forget_fact(self, key: str) -> bool:
        normalized_key = DialogMemoryFact.normalize_key(key)
        return self.repository.delete_fact(self.dialog_key, normalized_key)

    def stats(self) -> dict[str, int]:
        return {
            "messages_persisted": self.repository.count_messages(self.dialog_key),
            "cached_messages": len(self._ensure_cache()),
            "facts": self.repository.count_facts(self.dialog_key),
            "history_window": self.history_window or 0,
        }
