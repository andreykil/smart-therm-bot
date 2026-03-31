"""Test-only in-memory dialog state implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.chat.domain.models import DialogMemoryFact, DialogMessage, build_turn_messages
from src.chat.domain.ports import DialogState


class InMemoryDialogState(DialogState):
    """Эфемерная реализация состояния диалога без persistent storage."""

    def __init__(self, *, history_window: int | None = None):
        self.history_window = history_window
        self._messages: list[DialogMessage] = []
        self._facts: dict[str, DialogMemoryFact] = {}

    def _trim_messages(self) -> None:
        if self.history_window is None or self.history_window <= 0:
            return
        overflow = len(self._messages) - self.history_window
        if overflow > 0:
            self._messages = self._messages[overflow:]

    def recent_messages(self) -> list[DialogMessage]:
        return list(self._messages)

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
        self._messages.extend(
            build_turn_messages(
                user_message=user_message,
                assistant_message=assistant_message,
                rag_enabled=rag_enabled,
                rag_query=rag_query,
                rag_total_found=rag_total_found,
                user_metadata=user_metadata,
                assistant_metadata=assistant_metadata,
            )
        )
        self._trim_messages()

    def clear(self) -> None:
        self._messages.clear()
        self._facts.clear()

    def clear_history(self) -> None:
        self._messages.clear()

    def list_facts(self) -> list[DialogMemoryFact]:
        return list(sorted(self._facts.values(), key=lambda fact: fact.key))

    def remember_fact(self, key: str, value: str) -> DialogMemoryFact:
        fact = DialogMemoryFact.create(key, value)
        self._facts[fact.key] = fact
        return fact

    def forget_fact(self, key: str) -> bool:
        normalized_key = DialogMemoryFact.normalize_key(key)
        return self._facts.pop(normalized_key, None) is not None

    def stats(self) -> dict[str, int]:
        return {
            "messages_persisted": len(self._messages),
            "cached_messages": len(self._messages),
            "facts": len(self._facts),
            "history_window": self.history_window or 0,
        }
