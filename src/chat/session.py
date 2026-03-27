"""Каноническая история чата и операции над ней."""

from __future__ import annotations

from typing import Iterable

from .models import ChatMessage, ChatRole


class ChatSession:
    """Единственный источник истины для истории диалога."""

    def __init__(self, messages: Iterable[ChatMessage] | None = None):
        self._messages: list[ChatMessage] = list(messages or [])

    @property
    def messages(self) -> list[ChatMessage]:
        """Получить копию канонической истории."""
        return list(self._messages)

    def append_message(
        self,
        role: ChatRole,
        content: str,
        *,
        rag_enabled: bool = False,
        rag_query: str | None = None,
        rag_total_found: int = 0,
        metadata: dict | None = None,
    ) -> ChatMessage:
        """Добавить сообщение в историю."""
        message = ChatMessage(
            role=role,
            content=content,
            rag_enabled=rag_enabled,
            rag_query=rag_query,
            rag_total_found=rag_total_found,
            metadata=dict(metadata or {}),
        )
        self._messages.append(message)
        return message

    def append_turn(
        self,
        user_message: str,
        assistant_message: str,
        *,
        rag_enabled: bool,
        rag_query: str | None,
        rag_total_found: int,
    ) -> None:
        """Добавить завершённый user/assistant ход."""
        self.append_message(
            "user",
            user_message,
            rag_enabled=rag_enabled,
            rag_query=rag_query,
            rag_total_found=rag_total_found,
        )
        self.append_message(
            "assistant",
            assistant_message,
            rag_enabled=rag_enabled,
            rag_query=rag_query,
            rag_total_found=rag_total_found,
        )

    def clear(self) -> None:
        """Очистить историю."""
        self._messages.clear()

    def to_llm_history(self) -> list[dict[str, str]]:
        """Представить историю в формате Ollama chat messages."""
        return [message.to_llm_message() for message in self._messages]

    def export_messages(self) -> list[dict]:
        """Экспортировать каноническую историю целиком."""
        return [message.to_dict() for message in self._messages]

    def stats(self) -> dict[str, int]:
        """Краткая статистика по истории."""
        return {
            "messages": len(self._messages),
            "turns": sum(1 for message in self._messages if message.role == "user"),
        }
