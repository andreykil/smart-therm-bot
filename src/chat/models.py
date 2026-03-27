"""Структуры данных для chat orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from src.rag.models import SearchResult

ChatRole = Literal["user", "assistant"]
StreamEventKind = Literal["debug", "token", "final"]
RetrievalErrorKind = Literal["retrieval_unavailable", "misconfiguration", "runtime_failure"]


def utc_now_iso() -> str:
    """Получить текущее время в UTC в ISO-формате."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ChatMessage:
    """Каноническое сообщение истории чата."""

    role: ChatRole
    content: str
    created_at: str = field(default_factory=utc_now_iso)
    rag_enabled: bool = False
    rag_query: str | None = None
    rag_total_found: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_message(self) -> dict[str, str]:
        """Сконвертировать запись истории в payload для LLM."""
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать сообщение для JSON/export."""
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "rag_enabled": self.rag_enabled,
            "rag_query": self.rag_query,
            "rag_total_found": self.rag_total_found,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RetrievedContext:
    """Структурированное представление retrieval-результата для одного хода."""

    enabled: bool
    query: str | None = None
    result: SearchResult | None = None
    context_text: str = ""
    error: str | None = None
    error_kind: RetrievalErrorKind | None = None

    @property
    def total_found(self) -> int:
        return self.result.total_found if self.result is not None else 0

    @property
    def used(self) -> bool:
        return bool(self.context_text.strip())

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(slots=True)
class ChatTurnRequest:
    """Параметры одного хода чата."""

    user_message: str
    max_tokens: int = 1024
    temperature: float = 0.7
    use_rag: bool = False
    system_prompt_override: str | None = None


@dataclass(slots=True)
class PreparedChatTurn:
    """Подготовленный ход до фактического вызова LLM."""

    request: ChatTurnRequest
    llm_messages: list[dict[str, str]]
    retrieved_context: RetrievedContext


@dataclass(slots=True)
class ChatTurnResponse:
    """Структурированный результат завершённого хода."""

    user_message: str
    assistant_message: str
    llm_messages: list[dict[str, str]]
    retrieved_context: RetrievedContext
    streamed: bool = False


@dataclass(slots=True)
class ChatStreamEvent:
    """Событие потокового сценария."""

    kind: StreamEventKind
    text: str = ""
    payload: Any = None
    response: ChatTurnResponse | None = None
