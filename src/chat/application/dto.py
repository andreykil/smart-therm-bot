"""DTO и структуры orchestration-слоя чата."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.chat.domain.models import RetrievalResult

StreamEventKind = Literal["debug", "token", "final"]
RetrievalErrorKind = Literal["retrieval_unavailable", "misconfiguration", "runtime_failure"]


@dataclass(slots=True)
class RetrievedContext:
    """Структурированное представление retrieval-результата для одного хода."""

    enabled: bool
    query: str | None = None
    result: RetrievalResult | None = None
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
