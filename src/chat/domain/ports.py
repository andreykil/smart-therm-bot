"""Доменные порты chat-подсистемы."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from .models import DialogMemoryFact, DialogMessage, RetrievalResult


@runtime_checkable
class ChatModelClient(Protocol):
    """Минимальный контракт LLM-клиента для chat-layer."""

    model: str

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        think: bool | None = None,
    ) -> str: ...

    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        think: bool | None = None,
    ) -> Iterable[str]: ...

    def load(self, strict: bool = True) -> None: ...

    def get_stats(self) -> dict: ...


@runtime_checkable
class ChatContextRetriever(Protocol):
    """Минимальный контракт retrieval runtime для chat-layer."""

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = False,
    ) -> RetrievalResult: ...

    def get_stats(self) -> dict[str, object]: ...


@runtime_checkable
class DialogState(Protocol):
    """Единый порт состояния диалога для application-слоя."""

    def recent_messages(self) -> list[DialogMessage]: ...

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
    ) -> None: ...

    def clear(self) -> None: ...

    def list_facts(self) -> list[DialogMemoryFact]: ...

    def remember_fact(self, key: str, value: str) -> DialogMemoryFact: ...

    def forget_fact(self, key: str) -> bool: ...

    def stats(self) -> dict[str, int]: ...
