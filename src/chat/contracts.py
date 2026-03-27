"""Общие контракты chat-layer для LLM и RAG зависимостей."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from src.rag.models import SearchResult


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
class ChatRAGPipeline(Protocol):
    """Минимальный контракт retrieval pipeline для chat-layer."""

    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False) -> SearchResult: ...

    def get_stats(self) -> dict: ...
