"""Domain-модели диалога, памяти и retrieval-контекста чата."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

DialogRole = Literal["user", "assistant"]


def utc_now_iso() -> str:
    """Получить текущее время в UTC в ISO-формате."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DialogMessage:
    """Каноническое сообщение диалога."""

    role: DialogRole
    content: str
    created_at: str = field(default_factory=utc_now_iso)
    rag_enabled: bool = False
    rag_query: str | None = None
    rag_total_found: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "rag_enabled": self.rag_enabled,
            "rag_query": self.rag_query,
            "rag_total_found": self.rag_total_found,
            "metadata": self.metadata,
        }


def build_turn_messages(
    *,
    user_message: str,
    assistant_message: str,
    rag_enabled: bool,
    rag_query: str | None,
    rag_total_found: int,
    user_metadata: Mapping[str, Any] | None = None,
    assistant_metadata: Mapping[str, Any] | None = None,
) -> list[DialogMessage]:
    """Собрать canonical user/assistant сообщения одного завершенного хода."""
    return [
        DialogMessage(
            role="user",
            content=user_message,
            rag_enabled=rag_enabled,
            rag_query=rag_query,
            rag_total_found=rag_total_found,
            metadata=dict(user_metadata or {}),
        ),
        DialogMessage(
            role="assistant",
            content=assistant_message,
            rag_enabled=rag_enabled,
            rag_query=rag_query,
            rag_total_found=rag_total_found,
            metadata=dict(assistant_metadata or {}),
        ),
    ]


@dataclass(slots=True, frozen=True)
class DialogMemoryFact:
    """Один сохранённый факт о пользователе или диалоге."""

    key: str
    value: str
    updated_at: str

    @staticmethod
    def normalize_key(key: str) -> str:
        normalized = key.strip().lower()
        if not normalized:
            raise ValueError("memory fact key must be non-empty")
        return normalized

    @staticmethod
    def normalize_value(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory fact value must be non-empty")
        return normalized

    @classmethod
    def create(cls, key: str, value: str, *, updated_at: str | None = None) -> "DialogMemoryFact":
        return cls(
            key=cls.normalize_key(key),
            value=cls.normalize_value(value),
            updated_at=updated_at or utc_now_iso(),
        )


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """Chat-owned retrieval chunk без зависимости от layout инфраструктуры."""

    text: str
    source: str = "telegram chat"
    tags: tuple[str, ...] = ()
    version: str | None = None
    confidence: float | None = None
    code: str = ""

    def to_context_string(self) -> str:
        tags_str = ", ".join(self.tags) if self.tags else "без тегов"
        version_str = f" (v{self.version})" if self.version else ""
        confidence_pct = int((self.confidence or 0.0) * 100)
        base = (
            f"[#{confidence_pct}%] {self.text}\n"
            f"Теги: {tags_str}{version_str}\n"
            f"Источник: {self.source}\n"
            f"---\n"
        )
        if self.code:
            return f"{base}{self.code}\n"
        return base


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Chat-owned retrieval result, пригодный для orchestration-слоя."""

    query: str
    chunks: tuple[RetrievedChunk, ...] = ()
    total_found: int = 0
    reranked: bool = False

    def to_context_string(self) -> str:
        if not self.chunks:
            return "Нет релевантной информации."

        context_parts = []
        for index, chunk in enumerate(self.chunks, 1):
            context_parts.append(f"\n--- Источник {index} ---\n{chunk.to_context_string()}")

        return "\n".join(context_parts)
