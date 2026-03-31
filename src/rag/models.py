"""
Pydantic модели для RAG системы
"""

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Метаданные RAG чанка"""
    source: str = Field(default="telegram chat", description="Источник данных")
    date: str = Field(..., description="Дата последнего сообщения")
    tags: list[str] = Field(default_factory=list, description="Теги чанка")
    version: Optional[str] = Field(None, description="Версия прошивки/софта")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Уверенность в качестве")


class ChunkContent(BaseModel):
    """Контент RAG чанка"""
    text: str = Field(..., description="Полный текст чанка")
    code: str = Field(default="", description="Извлечённый код/команды")


class RAGChunk(BaseModel):
    """RAG чанк из chat_chunks.py"""
    content: ChunkContent
    metadata: ChunkMetadata

    def to_text(self) -> str:
        """Конвертировать чанк в текст для индексации"""
        return f"{self.content.text} {self.content.code}".strip()

    def to_context_string(self) -> str:
        """Форматировать чанк как контекст для LLM"""
        tags_str = ", ".join(self.metadata.tags) if self.metadata.tags else "без тегов"
        version_str = f" (v{self.metadata.version})" if self.metadata.version else ""
        confidence_pct = int(self.metadata.confidence * 100)
        base = (
            f"[#{confidence_pct}%] {self.content.text}\n"
            f"Теги: {tags_str}{version_str}\n"
            f"---\n"
        )
        if self.content.code:
            return f"{base}{self.content.code}\n"
        return base


class RetrievalResult(BaseModel):
    """Результат поиска (до reranking)"""
    chunk: RAGChunk
    score: float = Field(..., description="Скоринг от поисковика (0-1)")
    source: str = Field(..., description="Источник: 'faiss' или 'bm25'")
    rank: int = Field(0, description="Ранг в результатах этого источника")

    def __hash__(self):
        return hash((self.chunk.content.text, self.source))


class Query(BaseModel):
    """Запрос к RAG системе"""
    text: str = Field(..., min_length=1, description="Текст запроса")
    top_k: int = Field(default=5, ge=1, le=100, description="Количество результатов")
    tags: Optional[list[str]] = Field(default=None, description="Фильтр по тегам")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Мин. уверенность")


class SearchResult(BaseModel):
    """Итоговый результат поиска"""
    chunks: list[RAGChunk] = Field(default_factory=list)
    query: str = Field(..., description="Оригинальный запрос")
    total_found: int = Field(0, description="Всего найдено")
    reranked: bool = Field(False, description="Был ли применен reranking")

    def to_context_string(self) -> str:
        """Форматировать результаты как контекст для LLM"""
        if not self.chunks:
            return "Нет релевантной информации."

        context_parts = []
        for i, chunk in enumerate(self.chunks, 1):
            context_parts.append(f"\n--- Источник {i} ---\n{chunk.to_context_string()}")

        return "\n".join(context_parts)


@dataclass
class IndexStats:
    """Статистика индекса"""
    total_chunks: int = 0
    faiss_vectors: int = 0
    bm25_documents: int = 0
    embedding_dim: int = 0
