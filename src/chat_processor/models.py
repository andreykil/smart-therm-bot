"""
Pydantic модели данных для процессора чата
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Входные данные (Telegram JSON)
# =============================================================================

class TelegramMessage(BaseModel):
    """Сообщение из Telegram экспорта"""
    id: int
    type: str = "message"
    date: str
    date_unixtime: str
    from_: Optional[str] = Field(None, alias="from")
    from_id: Optional[str] = None
    text: Optional[list | str] = None
    text_entities: Optional[list] = None
    reply_to_message_id: Optional[int] = None
    
    class Config:
        populate_by_name = True


class TelegramChat(BaseModel):
    """Telegram чат"""
    name: str
    type: str
    id: int
    messages: list[TelegramMessage]


# =============================================================================
# Промежуточные данные (Этап 0)
# =============================================================================

class FilteredMessage(BaseModel):
    """Отфильтрованное сообщение"""
    id: int
    date: str
    date_unixtime: int
    from_: str = Field(alias="from")
    text: str
    reply_to_message_id: Optional[int] = None
    is_from_developer: bool = False

    class Config:
        populate_by_name = True


# =============================================================================
# Ветки (Этап 1)
# =============================================================================

class Thread(BaseModel):
    """Ветка обсуждения"""
    thread_id: str
    topic: str
    message_ids: list[int]
    start_date: str
    end_date: str
    participant_count: int = Field(default=1)


class ThreadsResult(BaseModel):
    """Результат выделения веток"""
    threads: list[Thread]
    group_number: int
    total_messages: int


# =============================================================================
# Чанки для RAG (Этап 3)
# =============================================================================

class ChunkSource(BaseModel):
    """Источник чанка"""
    type: str = "telegram"
    message_ids: list[int]
    date_range: str


class ChunkContent(BaseModel):
    """Содержимое чанка"""
    summary: str = Field(max_length=500)
    text: str = Field(min_length=10, max_length=3000)


class ChunkMetadata(BaseModel):
    """Метаданные чанка"""
    tags: list[str]
    version: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class RAGChunk(BaseModel):
    """RAG чанк для векторного поиска"""
    chunk_id: str
    source: ChunkSource
    content: ChunkContent
    metadata: ChunkMetadata
    
    def to_jsonl(self) -> str:
        """Конвертировать в JSONL строку"""
        import json
        return self.model_dump_json(exclude_none=True)
    
    @classmethod
    def from_thread(cls, thread: Thread) -> "RAGChunk":
        """Создать чанк из ветки (заглушка)"""
        return cls(
            chunk_id=f"tg_{thread.thread_id}",
            source=ChunkSource(
                message_ids=thread.message_ids,
                date_range=f"{thread.start_date} — {thread.end_date}"
            ),
            content=ChunkContent(
                summary=thread.topic,
                text=thread.topic
            ),
            metadata=ChunkMetadata(
                tags=[],
                confidence=0.5
            )
        )
