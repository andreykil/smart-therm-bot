"""
Pydantic модели данных для обработки чата
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
# Промежуточные данные
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


class Group(BaseModel):
    """Группа сообщений"""
    group_id: str
    message_ids: list[int]
    start_date: str
    end_date: str
    participant_count: int = 1


# =============================================================================
# Чанки для RAG
# =============================================================================

class ChunkContent(BaseModel):
    """Содержимое чанка"""
    text: str = Field(..., min_length=10, description="Основной текст чанка")
    code: str = Field(default="", description="Извлечённый код/команды")


class ChunkMetadata(BaseModel):
    """Метаданные чанка"""
    source: str = Field(default="telegram chat", description="Источник данных")
    date: str = Field(..., description="Дата последнего сообщения в группе (YYYY-MM-DD)")
    tags: list[str] = Field(default_factory=list)
    version: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class RAGChunk(BaseModel):
    """RAG чанк для векторного поиска"""
    content: ChunkContent
    metadata: ChunkMetadata

    def to_jsonl(self) -> str:
        """Конвертировать в JSONL строку"""
        return self.model_dump_json(exclude_none=True)

    def to_text(self) -> str:
        """Конвертировать в текстовый формат для индексации"""
        return f"{self.content.text} {self.content.code}".strip()
