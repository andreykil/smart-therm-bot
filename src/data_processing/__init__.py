"""
Data Processing — обработка данных чата для RAG

Модуль для преобразования Telegram чата в RAG чанки.

Использование:
    from src.data_processing import run_filtering, run_chunks
    from src.utils.config import Config

    config = Config.load()
    run_filtering(config)
    run_chunks(config, llm)
"""

from .models import (
    TelegramMessage,
    TelegramChat,
    FilteredMessage,
    Group,
    RAGChunk,
    ChunkContent,
    ChunkMetadata
)
from .chat_filtering import run_filtering
from .chat_chunks import run_chunks

__all__ = [
    # Модели данных
    "TelegramMessage",
    "TelegramChat",
    "FilteredMessage",
    "Group",
    "RAGChunk",
    "ChunkContent",
    "ChunkMetadata",

    # Функции
    "run_filtering",
    "run_chunks",
]
