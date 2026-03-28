"""
Data Processing — обработка данных чата для RAG.

Использование:
    from src.config import Config
    from src.data_processing import run_filtering, run_chunks

    config = Config.load()
    run_filtering(config)
    run_chunks(config, llm)
"""

from .chat_chunks import run_chunks
from .chat_filtering import run_filtering
from .models import (
    ChunkContent,
    ChunkMetadata,
    FilteredMessage,
    Group,
    RAGChunk,
    TelegramChat,
    TelegramMessage,
)

__all__ = [
    "TelegramMessage",
    "TelegramChat",
    "FilteredMessage",
    "Group",
    "RAGChunk",
    "ChunkContent",
    "ChunkMetadata",
    "run_filtering",
    "run_chunks",
]
