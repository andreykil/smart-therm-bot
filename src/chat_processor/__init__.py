"""
Chat Processor — Обработка чата SmartTherm для RAG

Модуль для преобразования Telegram чата в RAG чанки.

Использование:
    from src.chat_processor import run_stage1, run_stage2, run_stage3
    from src.utils.config import Config

    config = Config.load()
    run_stage1(config)
    run_stage2(config, llm)
    run_stage3(config, llm)
"""

from .models import (
    TelegramMessage,
    TelegramChat,
    FilteredMessage,
    Thread,
    ThreadsResult,
    RAGChunk,
    ChunkSource,
    ChunkContent,
    ChunkMetadata
)
from .stage1_filter import run_stage1
from .stage2_threads import run_stage2
from .stage3_chunks import run_stage3

__all__ = [
    # Модели данных
    "TelegramMessage",
    "TelegramChat",
    "FilteredMessage",
    "Thread",
    "ThreadsResult",
    "RAGChunk",
    "ChunkSource",
    "ChunkContent",
    "ChunkMetadata",

    # Отдельные этапы
    "run_stage1",
    "run_stage2",
    "run_stage3",
]
