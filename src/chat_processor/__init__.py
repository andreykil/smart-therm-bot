"""
Chat Processor — Обработка чата SmartTherm для RAG

Модуль для преобразования Telegram чата в RAG чанки.

Использование:
    from src.chat_processor import run_stage0, run_stage1, run_stage2, run_stage3
    from src.utils.config import Config

    config = Config.load()
    run_stage0(config)
    run_stage1(config)
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
from .stage0_filter import run_stage0
from .stage1_threads import run_stage1
from .stage2_dedup import run_stage2
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
    "run_stage0",
    "run_stage1",
    "run_stage2",
    "run_stage3",
]
