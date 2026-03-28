"""Lifecycle менеджер индексов RAG-подсистемы."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.rag.bm25_store import BM25Store
from src.rag.embedder import BgeM3Embedder
from src.rag.models import IndexStats, RAGChunk
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexManager:
    """Управляет загрузкой, индексацией и сохранением RAG-индексов."""

    def __init__(
        self,
        *,
        embedder: BgeM3Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self._indices_loaded = False

    @property
    def indices_loaded(self) -> bool:
        return self._indices_loaded

    def load(self) -> None:
        self.vector_store.load()
        self.bm25_store.load()
        self._indices_loaded = True
        logger.info("Индексы загружены с диска")

    def ensure_loaded(self) -> bool:
        if self._indices_loaded:
            return True
        try:
            self.load()
            return True
        except FileNotFoundError:
            logger.warning("Индексы не найдены на диске. Сначала выполните индексацию")
            self._indices_loaded = False
            return False

    def save(self) -> None:
        self.vector_store.save()
        self.bm25_store.save()
        logger.info("Индексы сохранены на диск")

    def index_chunks(self, chunks: list[RAGChunk], *, save: bool = True) -> IndexStats:
        logger.info("Индексация %s чанков...", len(chunks))
        self.vector_store.add_chunks(chunks)
        self.bm25_store.add_chunks(chunks)
        self._indices_loaded = True
        if save:
            self.save()
        return self.get_stats()

    def index_from_file(self, chunks_file: str, *, save: bool = True) -> IndexStats:
        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Файл не найден: {chunks_file}")

        chunks: list[RAGChunk] = []
        with chunks_path.open("r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    chunks.append(RAGChunk(**json.loads(payload)))
                except Exception as error:
                    logger.warning("Ошибка парсинга строки %s: %s", line_num, error)

        logger.info("Загружено %s чанков", len(chunks))
        return self.index_chunks(chunks, save=save)

    def get_stats(self) -> IndexStats:
        return IndexStats(
            total_chunks=len(self.vector_store),
            faiss_vectors=self.vector_store.size,
            bm25_documents=self.bm25_store.size,
            embedding_dim=self.embedder.embedding_dim,
        )

    def as_dict(self) -> dict[str, object]:
        stats = self.get_stats()
        return {
            "total_chunks": stats.total_chunks,
            "faiss_vectors": stats.faiss_vectors,
            "bm25_documents": stats.bm25_documents,
            "embedding_dim": stats.embedding_dim,
            "indices_loaded": self._indices_loaded,
        }
