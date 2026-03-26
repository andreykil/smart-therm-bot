"""
FAISS Vector Store — векторный поиск через FAISS
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.rag.models import RAGChunk, RetrievalResult
from src.rag.embedder import BgeM3Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS векторный индекс для семантического поиска.

    Поддерживает:
    - Создание индекса из списка чанков
    - Сохранение/загрузку индекса на диск
    - Поиск ближайших соседей

    Args:
        embedder: BgeM3Embedder для генерации эмбеддингов
        index_path: Путь для сохранения индекса (опционально)
        metric: Метрика близости ('cosine' или 'euclidean')
    """

    def __init__(
        self,
        embedder: BgeM3Embedder,
        index_path: Optional[str] = None,
        metric: str = "cosine"
    ):
        self.embedder = embedder
        self.index_path = Path(index_path) if index_path else None
        self.metric = metric
        self._index = None
        self._chunks: list[RAGChunk] = []
        self._id_to_chunk: dict[int, RAGChunk] = {}

    def _init_faiss(self, dim: int) -> None:
        """Инициализировать FAISS индекс"""
        import faiss

        if self.metric == "cosine":
            # Inner Product с нормализованными векторами эквивалентен cosine similarity
            self._index = faiss.IndexFlatIP(dim)
        elif self.metric == "euclidean":
            self._index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        logger.info(f"FAISS индекс инициализирован: metric={self.metric}, dim={dim}")

    def add_chunks(self, chunks: list[RAGChunk]) -> int:
        """
        Добавить чанки в индекс.

        Args:
            chunks: Список RAG чанков

        Returns:
            Количество добавленных чанков
        """
        if not chunks:
            return 0

        # Получаем тексты для эмбеддинга
        texts = [chunk.to_text() for chunk in chunks]

        # Генерируем эмбеддинги
        logger.info(f"Генерация эмбеддингов для {len(chunks)} чанков...")
        embeddings = self.embedder.embed(texts)

        # FAISS ожидает 2D массив (N, dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Инициализируем индекс если нужно
        if self._index is None:
            self._init_faiss(embeddings.shape[1])
        
        assert self._index is not None, "FAISS index must be initialized"

        # Добавляем в индекс
        start_id = len(self._chunks)
        self._index.add(embeddings.astype(np.float32))  # type: ignore[call-arg]

        # Сохраняем маппинг id -> chunk
        for i, chunk in enumerate(chunks):
            self._chunks.append(chunk)
            self._id_to_chunk[start_id + i] = chunk

        logger.info(f"Добавлено {len(chunks)} чанков в FAISS индекс (всего: {self._index.ntotal})")
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Найти ближайшие чанки к запросу.

        Args:
            query: Текст запроса
            top_k: Количество результатов

        Returns:
            Список RetrievalResult отсортированных по релевантности
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS индекс пуст")
            return []

        # Генерируем эмбеддинг запроса
        query_vector = self.embedder.embed_query(query)

        # FAISS ожидает 2D массив (1, dim)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ищем
        assert self._index is not None, "FAISS index must be initialized"
        top_k = min(top_k, self._index.ntotal)  # type: ignore[attr-defined]
        scores, indices = self._index.search(query_vector.astype(np.float32), top_k)  # type: ignore[call-arg]

        # Формируем результаты
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            chunk = self._id_to_chunk.get(int(idx))
            if chunk:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    source="faiss",
                    rank=rank
                ))

        return results

    def save(self, path: Optional[str] = None) -> None:
        """
        Сохранить индекс и чанки на диск.

        Args:
            path: Путь для сохранения (по умолчанию self.index_path)
        """
        if self._index is None:
            logger.warning("Нечего сохранять - индекс не инициализирован")
            return

        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise ValueError("Не указан путь для сохранения")

        save_path.mkdir(parents=True, exist_ok=True)

        # Сохраняем FAISS индекс
        import faiss
        faiss.write_index(self._index, str(save_path / "index.faiss"))

        # Сохраняем чанки
        chunks_data = [chunk.model_dump() for chunk in self._chunks]
        with open(save_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"FAISS индекс сохранён: {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """
        Загрузить индекс и чанки с диска.

        Args:
            path: Путь для загрузки (по умолчанию self.index_path)
        """
        import faiss

        load_path = Path(path) if path else self.index_path
        if load_path is None:
            raise ValueError("Не указан путь для загрузки")

        index_file = load_path / "index.faiss"
        chunks_file = load_path / "chunks.json"

        if not index_file.exists() or not chunks_file.exists():
            raise FileNotFoundError(f"Файлы индекса не найдены: {load_path}")

        # Загружаем FAISS индекс
        self._index = faiss.read_index(str(index_file))

        # Загружаем чанки
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self._chunks = [RAGChunk(**chunk_data) for chunk_data in chunks_data]
        self._id_to_chunk = {i: chunk for i, chunk in enumerate(self._chunks)}

        logger.info(f"FAISS индекс загружен: {load_path} ({len(self._chunks)} чанков)")

    @property
    def size(self) -> int:
        """Количество чанков в индексе"""
        return len(self._chunks)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"VectorStore(size={self.size}, metric={self.metric})"
