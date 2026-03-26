"""
BM25 Store — классический текстовый поиск через rank_bm25
"""

import json
import logging
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from src.rag.models import RAGChunk, RetrievalResult

logger = logging.getLogger(__name__)


class BM25Store:
    """
    BM25 текстовый индекс для классического поиска.

    BM25 — это probabilistic relevance model, хорошо работает для
    коротких запросов и точного совпадения ключевых слов.

    Поддерживает:
    - Создание индекса из списка чанков
    - Сохранение/загрузку индекса на диск
    - Поиск по ключевым словам

    Args:
        index_path: Путь для сохранения индекса (опционально)
        k1: BM25 k1 параметр (по умолчанию 1.5)
        b: BM25 b параметр (по умолчанию 0.75)
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.index_path = Path(index_path) if index_path else None
        self.k1 = k1
        self.b = b
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[RAGChunk] = []
        self._tokenized_texts: list[list[str]] = []

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

        # Получаем тексты и токенизируем
        texts = [chunk.to_text() for chunk in chunks]
        tokenized = [self._tokenize(text) for text in texts]

        # BM25Okapi не поддерживает инкрементальное добавление,
        # поэтому обновляем корпус и пересоздаём индекс целиком.
        self._chunks.extend(chunks)
        self._tokenized_texts.extend(tokenized)
        self._bm25 = BM25Okapi(self._tokenized_texts, k1=self.k1, b=self.b)

        logger.info(f"Добавлено {len(chunks)} чанков в BM25 индекс (всего: {len(self._chunks)})")
        return len(chunks)

    def _tokenize(self, text: str) -> list[str]:
        """
        Токенизация текста.

        Используем простой regex-based токенизатор.
        Для русского языка можно использовать pymorphy2 или stanza.

        Args:
            text: Входной текст

        Returns:
            Список токенов
        """
        import re
        # Удаляем пунктуацию и приводим к lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Разбиваем по пробелам
        tokens = text.split()
        # Фильтруем слишком короткие токены
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Найти релевантные чанки по запросу.

        Args:
            query: Текст запроса
            top_k: Количество результатов

        Returns:
            Список RetrievalResult отсортированных по релевантности
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 индекс пуст")
            return []

        # Токенизируем запрос
        tokenized_query = self._tokenize(query)

        # Получаем scores
        scores = self._bm25.get_scores(tokenized_query)

        # Получаем топ-k индексов
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Формируем результаты
        results = []
        for rank, idx in enumerate(top_indices):
            score = scores[idx]
            if score > 0:  # Фильтруем нулевые результаты
                results.append(RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    source="bm25",
                    rank=rank
                ))

        return results

    def save(self, path: Optional[str] = None) -> None:
        """
        Сохранить индекс и чанки на диск.

        Args:
            path: Путь для сохранения (по умолчанию self.index_path)
        """
        if self._bm25 is None:
            logger.warning("Нечего сохранять - индекс не инициализирован")
            return

        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise ValueError("Не указан путь для сохранения")

        save_path.mkdir(parents=True, exist_ok=True)

        # Сохраняем только безопасный JSON-формат
        chunks_data = [chunk.model_dump() for chunk in self._chunks]
        with open(save_path / "bm25.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": chunks_data,
                    "tokenized_texts": self._tokenized_texts,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"BM25 индекс сохранён: {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """
        Загрузить индекс и чанки с диска.

        Args:
            path: Путь для загрузки (по умолчанию self.index_path)
        """
        load_path = Path(path) if path else self.index_path
        if load_path is None:
            raise ValueError("Не указан путь для загрузки")

        index_file = load_path / "bm25.json"

        if not index_file.exists():
            raise FileNotFoundError(f"BM25 файл не найден: {load_path}")

        # Загружаем BM25 данные и пересобираем индекс
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._chunks = [RAGChunk(**chunk_data) for chunk_data in data.get("chunks", [])]
        self._tokenized_texts = data.get("tokenized_texts", [])
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        self._bm25 = BM25Okapi(self._tokenized_texts, k1=self.k1, b=self.b)

        logger.info(f"BM25 индекс загружен: {load_path} ({len(self._chunks)} чанков)")

    @property
    def size(self) -> int:
        """Количество чанков в индексе"""
        return len(self._chunks)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"BM25Store(size={self.size}, k1={self.k1}, b={self.b})"
