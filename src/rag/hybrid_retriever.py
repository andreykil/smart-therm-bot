"""
Hybrid Retriever — объединяет FAISS + BM25 с поддержкой reranker
"""

import logging
from typing import Optional

from src.rag.models import (
    RAGChunk,
    RetrievalResult,
    RerankedResult,
    Query,
    SearchResult
)
from src.rag.vector_store import VectorStore
from src.rag.bm25_store import BM25Store
from src.rag.reranker import Reranker, NoOpReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Гибридный поисковик, объединяющий FAISS и BM25.

    Поддерживает:
    - Взвешенное объединение результатов FAISS и BM25
    - Deduplication результатов
    - Интеграция с reranker для улучшения результатов

    Args:
        vector_store: FAISS VectorStore
        bm25_store: BM25 Store
        reranker: Reranker для улучшения результатов (опционально)
        vector_weight: Вес FAISS результатов (0-1)
        bm25_weight: Вес BM25 результатов (0-1)
        top_k: Количество результатов для поиска
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        reranker: Optional[Reranker] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker or NoOpReranker()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k

        # Нормализуем веса
        total = vector_weight + bm25_weight
        if total > 0:
            self.vector_weight = vector_weight / total
            self.bm25_weight = bm25_weight / total

    def search(
        self,
        query: str | Query,
        top_k: Optional[int] = None,
        use_reranker: bool = True
    ) -> SearchResult:
        """
        Найти релевантные чанки.

        Args:
            query: Строка запроса или Query объект
            top_k: Количество результатов (по умолчанию self.top_k)
            use_reranker: Использовать ли reranker

        Returns:
            SearchResult с найденными чанками
        """
        # Парсим запрос
        if isinstance(query, str):
            query_obj = Query(text=query, top_k=top_k or self.top_k)
        else:
            query_obj = query
            if top_k is not None:
                query_obj.top_k = top_k

        # Ищем в обоих индексах
        faiss_results = self.vector_store.search(query_obj.text, top_k=query_obj.top_k)
        bm25_results = self.bm25_store.search(query_obj.text, top_k=query_obj.top_k)

        logger.debug(f"FAISS: {len(faiss_results)}, BM25: {len(bm25_results)}")

        # Объединяем результаты с взвешиванием
        combined = self._combine_results(faiss_results, bm25_results)

        # Deduplication по summary
        deduplicated = self._deduplicate(combined)

        # Применяем фильтры
        filtered = self._apply_filters(deduplicated, query_obj)

        # Reranking если нужен
        reranked = use_reranker and self.reranker is not None
        if reranked and filtered:
            results = self._apply_reranking(filtered, query_obj.text)
        else:
            results = filtered

        # Финальное ограничение по top_k
        final_results = results[:query_obj.top_k]
        
        return SearchResult(
            chunks=[r.chunk for r in final_results],
            query=query_obj.text,
            total_found=len(final_results),
            reranked=reranked
        )

    def _combine_results(
        self,
        faiss_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Объединить результаты FAISS и BM25 с взвешиванием.

        Для RRF (Reciprocal Rank Fusion):
        score = w1 * norm_faiss + w2 * norm_bm25

        Returns:
            Объединённый список RetrievalResult
        """
        combined_map: dict[str, RetrievalResult] = {}

        # Нормализуем и добавляем FAISS результаты
        faiss_scores = [r.score for r in faiss_results]
        faiss_norm = self._normalize_scores(faiss_scores)
        for result, norm_score in zip(faiss_results, faiss_norm):
            key = self._chunk_key(result.chunk)
            combined_map[key] = RetrievalResult(
                chunk=result.chunk,
                score=self.vector_weight * norm_score,
                source="hybrid",
                rank=result.rank
            )

        # Нормализуем и добавляем BM25 результаты
        bm25_scores = [r.score for r in bm25_results]
        bm25_norm = self._normalize_scores(bm25_scores)
        for result, norm_score in zip(bm25_results, bm25_norm):
            key = self._chunk_key(result.chunk)
            if key in combined_map:
                # Увеличиваем существующий score
                existing = combined_map[key]
                existing.score += self.bm25_weight * norm_score
            else:
                combined_map[key] = RetrievalResult(
                    chunk=result.chunk,
                    score=self.bm25_weight * norm_score,
                    source="hybrid",
                    rank=result.rank
                )

        # Сортируем по score
        results = list(combined_map.values())
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        """Min-max нормализация в диапазон [0, 1]."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)
        scale = max_score - min_score

        if scale <= 1e-12:
            return [1.0 for _ in scores]

        return [(score - min_score) / scale for score in scores]

    def _chunk_key(self, chunk: RAGChunk) -> str:
        """Создать ключ для deduplication"""
        return chunk.content.text[:100]

    def _deduplicate(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Удалить дубликаты по summary.

        Returns:
            Deduplicated список
        """
        seen = set()
        unique = []

        for result in results:
            key = self._chunk_key(result.chunk)
            if key not in seen:
                seen.add(key)
                unique.append(result)

        return unique

    def _apply_filters(
        self,
        results: list[RetrievalResult],
        query: Query
    ) -> list[RetrievalResult]:
        """
        Применить фильтры (tags, min_confidence).

        Returns:
            Отфильтрованный список
        """
        filtered = results

        # Фильтр по тегам
        if query.tags:
            filtered = [
                r for r in filtered
                if any(tag in r.chunk.metadata.tags for tag in query.tags)
            ]

        # Фильтр по confidence
        if query.min_confidence > 0:
            filtered = [
                r for r in filtered
                if r.chunk.metadata.confidence >= query.min_confidence
            ]

        return filtered

    def _apply_reranking(
        self,
        results: list[RetrievalResult],
        query: str
    ) -> list[RetrievalResult]:
        """
        Применить reranking.

        Args:
            results: Результаты после объединения
            query: Оригинальный запрос

        Returns:
            Reranked результаты
        """
        if not results:
            return results

        # Подготовка данных для reranker
        reranked_results = self.reranker.rerank(
            query=query,
            results=results
        )

        return reranked_results

    @property
    def size(self) -> int:
        """Общее количество чанков"""
        return len(self.vector_store) + len(self.bm25_store)

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"faiss={self.vector_store.size}, "
            f"bm25={self.bm25_store.size}, "
            f"weights=({self.vector_weight:.2f}, {self.bm25_weight:.2f}))"
        )
