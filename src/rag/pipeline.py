"""
RAG Pipeline — объединяет все RAG компоненты
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.rag.models import RAGChunk, SearchResult, Query, IndexStats
from src.rag.embedder import BgeM3Embedder
from src.rag.vector_store import VectorStore
from src.rag.bm25_store import BM25Store
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import Reranker, NoOpReranker

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Полный RAG пайплайн для поиска по документам.

    Объединяет:
    - BgeM3Embedder для генерации эмбеддингов
    - VectorStore (FAISS) для семантического поиска
    - BM25Store для текстового поиска
    - HybridRetriever для объединения результатов
    - Reranker для улучшения порядка (опционально)

    Использование:
    ```python
    pipeline = RAGPipeline.from_config()

    # Индексация (один раз)
    pipeline.index_chunks(chunks)

    # Поиск
    results = pipeline.search("Как подключить датчик температуры?")
    ```

    Args:
        embedder: BgeM3Embedder
        vector_store: VectorStore
        bm25_store: BM25Store
        hybrid_retriever: HybridRetriever
        data_dir: Директория для сохранения индексов
    """

    def __init__(
        self,
        embedder: BgeM3Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        hybrid_retriever: HybridRetriever,
        data_dir: str = "data"
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.hybrid_retriever = hybrid_retriever
        self.data_dir = Path(data_dir)

        self._indices_loaded = False

    @classmethod
    def from_config(
        cls,
        config: Optional[dict] = None,
        data_dir: str = "data",
        ollama_base_url: str = "http://localhost:11434",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_k: int = 5,
        reranker_type: str = "no-op"
    ) -> "RAGPipeline":
        """
        Создать RAGPipeline из конфигурации.

        Args:
            config: Конфигурация (если None — загружается из default.yaml)
            data_dir: Директория данных
            ollama_base_url: URL Ollama
            vector_weight: Вес FAISS
            bm25_weight: Вес BM25
            top_k: Количество результатов
            reranker_type: Тип reranker ('no-op', 'bge', 'cross-encoder')
        """
        if config is None:
            from src.utils.config import Config
            config = Config.load().model_dump()

        # RAG config
        rag_config = config.get("rag", {})
        embedding_model = rag_config.get("embedding_model", "bge-m3")

        # Пути к индексам
        indices_dir = Path(data_dir) / "indices"
        faiss_dir = indices_dir / "faiss"
        bm25_dir = indices_dir / "bm25"

        # Создаём компоненты
        embedder = BgeM3Embedder(
            model=embedding_model,
            base_url=ollama_base_url
        )

        vector_store = VectorStore(
            embedder=embedder,
            index_path=str(faiss_dir),
            metric="cosine"
        )

        bm25_store = BM25Store(
            index_path=str(bm25_dir)
        )

        # Reranker
        reranker_kwargs = {}
        if reranker_type == "bge":
            reranker_kwargs = {"base_url": ollama_base_url}
        reranker = _create_reranker(reranker_type, **reranker_kwargs)

        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            reranker=reranker,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=top_k
        )

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            bm25_store=bm25_store,
            hybrid_retriever=hybrid_retriever,
            data_dir=data_dir
        )

    def index_chunks(
        self,
        chunks: list[RAGChunk],
        save: bool = True
    ) -> IndexStats:
        """
        Проиндексировать чанки.

        Args:
            chunks: Список RAG чанков
            save: Сохранить индексы на диск

        Returns:
            Статистика индекса
        """
        logger.info(f"Индексация {len(chunks)} чанков...")

        # Добавляем в оба индекса
        self.vector_store.add_chunks(chunks)
        self.bm25_store.add_chunks(chunks)

        # Сохраняем если нужно
        if save:
            self.vector_store.save()
            self.bm25_store.save()
            logger.info("Индексы сохранены на диск")

        self._indices_loaded = True

        return IndexStats(
            total_chunks=len(chunks),
            faiss_vectors=self.vector_store.size,
            bm25_documents=self.bm25_store.size,
            embedding_dim=self.embedder.embedding_dim
        )

    def index_from_file(
        self,
        chunks_file: str = "data/processed/chat/chunks_rag.jsonl",
        save: bool = True
    ) -> IndexStats:
        """
        Проиндексировать чанки из JSONL файла.

        Args:
            chunks_file: Путь к файлу с чанками
            save: Сохранить индексы на диск

        Returns:
            Статистика индекса
        """
        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Файл не найден: {chunks_file}")

        logger.info(f"Загрузка чанков из {chunks_file}...")
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = RAGChunk(**data)
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Ошибка парсинга строки {line_num}: {e}")

        logger.info(f"Загружено {len(chunks)} чанков")
        return self.index_chunks(chunks, save=save)

    def search(
        self,
        query: str | Query,
        top_k: Optional[int] = None,
        use_reranker: bool = False
    ) -> SearchResult:
        """
        Найти релевантные чанки.

        Args:
            query: Строка запроса или Query объект
            top_k: Количество результатов
            use_reranker: Использовать ли reranker

        Returns:
            SearchResult с чанками
        """
        query_text = query.text if isinstance(query, Query) else query

        # Загружаем индексы если нужно
        if not self._indices_loaded:
            loaded = self.ensure_indices_loaded()
            if not loaded:
                return SearchResult(
                    chunks=[],
                    query=query_text,
                    total_found=0,
                    reranked=False,
                )

        return self.hybrid_retriever.search(
            query=query,
            top_k=top_k,
            use_reranker=use_reranker
        )

    def ensure_indices_loaded(self) -> bool:
        """Публично обеспечить загрузку индексов перед retrieval."""
        if self._indices_loaded:
            return True
        return self._load_indices()

    def _load_indices(self) -> bool:
        """Загрузить индексы с диска"""
        try:
            self.vector_store.load()
            self.bm25_store.load()
            self._indices_loaded = True
            logger.info("Индексы загружены с диска")
            return True
        except FileNotFoundError:
            logger.warning("Индексы не найдены на диске. Сначала вызовите index_chunks()")
            self._indices_loaded = False
            return False

    def get_stats(self) -> dict:
        """Получить статистику индексов"""
        return {
            "total_chunks": len(self.vector_store),
            "faiss_vectors": self.vector_store.size,
            "bm25_documents": self.bm25_store.size,
            "embedding_dim": self.embedder.embedding_dim,
            "indices_loaded": self._indices_loaded,
            "reranker": self.hybrid_retriever.reranker.name,
            "weights": {
                "vector": self.hybrid_retriever.vector_weight,
                "bm25": self.hybrid_retriever.bm25_weight
            }
        }

    def __repr__(self) -> str:
        return (
            f"RAGPipeline("
            f"chunks={self.vector_store.size}, "
            f"dim={self.embedder.embedding_dim}, "
            f"reranker={self.hybrid_retriever.reranker.name})"
        )


def _create_reranker(reranker_type: str, **kwargs) -> Reranker:
    """Создать reranker по типу"""
    from src.rag.reranker import create_reranker as factory
    return factory(reranker_type, **kwargs)
