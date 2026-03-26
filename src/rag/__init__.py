"""
RAG система для SmartTherm Bot.

Модуль RAG обеспечивает:
- Генерацию эмбеддингов через BGE-M3
- FAISS векторный поиск
- BM25 текстовый поиск
- Гибридное объединение результатов
- Reranker (заготовка)

Пример использования:
    from src.rag import RAGPipeline

    pipeline = RAGPipeline.from_config()
    pipeline.index_from_file("data/processed/chat/chunks_rag.jsonl")

    results = pipeline.search("Как подключить датчик DS18B20?")
    print(results.to_context_string())
"""

from src.rag.models import (
    ChunkMetadata,
    ChunkContent,
    RAGChunk,
    RetrievalResult,
    RerankedResult,
    Query,
    SearchResult,
    IndexStats,
)

from src.rag.embedder import BgeM3Embedder
from src.rag.vector_store import VectorStore
from src.rag.bm25_store import BM25Store
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import (
    Reranker,
    NoOpReranker,
    BGEReranker,
    CrossEncoderReranker,
    create_reranker,
)
from src.rag.pipeline import RAGPipeline

__all__ = [
    # Models
    "ChunkMetadata",
    "ChunkContent",
    "RAGChunk",
    "RetrievalResult",
    "RerankedResult",
    "Query",
    "SearchResult",
    "IndexStats",
    # Core components
    "BgeM3Embedder",
    "VectorStore",
    "BM25Store",
    "HybridRetriever",
    # Reranker
    "Reranker",
    "NoOpReranker",
    "BGEReranker",
    "CrossEncoderReranker",
    "create_reranker",
    # Pipeline
    "RAGPipeline",
]
