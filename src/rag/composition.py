"""Composition root для RAG-подсистемы."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config
from src.rag.bm25_store import BM25Store
from src.rag.embedder import BgeM3Embedder
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.index_manager import IndexManager
from src.rag.reranker import HuggingFaceReranker, Reranker
from src.rag.retrieval_service import RetrievalService
from src.rag.vector_store import VectorStore


@dataclass(slots=True)
class RAGRuntime:
    retrieval_service: RetrievalService
    index_manager: IndexManager


@dataclass(slots=True)
class RAGInitializationResult:
    retrieval_service: RetrievalService | None
    index_manager: IndexManager | None
    error: str | None = None


def _resolve_chunks_path(config: Config, chunks_file: str) -> str:
    return str(config.resolve_path(chunks_file))


def build_reranker(*, config: Config) -> Reranker:
    reranker_config = config.rag.reranker
    return HuggingFaceReranker(
        model_name=reranker_config.model,
        models_dir=config.models_dir_path,
        device=reranker_config.device,
        batch_size=reranker_config.batch_size,
        max_length=reranker_config.max_length,
        candidate_pool_size=reranker_config.candidate_pool_size,
    )


def build_rag_runtime(
    *,
    config: Config,
    base_url: str,
    top_k: int,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    reranker: Reranker | None = None,
) -> RAGRuntime:
    rag_config = config.rag
    faiss_dir = config.indices_dir / "faiss"
    bm25_dir = config.indices_dir / "bm25"

    embedder = BgeM3Embedder(model=rag_config.embedding_model, base_url=base_url)
    vector_store = VectorStore(embedder=embedder, index_path=str(faiss_dir), metric="cosine")
    bm25_store = BM25Store(index_path=str(bm25_dir))
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        reranker=reranker,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        top_k=top_k,
    )
    return RAGRuntime(
        retrieval_service=RetrievalService(hybrid_retriever=hybrid_retriever, default_top_k=top_k),
        index_manager=IndexManager(embedder=embedder, vector_store=vector_store, bm25_store=bm25_store),
    )


def initialize_retrieval_service(
    *,
    config: Config,
    base_url: str,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    chunks_file: str | None = None,
    test_mode: bool = False,
) -> RAGInitializationResult:
    try:
        reranker = build_reranker(config=config)
        runtime = build_rag_runtime(
            config=config,
            base_url=base_url,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            reranker=reranker,
        )

        effective_chunks_file = chunks_file
        if test_mode:
            effective_chunks_file = "data/processed/chat/test/chunks_rag_test.jsonl"

        if effective_chunks_file:
            effective_chunks_file = _resolve_chunks_path(config, effective_chunks_file)
            runtime.index_manager.index_from_file(effective_chunks_file, save=True)
            return RAGInitializationResult(
                retrieval_service=runtime.retrieval_service,
                index_manager=runtime.index_manager,
            )

        if runtime.index_manager.ensure_loaded():
            return RAGInitializationResult(
                retrieval_service=runtime.retrieval_service,
                index_manager=runtime.index_manager,
            )

        return RAGInitializationResult(
            retrieval_service=None,
            index_manager=runtime.index_manager,
            error="RAG индексы не найдены или не удалось загрузить с диска",
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return RAGInitializationResult(retrieval_service=None, index_manager=None, error=message)
