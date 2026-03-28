"""Runtime retrieval-сервис без lifecycle ответственности."""

from __future__ import annotations

from src.chat.domain.models import RetrievalResult, RetrievedChunk
from src.chat.domain.ports import ChatContextRetriever

from src.rag.hybrid_retriever import HybridRetriever
from src.rag.models import SearchResult


class RetrievalService(ChatContextRetriever):
    """Узкий runtime-контракт поиска по уже загруженным индексам."""

    def __init__(self, *, hybrid_retriever: HybridRetriever, default_top_k: int = 5):
        self.hybrid_retriever = hybrid_retriever
        self.default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = False,
    ) -> RetrievalResult:
        result = self.hybrid_retriever.search(
            query=query,
            top_k=top_k or self.default_top_k,
            use_reranker=use_reranker,
        )
        return self._map_search_result(result)

    @staticmethod
    def _map_search_result(result: SearchResult) -> RetrievalResult:
        return RetrievalResult(
            query=result.query,
            chunks=tuple(
                RetrievedChunk(
                    text=chunk.content.text,
                    source=chunk.metadata.source,
                    tags=tuple(chunk.metadata.tags),
                    version=chunk.metadata.version,
                    confidence=chunk.metadata.confidence,
                    code=chunk.content.code,
                )
                for chunk in result.chunks
            ),
            total_found=result.total_found,
            reranked=result.reranked,
        )

    def get_stats(self) -> dict[str, object]:
        vector_size = self.hybrid_retriever.vector_store.size
        bm25_size = self.hybrid_retriever.bm25_store.size
        return {
            "total_chunks": vector_size,
            "faiss_vectors": vector_size,
            "bm25_documents": bm25_size,
            "reranker": self.hybrid_retriever.reranker.name,
            "weights": {
                "vector": self.hybrid_retriever.vector_weight,
                "bm25": self.hybrid_retriever.bm25_weight,
            },
            "top_k": self.hybrid_retriever.top_k,
        }
