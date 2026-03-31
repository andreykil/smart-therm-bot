from __future__ import annotations

from typing import cast

import torch

from src.rag.hybrid_retriever import HybridRetriever
from src.rag.bm25_store import BM25Store
from src.rag.models import ChunkContent, ChunkMetadata, RAGChunk, RetrievalResult
from src.rag.reranker import HuggingFaceReranker, Reranker
from src.rag.vector_store import VectorStore


def build_result(text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=RAGChunk(
            content=ChunkContent(text=text, code=""),
            metadata=ChunkMetadata(date="2024-01-01", tags=[], version=None, confidence=0.5),
        ),
        score=score,
        source="hybrid",
        rank=0,
    )


def test_huggingface_reranker_sorts_by_model_score(monkeypatch, tmp_path) -> None:
    HuggingFaceReranker._model_cache.clear()

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "[EOS]"

        def __call__(self, queries, documents, **kwargs):
            del queries, kwargs
            values = torch.tensor([[float(len(document))] for document in documents], dtype=torch.float32)
            return {
                "input_ids": values,
                "attention_mask": torch.ones_like(values),
            }

    class FakeModel:
        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, **encoded):
            return type("FakeOutput", (), {"logits": encoded["input_ids"]})()

    monkeypatch.setattr(
        "src.rag.reranker.AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: FakeTokenizer())}),
    )
    monkeypatch.setattr(
        "src.rag.reranker.AutoModelForSequenceClassification",
        type("FakeAutoModel", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: FakeModel())}),
    )

    reranker = HuggingFaceReranker(
        model_name="BAAI/bge-reranker-base",
        models_dir=tmp_path,
        device="cpu",
        batch_size=2,
        max_length=128,
        candidate_pool_size=6,
    )

    results = [
        build_result("short", 0.9),
        build_result("a bit longer", 0.5),
        build_result("the longest document of the batch", 0.1),
    ]

    reranked = reranker.rerank("wifi setup", results, top_k=2)

    assert [result.chunk.content.text for result in reranked] == [
        "the longest document of the batch",
        "a bit longer",
    ]
    assert [result.rank for result in reranked] == [1, 2]


def test_hybrid_retriever_uses_candidate_pool_before_final_top_k() -> None:
    class FakeStore:
        def __init__(self, results: list[RetrievalResult]) -> None:
            self.results = results
            self.calls: list[int] = []

        def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            del query
            self.calls.append(top_k)
            return self.results[:top_k]

        @property
        def size(self) -> int:
            return len(self.results)

        def __len__(self) -> int:
            return len(self.results)

    class FakeReranker(Reranker):
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int | None]] = []

        def rerank(
            self,
            query: str,
            results: list[RetrievalResult],
            top_k: int | None = None,
        ) -> list[RetrievalResult]:
            self.calls.append((query, len(results), top_k))
            return list(reversed(results))[:top_k]

        @property
        def name(self) -> str:
            return "fake-reranker"

        @property
        def candidate_pool_size(self) -> int:
            return 4

    vector_results = [
        build_result("doc-1", 0.9),
        build_result("doc-2", 0.8),
        build_result("doc-3", 0.7),
        build_result("doc-4", 0.6),
    ]
    vector_store = FakeStore(vector_results)
    bm25_store = FakeStore([])
    reranker = FakeReranker()

    retriever = HybridRetriever(
        vector_store=cast(VectorStore, vector_store),
        bm25_store=cast(BM25Store, bm25_store),
        reranker=cast(Reranker, reranker),
        top_k=2,
    )

    result = retriever.search("wifi setup", top_k=2, use_reranker=True)

    assert vector_store.calls == [4]
    assert bm25_store.calls == [4]
    assert reranker.calls == [("wifi setup", 4, 2)]
    assert [chunk.content.text for chunk in result.chunks] == ["doc-4", "doc-3"]
    assert result.reranked is True
