"""Cross-encoder reranker for hybrid RAG retrieval."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import cast

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.rag.models import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Reranks hybrid retrieval candidates against the original query."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return results sorted by reranker score."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable reranker identifier."""

    @property
    @abstractmethod
    def candidate_pool_size(self) -> int:
        """How many hybrid candidates should be scored before final truncation."""


class HuggingFaceReranker(Reranker):
    """Cross-encoder reranker backed by Hugging Face `transformers`."""

    _model_cache: dict[
        tuple[str, str, str],
        tuple[PreTrainedTokenizerBase, PreTrainedModel],
    ] = {}
    _cache_lock = Lock()

    def __init__(
        self,
        *,
        model_name: str,
        models_dir: Path,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        candidate_pool_size: int = 20,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._candidate_pool_size = candidate_pool_size
        self.cache_dir = models_dir / "huggingface"
        self.device = self._resolve_device(device)
        self._tokenizer, self._model = self._load_components()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def _load_components(self) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        cache_key = (self.model_name, self.device, str(self.cache_dir))

        with self._cache_lock:
            cached = self._model_cache.get(cache_key)
            if cached is not None:
                return cached

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )
            model.to(self.device)
            model.eval()

            components = (tokenizer, model)
            self._model_cache[cache_key] = components
            return components

    def _score(self, query: str, results: list[RetrievalResult]) -> list[float]:
        scores: list[float] = []

        for start in range(0, len(results), self.batch_size):
            batch = results[start:start + self.batch_size]
            documents = [result.chunk.to_text() for result in batch]
            encoded = self._tokenizer(
                [query] * len(batch),
                documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

            with torch.inference_mode():
                logits = cast(torch.Tensor, self._model(**encoded).logits)

            if logits.ndim == 2:
                if logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits[:, -1]
            else:
                batch_scores = logits.squeeze(-1)

            scores.extend(batch_scores.detach().float().cpu().tolist())

        return scores

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        limit = len(results) if top_k is None else top_k
        scored_results = zip(results, self._score(query, results), strict=False)
        ordered = sorted(scored_results, key=lambda item: item[1], reverse=True)

        return [
            result.model_copy(update={"score": float(score), "rank": rank})
            for rank, (result, score) in enumerate(ordered[:limit], start=1)
        ]

    @property
    def name(self) -> str:
        return f"huggingface({self.model_name})"

    @property
    def candidate_pool_size(self) -> int:
        return self._candidate_pool_size
