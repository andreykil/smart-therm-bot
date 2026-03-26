"""
BgeM3 Embedder — генерация эмбеддингов через Ollama API
"""

import logging
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class BgeM3Embedder:
    """
    Генератор эмбеддингов через Ollama API.

    Использует /api/embeddings endpoint Ollama для получения векторных
    представлений текста. Модель указывается в конструкторе.

    Args:
        model: Название модели эмбеддингов в Ollama (например, "bge-m3")
        base_url: URL Ollama сервера
        normalize: Нормализовать векторы (L2)
        batch_size: Размер батча для batchовой обработки
    """

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        normalize: bool = True,
        batch_size: int = 32
    ):
        self.model = model
        self.base_url = base_url
        self.normalize = normalize
        self.batch_size = batch_size
        self._embedding_dim: Optional[int] = None

    @staticmethod
    def _align_embedding_dim(embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Привести вектор к целевой размерности (padding/truncate)."""
        current_dim = embedding.shape[0]
        if current_dim == target_dim:
            return embedding

        if current_dim > target_dim:
            return embedding[:target_dim]

        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:current_dim] = embedding
        return padded

    def _get_embedding_single(self, text: str) -> np.ndarray:
        """Получить эмбеддинг для одного текста"""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        embedding = np.array(response.json()["embedding"], dtype=np.float32)

        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """
        Получить эмбеддинги для текста или списка текстов.

        Args:
            texts: Один текст или список текстов

        Returns:
            Массив эмбеддингов формы (N, dim) для списка или (dim,) для одного текста
        """
        if isinstance(texts, str):
            embedding = self._get_embedding_single(texts)
            self._embedding_dim = len(embedding)
            return embedding

        # Batchовая обработка для списка текстов
        expected_dim = self._embedding_dim
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for text in batch:
                try:
                    emb = self._get_embedding_single(text)
                    if expected_dim is None:
                        expected_dim = len(emb)
                        self._embedding_dim = expected_dim
                    elif len(emb) != expected_dim:
                        logger.warning(
                            "Получен эмбеддинг с размерностью %s вместо %s. "
                            "Вектор будет приведён к ожидаемой размерности.",
                            len(emb),
                            expected_dim,
                        )
                        emb = self._align_embedding_dim(emb, expected_dim)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Ошибка эмбеддинга для текста '{text[:50]}...': {e}")
                    # Используем нулевой вектор как fallback с корректной размерностью
                    if expected_dim is None:
                        expected_dim = self.embedding_dim
                        self._embedding_dim = expected_dim
                    embeddings.append(np.zeros(expected_dim, dtype=np.float32))

        if not embeddings:
            dim = expected_dim or self.embedding_dim
            return np.empty((0, dim), dtype=np.float32)

        result = np.stack(embeddings)
        if self._embedding_dim is None:
            self._embedding_dim = result.shape[1]

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Получить эмбеддинг для запроса.

        Эквивалентно embed(), но с понятным именем для использования в поиске.

        Args:
            query: Текст запроса

        Returns:
            Вектор запроса формы (dim,)
        """
        return self.embed(query)

    @property
    def embedding_dim(self) -> int:
        """Размерность эмбеддингов"""
        if self._embedding_dim is None:
            # Пробуем получить через Ollama
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": "test"},
                    timeout=60
                )
                self._embedding_dim = len(response.json()["embedding"])
            except Exception as e:
                logger.warning(f"Не удалось определить размерность: {e}")
                # bge-m3 имеет размерность 1024
                self._embedding_dim = 1024

        return self._embedding_dim

    def __repr__(self) -> str:
        return f"BgeM3Embedder(model={self.model!r}, dim={self.embedding_dim}, normalize={self.normalize})"
