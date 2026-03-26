"""
Reranker — улучшение порядка результатов поиска

Этот модуль содержит интерфейс Reranker и его реализации.
Reranker используется после гибридного поиска для более точного ранжирования.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from src.rag.models import RetrievalResult, RAGChunk

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """
    Абстрактный базовый класс для reranker.

    Reranker получает результаты поиска и переранжирует их
    на основе более глубокого анализа запроса и документов.

    Note:
        Это ЗАГОТОВКА — реальная реализация требует модели типа
        BAAI/bge-reranker-base или подобной.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        """
        Переранжировать результаты поиска.

        Args:
            query: Оригинальный запрос
            results: Результаты поиска
            top_k: Ограничить количество результатов

        Returns:
            Переранжированные результаты
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя reranker"""
        pass


class NoOpReranker(Reranker):
    """
    Dummy reranker — просто возвращает результаты как есть.

    Используется по умолчанию, когда real reranker не подключен.
    """

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        """Возвращает результаты без изменений"""
        if top_k is not None:
            return results[:top_k]
        return results

    @property
    def name(self) -> str:
        return "no-op"


class BGEReranker(Reranker):
    """
    BGE Reranker через Ollama API.

    Заготовка для интеграции BAAI/bge-reranker-base или подобной модели.

    Note:
        ДЛЯ РЕАЛИЗАЦИИ требуется:
        1. Ollama модель типа bge-reranker
        2. Реализация pairwise scoring
        3. Подключение к Ollama API

    Пример использования Ollama для reranking:
    ```python
    # Ollama не имеет нативной поддержки reranking,
    # но можно использовать модель в режиме сравнения пар
    response = ollama.chat(
        model="bge-reranker",
        messages=[{
            "role": "user",
            "content": f"Score: 1 if relevant else 0\nQuery: {query}\nDoc: {doc}"
        }]
    )
    ```
    """

    def __init__(
        self,
        model: str = "bge-reranker",
        base_url: str = "http://localhost:11434",
        top_k: int = 10
    ):
        self.model = model
        self.base_url = base_url
        self.top_k = top_k
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Проверить доступность модели"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in response.json().get("models", [])]
            available = self.model in models
            if not available:
                logger.warning(
                    f"Reranker модель '{self.model}' не найдена в Ollama. "
                    f"Будет использован NoOpReranker."
                )
            return available
        except Exception as e:
            logger.warning(f"Не удалось проверить reranker: {e}")
            return False

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        """
        Переранжировать результаты с помощью BGE reranker.

        Args:
            query: Оригинальный запрос
            results: Результаты поиска
            top_k: Количество возвращаемых результатов

        Returns:
            Переранжированные результаты
        """
        if not results:
            return results

        if not self._available:
            logger.info("Reranker недоступен, возвращаю результаты без изменений")
            return results[:top_k] if top_k else results

        # TODO: Реализовать pairwise reranking
        # Для каждой пары (query, doc) получить score
        # и переранжировать по сумме scores
        raise NotImplementedError(
            "BGE Reranker требует реализации pairwise scoring. "
            "Используйте NoOpReranker или реализуйте."
        )

    @property
    def name(self) -> str:
        return f"bge-reranker({self.model})"


class CrossEncoderReranker(Reranker):
    """
    Cross-Encoder Reranker через sentence-transformers.

    Альтернативная реализация с использованием sentence-transformers
    для кросс-encoder reranking.

    Note:
        ДЛЯ РЕАЛИЗАЦИИ требуется:
        1. sentence-transformers библиотека
        2. Модель типа cross-encoder/ms-marco
        3. GPU для эффективности

    ```python
    from sentence_transformers import CrossEncoder

    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = model.predict([(query, doc) for doc in docs])
    ```
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        top_k: int = 10
    ):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._model = None

    def _load_model(self):
        """Ленивая загрузка модели"""
        if self._model is None:
            # TODO: Реализовать загрузку
            raise NotImplementedError(
                "CrossEncoder требует sentence-transformers. "
                "Установите: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        """Переранжировать с помощью cross-encoder"""
        if not results:
            return results

        top_k = top_k or self.top_k

        # TODO: Реализовать reranking
        raise NotImplementedError(
            "CrossEncoder reranking требует реализации"
        )

    @property
    def name(self) -> str:
        return f"cross-encoder({self.model_name})"


def create_reranker(
    reranker_type: str = "no-op",
    **kwargs
) -> Reranker:
    """
    Фабрика для создания reranker.

    Args:
        reranker_type: Тип reranker ('no-op', 'bge', 'cross-encoder')
        **kwargs: Параметры для конкретного reranker

    Returns:
        Reranker instance
    """
    rerankers = {
        "no-op": NoOpReranker,
        "bge": BGEReranker,
        "cross-encoder": CrossEncoderReranker,
    }

    if reranker_type not in rerankers:
        logger.warning(
            f"Неизвестный reranker type: {reranker_type}. "
            f"Используется NoOpReranker."
        )
        return NoOpReranker()

    return rerankers[reranker_type](**kwargs)
