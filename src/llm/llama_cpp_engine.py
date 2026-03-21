"""
Llama-cpp-python движок для локального inference

Поддерживает:
- Все GGUF модели из реестра
- MPS ускорение для Apple Silicon
- Потоковый и непотоковый режимы
"""

import logging
import os
from pathlib import Path
from typing import Generator, Optional

from llama_cpp import Llama

from .base import LLMEngine
from .registry import (
    get_model,
    get_model_file_path,
    get_model_url,
    get_recommended_quantization,
)
from ..utils.config import Config

logger = logging.getLogger(__name__)


class LlamaCppEngine(LLMEngine):
    """LLM движок на основе llama-cpp-python"""
    
    def __init__(
        self,
        model_id: str,
        quantization: Optional[str] = None,
        n_ctx: int = 8192,
        n_batch: Optional[int] = None,  # None = использовать из конфига
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,  # Все слои на GPU
        verbose: bool = False
    ):
        """
        Инициализация движка

        Args:
            model_id: ID модели из реестра (например, "vikhr-nemo-12b-instruct-r")
            quantization: Уровень квантования (по умолчанию recommended для модели)
            n_ctx: Размер контекста
            n_batch: Размер батча для inference (None = использовать из конфига, 2048 оптимально для M2 Max)
            n_threads: Количество потоков (None = авто)
            n_gpu_layers: Количество слоёв на GPU (-1 = все слои, оптимально для M2 Max)
            verbose: Включить логирование llama.cpp
        """
        # Получить информацию о модели
        model_info = get_model(model_id)
        if not model_info:
            raise ValueError(f"Модель '{model_id}' не найдена в реестре")

        self.model_info = model_info

        # Использовать recommended quantization если не указан
        if quantization is None:
            quantization = get_recommended_quantization(model_id)

        super().__init__(
            model_id=model_id,
            quantization=quantization,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose
        )

        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.model: Optional[Llama] = None

        # Загрузить конфиг
        self.config = Config.load()
        
        # Использовать n_batch из конфига если не передан
        if self.n_batch is None:
            self.n_batch = self.config.llm.get("n_batch", 2048)
    
    @property
    def models_dir(self) -> Path:
        """Директория для хранения моделей"""
        return self.config.models_dir_path
    
    def get_model_path(self) -> Path:
        """Получить путь к файлу модели"""
        filename = get_model_file_path(self.model_id, self.quantization)
        return self.models_dir / filename
    
    def model_exists(self) -> bool:
        """Проверить наличие модели"""
        return self.get_model_path().exists()
    
    def get_model_url(self) -> str:
        """Получить URL для скачивания"""
        return get_model_url(self.model_id, self.quantization)
    
    def get_model_size_gb(self) -> float:
        """Получить размер модели в GB"""
        from .registry import get_model_size
        return get_model_size(self.model_id, self.quantization)
    
    def _get_optimal_threads(self) -> int:
        """Оптимальное количество потоков для M2 Max"""
        cpu_count = os.cpu_count() or 12
        return min(cpu_count, 12)
    
    def load(self) -> None:
        """Загрузить модель в память"""
        model_path = self.get_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"Скачайте модель:\n"
                f"   python scripts/download_model.py --model {self.model_id} --quantization {self.quantization}"
            )
        
        logger.info(f"Загрузка модели: {self.model_info['display_name']}")
        logger.info(f"Путь: {model_path.name}")
        logger.info(f"Квантование: {self.quantization}")
        logger.info(f"Размер: ~{self.get_model_size_gb()} GB")
        logger.info(f"Потоков: {self.n_threads or self._get_optimal_threads()}")
        logger.info(f"Контекст: {self.n_ctx} токенов")
        logger.info(f"n_batch: {self.n_batch}")
        logger.info(f"n_gpu_layers: {self.n_gpu_layers} (-1 = все слои на GPU)")

        # Инициализация модели с MPS поддержкой
        assert self.n_batch is not None, "n_batch must be set"
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.n_threads or self._get_optimal_threads(),
            n_gpu_layers=self.n_gpu_layers,  # -1 = все слои на GPU
            verbose=self.verbose,
            use_mmap=True,  # Memory mapping для экономии RAM
            use_mlock=False,  # Не блокировать в RAM
            add_bos_token=False  # Не добавлять BOS автоматически (используем chat template)
        )
        
        self._loaded = True
        logger.info("Модель успешно загружена")
    
    def unload(self) -> None:
        """Выгрузить модель из памяти"""
        if self.model:
            del self.model
            self.model = None
            self._loaded = False
            logger.info("Модель выгружена из памяти")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> str:
        """Сгенерировать ответ на промпт"""
        if not self._loaded:
            self.load()

        if self.model is None:
            raise RuntimeError("Модель не загружена")

        # Stop sequences — только если явно переданы
        stop_sequences = stop if stop else []

        # BOS уже добавлен в промпт (chat template)
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
            echo=False
        )

        return response["choices"][0]["text"].strip()  # type: ignore[index]
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> Generator[str, None, None]:
        """Сгенерировать ответ с потоковой отдачей"""
        if not self._loaded:
            self.load()

        if self.model is None:
            raise RuntimeError("Модель не загружена")

        # BOS уже добавлен в промпт (chat template)
        for token in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
            stream=True
        ):
            yield token["choices"][0]["text"]  # type: ignore[index]

    def get_stats(self) -> dict:
        """Получить статистику модели"""
        if not self._loaded:
            return {
                "model_id": self.model_id,
                "display_name": self.model_info["display_name"],
                "quantization": self.quantization,
                "loaded": False,
            }
        
        return {
            "model_id": self.model_id,
            "display_name": self.model_info["display_name"],
            "quantization": self.quantization,
            "context": self.n_ctx,
            "threads": self.n_threads or self._get_optimal_threads(),
            "loaded": True,
            "model_path": str(self.get_model_path()),
        }
