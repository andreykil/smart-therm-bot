"""
Базовый класс для LLM движков

Определяет интерфейс для всех backend'ов:
- llama-cpp-python (локальный inference)
- Ollama
- vLLM
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional


class LLMEngine(ABC):
    """Абстрактный базовый класс для LLM inference"""
    
    def __init__(
        self,
        model_id: str,
        quantization: str = "Q4_K_M",
        n_ctx: int = 8192,
        n_threads: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Инициализация движка
        
        Args:
            model_id: ID модели из реестра
            quantization: Уровень квантования
            n_ctx: Размер контекста
            n_threads: Количество потоков
            verbose: Включить логирование
        """
        self.model_id = model_id
        self.quantization = quantization
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self._loaded = False
    
    @property
    @abstractmethod
    def models_dir(self) -> Path:
        """Директория для хранения моделей"""
        pass
    
    @abstractmethod
    def get_model_path(self) -> Path:
        """Получить путь к файлу модели"""
        pass
    
    @abstractmethod
    def model_exists(self) -> bool:
        """Проверить наличие модели"""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Загрузить модель в память"""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Выгрузить модель из памяти"""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> str:
        """
        Сгенерировать ответ на промпт
        
        Args:
            prompt: Входной текст
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling
            stop: Стоп-токены
            
        Returns:
            Сгенерированный текст
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> Generator[str, None, None]:
        """
        Сгенерировать ответ с потоковой отдачей

        Args:
            prompt: Входной текст
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling
            stop: Стоп-токены

        Yields:
            Сгенерированные токены
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Получить статистику/информацию о модели"""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Проверить, загружена ли модель"""
        return self._loaded
