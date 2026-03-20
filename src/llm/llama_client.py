"""
Llama 3.1 8B Instruct клиент для Apple Silicon (M2 Max)
Использует llama-cpp-python с MPS ускорением
"""

import logging
from pathlib import Path
from typing import Generator, Optional

from llama_cpp import Llama

from ..utils.config import Config

logger = logging.getLogger(__name__)


class LlamaClient:
    """Клиент для локальной Llama 3.1 8B Instruct модели"""
    
    DEFAULT_MODEL = "llama-3.1-8b-instruct.Q5_K_M.gguf"
    MODEL_URL = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
    
    # Альтернативные модели (разные уровни квантования)
    MODELS = {
        "Q4_K_M": {
            "file": "llama-3.1-8b-instruct.Q4_K_M.gguf",
            "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "size_gb": 5.5,
            "quality": "good"
        },
        "Q5_K_M": {
            "file": "llama-3.1-8b-instruct.Q5_K_M.gguf",
            "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
            "size_gb": 6.5,
            "quality": "better"
        },
        "Q6_K": {
            "file": "llama-3.1-8b-instruct.Q6_K.gguf",
            "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
            "size_gb": 7.5,
            "quality": "even_better"
        },
        "Q8_0": {
            "file": "llama-3.1-8b-instruct.Q8_0.gguf",
            "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            "size_gb": 9.0,
            "quality": "best"
        }
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        quantization: str = "Q5_K_M",
        n_ctx: int = 8192,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Инициализация клиента
        
        Args:
            model_path: Путь к модели (если None, используется default)
            quantization: Уровень квантования (Q4_K_M, Q5_K_M, Q6_K, Q8_0)
            n_ctx: Размер контекста (по умолчанию 8192)
            n_batch: Размер батча для inference
            n_threads: Количество потоков (None = авто)
            verbose: Включить логирование llama.cpp
        """
        self.config = Config.load()
        self.model_path = Path(model_path) if model_path else None
        self.quantization = quantization
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads or self._get_optimal_threads()
        self.verbose = verbose
        
        self.model: Optional[Llama] = None
        self._initialized = False
        
    def _get_optimal_threads(self) -> int:
        """Оптимальное количество потоков для M2 Max"""
        # M2 Max имеет 12 ядер (8 performance + 4 efficiency)
        # Для inference лучше использовать performance ядра
        import os
        cpu_count = os.cpu_count() or 12
        return min(cpu_count, 12)
    
    @property
    def models_dir(self) -> Path:
        """Директория для хранения моделей"""
        return self.config.models_dir_path
    
    def get_model_path(self, quantization: Optional[str] = None) -> Path:
        """Получить путь к модели"""
        q = quantization or self.quantization
        model_info = self.MODELS.get(q, self.MODELS["Q5_K_M"])
        return self.models_dir / model_info["file"]
    
    def model_exists(self, quantization: Optional[str] = None) -> bool:
        """Проверить наличие модели"""
        return self.get_model_path(quantization).exists()
    
    def get_model_info(self, quantization: Optional[str] = None) -> dict:
        """Получить информацию о модели"""
        q = quantization or self.quantization
        return self.MODELS.get(q, self.MODELS["Q5_K_M"])
    
    def load(self, quantization: Optional[str] = None) -> None:
        """
        Загрузить модель в память
        
        Args:
            quantization: Уровень квантования (переопределить при инициализации)
        """
        q = quantization or self.quantization
        model_path = self.get_model_path(q)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"Скачайте модель: python scripts/download_model.py --quantization {q}"
            )
        
        logger.info(f"Загрузка модели: {model_path.name}")
        logger.info(f"Квантование: {q}")
        logger.info(f"Потоков: {self.n_threads}")
        logger.info(f"Контекст: {self.n_ctx} токенов")
        
        # Инициализация модели с MPS поддержкой
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            n_gpu_layers=0,  # MPS используется автоматически
            verbose=self.verbose,
            use_mmap=True,  # Memory mapping для экономии RAM
            use_mlock=False  # Не блокировать в RAM
        )
        
        self._initialized = True
        logger.info("Модель успешно загружена")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,  # Увеличено для более полных ответов
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> str:
        """
        Сгенерировать ответ
        
        Args:
            prompt: Входной промпт
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling
            stop: Стоп-токены

        Returns:
            Сгенерированный текст
        """
        if not self._initialized:
            self.load()
        
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False
        )

        return response["choices"][0]["text"].strip()  # type: ignore[index]

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,  # Увеличено для более полных ответов
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> Generator[str, None, None]:
        """
        Сгенерировать ответ с потоковой отдачей

        Args:
            prompt: Входной промпт
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling
            stop: Стоп-токены

        Yields:
            Сгенерированные токены
        """
        if not self._initialized:
            self.load()
        
        if self.model is None:
            raise RuntimeError("Модель не загружена")

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
    
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,  # Увеличено для более полных ответов
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Чат с моделью (формат Llama 3.1 Instruct)

        Args:
            messages: Список сообщений [{"role": "user|assistant", "content": "..."}]
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling

        Returns:
            Ответ модели
        """
        if not self._initialized:
            self.load()
        
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        # Форматирование промпта для Llama 3.1 Instruct
        prompt = self._format_chat_prompt(messages)

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|end_of_text|>", "<|eot_id|>"],
            echo=False
        )

        text = response["choices"][0]["text"].strip()  # type: ignore[index]
        # Убрать <|begin_of_text|> из начала ответа модели
        if text.startswith("<|begin_of_text|>"):
            text = text[len("<|begin_of_text|>"):]
        return text

    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 1024,  # Увеличено для более полных ответов
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Generator[str, None, None]:
        """
        Чат с потоковой отдачей (формат Llama 3.1 Instruct)

        Args:
            messages: Список сообщений [{"role": "user|assistant", "content": "..."}]
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling

        Yields:
            Сгенерированные токены
        """
        if not self._initialized:
            self.load()

        if self.model is None:
            raise RuntimeError("Модель не загружена")

        # Форматирование промпта для Llama 3.1 Instruct
        prompt = self._format_chat_prompt(messages)

        for token in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|end_of_text|>", "<|eot_id|>"],
            echo=False,
            stream=True
        ):
            text = token["choices"][0]["text"]  # type: ignore[index]
            # Убрать <|begin_of_text|> из начала ответа модели
            if text.startswith("<|begin_of_text|>"):
                text = text[len("<|begin_of_text|>"):]
            yield text

    def _format_chat_prompt(self, messages: list[dict]) -> str:
        """
        Форматирование чата для Llama 3.1 Instruct

        Формат:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        # Llama 3.1 сама добавляет <|begin_of_text|>, не добавляем его явно
        prompt = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()

            # Убрать все специальные токены из начала контента
            for special_token in ["<|begin_of_text|>", "<|start_header_id|>", "<|eot_id|>"]:
                while content.startswith(special_token):
                    content = content[len(special_token):]
            
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

        # Добавить заголовок для ответа ассистента
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

        return prompt
    
    def get_stats(self) -> dict:
        """Получить статистику модели"""
        if not self._initialized:
            return {}
        
        return {
            "model": self.get_model_path().name,
            "quantization": self.quantization,
            "context": self.n_ctx,
            "threads": self.n_threads,
            "initialized": self._initialized
        }
    
    def unload(self) -> None:
        """Выгрузить модель из памяти"""
        if self.model:
            del self.model
            self.model = None
            self._initialized = False
            logger.info("Модель выгружена из памяти")
