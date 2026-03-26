"""
Ollama LLM клиент

Простой HTTP клиент для Ollama API.
"""

import json
import logging
from pathlib import Path
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Клиент для Ollama API"""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        verbose: bool = False,
        think: Optional[bool] = None
    ):
        """
        Инициализация клиента

        Args:
            model: Название модели в Ollama (например, "llama3.1")
            base_url: URL Ollama сервера
            verbose: Включить логирование
            think: Включить/выключить thinking mode (None - не передавать, True/False - передать)
        """
        self.model = model
        self.base_url = base_url
        self.verbose = verbose
        self.think = think
        self._loaded = False

    def model_exists(self) -> bool:
        """Проверить наличие модели в Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in response.json().get("models", [])]
            # Проверяем полное имя или base name (без :latest и т.д.)
            model_base = self.model.split(":")[0]
            return self.model in models or any(
                m.split(":")[0] == model_base for m in models
            )
        except requests.RequestException as e:
            logger.error(f"Ошибка проверки модели: {e}")
            return False

    def load(self, strict: bool = True) -> None:
        """Загрузить модель.

        Args:
            strict: Если True, выбрасывать ошибку при отсутствии модели.
        """
        if not self.model_exists():
            message = (
                f"Модель '{self.model}' не найдена в Ollama. "
                f"Скачайте модель: ollama pull {self.model}"
            )
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        self._loaded = True
        if self.verbose:
            logger.info(f"Модель '{self.model}' готова к работе")

    def unload(self) -> None:
        """Выгрузить модель (Ollama управляет сам)"""
        self._loaded = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
        think: Optional[bool] = None
    ) -> str:
        """
        Сгенерировать ответ на промпт

        Args:
            prompt: Входной текст
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling
            stop: Стоп-токены
            think: Переопределить thinking mode (None - использовать из __init__)

        Returns:
            Сгенерированный текст
        """
        if not self._loaded:
            self.load(strict=True)

        effective_think = think if think is not None else self.think

        # Формируем запрос к /api/generate для raw prompt
        request_json = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop or []
            },
            "stream": False
        }

        if effective_think is not None:
            request_json["think"] = effective_think

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=request_json,
            timeout=600
        )
        response.raise_for_status()
        data = response.json()
        if "response" in data:
            return data["response"]
        return data["message"]["content"]

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
        think: Optional[bool] = None
    ) -> Generator[str, None, None]:
        """
        Сгенерировать ответ с потоковой отдачей

        Args:
            prompt: Входной текст
            max_tokens: Максимум токенов на выходе
            temperature: Температура генерации
            top_p: Top-p sampling
            stop: Стоп-токены
            think: Переопределить thinking mode (None - использовать из __init__)

        Yields:
            Сгенерированные токены
        """
        if not self._loaded:
            self.load(strict=True)

        effective_think = think if think is not None else self.think

        # Формируем запрос к /api/generate для raw prompt
        request_json = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop or []
            },
            "stream": True
        }

        if effective_think is not None:
            request_json["think"] = effective_think

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=request_json,
            stream=True,
            timeout=600
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                elif "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]

    def get_stats(self) -> dict:
        """Получить статистику модели"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "loaded": self._loaded,
            "provider": "ollama"
        }
