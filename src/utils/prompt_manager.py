"""
PromptManager — централизованный доступ к шаблонам промптов.

Отвечает только за:
- загрузку шаблонов из configs/prompts.yaml
- проверку наличия шаблонов
- форматирование шаблонов с подстановкой переменных
"""

import logging
import re
from typing import Any

import yaml

from .config import Config

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Менеджер промптов проекта

    Универсальный слой доступа к шаблонам промптов.
    Не содержит доменной логики (chat/chunks/RAG).
    """

    _instance: "PromptManager | None" = None
    _prompts: dict[str, Any] | None = None

    def __new__(cls) -> "PromptManager":
        """Singleton для предотвращения повторной загрузки"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Инициализация (вызывается один раз благодаря singleton)"""
        if self._prompts is None:
            self._load_prompts()

    def _load_prompts(self) -> None:
        """Загрузить промпты из YAML файла"""
        config = Config.load()
        prompts_path = config.project_root / "configs" / "prompts.yaml"

        if not prompts_path.exists():
            logger.error(f"Файл промптов не найден: {prompts_path}")
            self._prompts = {}
            return

        logger.info(f"Загрузка промптов из {prompts_path}")

        with open(prompts_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            self._prompts = loaded if isinstance(loaded, dict) else {}

        if self._prompts:
            logger.info(f"Загружено {len(self._prompts)} промптов")

    def _ensure_loaded(self) -> dict[str, Any]:
        """Убедиться, что промпты загружены"""
        if self._prompts is None:
            self._load_prompts()

        if self._prompts is None:
            raise RuntimeError("Failed to load prompts")

        return self._prompts

    @staticmethod
    def _extract_placeholders(template: str) -> set[str]:
        """
        Извлечь имена placeholder-переменных из строки шаблона.

        Учитывает экранированные скобки '{{' и '}}'.
        """
        return set(re.findall(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})", template))

    def has_prompt(self, name: str) -> bool:
        """Проверить, существует ли шаблон"""
        prompts = self._ensure_loaded()
        return name in prompts

    def list_prompts(self) -> list[str]:
        """Получить список доступных шаблонов"""
        prompts = self._ensure_loaded()
        return sorted(prompts.keys())

    def get_prompt(self, name: str, **kwargs: Any) -> str:
        """
        Получить промпт по имени с подстановкой переменных

        Args:
            name: Имя промпта (ключ в YAML)
            **kwargs: Переменные для подстановки в промпт

        Returns:
            Промпт с подставленными переменными

        Raises:
            KeyError: Если промпт не найден
            ValueError: Если не все переменные переданы
        """
        prompts = self._ensure_loaded()

        if name not in prompts:
            raise KeyError(f"Промпт '{name}' не найден. Доступные: {list(prompts.keys())}")

        template = prompts[name]
        if not isinstance(template, str):
            raise TypeError(f"Промпт '{name}' должен быть строкой")

        placeholders = self._extract_placeholders(template)
        provided_keys = set(kwargs.keys())
        missing = [var for var in sorted(placeholders) if var not in kwargs or kwargs[var] is None]
        unexpected = sorted(provided_keys - placeholders)

        if missing:
            raise ValueError(f"Не переданы обязательные переменные для промпта '{name}': {missing}")

        if unexpected:
            raise ValueError(
                f"Переданы лишние переменные для промпта '{name}': {unexpected}. "
                f"Ожидаются только: {sorted(placeholders)}"
            )

        return template.format(**kwargs)

    def reload(self) -> None:
        """Перезагрузить промпты из файла"""
        self._prompts = None
        self._load_prompts()

    @classmethod
    def reset(cls) -> None:
        """Сбросить singleton (для тестов)"""
        cls._instance = None
        cls._prompts = None
