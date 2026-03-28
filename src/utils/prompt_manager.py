"""PromptManager — централизованный доступ к шаблонам промптов."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """Универсальный слой доступа к шаблонам промптов без глобального состояния."""

    def __init__(
        self,
        *,
        prompts_path: str | Path | None = None,
        prompts: dict[str, Any] | None = None,
    ):
        self._prompts_path = Path(prompts_path) if prompts_path is not None else self._default_prompts_path()
        self._prompts: dict[str, Any] | None = dict(prompts) if prompts is not None else None
        if self._prompts is None:
            self._load_prompts()

    @staticmethod
    def _default_prompts_path() -> Path:
        return Path(__file__).resolve().parent.parent.parent / "configs" / "prompts.yaml"

    def _load_prompts(self) -> None:
        """Загрузить промпты из YAML файла"""
        prompts_path = self._prompts_path

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
