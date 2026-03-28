"""Загрузка конфигурации из YAML без глобального состояния."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def default_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "default.yaml"


def load_config_data(
    config_file: str | Path | None = None,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Прочитать YAML-конфиг и вернуть словарь для моделей конфигурации."""
    if config_file is None:
        candidate = default_config_path(project_root)
        if not candidate.exists():
            return {}
        config_path = candidate
    else:
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    return data
