"""Загрузка конфигурации из YAML без глобального состояния."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def default_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "default.yaml"


def default_qlora_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "qlora.yaml"


def resolve_config_path(config_path: str | Path, *, project_root: Path) -> Path:
    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    return data


def _merge_mappings(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged


def _load_layered_config_data(
    default_path: Path,
    *,
    config_file: str | Path | None,
    project_root: Path,
    override_env_var: str,
) -> dict[str, Any]:
    config_paths: list[Path] = []

    if default_path.exists():
        config_paths.append(default_path)

    override_reference = config_file or os.getenv(override_env_var)
    if override_reference:
        override_path = resolve_config_path(override_reference, project_root=project_root)
        if not override_path.exists():
            raise FileNotFoundError(f"Config file not found: {override_path}")
        if override_path not in config_paths:
            config_paths.append(override_path)

    if not config_paths:
        return {}

    payload: dict[str, Any] = {}
    for config_path in config_paths:
        payload = _merge_mappings(payload, _load_yaml_mapping(config_path))

    return payload


def load_config_data(
    config_file: str | Path | None = None,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Прочитать базовый YAML и при необходимости наложить override-конфиг.

    Порядок загрузки:
    1. [`configs/default.yaml`](configs/default.yaml)
    2. файл из аргумента `config_file` или из env `SMART_THERM_CONFIG`
    """
    return _load_layered_config_data(
        default_config_path(project_root),
        config_file=config_file,
        project_root=project_root,
        override_env_var="SMART_THERM_CONFIG",
    )


def load_qlora_config_data(
    config_file: str | Path | None = None,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Прочитать базовый YAML для QLoRA и при необходимости наложить override-конфиг.

    Порядок загрузки:
    1. [`configs/qlora.yaml`](configs/qlora.yaml)
    2. файл из аргумента `config_file`
    """
    config_paths: list[Path] = []

    default_path = default_qlora_config_path(project_root)
    if default_path.exists():
        config_paths.append(default_path)

    if config_file:
        override_path = resolve_config_path(config_file, project_root=project_root)
        if not override_path.exists():
            raise FileNotFoundError(f"Config file not found: {override_path}")
        if override_path not in config_paths:
            config_paths.append(override_path)

    if not config_paths:
        return {}

    payload: dict[str, Any] = {}
    for config_path in config_paths:
        payload = _merge_mappings(payload, _load_yaml_mapping(config_path))

    return payload
