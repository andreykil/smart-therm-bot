from __future__ import annotations

from src.config import Config


def test_config_load_merges_default_and_explicit_override() -> None:
    config = Config.load("configs/docker.yaml")

    assert config.llm.base_url == "http://host.docker.internal:11434"
    assert config.llm.model == "qwen3.5:9b"


def test_config_load_uses_smart_therm_config_env(monkeypatch) -> None:
    monkeypatch.setenv("SMART_THERM_CONFIG", "configs/docker.yaml")

    config = Config.load()

    assert config.llm.base_url == "http://host.docker.internal:11434"
