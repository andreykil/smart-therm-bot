from __future__ import annotations

from pathlib import Path

from src.config import Config


def test_config_load_merges_default_and_explicit_override() -> None:
    config = Config.load("configs/docker.yaml")

    assert config.llm.base_url == "http://host.docker.internal:11434"
    assert config.llm.model == "qwen3.5:9b"
    assert config.rag.chunks_file == "data/processed/chat/chunks_rag_extended.jsonl"
    assert config.rag.reranker.model == "BAAI/bge-reranker-base"


def test_config_load_uses_smart_therm_config_env(monkeypatch) -> None:
    monkeypatch.setenv("SMART_THERM_CONFIG", "configs/docker.yaml")

    config = Config.load()

    assert config.llm.base_url == "http://host.docker.internal:11434"


def test_config_load_overrides_rag_chunks_file_from_env(monkeypatch, tmp_path) -> None:
    override_config = tmp_path / "override.yaml"
    override_config.write_text('rag:\n  chunks_file: "data/processed/chat/custom_chunks.jsonl"\n', encoding="utf-8")
    monkeypatch.setenv("SMART_THERM_CONFIG", str(override_config))

    config = Config.load()

    assert config.rag.chunks_file == "data/processed/chat/custom_chunks.jsonl"
    assert config.rag_chunks_path == config.project_root / "data" / "processed" / "chat" / "custom_chunks.jsonl"


def test_config_exposes_resolved_rag_chunks_path() -> None:
    config = Config()

    assert config.rag_chunks_path == config.project_root / "data" / "processed" / "chat" / "chunks_rag.jsonl"


def test_config_resolves_relative_paths_from_project_root() -> None:
    config = Config()

    assert config.resolve_path("data/test.jsonl") == config.project_root / "data" / "test.jsonl"
    assert config.resolve_path(Path("/tmp/absolute.jsonl")) == Path("/tmp/absolute.jsonl")
