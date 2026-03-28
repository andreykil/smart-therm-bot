"""Pydantic-модели конфигурации проекта."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class RAGConfig(BaseModel):
    embedding_model: str = "BAAI/bge-m3"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    think: bool | None = None
    temperature: float = 0.3
    max_tokens: int = 2048
    context_size: int = 8192


class ChatProcessingConfig(BaseModel):
    group_size: int = 50
    overlap_size: int = 5
    stop_words: list[str] = Field(
        default_factory=lambda: [
            "спасибо",
            "ок",
            "понял",
            "благодарю",
            "+",
            "да",
            "спс",
            "благодарка",
            "круто",
            "супер",
            "хорошо",
        ]
    )


class TruncateConfig(BaseModel):
    limit: int = 20


class BotConfig(BaseModel):
    token: str = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    admin_ids: list[int] = Field(default_factory=list)
    use_rag: bool = True

    @field_validator("token", mode="before")
    @classmethod
    def use_env_token_when_config_is_empty(cls, value: object) -> object:
        if isinstance(value, str) and value.strip():
            return value
        return os.getenv("TELEGRAM_BOT_TOKEN", "")


class MemoryConfig(BaseModel):
    sqlite_path: str = "data/runtime/chat_memory.sqlite3"
    session_cache_limit: int = 12
    registry_max_contexts: int = 1000
    registry_idle_ttl_seconds: int = 3600


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class LoRAConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4


class Config(BaseModel):
    """Корневая конфигурация проекта."""

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: str = "data"
    models_dir: str = "data/models"
    rag: RAGConfig = Field(default_factory=RAGConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chat_processing: ChatProcessingConfig = Field(default_factory=ChatProcessingConfig)
    truncate: TruncateConfig = Field(default_factory=TruncateConfig)
    bot: BotConfig = Field(default_factory=BotConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    @field_validator("data_dir", "models_dir", mode="before")
    @classmethod
    def convert_to_string(cls, value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        return value

    @property
    def data_dir_path(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def models_dir_path(self) -> Path:
        return self.project_root / self.models_dir

    @property
    def processed_dir(self) -> Path:
        return self.data_dir_path / "processed"

    @property
    def indices_dir(self) -> Path:
        return self.data_dir_path / "indices"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir_path / "runtime"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir_path / "raw"

    @classmethod
    def load(cls, config_file: str | Path | None = None) -> "Config":
        from .loader import load_config_data

        payload = load_config_data(config_file, project_root=Path(__file__).resolve().parent.parent.parent)
        return cls.model_validate(payload)
