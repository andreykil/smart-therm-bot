"""Pydantic-модели конфигурации проекта."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class RAGConfig(BaseModel):
    class RerankerConfig(BaseModel):
        model: str = "BAAI/bge-reranker-base"
        device: str = "auto"
        batch_size: int = 8
        max_length: int = 512
        candidate_pool_size: int = 20

    embedding_model: str = "BAAI/bge-m3"
    chunks_file: str = "data/processed/chat/chunks_rag.jsonl"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


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
    streaming: "BotStreamingConfig" = Field(default_factory=lambda: BotStreamingConfig())

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


class BotStreamingConfig(BaseModel):
    enabled: bool = False
    private_native_drafts: bool = True
    flush_interval_ms: int = 400
    min_chars_delta: int = 120
    max_draft_chars: int = 4000
    max_draft_seconds: int = 30


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


class TrainBackendConfig(BaseModel):
    device_preference: str = "auto"
    trust_remote_code: bool = True
    use_mps_bitsandbytes: bool = False


class QLoRAConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"


class TrainLoopConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 1
    save_total_limit: int = 2


class TrainExportConfig(BaseModel):
    output_root: str = "qlora/artifacts"
    adapter_dir_name: str = "adapter"
    merged_dir_name: str = "merged"
    gguf_dir_name: str = "gguf"
    ollama_dir_name: str = "ollama"
    merge_after_train: bool = True
    gguf_enabled: bool = True
    gguf_converter_script: str | None = None
    gguf_outtype: str = "q8_0"
    ollama_modelfile_enabled: bool = True
    ollama_model_name: str = "smart-therm-qwen3.5-9b"


class TrainMicroTestConfig(BaseModel):
    sample_pairs: int = 3
    epochs: int = 1


class TrainConfig(BaseModel):
    enabled: bool = False
    base_model: str = "Qwen/Qwen3.5-9B"
    dataset_path: str = "data/processed/chat/lora_pairs.jsonl"
    max_seq_length: int = 2048
    seed: int = 42
    backend: TrainBackendConfig = Field(default_factory=TrainBackendConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    training: TrainLoopConfig = Field(default_factory=TrainLoopConfig)
    export: TrainExportConfig = Field(default_factory=TrainExportConfig)
    micro_test: TrainMicroTestConfig = Field(default_factory=TrainMicroTestConfig)


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
    train: TrainConfig = Field(default_factory=TrainConfig)

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

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate

    @property
    def rag_chunks_path(self) -> Path:
        return self.resolve_path(self.rag.chunks_file)

    @classmethod
    def load(cls, config_file: str | Path | None = None) -> "Config":
        from .loader import load_config_data

        payload = load_config_data(config_file, project_root=Path(__file__).resolve().parent.parent.parent)
        return cls.model_validate(payload)
