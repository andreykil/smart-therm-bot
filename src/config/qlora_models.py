"""Pydantic-модели отдельного конфига для QLoRA-обучения."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class QLoRABackendConfig(BaseModel):
    device_preference: str = "auto"
    trust_remote_code: bool = True
    use_mps_bitsandbytes: bool = False


class QLoRAQuantizationConfig(BaseModel):
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


class QLoRATrainingConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 1
    save_total_limit: int = 2


class QLoRAExportConfig(BaseModel):
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
    modelfile_temperature: float = 0.1


class QLoRAMicroTestConfig(BaseModel):
    sample_pairs: int = 3
    epochs: int = 1


class QLoRAWorkspaceConfig(BaseModel):
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: str = "data"
    models_dir: str = "data/models"
    mode: str = "inspect"
    enabled: bool = False
    base_model: str = "Qwen/Qwen3.5-9B"
    dataset_path: str = "data/processed/chat/lora_pairs.jsonl"
    tmp_dir: str = "qlora/tmp"
    max_seq_length: int = 2048
    seed: int = 42
    backend: QLoRABackendConfig = Field(default_factory=QLoRABackendConfig)
    qlora: QLoRAQuantizationConfig = Field(default_factory=QLoRAQuantizationConfig)
    training: QLoRATrainingConfig = Field(default_factory=QLoRATrainingConfig)
    export: QLoRAExportConfig = Field(default_factory=QLoRAExportConfig)
    micro_test: QLoRAMicroTestConfig = Field(default_factory=QLoRAMicroTestConfig)

    @property
    def data_dir_path(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def models_dir_path(self) -> Path:
        return self.project_root / self.models_dir

    @property
    def dataset_path_resolved(self) -> Path:
        return self.project_root / self.dataset_path

    @property
    def tmp_dir_path(self) -> Path:
        return self.project_root / self.tmp_dir

    @classmethod
    def load(cls, config_file: str | Path | None = None) -> "QLoRAWorkspaceConfig":
        from .loader import load_qlora_config_data

        payload = load_qlora_config_data(config_file, project_root=Path(__file__).resolve().parent.parent.parent)
        return cls.model_validate(payload)
