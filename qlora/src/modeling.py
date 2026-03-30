from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config.qlora_models import QLoRAWorkspaceConfig


@dataclass(frozen=True)
class LocalBaseModelInfo:
    model_id: str
    snapshot_dir: Path
    metadata_path: Path
    ready: bool
    downloaded_now: bool


def _torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype for QLoRA: {dtype_name}")
    return mapping[dtype_name]


def resolve_device_map(config: QLoRAWorkspaceConfig) -> Any:
    if config.backend.use_mps_bitsandbytes:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS bitsandbytes mode requested, but MPS is not available")
        return {"": "mps"}

    if torch.cuda.is_available():
        return "auto"

    raise RuntimeError(
        "QLoRA with bitsandbytes in this pipeline requires CUDA or experimental MPS bitsandbytes mode"
    )


def _sanitize_model_name(model_name: str) -> str:
    return quote(model_name, safe="")


def local_model_snapshot_dir(config: QLoRAWorkspaceConfig) -> Path:
    return config.models_dir_path / "huggingface" / _sanitize_model_name(config.base_model)


def _snapshot_has_weights(snapshot_dir: Path) -> bool:
    return any(snapshot_dir.glob("*.safetensors")) or any(snapshot_dir.glob("pytorch_model*.bin"))


def _snapshot_has_tokenizer(snapshot_dir: Path) -> bool:
    return (snapshot_dir / "tokenizer.json").exists() or (snapshot_dir / "tokenizer_config.json").exists()


def _snapshot_is_complete(snapshot_dir: Path) -> bool:
    return (snapshot_dir / "config.json").exists() and _snapshot_has_weights(snapshot_dir) and _snapshot_has_tokenizer(snapshot_dir)


def _snapshot_metadata_path(config: QLoRAWorkspaceConfig) -> Path:
    return local_model_snapshot_dir(config) / ".snapshot-info.json"


def _write_snapshot_metadata(config: QLoRAWorkspaceConfig, snapshot_dir: Path) -> Path:
    metadata_path = _snapshot_metadata_path(config)
    payload = {
        "model_id": config.base_model,
        "snapshot_dir": str(snapshot_dir),
        "files": sorted(path.name for path in snapshot_dir.iterdir() if path.is_file()),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def prepare_local_base_model(config: QLoRAWorkspaceConfig, *, force: bool = False) -> LocalBaseModelInfo:
    snapshot_dir = local_model_snapshot_dir(config)
    metadata_path = _snapshot_metadata_path(config)
    was_ready = _snapshot_is_complete(snapshot_dir)
    if was_ready and not force:
        if not metadata_path.exists():
            metadata_path = _write_snapshot_metadata(config, snapshot_dir)
        return LocalBaseModelInfo(
            model_id=config.base_model,
            snapshot_dir=snapshot_dir,
            metadata_path=metadata_path,
            ready=True,
            downloaded_now=False,
        )

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.base_model,
        local_dir=str(snapshot_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if not _snapshot_is_complete(snapshot_dir):
        raise RuntimeError(
            f"Local base model snapshot is incomplete: {snapshot_dir}. "
            f"Expected config, tokenizer, and weight files for {config.base_model}."
        )
    metadata_path = _write_snapshot_metadata(config, snapshot_dir)
    return LocalBaseModelInfo(
        model_id=config.base_model,
        snapshot_dir=snapshot_dir,
        metadata_path=metadata_path,
        ready=True,
        downloaded_now=not was_ready,
    )


def ensure_local_model_snapshot(config: QLoRAWorkspaceConfig) -> Path:
    return prepare_local_base_model(config).snapshot_dir


def load_tokenizer(config: QLoRAWorkspaceConfig):
    model_path = ensure_local_model_snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=config.backend.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_peft_config(config: QLoRAWorkspaceConfig) -> LoraConfig:
    return LoraConfig(
        r=config.qlora.r,
        lora_alpha=config.qlora.alpha,
        lora_dropout=config.qlora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.qlora.target_modules),
    )


def load_quantized_model(config: QLoRAWorkspaceConfig):
    model_path = ensure_local_model_snapshot(config)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.qlora.load_in_4bit,
        bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_torch_dtype(config.qlora.bnb_4bit_compute_dtype),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=resolve_device_map(config),
        local_files_only=True,
        trust_remote_code=config.backend.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, create_peft_config(config))
    return model
