from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.qlora_models import QLoRAWorkspaceConfig

from .dataset import QLoRAExample
from .formatting import build_prompt
from .paths import ensure_artifact_directories


def validate_artifacts(config: QLoRAWorkspaceConfig) -> dict[str, bool]:
    paths = ensure_artifact_directories(config)
    return {
        "adapter_exists": paths.adapter_dir.exists(),
        "merged_exists": paths.merged_dir.exists(),
        "gguf_exists": any(paths.gguf_dir.glob("*.gguf")),
        "modelfile_exists": (paths.ollama_dir / "Modelfile").exists(),
    }


def run_smoke_inference(config: QLoRAWorkspaceConfig, merged_path: Path, example: QLoRAExample) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        merged_path,
        local_files_only=True,
        trust_remote_code=config.backend.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        local_files_only=True,
        trust_remote_code=config.backend.trust_remote_code,
    )
    prompt = build_prompt(example.instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    if not torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
