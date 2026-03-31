from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from src.config.qlora_models import QLoRAWorkspaceConfig

from .paths import ensure_artifact_directories


def merge_adapter(config: QLoRAWorkspaceConfig, adapter_path: Path) -> Path:
    paths = ensure_artifact_directories(config)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        local_files_only=True,
        trust_remote_code=config.backend.trust_remote_code,
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(paths.merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        local_files_only=True,
        trust_remote_code=config.backend.trust_remote_code,
    )
    tokenizer.save_pretrained(paths.merged_dir)
    return paths.merged_dir


def export_gguf(config: QLoRAWorkspaceConfig, merged_path: Path) -> Path | None:
    paths = ensure_artifact_directories(config)
    if not config.export.gguf_enabled:
        return None
    converter_script = config.export.gguf_converter_script
    if not converter_script:
        return None

    output_path = paths.gguf_dir / f"model-{config.export.gguf_outtype}.gguf"
    command = [
        sys.executable,
        converter_script,
        str(merged_path),
        "--outfile",
        str(output_path),
        "--outtype",
        config.export.gguf_outtype,
    ]
    subprocess.run(command, check=True)
    return output_path


def write_ollama_modelfile(config: QLoRAWorkspaceConfig, gguf_path: Path | None) -> Path | None:
    paths = ensure_artifact_directories(config)
    if not config.export.ollama_modelfile_enabled or gguf_path is None:
        return None

    modelfile_path = paths.ollama_dir / "Modelfile"
    content = "\n".join(
        [
            f"FROM {gguf_path}",
            f"PARAMETER temperature {config.export.modelfile_temperature}",
            "",
        ]
    )
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path
