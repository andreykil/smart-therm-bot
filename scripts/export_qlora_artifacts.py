"""CLI entrypoint for post-train QLoRA export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qlora.src.paths import artifact_paths
from src.config.qlora_models import QLoRAWorkspaceConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GGUF and Ollama Modelfile from finished QLoRA artifacts.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to qlora config YAML.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = QLoRAWorkspaceConfig.load(args.config)
    paths = artifact_paths(config)

    if not paths.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {paths.adapter_dir}")
    if not paths.merged_dir.exists():
        raise FileNotFoundError(f"Merged directory not found: {paths.merged_dir}")
    if not config.export.gguf_enabled:
        raise RuntimeError("GGUF export is disabled in config.export.gguf_enabled")
    if not config.export.gguf_converter_script:
        raise RuntimeError("config.export.gguf_converter_script is not set")
    if not config.export.ollama_modelfile_enabled:
        raise RuntimeError("Ollama Modelfile export is disabled in config.export.ollama_modelfile_enabled")

    try:
        from qlora.src.export import export_gguf, write_ollama_modelfile
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing runtime dependency for QLoRA export. Install the training/export stack before running this CLI."
        ) from exc

    gguf_path = export_gguf(config, paths.merged_dir)
    modelfile_path = write_ollama_modelfile(config, gguf_path)

    payload = {
        "adapter_dir": str(paths.adapter_dir),
        "merged_dir": str(paths.merged_dir),
        "gguf_path": str(gguf_path) if gguf_path else None,
        "modelfile_path": str(modelfile_path) if modelfile_path else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
