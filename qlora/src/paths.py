from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.qlora_models import QLoRAWorkspaceConfig


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    adapter_dir: Path
    merged_dir: Path
    gguf_dir: Path
    ollama_dir: Path
    tmp_dir: Path


def artifact_paths(config: QLoRAWorkspaceConfig) -> ArtifactPaths:
    root = config.project_root / config.export.output_root
    return ArtifactPaths(
        root=root,
        adapter_dir=root / config.export.adapter_dir_name,
        merged_dir=root / config.export.merged_dir_name,
        gguf_dir=root / config.export.gguf_dir_name,
        ollama_dir=root / config.export.ollama_dir_name,
        tmp_dir=config.tmp_dir_path,
    )


def ensure_artifact_directories(config: QLoRAWorkspaceConfig) -> ArtifactPaths:
    paths = artifact_paths(config)
    for path in [
        paths.root,
        paths.adapter_dir,
        paths.merged_dir,
        paths.gguf_dir,
        paths.ollama_dir,
        paths.tmp_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dataset_path(config: QLoRAWorkspaceConfig) -> Path:
    return config.dataset_path_resolved
