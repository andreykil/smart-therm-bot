from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.qlora_models import QLoRAWorkspaceConfig

from .dataset import QLoRAExample, load_examples, save_examples_as_jsonl
from .export import export_gguf, merge_adapter, write_ollama_modelfile
from .modeling import LocalBaseModelInfo
from .paths import dataset_path, ensure_artifact_directories
from .training import TrainRunResult, run_training
from .validation import run_smoke_inference, validate_artifacts


@dataclass(frozen=True)
class PipelineResult:
    base_model: LocalBaseModelInfo
    adapter_path: Path
    merged_path: Path | None
    gguf_path: Path | None
    modelfile_path: Path | None
    smoke_output: str | None
    sample_count: int
    artifact_status: dict[str, bool]


def finalize_training_cycle(
    config: QLoRAWorkspaceConfig,
    train_result: TrainRunResult,
    *,
    examples: list[QLoRAExample],
) -> PipelineResult:
    merged_path: Path | None = None
    gguf_path: Path | None = None
    modelfile_path: Path | None = None
    smoke_output: str | None = None

    if config.export.merge_after_train:
        merged_path = merge_adapter(config, train_result.adapter_path)
        gguf_path = export_gguf(config, merged_path)
        modelfile_path = write_ollama_modelfile(config, gguf_path)
        if examples:
            smoke_output = run_smoke_inference(config, merged_path, examples[0])

    return PipelineResult(
        base_model=train_result.base_model,
        adapter_path=train_result.adapter_path,
        merged_path=merged_path,
        gguf_path=gguf_path,
        modelfile_path=modelfile_path,
        smoke_output=smoke_output,
        sample_count=train_result.sample_count,
        artifact_status=validate_artifacts(config),
    )


def run_full_training_cycle(config: QLoRAWorkspaceConfig) -> PipelineResult:
    train_result = run_training(config)
    examples = load_examples(dataset_path(config))
    return finalize_training_cycle(config, train_result, examples=examples)


def run_micro_test_cycle(config: QLoRAWorkspaceConfig) -> PipelineResult:
    paths = ensure_artifact_directories(config)
    examples = load_examples(dataset_path(config))[: config.micro_test.sample_pairs]
    micro_dataset_path = paths.tmp_dir / "micro_test_pairs.jsonl"
    save_examples_as_jsonl(examples, micro_dataset_path)

    micro_config = config.model_copy(deep=True)
    micro_config.dataset_path = str(micro_dataset_path.relative_to(config.project_root))
    train_result = run_training(
        micro_config,
        examples_limit=config.micro_test.sample_pairs,
        epochs_override=config.micro_test.epochs,
    )
    return finalize_training_cycle(micro_config, train_result, examples=examples)
