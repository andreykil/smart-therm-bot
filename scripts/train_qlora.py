"""Notebook-style CLI for QLoRA training stages."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qlora.src.dataset import load_examples
from qlora.src.formatting import encode_example
from qlora.src.modeling import create_peft_config, load_tokenizer, prepare_local_base_model
from qlora.src.paths import dataset_path
from qlora.src.pipeline import finalize_training_cycle
from qlora.src.training import run_training
from qlora.src.validation import validate_artifacts
from src.config.qlora_models import QLoRAWorkspaceConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run notebook-like QLoRA stages from CLI.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to qlora config YAML.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the base model snapshot into the local project cache.",
    )
    return parser


def _stage(name: str, payload: dict[str, object]) -> None:
    print(f"\n[{name}]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    args = _build_parser().parse_args()

    config = QLoRAWorkspaceConfig.load(args.config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.mode not in {"micro_test", "full_train", "prepare", "inspect"}:
        raise ValueError(f"Unsupported qlora mode: {config.mode}")

    data_path = dataset_path(config)
    all_examples = load_examples(data_path)
    base_model = prepare_local_base_model(config, force=args.force_download)
    _stage(
        "prepare",
        {
            "mode": config.mode,
            "dataset": str(data_path),
            "examples": len(all_examples),
            "base_model_id": base_model.model_id,
            "base_model_dir": str(base_model.snapshot_dir),
            "metadata_path": str(base_model.metadata_path),
            "downloaded_now": base_model.downloaded_now,
            "artifact_status": validate_artifacts(config),
        },
    )

    if config.mode == "micro_test":
        selected_examples = all_examples[: config.micro_test.sample_pairs]
        effective_epochs = config.micro_test.epochs
    else:
        selected_examples = all_examples
        effective_epochs = config.training.epochs

    _stage(
        "select",
        {
            "selected_examples": len(selected_examples),
            "effective_epochs": effective_epochs,
        },
    )

    tokenizer = load_tokenizer(config)
    peft_config = create_peft_config(config)
    target_modules = peft_config.target_modules
    if isinstance(target_modules, str):
        target_modules_payload = [target_modules]
    elif target_modules is None:
        target_modules_payload = []
    else:
        target_modules_payload = list(target_modules)
    preview_count = len(
        [
            encode_example(example, tokenizer, max_seq_length=config.max_seq_length)
            for example in selected_examples[:1]
        ]
    )
    _stage(
        "tokenizer",
        {
            "pad_token": tokenizer.pad_token,
            "target_modules": target_modules_payload,
            "preview_encoded_samples": preview_count,
        },
    )

    train_result = None
    if config.mode == "micro_test":
        train_result = run_training(
            config,
            examples_limit=config.micro_test.sample_pairs,
            epochs_override=config.micro_test.epochs,
        )
    elif config.mode == "full_train":
        train_result = run_training(config)
    else:
        _stage("train", {"message": "inspect/prepare mode: training skipped"})

    if train_result is not None:
        _stage(
            "train",
            {
                "adapter_path": str(train_result.adapter_path),
                "sample_count": train_result.sample_count,
                "base_model_dir": str(train_result.base_model.snapshot_dir),
                "downloaded_now": train_result.base_model.downloaded_now,
            },
        )

    pipeline_result = None
    if train_result is not None:
        smoke_examples = (
            selected_examples[: config.micro_test.sample_pairs]
            if config.mode == "micro_test"
            else selected_examples
        )
        pipeline_result = finalize_training_cycle(config, train_result, examples=smoke_examples)
        _stage(
            "export",
            {
                "merged_path": str(pipeline_result.merged_path) if pipeline_result.merged_path else None,
                "gguf_path": str(pipeline_result.gguf_path) if pipeline_result.gguf_path else None,
                "modelfile_path": str(pipeline_result.modelfile_path) if pipeline_result.modelfile_path else None,
                "smoke_output": pipeline_result.smoke_output,
            },
        )
    else:
        _stage("export", {"message": "inspect/prepare mode: export skipped"})

    payload: dict[str, object] = {
        "mode": config.mode,
        "base_model": {
            "model_id": base_model.model_id,
            "snapshot_dir": str(base_model.snapshot_dir),
            "metadata_path": str(base_model.metadata_path),
            "ready": base_model.ready,
            "downloaded_now": base_model.downloaded_now,
        },
        "result": {
            "adapter_path": str(pipeline_result.adapter_path) if pipeline_result else None,
            "merged_path": str(pipeline_result.merged_path) if pipeline_result and pipeline_result.merged_path else None,
            "gguf_path": str(pipeline_result.gguf_path) if pipeline_result and pipeline_result.gguf_path else None,
            "modelfile_path": str(pipeline_result.modelfile_path)
            if pipeline_result and pipeline_result.modelfile_path
            else None,
            "sample_count": pipeline_result.sample_count if pipeline_result else 0,
            "artifact_status": pipeline_result.artifact_status if pipeline_result else validate_artifacts(config),
        },
    }

    print("\n[result]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
