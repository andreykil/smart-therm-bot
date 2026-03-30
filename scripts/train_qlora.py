"""CLI entrypoint for QLoRA base-model preparation and training."""

from __future__ import annotations

import argparse
import json

from qlora.src.modeling import prepare_local_base_model
from qlora.src.pipeline import run_full_training_cycle, run_micro_test_cycle
from src.config.qlora_models import QLoRAWorkspaceConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local QLoRA base model and run training flows.")
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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = QLoRAWorkspaceConfig.load(args.config)
    base_model = prepare_local_base_model(config, force=args.force_download)

    payload: dict[str, object] = {
        "mode": args.mode,
        "base_model": {
            "model_id": base_model.model_id,
            "snapshot_dir": str(base_model.snapshot_dir),
            "metadata_path": str(base_model.metadata_path),
            "ready": base_model.ready,
            "downloaded_now": base_model.downloaded_now,
        },
    }

    if config.mode == "micro_test":
        result = run_micro_test_cycle(config)
        payload["result"] = {
            "adapter_path": str(result.adapter_path),
            "merged_path": str(result.merged_path) if result.merged_path else None,
            "gguf_path": str(result.gguf_path) if result.gguf_path else None,
            "modelfile_path": str(result.modelfile_path) if result.modelfile_path else None,
            "sample_count": result.sample_count,
            "artifact_status": result.artifact_status,
        }
    elif config.mode == "full_train":
        result = run_full_training_cycle(config)
        payload["result"] = {
            "adapter_path": str(result.adapter_path),
            "merged_path": str(result.merged_path) if result.merged_path else None,
            "gguf_path": str(result.gguf_path) if result.gguf_path else None,
            "modelfile_path": str(result.modelfile_path) if result.modelfile_path else None,
            "sample_count": result.sample_count,
            "artifact_status": result.artifact_status,
        }
    elif config.mode in {"prepare", "inspect"}:
        payload["result"] = {
            "message": "QLoRA mode is inspect/prepare; training was skipped.",
        }
    else:
        raise ValueError(f"Unsupported qlora mode: {config.mode}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
