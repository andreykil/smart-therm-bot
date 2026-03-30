from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from src.config.qlora_models import QLoRAWorkspaceConfig

from .dataset import QLoRAExample, load_examples
from .formatting import EncodedSample, encode_example
from .modeling import LocalBaseModelInfo, load_quantized_model, load_tokenizer, prepare_local_base_model
from .paths import dataset_path, ensure_artifact_directories


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[EncodedSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EncodedSample:
        return self.samples[index]


class PaddingCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length)
            attention_masks.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass(frozen=True)
class TrainRunResult:
    adapter_path: Path
    sample_count: int
    base_model: LocalBaseModelInfo


def _encode_examples(
    examples: list[QLoRAExample],
    tokenizer,
    max_seq_length: int,
) -> list[EncodedSample]:
    return [encode_example(example, tokenizer, max_seq_length=max_seq_length) for example in examples]


def run_training(
    config: QLoRAWorkspaceConfig,
    *,
    examples_limit: int | None = None,
    epochs_override: int | None = None,
) -> TrainRunResult:
    artifact_dirs = ensure_artifact_directories(config)
    base_model = prepare_local_base_model(config)
    all_examples = load_examples(dataset_path(config))
    examples = all_examples[:examples_limit] if examples_limit is not None else all_examples
    if not examples:
        raise RuntimeError("QLoRA dataset is empty after preprocessing")

    tokenizer = load_tokenizer(config)
    encoded_examples = _encode_examples(examples, tokenizer, config.max_seq_length)
    dataset = SupervisedDataset(encoded_examples)
    model = load_quantized_model(config)

    effective_epochs = epochs_override if epochs_override is not None else config.training.epochs
    training_args = TrainingArguments(
        output_dir=str(artifact_dirs.adapter_dir),
        num_train_epochs=effective_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        save_strategy="epoch",
        save_total_limit=config.training.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        fp16=config.qlora.bnb_4bit_compute_dtype == "float16",
        bf16=config.qlora.bnb_4bit_compute_dtype == "bfloat16",
        dataloader_pin_memory=False,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PaddingCollator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(str(artifact_dirs.adapter_dir))
    tokenizer.save_pretrained(artifact_dirs.adapter_dir)
    return TrainRunResult(
        adapter_path=artifact_dirs.adapter_dir,
        sample_count=len(examples),
        base_model=base_model,
    )
