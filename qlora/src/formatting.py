from __future__ import annotations

from typing import TypedDict

from .dataset import QLoRAExample


class EncodedSample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def build_prompt(instruction: str) -> str:
    return f"Instruction:\n{instruction}\n\nResponse:\n"


def build_training_text(example: QLoRAExample, eos_token: str) -> str:
    return f"{build_prompt(example.instruction)}{example.response}{eos_token}"


def encode_example(example: QLoRAExample, tokenizer, max_seq_length: int) -> EncodedSample:
    eos_token = tokenizer.eos_token or ""
    prompt = build_prompt(example.instruction)
    full_text = build_training_text(example, eos_token=eos_token)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    labels = list(input_ids)
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
