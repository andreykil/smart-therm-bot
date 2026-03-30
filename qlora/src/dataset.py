from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QLoRAExample:
    instruction: str
    response: str


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_examples(payload: dict[str, object]) -> list[QLoRAExample]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return []

    examples: list[QLoRAExample] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        instruction = _normalize_text(raw_item.get("instruction"))
        response = _normalize_text(raw_item.get("response"))
        if instruction and response:
            examples.append(QLoRAExample(instruction=instruction, response=response))
    return examples


def load_examples(path: Path) -> list[QLoRAExample]:
    examples: list[QLoRAExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                examples.extend(_extract_examples(payload))
    return examples


def save_examples_as_jsonl(examples: list[QLoRAExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        record = {"items": [asdict(example) for example in examples]}
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
