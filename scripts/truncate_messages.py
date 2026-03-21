#!/usr/bin/env python3
"""
Обрезка отфильтрованных сообщений до заданного числа

Используется для быстрого тестирования этапов обработки.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def truncate_messages(
    input_path: Path,
    output_path: Path,
    limit: int = 250
) -> dict:
    """
    Обрезать сообщения до заданного числа

    Args:
        input_path: Входной файл
        output_path: Выходной файл
        limit: Максимальное число сообщений

    Returns:
        Статистика
    """
    logger.info(f"Чтение из {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_messages = len(data.get("messages", []))
    logger.info(f"Всего сообщений: {total_messages}")

    if limit >= total_messages:
        logger.info(f"Limit {limit} >= {total_messages}, копируем как есть")
        truncated_messages = data["messages"]
    else:
        truncated_messages = data["messages"][:limit]
        logger.info(f"Обрезка до {limit} сообщений")

    # Создание выходного файла
    output_data = {
        "total_original": data.get("total_original", total_messages),
        "total_filtered": len(truncated_messages),
        "removed": data.get("removed", {}),
        "saved_developer": data.get("saved_developer", 0),
        "messages": truncated_messages,
        "truncated": True,
        "original_count": total_messages,
        "limit": limit
    }

    logger.info(f"Сохранение {len(truncated_messages)} сообщений в {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    stats = {
        "original": total_messages,
        "truncated": len(truncated_messages),
        "removed": total_messages - len(truncated_messages),
        "percentage": len(truncated_messages) / total_messages * 100 if total_messages > 0 else 0
    }

    logger.info(f"Готово: {stats['truncated']}/{stats['original']} ({stats['percentage']:.1f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Обрезка сообщений для тестирования",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Входной файл (по умолчанию: data/processed/chat/messages_filtered.json)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Выходной файл (по умолчанию: data/processed/chat/test/messages_test.json)"
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=250,
        help="Максимальное число сообщений (по умолчанию: 250)"
    )

    args = parser.parse_args()

    # Дефолтные пути
    project_root = Path(__file__).parent.parent
    input_path = args.input or project_root / "data" / "processed" / "chat" / "messages_filtered.json"
    output_path = args.output or project_root / "data" / "processed" / "chat" / "test" / "messages_test.json"

    if not input_path.exists():
        logger.error(f"Файл не найден: {input_path}")
        logger.error("Сначала запустите: make process-stage0")
        return 1

    truncate_messages(input_path, output_path, args.limit)

    return 0


if __name__ == "__main__":
    exit(main())
