#!/usr/bin/env python3
"""
Тестовый скрипт для Этапа 1 (Выделение веток)

Запускает обработку только первой группы (50 сообщений)
и выводит ответ LLM для отладки.

Использование:
    python scripts/test_stage1.py
    python scripts/test_stage1.py --model vikhr-nemo-12b-instruct-r --quantization Q8_0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat_processor.models import FilteredMessage
from src.chat_processor.stage1_threads import (
    THREAD_EXTRACTION_PROMPT,
    extract_threads_from_group,
    create_groups,
    load_filtered_messages
)
from src.llm.factory import create_llm_engine
from src.utils.config import Config


def setup_logging():
    """Настроить логирование"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Тест Этапа 1 (Выделение веток)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="ID модели (по умолчанию из конфига)"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default=None,
        help="Квантование (по умолчанию из конфига)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Входной файл (по умолчанию: data/processed/chat/messages_filtered.json)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Количество сообщений для теста (по умолчанию: 50)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый тест (5 сообщений)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Сохранить ответ LLM в файл"
    )

    args = parser.parse_args()

    setup_logging()

    print("=" * 70)
    print("ТЕСТ ЭТАПА 1: Выделение веток")
    print("=" * 70)
    print()

    # Загрузка конфига для получения LLM по умолчанию
    config = Config.load()
    llm_config = config.llm
    
    model_id = args.model or llm_config.get("model")
    quantization = args.quantization or llm_config.get("quantization")
    
    # Быстрый тест = 10 сообщений (минимум для группы)
    limit = 10 if args.quick else args.limit
    
    # Для быстрого теста используем меньший размер группы
    group_size = 10 if args.quick else 50
    overlap_size = 3 if args.quick else 5

    # Загрузка сообщений
    input_path = Path(args.input) if args.input else Path("data/processed/chat/messages_filtered.json")

    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        print("   Сначала запустите: make process-stage0")
        sys.exit(1)

    print(f"📖 Загрузка сообщений из {input_path}...")
    messages = load_filtered_messages(input_path)
    print(f"✅ Загружено {len(messages)} сообщений")

    # Ограничение количества
    if limit and len(messages) > limit:
        messages = messages[:limit]
        print(f"✂️  Обрезано до {len(messages)} сообщений")

    print()
    print("=" * 70)
    print("ПАРАМЕТРЫ")
    print("=" * 70)
    print(f"Модель: {model_id} ({quantization})")
    print(f"Сообщений: {len(messages)}")
    print()

    # Инициализация LLM
    print("🔄 Инициализация LLM...")
    llm = create_llm_engine(
        model_id=model_id,
        quantization=quantization,
        n_ctx=8192,
        verbose=False
    )

    if not llm.model_exists():
        print(f"❌ Модель не найдена: {llm.get_model_path()}")
        print("   Скачайте: python scripts/download_model.py")
        sys.exit(1)

    llm.load()
    print("✅ Модель загружена")
    print()

    # Создание одной группы
    print("=" * 70)
    print("ОБРАБОТКА")
    print("=" * 70)

    groups = create_groups(messages, group_size=group_size, overlap_size=overlap_size)
    print(f"Создано групп: {len(groups)} (размер={group_size}, перекрытие={overlap_size})")
    print()

    if not groups:
        print("❌ Нет групп для обработки")
        sys.exit(1)

    # Обработка первой группы
    print("📝 Обработка группы 1...")
    print()

    # Формирование промпта для отображения
    group = groups[0]
    date_range = f"{group[0].date[:10]} — {group[-1].date[:10]}"

    messages_json = json.dumps(
        [
            {
                "id": msg.id,
                "from": msg.from_,
                "date": msg.date[:10],
                "text": msg.text[:200],
                "reply_to": msg.reply_to_message_id
            }
            for msg in group
        ],
        ensure_ascii=False,
        indent=2
    )

    prompt = THREAD_EXTRACTION_PROMPT.format(
        num_messages=len(group),
        date_range=date_range,
        group_num=1,
        messages_json=messages_json
    )

    print("📤 PROMPT:")
    print(prompt[:4000])
    print()
    print("Промпт отправлен...")

    # Вызов LLM
    response = llm.generate(
        prompt=prompt,
        max_tokens=2048,
        temperature=0.3,
        top_p=0.9
    )

    print("📥 Ответ получен!")
    print()
    print("=" * 70)
    print("ОТВЕТ LLM (СЫРОЙ)")
    print("=" * 70)
    print(response)
    print()

    # Сохранение ответа
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"💾 Ответ сохранён в {args.output}")
        print()

    # Парсинг и вывод результатов
    print("=" * 70)
    print("РЕЗУЛЬТАТ (распарсенные ветки)")
    print("=" * 70)

    result = extract_threads_from_group(
        group=group,
        group_number=1,
        llm=llm,
        config=config,
        max_tokens=2048
    )

    print()
    print(f"Найдено веток: {len(result.threads)}")
    print()

    for i, thread in enumerate(result.threads, 1):
        print(f"{'-' * 70}")
        print(f"Ветка #{i}: {thread.thread_id}")
        print(f"  Тема: {thread.topic}")
        print(f"  Сообщения: {len(thread.message_ids)} шт.")
        print(f"  Даты: {thread.start_date} — {thread.end_date}")
        print(f"  Решение: {'✅' if thread.has_solution else '❌'}")
        print(f"  Кратко: {thread.summary}")
        print()

    print("=" * 70)
    print("ТЕСТ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == "__main__":
    main()
