#!/usr/bin/env python3
"""
Скрипт для валидации результатов обработки чата

Выводит выборку результатов каждого этапа для ручной проверки.
"""

import json
import logging
import sys
from pathlib import Path

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def validate_stage0(config: Config):
    """Валидация Этапа 0"""
    path = config.processed_dir / "chat" / "messages_filtered.json"
    
    if not path.exists():
        logger.info(f"❌ Этап 0: Файл не найден ({path})")
        return
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info("=" * 70)
    logger.info("ЭТАП 0: Фильтрация шума")
    logger.info("=" * 70)
    logger.info(f"✅ Файл: {path}")
    logger.info(f"✅ Всего исходно: {data['total_original']}")
    logger.info(f"✅ После фильтрации: {data['total_filtered']}")
    logger.info(f"✅ Удалено: {sum(data['removed'].values())}")
    logger.info(f"   - Сервисные: {data['removed']['service']}")
    logger.info(f"   - Эмодзи: {data['removed']['emoji']}")
    logger.info(f"   - Флуд: {data['removed']['flood']}")
    logger.info(f"   - Дубликаты: {data['removed']['duplicate']}")
    logger.info(f"✅ Процент сохранённых: {data['total_filtered']/data['total_original']*100:.1f}%")
    
    # Выборка
    logger.info("\n📋 Выборка сообщений (первые 5):")
    for msg in data['messages'][:5]:
        from_name = msg.get('from_', msg.get('from', 'unknown'))
        text = msg.get('text', '')[:100]
        logger.info(f"  [{msg['id']}] {from_name}: {text}...")


def validate_stage1(config: Config):
    """Валидация Этапа 1"""
    path = config.processed_dir / "chat" / "threads.json"
    
    if not path.exists():
        logger.info(f"❌ Этап 1: Файл не найден ({path})")
        return
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info("\n" + "=" * 70)
    logger.info("ЭТАП 1: Выделение веток")
    logger.info("=" * 70)
    logger.info(f"✅ Файл: {path}")
    logger.info(f"✅ Групп обработано: {data['total_groups']}")
    logger.info(f"✅ Веток найдено: {data['total_threads']}")
    logger.info(f"✅ Сообщений покрыто: {data['total_messages_covered']}")
    
    # Выборка
    logger.info(f"\n📋 Выборка веток (первые 5 из {len(data['threads'])}):")
    for thread in data['threads'][:5]:
        logger.info(f"  {thread['thread_id']}: {thread['topic']}")
        logger.info(f"    Сообщения: {len(thread['message_ids'])} шт.")
        logger.info(f"    Даты: {thread['start_date']} — {thread['end_date']}")
        logger.info(f"    Решение: {'✅' if thread['has_solution'] else '❌'}")
        logger.info(f"    Кратко: {thread['summary'][:100]}...")


def validate_stage2(config: Config):
    """Валидация Этапа 2"""
    path = config.processed_dir / "chat" / "threads_deduped.json"
    
    if not path.exists():
        logger.info(f"❌ Этап 2: Файл не найден ({path})")
        return
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info("\n" + "=" * 70)
    logger.info("ЭТАП 2: Дедупликация веток")
    logger.info("=" * 70)
    logger.info(f"✅ Файл: {path}")
    logger.info(f"✅ Было веток: {data['total_original']}")
    logger.info(f"✅ После дедупликации: {data['total_deduped']}")
    logger.info(f"✅ Удалено дубликатов: {data['removed_duplicates']}")
    if data['total_original'] > 0:
        pct = (1 - data['total_deduped'] / data['total_original']) * 100
        logger.info(f"✅ Процент дубликатов: {pct:.1f}%")
    
    # Выборка
    logger.info(f"\n📋 Выборка веток (первые 5 из {len(data['threads'])}):")
    for thread in data['threads'][:5]:
        logger.info(f"  {thread['thread_id']}: {thread['topic']}")
        logger.info(f"    Сообщения: {len(thread['message_ids'])} шт.")
        logger.info(f"    Даты: {thread['start_date']} — {thread['end_date']}")
        logger.info(f"    Кратко: {thread['summary'][:100]}...")


def validate_stage3(config: Config):
    """Валидация Этапа 3"""
    path = config.processed_dir / "chat" / "chunks_rag.jsonl"
    sample_path = config.processed_dir / "chat" / "chunks_sample.json"
    
    if not path.exists():
        logger.info(f"❌ Этап 3: Файл не найден ({path})")
        return
    
    # Подсчёт чанков
    with open(path, "r", encoding="utf-8") as f:
        chunk_count = sum(1 for _ in f)
    
    logger.info("\n" + "=" * 70)
    logger.info("ЭТАП 3: Создание RAG чанков")
    logger.info("=" * 70)
    logger.info(f"✅ Файл: {path}")
    logger.info(f"✅ Создано чанков: {chunk_count}")
    
    # Выборка
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        
        logger.info(f"\n📋 Выборка чанков ({len(sample)} шт.):")
        for chunk in sample[:3]:
            logger.info(f"  {chunk['chunk_id']}:")
            logger.info(f"    Summary: {chunk['content']['summary'][:100]}...")
            logger.info(f"    Text: {chunk['content']['text'][:150]}...")
            logger.info(f"    Tags: {chunk['metadata'].get('tags', [])}")
            logger.info(f"    Confidence: {chunk['metadata'].get('confidence', 0):.2f}")
    else:
        logger.info(f"⚠️  Выборка не найдена ({sample_path})")


def main():
    logger.info("=" * 70)
    logger.info("ВАЛИДАЦИЯ ОБРАБОТКИ ЧАТА")
    logger.info("=" * 70)
    
    config = Config.load()
    
    # Проверка наличия исходного файла
    raw_chat = config.raw_dir / "chat_history.json"
    if not raw_chat.exists():
        logger.error(f"❌ Исходный чат не найден: {raw_chat}")
        sys.exit(1)
    
    logger.info(f"✅ Исходный чат: {raw_chat}")
    logger.info("")
    
    # Валидация этапов
    validate_stage0(config)
    validate_stage1(config)
    validate_stage2(config)
    validate_stage3(config)
    
    logger.info("\n" + "=" * 70)
    logger.info("ВАЛИДАЦИЯ ЗАВЕРШЕНА")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
