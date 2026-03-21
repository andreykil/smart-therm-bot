"""
Этап 2: Дедупликация веток

Объединяет дублирующиеся ветки на границах групп
"""

import json
import logging
from pathlib import Path

from .models import Thread
from ..llm import LLMEngine, create_llm_engine
from ..utils.config import Config

logger = logging.getLogger(__name__)

# Промпт для дедупликации
DEDUP_PROMPT = """
Ты дедуплицируешь ветки чата SmartTherm.

Ветка A:
Тема: {topic_a}
Сообщения: {ids_a}
Даты: {dates_a}
Кратко: {summary_a}

Ветка B:
Тема: {topic_b}
Сообщения: {ids_b}
Даты: {dates_b}
Кратко: {summary_b}

Ответь ОДНИМ словом:
- merge — если это одна ветка (overlap > 50% или та же тема)
- different — если разные ветки

ВЫВОД (только одно слово):
"""


def load_threads(path: Path) -> list[Thread]:
    """Загрузить ветки из файла"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [Thread(**t) for t in data["threads"]]


def calculate_overlap(thread_a: Thread, thread_b: Thread) -> float:
    """
    Рассчитать overlap между ветками
    
    Returns:
        Коэффициент overlap (0.0 - 1.0)
    """
    ids_a = set(thread_a.message_ids)
    ids_b = set(thread_b.message_ids)
    
    if not ids_a or not ids_b:
        return 0.0
    
    intersection = len(ids_a & ids_b)
    min_size = min(len(ids_a), len(ids_b))
    
    return intersection / min_size if min_size > 0 else 0.0


def should_merge_llm(
    thread_a: Thread,
    thread_b: Thread,
    llm: LLMEngine,
    temperature: float = 0.1,
    max_tokens: int = 50
) -> bool:
    """
    Спросить LLM, нужно ли объединять ветки
    """
    prompt = DEDUP_PROMPT.format(
        topic_a=thread_a.topic,
        ids_a=thread_a.message_ids[:10],  # Первые 10 для краткости
        dates_a=f"{thread_a.start_date} — {thread_a.end_date}",
        summary_a=thread_a.summary[:200],
        topic_b=thread_b.topic,
        ids_b=thread_b.message_ids[:10],
        dates_b=f"{thread_b.start_date} — {thread_b.end_date}",
        summary_b=thread_b.summary[:200]
    )

    response = llm.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    response = response.strip().lower()

    return "merge" in response


def merge_threads(thread_a: Thread, thread_b: Thread) -> Thread:
    """
    Объединить две ветки в одну
    """
    # Объединение message_ids (уникальные, отсортированные)
    merged_ids = sorted(set(thread_a.message_ids) | set(thread_b.message_ids))
    
    # Объединение дат
    start_date = min(thread_a.start_date, thread_b.start_date)
    end_date = max(thread_a.end_date, thread_b.end_date)
    
    # Объединение summary
    merged_summary = f"{thread_a.summary} {thread_b.summary}"
    if len(merged_summary) > 500:
        merged_summary = merged_summary[:497] + "..."
    
    # Объединение has_solution
    has_solution = thread_a.has_solution or thread_b.has_solution
    
    # Participant count
    participants = set()
    # Примечание: точное количество участников сложно определить без исходных сообщений
    # Используем оценку
    participant_count = max(thread_a.participant_count, thread_b.participant_count)
    
    return Thread(
        thread_id=thread_a.thread_id,  # Сохраняем ID первой ветки
        topic=thread_a.topic,
        message_ids=merged_ids,
        start_date=start_date,
        end_date=end_date,
        has_solution=has_solution,
        summary=merged_summary,
        participant_count=participant_count
    )


def deduplicate_threads(
    threads: list[Thread],
    llm: LLMEngine,
    overlap_threshold: float = 0.5,
    temperature: float = 0.1,
    max_tokens: int = 50
) -> list[Thread]:
    """
    Дедупликация веток

    Args:
        threads: Список веток
        llm: LLM движок
        overlap_threshold: Порог overlap для автоматического объединения
        temperature: Температура для LLM
        max_tokens: Максимум токенов для LLM
    """
    logger.info(f"Дедупликация {len(threads)} веток")

    # Сортировка по start_date
    threads_sorted = sorted(threads, key=lambda t: t.start_date)

    merged_threads = []
    skip_indices = set()

    for i, thread_a in enumerate(threads_sorted):
        if i in skip_indices:
            continue

        current_thread = thread_a

        for j, thread_b in enumerate(threads_sorted[i+1:], i+1):
            if j in skip_indices:
                continue

            # Расчёт overlap
            overlap = calculate_overlap(current_thread, thread_b)

            if overlap >= overlap_threshold:
                # Автоматическое объединение
                logger.debug(f"Объединение (overlap={overlap:.2f}): {current_thread.thread_id} + {thread_b.thread_id}")
                current_thread = merge_threads(current_thread, thread_b)
                skip_indices.add(j)
            elif overlap > 0.1:  # Есть небольшой overlap, спрашиваем LLM
                logger.debug(f"Проверка LLM (overlap={overlap:.2f}): {current_thread.thread_id} + {thread_b.thread_id}")

                if should_merge_llm(current_thread, thread_b, llm, temperature=temperature, max_tokens=max_tokens):
                    logger.debug(f"  → LLM решил объединить")
                    current_thread = merge_threads(current_thread, thread_b)
                    skip_indices.add(j)
                else:
                    logger.debug(f"  → LLM решил не объединять")

        merged_threads.append(current_thread)

    logger.info(f"Дедупликация завершена: {len(threads)} → {len(merged_threads)} веток")

    return merged_threads


def run_stage2(
    config: Config,
    llm: LLMEngine | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None
) -> dict:
    """
    Запустить Этап 2

    Args:
        config: Конфигурация
        llm: LLM движок
        input_path: Входной файл (по умолчанию: processed/chat/threads.json)
        output_path: Выходной файл (по умолчанию: processed/chat/threads_deduped.json)
    """
    logger.info("=" * 60)
    logger.info("ЭТАП 2: Дедупликация веток")
    logger.info("=" * 60)

    # Загрузка веток
    input_path = input_path or config.processed_dir / "chat" / "threads.json"
    
    logger.info(f"Загрузка веток из {input_path}")
    threads = load_threads(input_path)
    logger.info(f"Загружено {len(threads)} веток")

    # Инициализация LLM (если нужен)
    need_llm = True

    if llm is None and need_llm:
        logger.info(f"Инициализация LLM: {config.llm['model']} ({config.llm['quantization']})")
        llm = create_llm_engine(
            model_id=config.llm.get("model"),
            quantization=config.llm.get("quantization"),
            n_ctx=config.llm.get("context_size") or 8192,
            verbose=False
        )

        if not llm.model_exists():
            logger.error(f"Модель не найдена")
            raise RuntimeError("Модель не найдена")

        llm.load()

    # Дедупликация
    if llm is None:
        raise RuntimeError("LLM engine is required for deduplication")

    deduped_threads = deduplicate_threads(
        threads,
        llm,
        overlap_threshold=0.5,
        temperature=config.llm.get("stage2", {}).get("temperature") or 0.1,
        max_tokens=config.llm.get("stage2", {}).get("max_tokens") or 50
    )

    # Сохранение результатов
    output_path = output_path or config.processed_dir / "chat" / "threads_deduped.json"
    
    logger.info(f"Сохранение {len(deduped_threads)} веток в {output_path}")

    output_data = {
        "total_original": len(threads),
        "total_deduped": len(deduped_threads),
        "removed_duplicates": len(threads) - len(deduped_threads),
        "threads": [t.model_dump() for t in deduped_threads]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    stats = {
        "threads_original": len(threads),
        "threads_deduped": len(deduped_threads),
        "duplicates_removed": len(threads) - len(deduped_threads),
        "dedup_percentage": (1 - len(deduped_threads) / len(threads) * 100) if threads else 0
    }

    logger.info(f"Удалено дубликатов: {stats['duplicates_removed']} ({stats['dedup_percentage']:.1f}%)")

    return stats


# Import create_llm_engine
from ..llm.factory import create_llm_engine
