"""
Этап 1: Выделение веток

Разбивает отфильтрованные сообщения на группы с перекрытием
и использует LLM для выделения независимых веток обсуждения
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .models import FilteredMessage, Thread, ThreadsResult
from ..llm import LLMEngine
from ..utils.config import Config
from ..utils.json_utils import extract_json_from_text

logger = logging.getLogger(__name__)

# Промпт для выделения веток
THREAD_EXTRACTION_PROMPT = """
Ты анализируешь историю чата технической поддержки SmartTherm.

ВХОД: {num_messages} сообщений за период {date_range}

ЗАДАЧА: Найди все независимые ветки обсуждения.

Ветка — это:
1. Вопрос пользователя + ответы (разработчика или других пользователей)
2. Объявление разработчика + вопросы/комментарии
3. Обсуждение одной технической проблемы
4. Отчёт пользователя об опыте + уточнения
5. Ссылка на ресурс + обсуждение этого ресурса

КРИТЕРИИ разделения на разные ветки:
- Новая тема (другой вопрос/проблема/котёл)
- Прошло >12 часов между сообщениями без обсуждения
- Разные участники без семантической связи
- Смена контекста (например, с прошивки на подключение)

КРИТЕРИИ объединения в одну ветку:
- Ответы на один вопрос (даже с интервалом)
- Уточнения проблемы тем же пользователем
- Цепочка reply_to_message_id (явная связь между сообщениями)
- Обсуждение одного компонента (WiFi, OpenTherm, датчики)
- Ссылка + комментарии к ней

ОБРАТИ ВНИМАНИЕ НА reply_to_message_id:
- Если сообщение имеет reply_to_message_id, оно продолжает ветку того сообщения
- Ответы разработчика (Evgen) часто имеют reply_to_message_id на вопрос пользователя
- Используй reply_to_message_id для точного определения связей

ВЕРНИ JSON СТРОГОЙ СТРУКТУРЫ:
{{
  "threads": [
    {{
      "thread_id": "t{group_num:03d}_0",
      "topic": "Краткая тема (3-5 слов)",
      "message_ids": [12345, 12346, 12350],
      "start_date": "2024-01-15",
      "end_date": "2024-01-17",
      "has_solution": true,
      "summary": "Пользователь спрашивал про X. Разработчик ответил Y. Проблема решена через Z."
    }}
  ]
}}

ТРЕБОВАНИЯ:
- topic: 3-5 слов, без артиклей
- message_ids: все ID сообщений ветки (отсортированы)
- has_solution: true если есть ответ от Evgen или подтверждение решения
- summary: 1-3 предложения, конкретно

КРИТИЧЕСКИ ВАЖНО:
1. ВЕРНИ ТОЛЬКО JSON - никаких пояснений до или после
2. НЕ используй markdown (никаких ```json)
3. ЗАКОНЧИ ответ сразу после закрывающей }}
4. ЕСЛИ не можешь выделить ветки, верни пустой список: {{"threads": []}}
5. НЕ пиши "Обратите внимание", "Пожалуйста", "С уважением" и т.д.
6. ДАЖЕ если сообщения это просто ссылки - группируй их по теме

СООБЩЕНИЯ ДЛЯ АНАЛИЗА:
{messages_json}

ТВОЙ ОТВЕТ (JSON):
"""


def load_filtered_messages(path: Path) -> list[FilteredMessage]:
    """Загрузить отфильтрованные сообщения"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [FilteredMessage(**msg) for msg in data["messages"]]


def create_groups(
    messages: list[FilteredMessage],
    group_size: int,
    overlap_size: int
) -> list[list[FilteredMessage]]:
    """
    Разбить сообщения на группы с перекрытием

    Args:
        messages: Список сообщений
        group_size: Размер группы
        overlap_size: Размер перекрытия
    """
    groups = []
    step = group_size - overlap_size

    for i in range(0, len(messages), step):
        group = messages[i:i + group_size]
        if len(group) >= 10:  # Минимальный размер группы
            groups.append(group)

    logger.info(f"Создано {len(groups)} групп (размер={group_size}, перекрытие={overlap_size})")

    return groups


def extract_threads_from_group(
    group: list[FilteredMessage],
    group_number: int,
    llm: LLMEngine,
    config: Config,
    max_tokens: int = 2048
) -> ThreadsResult:
    """
    Выделить ветки из группы сообщений

    Args:
        group: Группа сообщений
        group_number: Номер группы
        llm: LLM движок
        config: Конфигурация
        max_tokens: Максимум токенов на ответ
    """
    # Подготовка сообщений для промпта
    messages_json = json.dumps(
        [
            {
                "id": msg.id,
                "from": msg.from_,
                "date": msg.date[:10],  # YYYY-MM-DD
                "text": msg.text[:200],  # Обрезаем длинные сообщения
                "reply_to": msg.reply_to_message_id  # Явная связь с другим сообщением
            }
            for msg in group
        ],
        ensure_ascii=False,
        indent=2
    )

    # Даты группы
    date_range = f"{group[0].date[:10]} — {group[-1].date[:10]}"

    # Формирование промпта
    prompt = THREAD_EXTRACTION_PROMPT.format(
        num_messages=len(group),
        date_range=date_range,
        group_num=group_number,
        messages_json=messages_json
    )

    # Вызов LLM
    logger.info(f"Группа {group_number}: Запрос к LLM ({len(group)} сообщений)...")

    # Stop sequences для JSON
    stop_sequences = [
        "\n\n",  # Двойной newline
        "Объяснение",
        "Примечание",
        "Закрытие",
        "В данном случае",
        "Пожалуйста",
        "Обратите внимание"
    ]

    stage1_cfg = config.llm.get("stage1", {})
    response = llm.generate(
        prompt=prompt,
        max_tokens=stage1_cfg.get("max_tokens") or 1000,
        temperature=stage1_cfg.get("temperature") or 0.5,
        top_p=0.9,
        stop=stop_sequences
    )

    # Парсинг ответа
    logger.info(f"Группа {group_number}: Парсинг ответа...")

    try:
        # Извлечь JSON из ответа
        json_str = extract_json_from_text(response)
        result_data = json.loads(json_str)

        threads = [
            Thread(
                thread_id=t.get("thread_id", f"t{group_number:03d}_0"),
                topic=t.get("topic", "Без темы"),
                message_ids=t.get("message_ids", []),
                start_date=t.get("start_date", group[0].date[:10]),
                end_date=t.get("end_date", group[-1].date[:10]),
                has_solution=t.get("has_solution", False),
                summary=t.get("summary", ""),
                participant_count=len(set(
                    msg.from_ for msg in group if msg.id in t.get("message_ids", [])
                ))
            )
            for t in result_data.get("threads", [])
        ]

        logger.info(f"Группа {group_number}: Найдено {len(threads)} веток")

    except Exception as e:
        logger.error(f"Группа {group_number}: Ошибка парсинга: {e}")
        logger.error(f"Ответ LLM: {response[:500]}...")

        # Fallback: создать одну ветку из всех сообщений
        threads = [
            Thread(
                thread_id=f"t{group_number:03d}_0",
                topic="Обсуждение в чате",
                message_ids=[msg.id for msg in group],
                start_date=group[0].date[:10],
                end_date=group[-1].date[:10],
                has_solution=any(msg.is_from_developer for msg in group),
                summary=f"Группа сообщений {group[0].date[:10]} — {group[-1].date[:10]}",
                participant_count=len(set(msg.from_ for msg in group))
            )
        ]

    return ThreadsResult(
        threads=threads,
        group_number=group_number,
        total_messages=len(group)
    )


def run_stage1(
    config: Config,
    llm: Optional[LLMEngine] = None
) -> dict:
    """
    Запустить Этап 1

    Args:
        config: Конфигурация
        llm: LLM движок (если None, создаётся новый)
    """
    logger.info("=" * 60)
    logger.info("ЭТАП 1: Выделение веток")
    logger.info("=" * 60)

    # Загрузка отфильтрованных сообщений
    logger.info(f"Загрузка сообщений из {config.processed_dir / 'chat' / 'messages_filtered.json'}")
    messages = load_filtered_messages(config.processed_dir / "chat" / "messages_filtered.json")
    logger.info(f"Загружено {len(messages)} сообщений")

    # Создание групп
    chat_cfg = config.chat_processing
    groups = create_groups(
        messages,
        chat_cfg.get("group_size") or 50,
        chat_cfg.get("overlap_size") or 5
    )

    # Инициализация LLM
    need_llm = llm is None

    if need_llm:
        logger.info(f"Инициализация LLM: {config.llm['model']} ({config.llm['quantization']})")
        from ..llm.factory import create_llm_engine

        llm = create_llm_engine(
            model_id=config.llm.get("model"),
            quantization=config.llm.get("quantization"),
            n_ctx=config.llm.get("context_size") or 8192,
            verbose=False
        )

        if not llm.model_exists():
            logger.error(f"Модель не найдена: {llm.get_model_path()}")
            logger.error("Скачайте модель: python scripts/download_model.py")
            raise RuntimeError("Модель не найдена")

        llm.load()

    # Обработка групп
    all_threads = []
    stats = {
        "groups_processed": 0,
        "threads_found": 0,
        "messages_covered": 0
    }

    assert llm is not None, "LLM engine is required"

    for i, group in enumerate(groups, 1):
        logger.info(f"Обработка группы {i}/{len(groups)}")

        result = extract_threads_from_group(
            group,
            group_number=i,
            llm=llm,
            config=config,
            max_tokens=config.llm.get("max_tokens") or 2048
        )

        all_threads.extend(result.threads)

        stats["groups_processed"] += 1
        stats["threads_found"] += len(result.threads)
        stats["messages_covered"] += result.total_messages

    # Сохранение результатов
    threads_path = config.processed_dir / "chat" / "threads.json"
    logger.info(f"Сохранение {len(all_threads)} веток в {threads_path}")

    output_data = {
        "total_groups": stats["groups_processed"],
        "total_threads": stats["threads_found"],
        "total_messages_covered": stats["messages_covered"],
        "threads": [t.model_dump() for t in all_threads]
    }

    with open(threads_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info("Этап 1 завершён")

    return stats
