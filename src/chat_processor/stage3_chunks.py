"""
Этап 3: Создание RAG чанков

Преобразует дедуплицированные ветки в чанки для векторного поиска
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .models import RAGChunk, Thread
from ..llm import LLMEngine 
from ..llm.factory import create_llm_engine
from ..utils.config import Config

logger = logging.getLogger(__name__)

# Промпт для создания чанка
CHUNK_CREATION_PROMPT = """
Ты создаёшь RAG чанк из набоора сообщений чата SmartTherm для технического помощника.

ВХОД: Сообщения чата (вопросы + ответы + обсуждение + пользовательский опыт)

НАБОР СООБЩЕНИЙ:
Тема: {topic}
Сообщения: {num_messages}
Даты: {date_range}
Есть решение: {has_solution}
Кратко: {summary}

ПОЛНЫЕ СООБЩЕНИЯ:
{messages_text}

ЗАДАЧА: Создай 1 RAG чанк.

ТРЕБОВАНИЯ К content.text:
1. Самодостаточность: текст понятен без контекста чата
2. Полнота: включи проблему, решение, технические детали
3. Конкретика: версии, пины, команды, ссылки на инструкцию
4. Стиль: технический, нейтральный, без "пользователь спросил"
5. Объём: до 500 слов

ТРЕБОВАНИЯ К metadata:
1. tags: 3-10 тегов из списка:
   opentherm, wifi, прошивка, датчики, ds18b20, esp8266, esp32,
   navien, baxi, immergas, котёл, pid, homeassistant, mqtt,
   подключение, ошибка, диагностика, распиновка, веб-интерфейс
2. version: укажи если упомянута (например "0.73")
3. confidence: 0.0-1.0 (полезность обсуждения/полноту ответа)
   - 0.9-1.0: есть ответ от Evgen + подтверждение
   - 0.7-0.9: есть ответ от Evgen
   - 0.5-0.7: есть ответ от пользователя с опытом
   - <0.5: нет решения

ВЕРНИ JSON СТРОГОЙ СТРУКТУРЫ:
{{
  "content": {{
    "summary": "...",
    "text": "..."
  }},
  "metadata": {{
    "tags": ["...", "..."],
    "version": "...",
    "confidence": 0.9,
  }}
}}

ВЫВОД (только JSON, без пояснений):
"""


def load_deduped_threads(path: Path) -> list[Thread]:
    """Загрузить дедуплицированные ветки"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [Thread(**t) for t in data["threads"]]


def load_filtered_messages(path: Path) -> dict[int, dict]:
    """
    Загрузить отфильтрованные сообщения как словарь
    
    Returns:
        {message_id: message_data}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return {msg["id"]: msg for msg in data["messages"]}


def create_chunk_from_thread(
    thread: Thread,
    messages: dict[int, dict],
    llm: LLMEngine,
    max_tokens: int = 2048,
    temperature: float = 0.5
) -> Optional[RAGChunk]:
    """
    Создать RAG чанк из ветки

    Args:
        thread: Ветка
        messages: Словарь сообщений
        llm: LLM движок
        max_tokens: Максимум токенов
        temperature: Температура генерации
    """
    # Получение сообщений ветки
    thread_messages = [
        messages.get(mid) for mid in thread.message_ids
        if mid in messages
    ]
    
    if not thread_messages:
        logger.warning(f"Ветка {thread.thread_id} не имеет сообщений")
        return None

    # Формирование текста сообщений
    messages_text = "\n".join(
        f"[{m.get('date', '')[:10]}] {m.get('from', 'unknown')}: {m.get('text', '')[:300]}"
        for m in thread_messages
        if m is not None
    )
    
    # Формирование промпта
    prompt = CHUNK_CREATION_PROMPT.format(
        topic=thread.topic,
        num_messages=len(thread.message_ids),
        date_range=f"{thread.start_date} — {thread.end_date}",
        has_solution="Да" if thread.has_solution else "Нет",
        summary=thread.summary,
        messages_text=messages_text
    )
    
    # Вызов LLM
    logger.debug(f"Создание чанка для {thread.thread_id}...")

    response = llm.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )
    
    # Парсинг ответа
    try:
        # Очистка от markdown
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        chunk_data = json.loads(response)

        # Создание чанка с явными объектами
        from .models import ChunkSource, ChunkContent, ChunkMetadata
        
        chunk = RAGChunk(
            chunk_id=f"tg_{thread.thread_id}",
            source=ChunkSource(
                type="telegram",
                message_ids=thread.message_ids,
                date_range=f"{thread.start_date} — {thread.end_date}"
            ),
            content=ChunkContent(
                summary=chunk_data.get("content", {}).get("summary", thread.summary),
                text=chunk_data.get("content", {}).get("text", thread.summary)
            ),
            metadata=ChunkMetadata(
                tags=chunk_data.get("metadata", {}).get("tags", []),
                version=chunk_data.get("metadata", {}).get("version"),
                confidence=chunk_data.get("metadata", {}).get("confidence", 0.5)
            )
        )

        return chunk

    except Exception as e:
        logger.error(f"Ошибка парсинга чанка {thread.thread_id}: {e}")

        # Fallback: создать чанк из summary
        from .models import ChunkSource, ChunkContent, ChunkMetadata
        
        return RAGChunk(
            chunk_id=f"tg_{thread.thread_id}",
            source=ChunkSource(
                type="telegram",
                message_ids=thread.message_ids,
                date_range=f"{thread.start_date} — {thread.end_date}"
            ),
            content=ChunkContent(
                summary=thread.summary,
                text=thread.summary
            ),
            metadata=ChunkMetadata(
                tags=[],
                version=None,
                confidence=0.3
            )
        )


def run_stage3(
    config: Config,
    llm: Optional[LLMEngine] = None,
    sample_size: int = 5
) -> dict:
    """
    Запустить Этап 3

    Args:
        config: Конфигурация
        llm: LLM движок
        sample_size: Размер выборки для валидации
    """
    logger.info("=" * 60)
    logger.info("ЭТАП 3: Создание RAG чанков")
    logger.info("=" * 60)

    # Загрузка дедуплицированных веток
    threads_path = config.processed_dir / "chat" / "threads_deduped.json"
    logger.info(f"Загрузка веток из {threads_path}")
    threads = load_deduped_threads(threads_path)
    logger.info(f"Загружено {len(threads)} веток")

    # Загрузка сообщений
    filtered_path = config.processed_dir / "chat" / "messages_filtered.json"
    logger.info(f"Загрузка сообщений из {filtered_path}")
    messages = load_filtered_messages(filtered_path)
    logger.info(f"Загружено {len(messages)} сообщений")

    # Инициализация LLM
    if llm is None:
        logger.info(f"Инициализация LLM: {config.llm['model']} ({config.llm['quantization']})")
        llm = create_llm_engine(
            model_id=config.llm.get("model") or "vikhr-nemo-12b-instruct-r",
            quantization=config.llm.get("quantization") or "Q8_0",
            n_ctx=config.llm.get("context_size") or 8192,
            verbose=False
        )

        if not llm.model_exists():
            logger.error(f"Модель не найдена")
            raise RuntimeError("Модель не найдена")

        llm.load()

    # Создание чанков
    chunks = []
    stats = {
        "threads_processed": 0,
        "chunks_created": 0,
        "fallback_chunks": 0
    }

    stage3_cfg = config.llm.get("stage3", {})

    for i, thread in enumerate(threads, 1):
        logger.debug(f"Обработка ветки {i}/{len(threads)}: {thread.thread_id}")

        chunk = create_chunk_from_thread(
            thread=thread,
            messages=messages,
            llm=llm,
            max_tokens=stage3_cfg.get("max_tokens") or 2048,
            temperature=stage3_cfg.get("temperature") or 0.5
        )

        if chunk:
            chunks.append(chunk)
            stats["chunks_created"] += 1

            # Проверка на fallback (низкая уверенность)
            if chunk.metadata.confidence < 0.4:
                stats["fallback_chunks"] += 1

        stats["threads_processed"] += 1

        # Прогресс
        if i % 100 == 0:
            logger.info(f"Прогресс: {i}/{len(threads)} веток обработано")

    # Сохранение результатов (JSONL)
    chunks_path = config.processed_dir / "chat" / "chunks_rag.jsonl"
    logger.info(f"Сохранение {len(chunks)} чанков в {chunks_path}")

    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_jsonl() + "\n")

    # Валидация: выборка для ручной проверки
    if sample_size > 0 and chunks:
        import random
        sample = random.sample(chunks, min(sample_size, len(chunks)))

        sample_path = chunks_path.parent / "chunks_sample.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.model_dump() for c in sample],
                f,
                ensure_ascii=False,
                indent=2
            )

        logger.info(f"Выборка сохранена в {sample_path}")

    logger.info("Этап 3 завершён")

    return stats
