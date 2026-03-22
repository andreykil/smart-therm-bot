"""
Этап 3: Создание RAG чанков

Преобразует дедуплицированные ветки в чанки для векторного поиска
"""

import json
import logging
from pathlib import Path

from .models import RAGChunk, Thread
from ..llm import LLMEngine
from ..llm.factory import create_llm_engine
from ..utils.config import Config
from ..utils.json_utils import extract_json_from_text

logger = logging.getLogger(__name__)

# Промпт для создания чанка
CHUNK_CREATION_PROMPT = """<|start_header_id|>system<|end_header_id|>

Ты создаёшь RAG чанк из сообщений чата SmartTherm для технического помощника.
Твоя задача — создать выжимку фактической информации.
Никаких пояснений, только JSON.

<|eot_id|><|start_header_id|>user<|end_header_id|>

ВХОД: Сообщения чата (вопросы + ответы + обсуждение + пользовательский опыт)

НАБОР СООБЩЕНИЙ:
Тема: {topic}
Сообщения: {num_messages}
Даты: {date_range}

ПОЛНЫЕ СООБЩЕНИЯ:
{messages_text}

ЗАДАЧА: Создай 1 RAG чанк.

ТРЕБОВАНИЯ К content.text:
1. ГЛАВНОЕ: собери все полезные факты об обсуждаемой теме
2. Полнота: включи проблему, решение, технические детали
3. Конкретика: ВСЕ факты, версии, пины, команды, термины
4. ТОЛЬКО ФАКТЫ: НЕ пиши "пользователь спросил ...", "В сообщении указано ...", "В тексте обсуждается ..."
5. Объём: до 500 слов

ТРЕБОВАНИЯ К metadata:
1. tags: 3-10 тегов из списка:
   opentherm, wifi, прошивка, датчики, ds18b20, esp8266, esp32,
   navien, baxi, immergas, котёл, pid, homeassistant, mqtt,
   подключение, ошибка, диагностика, распиновка, веб-интерфейс
2. version: укажи если упомянута (например "0.73")
3. confidence: 0.0-1.0 (полезность обсуждения)
   - 0.9-1.0: Много фактической информации, технических деталей
   - 0.7-0.9: Всё описано достаточно подробно, есть полезный опыт
   - 0.5-0.7: Есть полезные факты
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
    "confidence": 0.9
  }}
}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def load_threads(path: Path) -> list[Thread]:
    """Загрузить ветки из файла"""
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
    temperature: float = 0.5,
    debug: bool = False
) -> RAGChunk | None:
    """
    Создать RAG чанк из ветки

    Args:
        thread: Ветка
        messages: Словарь сообщений
        llm: LLM движок
        max_tokens: Максимум токенов
        temperature: Температура генерации
        debug: Если True, выводить сырой ответ LLM
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

    # Debug режим: вывод сырого ответа
    if debug:
        print(f"\n{'='*60}")
        print(f"ЧАНК {thread.thread_id} - СЫРОЙ ОТВЕТ LLM:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")

    # Парсинг ответа
    try:
        # Очистка от markdown
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Извлечь JSON из ответа (на случай лишнего текста)
        json_str = extract_json_from_text(response)
        chunk_data = json.loads(json_str)

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
                summary=chunk_data.get("content", {}).get("summary", thread.topic),
                text=chunk_data.get("content", {}).get("text", thread.topic)
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

        # Fallback: создать чанк из topic + метаданных
        from .models import ChunkSource, ChunkContent, ChunkMetadata

        # Создаём более длинный текст для fallback
        fallback_text = f"{thread.topic}. Сообщений: {len(thread.message_ids)}. Период: {thread.start_date} — {thread.end_date}."

        return RAGChunk(
            chunk_id=f"tg_{thread.thread_id}",
            source=ChunkSource(
                type="telegram",
                message_ids=thread.message_ids,
                date_range=f"{thread.start_date} — {thread.end_date}"
            ),
            content=ChunkContent(
                summary=fallback_text,
                text=fallback_text
            ),
            metadata=ChunkMetadata(
                tags=[],
                version=None,
                confidence=0.3
            )
        )


def run_stage3(
    config: Config,
    llm: LLMEngine | None = None,
    threads_path: Path | None = None,
    messages_path: Path | None = None,
    output_path: Path | None = None,
    sample_size: int = 5,
    debug: bool = False
) -> dict:
    """
    Запустить Этап 3

    Args:
        config: Конфигурация
        llm: LLM движок
        threads_path: Входной файл с ветками (по умолчанию: processed/chat/threads.json)
        messages_path: Входной файл с сообщениями (по умолчанию: processed/chat/messages_filtered.json)
        output_path: Выходной файл (по умолчанию: processed/chat/chunks_rag.jsonl)
        sample_size: Размер выборки для валидации
        debug: Если True, выводить сырые ответы LLM
    """
    logger.info("=" * 60)
    logger.info("ЭТАП 3: Создание RAG чанков")
    logger.info("=" * 60)

    # Загрузка веток
    threads_path = threads_path or config.processed_dir / "chat" / "threads.json"

    logger.info(f"Загрузка веток из {threads_path}")
    threads = load_threads(threads_path)
    logger.info(f"Загружено {len(threads)} веток")

    # Загрузка сообщений
    messages_path = messages_path or config.processed_dir / "chat" / "messages_filtered.json"
    
    logger.info(f"Загрузка сообщений из {messages_path}")
    messages = load_filtered_messages(messages_path)
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
            temperature=stage3_cfg.get("temperature") or 0.5,
            debug=debug
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
    output_path = output_path or config.processed_dir / "chat" / "chunks_rag.jsonl"
    
    logger.info(f"Сохранение {len(chunks)} чанков в {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_jsonl() + "\n")

    # Валидация: выборка для ручной проверки
    if sample_size > 0 and chunks:
        import random
        sample = random.sample(chunks, min(sample_size, len(chunks)))

        sample_path = output_path.parent / "chunks_sample.json"
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
