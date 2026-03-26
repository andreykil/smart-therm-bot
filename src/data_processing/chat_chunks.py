"""
Создание RAG чанков из чата

Логика:
1. Загрузка отфильтрованных сообщений
2. Разбиение на группы с перекрытием
3. Сохранение групп (опционально, для отладки)
4. Создание RAG чанков из групп
"""

import json
import logging
import shutil
from pathlib import Path

from .models import FilteredMessage, Group, RAGChunk, ChunkContent, ChunkMetadata
from ..llm import OllamaClient
from ..utils.config import Config
from ..utils.text_utils import extract_json_from_text
from ..utils.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

def load_filtered_messages(path: Path) -> list[FilteredMessage]:
    """Загрузить отфильтрованные сообщения"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [FilteredMessage(**msg) for msg in data["messages"]]


def create_groups(
    messages: list[FilteredMessage],
    group_size: int,
    overlap_size: int
) -> list[Group]:
    """
    Разбить сообщения на группы с перекрытием

    Args:
        messages: Список сообщений
        group_size: Размер группы
        overlap_size: Размер перекрытия

    Returns:
        Список групп
    """
    if group_size <= 0:
        raise ValueError(f"group_size должен быть > 0, получено: {group_size}")
    if overlap_size < 0:
        raise ValueError(f"overlap_size должен быть >= 0, получено: {overlap_size}")
    if overlap_size >= group_size:
        raise ValueError(
            f"overlap_size ({overlap_size}) должен быть меньше group_size ({group_size})"
        )

    groups = []
    step = group_size - overlap_size

    for i in range(0, len(messages), step):
        group_msgs = messages[i:i + group_size]
        if len(group_msgs) >= 10:  # Минимальный размер группы
            group = Group(
                group_id=f"g{len(groups) + 1:03d}",
                message_ids=[msg.id for msg in group_msgs],
                start_date=group_msgs[0].date[:10],
                end_date=group_msgs[-1].date[:10],
                participant_count=len(set(msg.from_ for msg in group_msgs))
            )
            groups.append(group)

    logger.info(f"Создано {len(groups)} групп (размер={group_size}, перекрытие={overlap_size})")
    return groups


def save_groups_to_files(groups: list[Group], messages: dict[int, FilteredMessage], output_dir: Path) -> None:
    """
    Сохранить группы в файлы (для отладки)

    Args:
        groups: Список групп
        messages: Словарь сообщений
        output_dir: Директория для сохранения
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        group_file = output_dir / f"{group.group_id}.json"
        group_messages = [
            {
                "id": msg.id,
                "from": msg.from_,
                "date": msg.date,
                "text": msg.text,
                "reply_to_message_id": msg.reply_to_message_id
            }
            for msg in messages.values()
            if msg.id in group.message_ids
        ]

        group_data = {
            "group_id": group.group_id,
            "message_count": len(group_messages),
            "date_range": f"{group.start_date} — {group.end_date}",
            "participant_count": group.participant_count,
            "messages": group_messages
        }

        with open(group_file, "w", encoding="utf-8") as f:
            json.dump(group_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Группы сохранены в {output_dir}")


def create_chunk_from_group(
    group: Group,
    messages: dict[int, FilteredMessage],
    llm: OllamaClient,
    last_message_date: str,
    max_tokens: int = 2048,
    temperature: float = 0.5
) -> RAGChunk | None:
    """
    Создать один RAG чанк из группы

    Args:
        group: Группа сообщений
        messages: Словарь сообщений
        llm: Ollama клиент
        last_message_date: Дата последнего сообщения в группе
        max_tokens: Максимум токенов
        temperature: Температура генерации
    """
    group_messages = [messages[mid] for mid in group.message_ids if mid in messages]

    if not group_messages:
        logger.warning(f"Группа {group.group_id} не имеет сообщений")
        return None

    # Формирование промпта через PromptBuilder
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_chunk_creation_prompt(
        group_messages=group_messages,
        last_message_date=last_message_date,
    )

    logger.debug(f"Создание чанков для {group.group_id}...")

    response = llm.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )

    # Парсинг ответа (ожидается один JSON-объект чанка)
    try:
        json_str = extract_json_from_text(response.strip())
        chunk_data = json.loads(json_str)
        return RAGChunk.model_validate(chunk_data)

    except Exception as e:
        logger.error(f"Ошибка парсинга чанков {group.group_id}: {e}")

        # Fallback возвращает один чанк в том же формате, что и LLM
        fallback_text = response.strip() or "Не удалось распарсить ответ LLM."

        return RAGChunk(
            content=ChunkContent(
                text=fallback_text,
                code=""
            ),
            metadata=ChunkMetadata(
                source="telegram chat",
                date=last_message_date,
                tags=[],
                version=None,
                confidence=0.3
            )
        )


def run_chunks(
    config: Config,
    llm: OllamaClient | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
    save_groups: bool = False,
    groups_dir: Path | None = None
) -> dict:
    """
    Запустить создание чанков

    Args:
        config: Конфигурация
        llm: Ollama клиент (если None, создаётся новый)
        input_path: Входной файл с отфильтрованными сообщениями
        output_path: Выходной файл с чанками (jsonl)
        save_groups: Сохранять ли группы в файлы
        groups_dir: Директория для групп
    """
    logger.info("=" * 60)
    logger.info("СОЗДАНИЕ RAG ЧАНКОВ")
    logger.info("=" * 60)

    # Загрузка сообщений
    input_path = input_path or config.processed_dir / "chat" / "messages_filtered.json"

    logger.info(f"Загрузка сообщений из {input_path}")
    messages_list = load_filtered_messages(input_path)
    logger.info(f"Загружено {len(messages_list)} сообщений")

    # Создание групп
    chat_cfg = config.chat_processing
    groups = create_groups(
        messages_list,
        chat_cfg.get("group_size", 50),
        chat_cfg.get("overlap_size", 5)
    )

    # Сохранение групп (опционально)
    if save_groups:
        groups_dir = groups_dir or config.processed_dir / "chat" / "groups"
        messages_dict = {msg.id: msg for msg in messages_list}
        save_groups_to_files(groups, messages_dict, groups_dir)

    # Инициализация LLM
    if llm is None:
        model_name = config.llm.get("model") or "llama3.1"
        logger.info(f"Инициализация Ollama: {model_name}")
        llm = OllamaClient(
            model=model_name,
            base_url=config.llm.get("base_url", "http://localhost:11434"),
            verbose=False,
            think=config.llm.get("think")
        )

        if not llm.model_exists():
            logger.error(f"Модель не найдена в Ollama: {model_name}")
            raise RuntimeError(f"Модель не найдена: {model_name}")

        llm.load()

    # Создание чанков
    messages_dict = {msg.id: msg for msg in messages_list}
    chunks = []
    stats = {
        "groups_processed": 0,
        "chunks_created": 0,
        "fallback_chunks": 0
    }

    for i, group in enumerate(groups, 1):
        logger.info(f"Обработка группы {i}/{len(groups)}: {group.group_id}")

        # Получить дату последнего сообщения в группе
        last_date = messages_dict[group.message_ids[-1]].date[:10]
        chunk = create_chunk_from_group(
            group=group,
            messages=messages_dict,
            llm=llm,
            last_message_date=last_date,
            max_tokens=config.llm.get("max_tokens") or 2048,
            temperature=config.llm.get("temperature") or 0.5
        )

        if chunk:
            chunks.append(chunk)
            stats["chunks_created"] += 1

            if chunk.metadata.confidence < 0.4:
                stats["fallback_chunks"] += 1

        stats["groups_processed"] += 1

        if i % 10 == 0:
            logger.info(f"Прогресс: {i}/{len(groups)} групп обработано")

    # Сохранение чанков (JSONL)
    output_path = output_path or config.processed_dir / "chat" / "chunks_rag.jsonl"

    logger.info(f"Сохранение {len(chunks)} чанков в {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_jsonl() + "\n")

    logger.info("Создание чанков завершено")

    return stats
