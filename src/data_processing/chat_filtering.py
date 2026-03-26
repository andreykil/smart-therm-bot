"""
Фильтрация чата — удаление шума

Удаляет:
- Сервисные сообщения
- Эмодзи/стикеры
- Флуд (спасибо, ок, и т.д.)
- Дубликаты
"""

import json
import logging
import re
from pathlib import Path

from .models import FilteredMessage, TelegramChat, TelegramMessage
from ..utils.config import Config

logger = logging.getLogger(__name__)


def extract_text_from_message(msg: TelegramMessage) -> str:
    """Извлечь текст из сообщения (поддержка массива и строки)"""
    if msg.text is None:
        return ""

    if isinstance(msg.text, str):
        return msg.text.strip()

    if isinstance(msg.text, list):
        texts = []
        for item in msg.text:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "plain":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "link":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "mention":
                    texts.append(item.get("text", ""))
        return " ".join(texts).strip()

    return ""


def is_emoji_only(text: str) -> bool:
    """Проверить, состоит ли текст только из эмодзи"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.fullmatch(text))


def is_flood_message(text: str, stop_words: list[str] | None = None) -> bool:
    """Проверить на флуд: стоп-слово и длина сообщения < 3 слов."""
    if stop_words is None:
        stop_words = []

    text_lower = re.sub(r"\s+", " ", text.lower()).strip()

    if "?" in text_lower or "почему" in text_lower or "как" in text_lower:
        return False

    word_count = len(text_lower.split())
    if word_count >= 3:
        return False

    for word in stop_words:
        normalized_word = word.lower().strip()
        if text_lower == normalized_word or text_lower.startswith(normalized_word + " "):
            return True

    return False


def filter_messages(
    input_path: Path,
    output_path: Path,
    stop_words: list | None = None
) -> dict:
    """
    Фильтрация сообщений

    Args:
        input_path: Входной файл
        output_path: Выходной файл
        stop_words: Стоп-слова для фильтрации

    Returns:
        Статистика фильтрации
    """
    logger.info(f"Чтение чата из {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chat_data = json.load(f)

    chat = TelegramChat(**chat_data)
    total_messages = len(chat.messages)
    logger.info(f"Всего сообщений: {total_messages}")

    filtered_messages = []
    removed_service = 0
    removed_emoji = 0
    removed_flood = 0
    removed_duplicate = 0
    saved_developer = 0

    seen_messages = {}
    developer_names = {"Evgen", "evgen", "Evgen2", "admin"}

    for msg in chat.messages:
        if msg.type == "service":
            removed_service += 1
            continue

        text = extract_text_from_message(msg)

        if not text:
            removed_emoji += 1
            continue

        if is_emoji_only(text):
            removed_emoji += 1
            continue

        if is_flood_message(text, stop_words):
            removed_flood += 1
            continue

        from_name = msg.from_ or "unknown"
        msg_key = (text, from_name)
        msg_timestamp = int(msg.date_unixtime)

        if msg_key in seen_messages:
            last_timestamp = seen_messages[msg_key]
            if abs(msg_timestamp - last_timestamp) < 60:
                removed_duplicate += 1
                continue

        seen_messages[msg_key] = msg_timestamp

        is_developer = from_name in developer_names
        if is_developer:
            saved_developer += 1

        filtered_msg = FilteredMessage(
            id=msg.id,
            date=msg.date,
            date_unixtime=int(msg.date_unixtime),
            **{"from": from_name},
            text=text,
            reply_to_message_id=msg.reply_to_message_id,
            is_from_developer=is_developer
        )
        filtered_messages.append(filtered_msg)

    filtered_messages.sort(key=lambda x: x.id)

    logger.info(f"Сохранение {len(filtered_messages)} сообщений в {output_path}")

    output_data = {
        "total_original": total_messages,
        "total_filtered": len(filtered_messages),
        "removed": {
            "service": removed_service,
            "emoji": removed_emoji,
            "flood": removed_flood,
            "duplicate": removed_duplicate
        },
        "saved_developer": saved_developer,
        "messages": [msg.model_dump() for msg in filtered_messages]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    stats = {
        "total_original": total_messages,
        "total_filtered": len(filtered_messages),
        "removed_total": removed_service + removed_emoji + removed_flood + removed_duplicate,
        "removed_service": removed_service,
        "removed_emoji": removed_emoji,
        "removed_flood": removed_flood,
        "removed_duplicate": removed_duplicate,
        "saved_developer": saved_developer,
        "filter_percentage": (len(filtered_messages) / total_messages * 100) if total_messages > 0 else 0
    }

    logger.info(f"Фильтрация завершена: {stats['total_filtered']}/{stats['total_original']} ({stats['filter_percentage']:.1f}%)")

    return stats


def run_filtering(
    config: Config,
    input_path: Path | None = None,
    output_path: Path | None = None
) -> dict:
    """
    Запустить фильтрацию

    Args:
        config: Конфигурация
        input_path: Входной файл (по умолчанию: raw/chat_history.json)
        output_path: Выходной файл (по умолчанию: processed/chat/messages_filtered.json)
    """
    logger.info("=" * 60)
    logger.info("ФИЛЬТРАЦИЯ ЧАТА")
    logger.info("=" * 60)

    chat_cfg = config.chat_processing

    input_path = input_path or config.raw_dir / "chat_history.json"
    output_path = output_path or config.processed_dir / "chat" / "messages_filtered.json"

    stats = filter_messages(
        input_path,
        output_path,
        stop_words=chat_cfg.get("stop_words")
    )

    logger.info("Статистика:")
    logger.info(f"  Всего исходно: {stats['total_original']}")
    logger.info(f"  После фильтрации: {stats['total_filtered']}")
    logger.info(f"  Удалено: {stats['removed_total']}")
    logger.info(f"    - Сервисные: {stats['removed_service']}")
    logger.info(f"    - Эмодзи: {stats['removed_emoji']}")
    logger.info(f"    - Флуд: {stats['removed_flood']}")
    logger.info(f"    - Дубликаты: {stats['removed_duplicate']}")
    logger.info(f"  Сохранено от разработчика: {stats['saved_developer']}")
    logger.info(f"  Процент сохранённых: {stats['filter_percentage']:.1f}%")

    return stats
