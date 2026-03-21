"""
Утилиты для работы с JSON

Функции для извлечения и обработки JSON из текста.
"""

import logging

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> str:
    """
    Извлечь первый JSON объект из текста

    Args:
        text: Текст, содержащий JSON

    Returns:
        Строка JSON
    """
    text = text.strip()

    # Найти начало JSON
    start = text.find("{")
    if start == -1:
        logger.warning("JSON не найден (нет открывающей скобки)")
        return text

    # Считаем скобки чтобы найти правильный конец JSON
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                # Найден конец JSON - возвращаем только JSON
                json_str = text[start:i+1]
                
                # Проверка: если после JSON есть текст, логируем
                remaining = text[i+1:].strip()
                if remaining:
                    logger.debug(f"После JSON найдено: {remaining[:50]}...")
                
                return json_str

    # Если не нашли закрывающую скобку, логируем ошибку
    logger.warning(f"Не найдена закрывающая скобка JSON. Depth={depth}")
    logger.warning(f"Текст: {text[:200]}...")
    
    # Пытаемся вернуть как есть
    return text
