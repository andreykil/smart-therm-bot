"""
Утилиты для работы с JSON

Функции для извлечения и обработки JSON из текста.
"""


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
                return text[start:i+1]

    # Если не нашли закрывающую скобку, возвращаем как есть
    return text
