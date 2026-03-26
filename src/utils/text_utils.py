"""
Утилиты для обработки текстов

Функции для работы с JSON, форматирования чата и очистки текста.
"""

import logging

logger = logging.getLogger(__name__)


# ============== JSON утилиты ==============

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


# ============== Chat форматирование ==============

def format_llama_chat_prompt(messages: list[dict]) -> str:
    """
    Форматирование чата для Llama 3.1 / Instruct моделей

    Формат:
    <|start_header_id|>system<|end_header_id|>
    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    {assistant}<|eot_id|>

    Args:
        messages: Список сообщений [{"role": "user|assistant", "content": "..."}]

    Returns:
        Отформатированный промпт
    """
    prompt = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        # Очистка от специальных токенов
        for special_token in ["<|begin_of_text|>", "<|start_header_id|>", "<|eot_id|>"]:
            while content.startswith(special_token):
                content = content[len(special_token):]

        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    # Добавить заголовок для ответа ассистента (с двумя newline)
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def clean_response_text(text: str, strip_spaces: bool = False) -> str:
    """
    Очистка ответа модели от специальных токенов

    Args:
        text: Текст ответа
        strip_spaces: Удалять ли специальные токены (True для первого токена)

    Returns:
        Очищенный текст
    """
    # Удалить специальные токены в начале (повторять пока есть совпадения)
    special_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

    # Повторять пока текст начинается со специального токена
    changed = True
    while changed:
        changed = False
        for token in special_tokens:
            if text.startswith(token):
                text = text[len(token):]
                changed = True
                break

    # Также удалить <|end_header_id|> если он остался после role токена
    # (например "assistant<|end_header_id|>Привет" -> "assistantПривет")
    text = text.replace("<|end_header_id|>", "")

    # Для первого токена: удалить только если это только пробелы
    # Пробелы важны для форматирования текста
    if strip_spaces:
        # Удалить только если это только пробелы
        if text.strip() == "" and text != "":
            text = ""

    return text
