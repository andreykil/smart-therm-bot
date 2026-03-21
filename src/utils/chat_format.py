"""
Утилиты для форматирования чата

Функции для преобразования сообщений в промпты для различных моделей.
"""


def format_llama_chat_prompt(messages: list[dict]) -> str:
    """
    Форматирование чата для Llama 3.1 / Vikhr Instruct

    Формат:
    <|start_header_id|>system<|end_header_id|>
    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

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

        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

    # Добавить заголовок для ответа ассистента
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    return prompt


def clean_response_text(text: str, strip_spaces: bool = False) -> str:
    """
    Очистка ответа модели от специальных токенов

    Args:
        text: Текст ответа
        strip_spaces: Удалять ли пробелы по краям (True только для первого токена)

    Returns:
        Очищенный текст
    """
    if strip_spaces:
        text = text.strip()

    # Удалить <|begin_of_text|> в начале
    if text.startswith("<|begin_of_text|>"):
        text = text[len("<|begin_of_text|>"):]

    return text
