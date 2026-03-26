"""
PromptBuilder — сборка финальных messages для chat-сценариев.

Использует PromptManager как источник шаблонов и формирует
итоговый список сообщений для LLM.
"""

from typing import Any, Optional

from .prompt_manager import PromptManager


class PromptBuilder:
    """Сборщик промптов для диалогового режима"""

    def __init__(self, prompt_manager: Optional[PromptManager] = None):
        self.prompt_manager = prompt_manager or PromptManager()

    def get_system_prompt(self, use_rag: bool, system_prompt_override: Optional[str] = None) -> str:
        """Собрать системный промпт по режиму RAG"""
        if system_prompt_override:
            return system_prompt_override.strip()

        base = self.prompt_manager.get_prompt("chat_system_base")
        policy_name = "chat_with_rag_policy" if use_rag else "chat_without_rag_policy"
        policy = self.prompt_manager.get_prompt(policy_name)
        return f"{base}\n\n{policy}".strip()

    @staticmethod
    def _normalize_history(history: list[dict]) -> list[dict[str, str]]:
        """Оставить только корректные user/assistant сообщения"""
        normalized: list[dict[str, str]] = []

        for msg in history:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()

            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue

            normalized.append({"role": role, "content": content})

        return normalized

    def build_chat_messages(
        self,
        user_question: str,
        history: list[dict],
        rag_context: Optional[str] = None,
        use_rag: bool = False,
        system_prompt_override: Optional[str] = None
    ) -> list[dict[str, str]]:
        """
        Собрать список сообщений для /api/chat.

        Args:
            user_question: Новый вопрос пользователя
            history: История диалога (только user/assistant)
            rag_context: Контекст из RAG (если найден)
            use_rag: Флаг режима RAG
            system_prompt_override: Переопределение системного промпта
        """
        system_prompt = self.get_system_prompt(use_rag=use_rag, system_prompt_override=system_prompt_override)
        normalized_history = self._normalize_history(history)

        user_blocks: list[str] = []

        if use_rag and rag_context and rag_context.strip():
            context_block = self.prompt_manager.get_prompt(
                "chat_context_block",
                rag_context=rag_context.strip()
            ).strip()
            if context_block:
                user_blocks.append(context_block)

        question_block = self.prompt_manager.get_prompt(
            "chat_question_block",
            user_question=user_question.strip()
        ).strip()
        user_blocks.append(question_block)

        user_message = "\n\n".join(user_blocks)

        return [
            {"role": "system", "content": system_prompt},
            *normalized_history,
            {"role": "user", "content": user_message}
        ]

    @staticmethod
    def _format_chunk_messages(group_messages: list[Any]) -> str:
        """Сформировать текст сообщений группы для chunk_creation"""
        return "\n".join(
            f"[{str(msg.date)[:10]}] {msg.from_}: {msg.text}"
            for msg in group_messages
        )

    def build_chunk_creation_messages(self, group_messages: list[Any], last_message_date: str) -> list[dict[str, str]]:
        """Собрать messages для создания RAG чанка из группы сообщений"""
        messages_text = self._format_chunk_messages(group_messages)
        system_prompt = self.prompt_manager.get_prompt("chunk_creation_system")
        user_prompt = self.prompt_manager.get_prompt(
            "chunk_creation_user",
            messages_text=messages_text,
            date=last_message_date,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
