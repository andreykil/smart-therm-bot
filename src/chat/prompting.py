"""Chat-specific prompt policy и composition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.utils.prompt_manager import PromptManager

from .models import ChatMessage


class ChatPrompting:
    """Сборка chat messages и выбор chat prompt policy."""

    def __init__(self, prompt_manager: PromptManager | None = None):
        self.prompt_manager = prompt_manager or PromptManager()

    def get_system_prompt(self, use_rag: bool, system_prompt_override: str | None = None) -> str:
        """Получить системный промпт для текущего режима чата."""
        if system_prompt_override:
            return system_prompt_override.strip()

        base = self.prompt_manager.get_prompt("chat_system_base")
        policy_name = "chat_with_rag_policy" if use_rag else "chat_without_rag_policy"
        policy = self.prompt_manager.get_prompt(policy_name)
        return f"{base}\n\n{policy}".strip()

    def build_user_message(self, user_question: str, rag_context: str | None = None, use_rag: bool = False) -> str:
        """Собрать финальный user block для модели."""
        user_blocks: list[str] = []

        if use_rag and rag_context and rag_context.strip():
            context_block = self.prompt_manager.get_prompt(
                "chat_context_block",
                rag_context=rag_context.strip(),
            ).strip()
            if context_block:
                user_blocks.append(context_block)

        question_block = self.prompt_manager.get_prompt(
            "chat_question_block",
            user_question=user_question.strip(),
        ).strip()
        user_blocks.append(question_block)

        return "\n\n".join(user_blocks)

    def normalize_history(self, history: Iterable[ChatMessage | Mapping[str, object]]) -> list[dict[str, str]]:
        """Нормализовать историю до user/assistant сообщений для LLM."""
        normalized: list[dict[str, str]] = []

        for message in history:
            if isinstance(message, ChatMessage):
                role = message.role
                content = message.content.strip()
            else:
                role = str(message.get("role", "")).strip().lower()
                content = str(message.get("content", "")).strip()

            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue

            normalized.append({"role": role, "content": content})

        return normalized

    def build_chat_messages(
        self,
        user_question: str,
        history: Iterable[ChatMessage | Mapping[str, object]],
        *,
        rag_context: str | None = None,
        use_rag: bool = False,
        system_prompt_override: str | None = None,
    ) -> list[dict[str, str]]:
        """Собрать итоговый список messages для Ollama `/api/chat`."""
        system_prompt = self.get_system_prompt(
            use_rag=use_rag,
            system_prompt_override=system_prompt_override,
        )
        user_message = self.build_user_message(
            user_question=user_question,
            rag_context=rag_context,
            use_rag=use_rag,
        )

        return [
            {"role": "system", "content": system_prompt},
            *self.normalize_history(history),
            {"role": "user", "content": user_message},
        ]
