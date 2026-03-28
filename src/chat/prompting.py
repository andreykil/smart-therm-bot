"""Chat-specific prompt policy и composition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.chat.domain.models import DialogMemoryFact
from src.utils.prompt_manager import PromptManager


class ChatPrompting:
    """Сборка chat messages и выбор chat prompt policy."""

    def __init__(self, *, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager

    def get_system_prompt(self, use_rag: bool, system_prompt_override: str | None = None) -> str:
        """Получить системный промпт для текущего режима чата."""
        if system_prompt_override:
            return system_prompt_override.strip()

        base = self.prompt_manager.get_prompt("chat_system_base")
        policy_name = "chat_with_rag_policy" if use_rag else "chat_without_rag_policy"
        policy = self.prompt_manager.get_prompt(policy_name)
        return f"{base}\n\n{policy}".strip()

    def build_user_message(
        self,
        user_question: str,
        rag_context: str | None = None,
        use_rag: bool = False,
        facts: Iterable[DialogMemoryFact] | None = None,
    ) -> str:
        """Собрать финальный user block для модели."""
        user_blocks: list[str] = []

        memory_context = self.render_memory_context(facts or [])
        if memory_context and memory_context.strip():
            memory_block = self.prompt_manager.get_prompt(
                "chat_memory_block",
                memory_context=memory_context.strip(),
            ).strip()
            if memory_block:
                user_blocks.append(memory_block)

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

    @staticmethod
    def render_memory_context(facts: Iterable[DialogMemoryFact]) -> str:
        """Сконвертировать raw facts в текстовый контекст для prompt."""
        normalized_facts = [fact for fact in facts if fact.value.strip()]
        if not normalized_facts:
            return ""
        return "\n".join(f"- {fact.key}: {fact.value}" for fact in normalized_facts)

    def normalize_history(self, history: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
        """Нормализовать историю до user/assistant сообщений для LLM."""
        normalized: list[dict[str, str]] = []

        for message in history:
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
        history: Iterable[Mapping[str, object]],
        *,
        rag_context: str | None = None,
        facts: Iterable[DialogMemoryFact] | None = None,
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
            facts=facts,
            use_rag=use_rag,
        )

        return [
            {"role": "system", "content": system_prompt},
            *self.normalize_history(history),
            {"role": "user", "content": user_message},
        ]
