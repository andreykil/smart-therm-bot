"""Runtime-настройки transport-независимого chat-flow."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ChatTurnRequest


@dataclass(slots=True)
class ChatRuntime:
    """Изменяемое runtime-состояние transport-слоя."""

    max_tokens: int = 1024
    temperature: float = 0.7
    use_rag: bool = False
    system_prompt_override: str | None = None
    debug: bool = False

    def build_request(self, user_message: str) -> ChatTurnRequest:
        """Собрать request одного хода из текущего runtime-состояния."""
        return ChatTurnRequest(
            user_message=user_message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            use_rag=self.use_rag,
            system_prompt_override=self.system_prompt_override,
        )

    def toggle_rag(self) -> bool:
        """Переключить флаг использования RAG."""
        self.use_rag = not self.use_rag
        return self.use_rag

    def stats(self) -> dict[str, object]:
        """Краткое runtime-представление текущих настроек."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "use_rag": self.use_rag,
            "system_prompt_override": bool(self.system_prompt_override),
            "debug": self.debug,
        }
