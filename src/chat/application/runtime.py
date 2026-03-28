"""Runtime-настройки transport-независимого chat-flow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChatRuntime:
    """Изменяемое runtime-состояние одной chat session."""

    max_tokens: int = 1024
    temperature: float = 0.7
    use_rag: bool = False
    system_prompt_override: str | None = None
    debug: bool = False

    def toggle_rag(self) -> bool:
        self.use_rag = not self.use_rag
        return self.use_rag

    def stats(self) -> dict[str, object]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "use_rag": self.use_rag,
            "system_prompt_override": bool(self.system_prompt_override),
            "debug": self.debug,
        }
