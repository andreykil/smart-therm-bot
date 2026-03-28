"""Тонкий multi-dialog transport-helper для Telegram поверх chat application."""

from __future__ import annotations

from dataclasses import dataclass

from src.chat.registry import DialogRegistry


@dataclass(slots=True, frozen=True)
class TelegramTransportRequest:
    """Нормализованный вход transport-слоя из Telegram update."""

    chat_id: int
    text: str
    user_id: int | None = None
    thread_id: int | None = None

    @property
    def dialog_key(self) -> str:
        """Построить ключ изолированного диалога для registry."""
        key = f"chat:{self.chat_id}"
        if self.thread_id is not None:
            key += f":thread:{self.thread_id}"
        return key


@dataclass(slots=True)
class TelegramTransportResponse:
    """Нормализованный результат обработки входящего Telegram текста."""

    text: str
    is_command: bool = False
    dialog_key: str = ""


class TelegramTransport:
    """Telegram transport без привязки к конкретной bot framework библиотеке."""

    def __init__(self, registry: DialogRegistry):
        self.registry = registry

    def handle_request(self, request: TelegramTransportRequest) -> TelegramTransportResponse:
        """Обработать Telegram request в рамках изолированного dialog context."""
        lease = self.registry.acquire(request.dialog_key)

        try:
            with lease.lock:
                session = lease.session
                command_result = session.try_execute_command(request.text)
                if command_result is not None:
                    return TelegramTransportResponse(
                        text="\n".join(command_result.lines),
                        is_command=True,
                        dialog_key=request.dialog_key,
                    )

                response = session.run_text(
                    request.text,
                    metadata={
                        "chat_id": request.chat_id,
                        "user_id": request.user_id,
                        "thread_id": request.thread_id,
                        "dialog_key": request.dialog_key,
                    },
                )
                return TelegramTransportResponse(
                    text=response.assistant_message,
                    dialog_key=request.dialog_key,
                )
        finally:
            self.registry.release(lease)

    def handle_text(
        self,
        text: str,
        *,
        chat_id: int,
        user_id: int | None = None,
        thread_id: int | None = None,
    ) -> TelegramTransportResponse:
        """Совместимый helper: собрать request и обработать его через registry."""
        return self.handle_request(
            TelegramTransportRequest(
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
                text=text,
            )
        )
