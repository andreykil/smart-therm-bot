"""Тонкий transport-helper для Telegram поверх универсального chat application."""

from __future__ import annotations

from dataclasses import dataclass

from src.chat import ChatApp, CommandParser


@dataclass(slots=True)
class TelegramTransportResponse:
    """Нормализованный результат обработки входящего Telegram текста."""

    text: str
    is_command: bool = False


class TelegramTransport:
    """Telegram transport без привязки к конкретной bot framework библиотеке."""

    def __init__(self, app: ChatApp):
        self.app = app

    def handle_text(self, text: str) -> TelegramTransportResponse:
        """Обработать входящий текст как команду или обычный chat turn."""
        if CommandParser.is_command(text):
            result = self.app.commands.execute(text)
            return TelegramTransportResponse(
                text="\n".join(result.lines),
                is_command=True,
            )

        request = self.app.runtime.build_request(text)
        prepared = self.app.service.prepare_turn(request) if self.app.runtime.debug else None
        response = self.app.service.run_turn(request, prepared=prepared)
        return TelegramTransportResponse(text=response.assistant_message)
