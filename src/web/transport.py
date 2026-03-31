"""Minimal web transport for SmartTherm chat application."""

from __future__ import annotations

from dataclasses import dataclass
import html

from src.chat.application.command_service import CommandParser, CommandService
from src.chat.registry import DialogRegistry


@dataclass(slots=True, frozen=True)
class WebTransportRequest:
    """Normalized request for web transport."""

    session_id: str
    text: str

    @property
    def dialog_key(self) -> str:
        return f"web:{self.session_id}"


@dataclass(slots=True)
class WebTransportResponse:
    """Normalized response for the web UI."""

    text: str
    render_mode: str = "text"
    is_command: bool = False
    dialog_key: str = ""
    reset_transcript: bool = False
    rag_enabled: bool = False


def build_web_start_html() -> str:
    """Render the `/start` greeting for the web UI."""
    lines = [
        "<b>Привет! Я бот поддержки SmartTherm.</b>",
        "Здесь доступен минимальный веб-интерфейс. Введите вопрос или slash-команду в одно поле ввода.",
        "",
        "<b>Команды:</b>",
        "<pre>",
        *(f"{html.escape(name):<16} — {html.escape(description)}" for name, description in CommandService.command_items()),
        "</pre>",
    ]
    return "\n".join(lines)


class WebTransport:
    """HTTP-facing transport over the shared chat application session model."""

    def __init__(self, registry: DialogRegistry):
        self.registry = registry

    def handle_request(self, request: WebTransportRequest) -> WebTransportResponse:
        lease = self.registry.acquire(request.dialog_key)

        try:
            with lease.lock:
                session = lease.session
                parsed = CommandParser.parse(request.text)
                if parsed is not None and parsed.name == "/start":
                    return WebTransportResponse(
                        text=build_web_start_html(),
                        render_mode="html",
                        is_command=True,
                        dialog_key=request.dialog_key,
                        rag_enabled=session.rag_enabled,
                    )

                command_result = session.try_execute_command(request.text)
                if command_result is not None:
                    return WebTransportResponse(
                        text="\n".join(command_result.lines),
                        render_mode="html" if command_result.parse_mode == "HTML" else "text",
                        is_command=True,
                        dialog_key=request.dialog_key,
                        reset_transcript=command_result.reset_transcript,
                        rag_enabled=session.rag_enabled,
                    )

                response = session.run_text(
                    request.text,
                    metadata={
                        "dialog_key": request.dialog_key,
                        "web_session_id": request.session_id,
                    },
                )
                return WebTransportResponse(
                    text=response.assistant_message,
                    render_mode="text",
                    is_command=False,
                    dialog_key=request.dialog_key,
                    rag_enabled=session.rag_enabled,
                )
        finally:
            self.registry.release(lease)

    def handle_text(self, text: str, *, session_id: str) -> WebTransportResponse:
        return self.handle_request(WebTransportRequest(session_id=session_id, text=text))
