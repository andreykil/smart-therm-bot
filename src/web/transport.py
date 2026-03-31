"""Minimal web transport for SmartTherm chat application."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
import html
import time
from typing import Callable
import uuid

from src.chat.application.command_service import CommandParser, CommandService
from src.chat.registry import DialogRegistry

from .markdown import render_web_markdown, split_renderable_markdown


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


@dataclass(slots=True)
class WebStreamEvent:
    """Event delivered to the web UI NDJSON stream."""

    event: str
    text: str = ""
    render_mode: str = "text"
    is_command: bool = False
    reset_transcript: bool = False
    rag_enabled: bool = False
    message_id: str = ""
    error: str = ""
    html: str = ""
    tail_text: str = ""

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": self.event,
            "text": self.text,
            "render_mode": self.render_mode,
            "is_command": self.is_command,
            "reset_transcript": self.reset_transcript,
            "rag_enabled": self.rag_enabled,
        }
        if self.message_id:
            payload["message_id"] = self.message_id
        if self.error:
            payload["error"] = self.error
        if self.html:
            payload["html"] = self.html
        if self.tail_text:
            payload["tail_text"] = self.tail_text
        return payload


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

    def __init__(
        self,
        registry: DialogRegistry,
        *,
        preview_interval_ms: int = 250,
        preview_min_chars: int = 160,
        clock: Callable[[], float] | None = None,
    ):
        self.registry = registry
        self.preview_interval_ms = preview_interval_ms
        self.preview_min_chars = preview_min_chars
        self._clock = clock or time.monotonic

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
                    text=render_web_markdown(response.assistant_message),
                    render_mode="markdown",
                    is_command=False,
                    dialog_key=request.dialog_key,
                    rag_enabled=session.rag_enabled,
                )
        finally:
            self.registry.release(lease)

    def handle_text(self, text: str, *, session_id: str) -> WebTransportResponse:
        return self.handle_request(WebTransportRequest(session_id=session_id, text=text))

    def stream_request(self, request: WebTransportRequest) -> Generator[WebStreamEvent, None, None]:
        lease = self.registry.acquire(request.dialog_key)

        try:
            with lease.lock:
                session = lease.session
                parsed = CommandParser.parse(request.text)
                if parsed is not None and parsed.name == "/start":
                    yield WebStreamEvent(
                        event="command",
                        text=build_web_start_html(),
                        render_mode="html",
                        is_command=True,
                        rag_enabled=session.rag_enabled,
                    )
                    return

                command_result = session.try_execute_command(request.text)
                if command_result is not None:
                    yield WebStreamEvent(
                        event="command",
                        text="\n".join(command_result.lines),
                        render_mode="html" if command_result.parse_mode == "HTML" else "text",
                        is_command=True,
                        reset_transcript=command_result.reset_transcript,
                        rag_enabled=session.rag_enabled,
                    )
                    return

                message_id = uuid.uuid4().hex
                yield WebStreamEvent(
                    event="start",
                    message_id=message_id,
                    render_mode="markdown",
                    rag_enabled=session.rag_enabled,
                )

                final_text = ""
                raw_text = ""
                preview_chars_since_emit = 0
                last_preview_at = self._clock()
                last_preview_source = ""
                for stream_event in session.stream_text(
                    request.text,
                    metadata={
                        "dialog_key": request.dialog_key,
                        "web_session_id": request.session_id,
                    },
                ):
                    if stream_event.kind == "token" and stream_event.text:
                        raw_text += stream_event.text
                        preview_chars_since_emit += len(stream_event.text)
                        yield WebStreamEvent(
                            event="token",
                            text=stream_event.text,
                            rag_enabled=session.rag_enabled,
                            message_id=message_id,
                        )

                        now = self._clock()
                        enough_time = (now - last_preview_at) * 1000 >= self.preview_interval_ms
                        enough_chars = preview_chars_since_emit >= self.preview_min_chars
                        if enough_time or enough_chars:
                            preview_prefix, preview_tail = split_renderable_markdown(raw_text)
                            if preview_prefix and preview_prefix != last_preview_source:
                                yield WebStreamEvent(
                                    event="preview",
                                    render_mode="markdown",
                                    rag_enabled=session.rag_enabled,
                                    message_id=message_id,
                                    html=render_web_markdown(preview_prefix),
                                    tail_text=preview_tail,
                                )
                                last_preview_source = preview_prefix
                            last_preview_at = now
                            preview_chars_since_emit = 0
                        continue

                    if stream_event.kind == "final" and stream_event.response is not None:
                        final_text = stream_event.response.assistant_message
                        yield WebStreamEvent(
                            event="final",
                            text=render_web_markdown(final_text),
                            render_mode="markdown",
                            rag_enabled=session.rag_enabled,
                            message_id=message_id,
                        )
                        return

                if not final_text:
                    raise RuntimeError("Streaming response finished without final event")
        except Exception as error:
            yield WebStreamEvent(event="error", error=str(error))
        finally:
            self.registry.release(lease)

    def stream_text(self, text: str, *, session_id: str) -> Generator[WebStreamEvent, None, None]:
        return self.stream_request(WebTransportRequest(session_id=session_id, text=text))
