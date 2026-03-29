from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

from src.bot.telegram_draft_sender import TelegramNativeStreamingSettings
from src.bot.telegram_runner import (
    DRAFT_SENDER_FACTORY_KEY,
    STREAMING_SETTINGS_KEY,
    IncomingTelegramText,
    _handle_private_stream_request,
    build_start_text,
    build_transport_request,
    route_transport_message,
)
from src.bot.telegram_transport import TelegramTransportRequest, TelegramTransportResponse
from src.chat.application.dto import ChatStreamEvent, ChatTurnResponse, RetrievedContext


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[TelegramTransportRequest] = []

    def handle_request(self, request: TelegramTransportRequest) -> TelegramTransportResponse:
        self.requests.append(request)
        return TelegramTransportResponse(
            text=f"handled:{request.text}",
            is_command=request.text.startswith("/"),
            dialog_key=request.dialog_key,
        )


class FakeSession:
    def command_lines(self) -> list[str]:
        return ["Команды:", "  /help             — показать команды"]

    def command_help_html(self) -> str:
        return "<b>Команды:</b>\n<pre>/help            — показать команды</pre>"


class FakeLease:
    def __init__(self) -> None:
        self.session = FakeSession()
        self.lock = threading.RLock()


class FakeRegistry:
    def __init__(self) -> None:
        self.acquired_keys: list[str] = []

    def acquire(self, dialog_key: str) -> FakeLease:
        self.acquired_keys.append(dialog_key)
        return FakeLease()

    def release(self, lease: FakeLease) -> None:
        del lease


class FakeTransportWithRegistry(FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.registry = FakeRegistry()


def test_route_transport_message_processes_private_text() -> None:
    transport = FakeTransport()
    response = route_transport_message(
        transport,
        IncomingTelegramText(chat_id=101, chat_type="private", text="Привет", user_id=10),
        bot_username="smart_therm_bot",
    )

    assert response is not None
    assert response.text == "handled:Привет"
    assert transport.requests[0].dialog_key == "chat:101"


def test_route_transport_message_ignores_unaddressed_group_text() -> None:
    transport = FakeTransport()
    response = route_transport_message(
        transport,
        IncomingTelegramText(chat_id=-1001, chat_type="group", text="Привет всем", user_id=10),
        bot_username="smart_therm_bot",
    )

    assert response is None
    assert transport.requests == []


def test_route_transport_message_passes_group_mention_to_transport() -> None:
    transport = FakeTransport()
    response = route_transport_message(
        transport,
        IncomingTelegramText(chat_id=-1001, chat_type="supergroup", text="@smart_therm_bot привет", user_id=10),
        bot_username="smart_therm_bot",
    )

    assert response is not None
    assert transport.requests[0].text == "привет"


def test_route_transport_message_passes_reply_to_bot_without_mention() -> None:
    transport = FakeTransport()
    response = route_transport_message(
        transport,
        IncomingTelegramText(
            chat_id=-1001,
            chat_type="supergroup",
            text="подскажи ещё раз",
            user_id=10,
            is_reply_to_bot=True,
        ),
        bot_username="smart_therm_bot",
    )

    assert response is not None
    assert transport.requests[0].text == "подскажи ещё раз"


def test_route_transport_message_normalizes_command_suffix_for_current_bot() -> None:
    transport = FakeTransport()
    response = route_transport_message(
        transport,
        IncomingTelegramText(chat_id=-1001, chat_type="group", text="/help@smart_therm_bot", user_id=10),
        bot_username="smart_therm_bot",
    )

    assert response is not None
    assert transport.requests[0].text == "/help"
    assert response.is_command is True


def test_route_transport_message_ignores_command_for_another_bot() -> None:
    transport = FakeTransport()
    response = route_transport_message(
        transport,
        IncomingTelegramText(chat_id=-1001, chat_type="group", text="/help@another_bot", user_id=10),
        bot_username="smart_therm_bot",
    )

    assert response is None
    assert transport.requests == []


def test_build_transport_request_returns_normalized_private_request() -> None:
    request = build_transport_request(
        IncomingTelegramText(chat_id=202, chat_type="private", text="  привет  ", user_id=55),
        bot_username="smart_therm_bot",
    )

    assert request is not None
    assert request.chat_id == 202
    assert request.text == "привет"


def test_build_start_text_uses_transport_command_lines() -> None:
    transport = FakeTransportWithRegistry()

    text = build_start_text(transport, chat_id=303)

    assert "<b>Привет! Я бот поддержки SmartTherm.</b>" in text
    assert "<pre>" in text
    assert transport.registry.acquired_keys == ["chat:303"]


class FakeStreamSession:
    def __init__(self) -> None:
        self.stream_requests: list[tuple[str, dict[str, object]]] = []

    def stream_text(self, text: str, *, metadata: dict[str, object]):
        self.stream_requests.append((text, metadata))
        yield ChatStreamEvent(kind="token", text="Часть ")
        yield ChatStreamEvent(
            kind="final",
            response=ChatTurnResponse(
                user_message=text,
                assistant_message="Часть ответа",
                llm_messages=[],
                retrieved_context=RetrievedContext(enabled=False),
                streamed=True,
            ),
        )


class FakeStreamingLease:
    def __init__(self, session: FakeStreamSession) -> None:
        self.session = session
        self.lock = threading.RLock()


class FakeStreamingRegistry:
    def __init__(self, session: FakeStreamSession) -> None:
        self.session = session
        self.acquired_keys: list[str] = []
        self.released = 0

    def acquire(self, dialog_key: str) -> FakeStreamingLease:
        self.acquired_keys.append(dialog_key)
        return FakeStreamingLease(self.session)

    def release(self, lease: FakeStreamingLease) -> None:
        self.released += 1
        del lease


class FakeStreamingTransport:
    def __init__(self, session: FakeStreamSession) -> None:
        self.registry = FakeStreamingRegistry(session)


class FakeBot:
    def __init__(self) -> None:
        self.token = "token"


class FakeApplication:
    def __init__(self, sender_factory: type[FakeDraftSender]) -> None:
        self.bot = FakeBot()
        self.bot_data = {
            STREAMING_SETTINGS_KEY: TelegramNativeStreamingSettings(enabled=True, flush_interval_ms=0, min_chars_delta=1),
            DRAFT_SENDER_FACTORY_KEY: sender_factory,
        }


class FakeContext:
    def __init__(self, application: FakeApplication) -> None:
        self.application = application


class FakeChat:
    type = "private"


class FakeMessage:
    def __init__(self) -> None:
        self.chat_id = 404
        self.chat = FakeChat()
        self.message_thread_id = None


class FakeDraftSender:
    instances: list[FakeDraftSender] = []

    def __init__(self, *, bot, bot_token: str, settings: TelegramNativeStreamingSettings) -> None:
        self.bot = bot
        self.bot_token = bot_token
        self.settings = settings
        self.events: list[ChatStreamEvent] = []
        FakeDraftSender.instances.append(self)

    async def send_stream(self, *, source_message: FakeMessage, events) -> ChatTurnResponse | None:
        self.events = list(events)
        return None


def test_handle_private_stream_request_uses_session_stream_and_sender() -> None:
    FakeDraftSender.instances.clear()
    session = FakeStreamSession()
    transport = FakeStreamingTransport(session)
    application = FakeApplication(FakeDraftSender)
    context = FakeContext(application)
    message = FakeMessage()
    request = TelegramTransportRequest(chat_id=404, text="привет", user_id=77)

    asyncio.run(
        _handle_private_stream_request(
            cast(Any, transport),
            cast(Any, context),
            message=cast(Any, message),
            request=request,
        )
    )

    assert session.stream_requests == [
        (
            "привет",
            {
                "chat_id": 404,
                "user_id": 77,
                "thread_id": None,
                "dialog_key": "chat:404",
            },
        )
    ]
    assert len(FakeDraftSender.instances) == 1
    assert [event.kind for event in FakeDraftSender.instances[0].events] == ["token", "final"]
    assert transport.registry.acquired_keys == ["chat:404"]
    assert transport.registry.released == 1
