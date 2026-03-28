from __future__ import annotations

import threading

from src.bot.telegram_runner import IncomingTelegramText, build_start_text, route_transport_message
from src.bot.telegram_transport import TelegramTransportRequest, TelegramTransportResponse


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


def test_build_start_text_uses_transport_command_lines() -> None:
    transport = FakeTransportWithRegistry()

    text = build_start_text(transport, chat_id=303)

    assert "<b>Привет! Я бот поддержки SmartTherm.</b>" in text
    assert "<pre>" in text
    assert transport.registry.acquired_keys == ["chat:303"]
