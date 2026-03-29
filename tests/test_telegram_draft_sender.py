from __future__ import annotations

import asyncio
from typing import Any, cast

from src.bot.telegram_draft_sender import (
    TelegramDraftRateLimitError,
    TelegramDraftSender,
    TelegramNativeStreamingSettings,
)
from src.chat.application.dto import ChatStreamEvent, ChatTurnResponse, RetrievedContext


class FakeBot:
    pass


class FakeChat:
    type = "private"


class FakeMessage:
    def __init__(self) -> None:
        self.chat_id = 101
        self.chat = FakeChat()
        self.message_thread_id = None
        self.replies: list[tuple[str, str | None]] = []

    async def reply_text(self, text: str, parse_mode: str | None = None) -> None:
        self.replies.append((text, parse_mode))


class RecordingDraftSender(TelegramDraftSender):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.draft_payloads: list[dict[str, object]] = []
        self.fail_drafts = False
        self.rate_limit_once = False

    async def _send_draft(self, *, chat_id: int, thread_id: int | None, draft_id: int, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "thread_id": thread_id,
            "draft_id": draft_id,
            "text": text,
        }
        self.draft_payloads.append(payload)
        if self.rate_limit_once:
            self.rate_limit_once = False
            raise TelegramDraftRateLimitError(retry_after=1)
        if self.fail_drafts:
            raise RuntimeError("draft failed")


def _build_events():
    yield ChatStreamEvent(kind="token", text="Привет")
    yield ChatStreamEvent(kind="token", text=", мир")
    yield ChatStreamEvent(
        kind="final",
        response=ChatTurnResponse(
            user_message="привет",
            assistant_message="Привет, мир",
            llm_messages=[],
            retrieved_context=RetrievedContext(enabled=False),
            streamed=True,
        ),
    )


def test_draft_sender_sends_drafts_and_final_message() -> None:
    sender = RecordingDraftSender(
        bot=FakeBot(),
        bot_token="token",
        settings=TelegramNativeStreamingSettings(
            enabled=True,
            flush_interval_ms=0,
            min_chars_delta=1,
        ),
    )
    message = FakeMessage()

    asyncio.run(sender.send_stream(source_message=cast(Any, message), events=_build_events()))

    assert sender.draft_payloads
    assert message.replies == [("Привет, мир", "HTML")]


def test_draft_sender_falls_back_to_final_only_when_draft_fails() -> None:
    sender = RecordingDraftSender(
        bot=FakeBot(),
        bot_token="token",
        settings=TelegramNativeStreamingSettings(
            enabled=True,
            flush_interval_ms=0,
            min_chars_delta=1,
        ),
    )
    sender.fail_drafts = True
    message = FakeMessage()

    asyncio.run(sender.send_stream(source_message=cast(Any, message), events=_build_events()))

    assert sender.draft_payloads
    assert message.replies == [("Привет, мир", "HTML")]


def test_draft_sender_keeps_streaming_after_rate_limit_backoff() -> None:
    sender = RecordingDraftSender(
        bot=FakeBot(),
        bot_token="token",
        settings=TelegramNativeStreamingSettings(
            enabled=True,
            flush_interval_ms=0,
            min_chars_delta=1,
        ),
    )
    sender.rate_limit_once = True
    message = FakeMessage()

    asyncio.run(sender.send_stream(source_message=cast(Any, message), events=_build_events()))

    assert len(sender.draft_payloads) >= 1
    assert message.replies == [("Привет, мир", "HTML")]


def test_draft_sender_uses_uniform_flush_thresholds() -> None:
    sender = RecordingDraftSender(
        bot=FakeBot(),
        bot_token="token",
        settings=TelegramNativeStreamingSettings(
            enabled=True,
            flush_interval_ms=999999,
            min_chars_delta=6,
        ),
    )
    message = FakeMessage()

    asyncio.run(sender.send_stream(source_message=cast(Any, message), events=_build_events()))

    assert sender.draft_payloads
    assert sender.draft_payloads[0]["text"] == "Привет"
