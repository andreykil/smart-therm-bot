"""Native draft-stream sender для private Telegram chats."""

from __future__ import annotations

import asyncio
import html
import logging
import time
from dataclasses import dataclass
from random import randint
from typing import Iterable

import requests
from telegram import Bot, Message

from src.chat.application.dto import ChatStreamEvent, ChatTurnResponse

from .telegram_markdown import render_telegram_html_from_markdown

LOGGER = logging.getLogger(__name__)

TELEGRAM_PUBLIC_API_BASE_URL = "https://api.telegram.org"


class TelegramDraftRateLimitError(RuntimeError):
    """Сигнализирует о rate limit на sendMessageDraft."""

    def __init__(self, retry_after: int | None = None):
        self.retry_after = retry_after
        suffix = f" retry_after={retry_after}" if retry_after is not None else ""
        super().__init__(f"sendMessageDraft rate limited.{suffix}")


@dataclass(slots=True, frozen=True)
class TelegramNativeStreamingSettings:
    """Настройки native draft streaming для private chats."""

    enabled: bool = False
    private_native_drafts: bool = True
    flush_interval_ms: int = 400
    min_chars_delta: int = 120
    max_draft_chars: int = 4000
    max_draft_seconds: int = 30


class TelegramDraftSender:
    """Доставляет частичный ответ через sendMessageDraft и финализирует sendMessage."""

    def __init__(
        self,
        *,
        bot: Bot,
        bot_token: str,
        settings: TelegramNativeStreamingSettings,
        api_base_url: str = TELEGRAM_PUBLIC_API_BASE_URL,
    ) -> None:
        self.bot = bot
        self.bot_token = bot_token
        self.settings = settings
        self.api_base_url = api_base_url.rstrip("/")

    def _draft_method_url(self) -> str:
        return f"{self.api_base_url}/bot{self.bot_token}/sendMessageDraft"

    @staticmethod
    def _generate_draft_id() -> int:
        return randint(1, 2_147_483_647)

    @staticmethod
    def _is_private_message(message: Message) -> bool:
        return message.chat.type == "private"

    def _drafts_allowed_for(self, message: Message) -> bool:
        return self.settings.enabled and self.settings.private_native_drafts and self._is_private_message(message)

    def _render_stream_html(self, raw_text: str) -> str:
        if not raw_text.strip():
            return ""

        rendered = render_telegram_html_from_markdown(raw_text)
        if rendered and len(rendered) <= self.settings.max_draft_chars:
            return rendered

        truncated = raw_text[: self.settings.max_draft_chars].rstrip()
        return html.escape(truncated, quote=False)

    def _post_draft_request(self, payload: dict[str, object]) -> None:
        response = requests.post(self._draft_method_url(), json=payload, timeout=15)

        if response.status_code == 429:
            retry_after: int | None = None
            try:
                body = response.json()
                parameters = body.get("parameters") if isinstance(body, dict) else None
                if isinstance(parameters, dict):
                    raw_retry_after = parameters.get("retry_after")
                    if isinstance(raw_retry_after, int):
                        retry_after = raw_retry_after
            except ValueError:
                retry_after = None
            raise TelegramDraftRateLimitError(retry_after=retry_after)

        response.raise_for_status()

        body = response.json()
        if not body.get("ok", False):
            raise RuntimeError(str(body.get("description", "sendMessageDraft failed")))

    async def _send_draft(self, *, chat_id: int, thread_id: int | None, draft_id: int, text: str) -> None:
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "draft_id": draft_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        await asyncio.to_thread(self._post_draft_request, payload)

    async def _send_final_message(self, source_message: Message, rendered_html: str) -> None:
        await source_message.reply_text(rendered_html, parse_mode="HTML")

    async def send_stream(
        self,
        *,
        source_message: Message,
        events: Iterable[ChatStreamEvent],
    ) -> ChatTurnResponse | None:
        raw_text = ""
        final_response: ChatTurnResponse | None = None
        drafts_enabled = self._drafts_allowed_for(source_message)
        draft_id = self._generate_draft_id() if drafts_enabled else 0
        last_flush_at = time.monotonic()
        last_sent_length = 0
        last_draft_text = ""
        draft_started_at = last_flush_at
        backoff_until = 0.0

        for event in events:
            if event.kind == "token":
                raw_text += event.text

                if not drafts_enabled:
                    continue

                now = time.monotonic()
                if now - draft_started_at >= self.settings.max_draft_seconds:
                    drafts_enabled = False
                    continue

                if now < backoff_until:
                    continue

                enough_time = (now - last_flush_at) * 1000 >= self.settings.flush_interval_ms
                enough_delta = len(raw_text) - last_sent_length >= self.settings.min_chars_delta
                if not enough_time and not enough_delta:
                    continue

                rendered_html = self._render_stream_html(raw_text)
                if not rendered_html or rendered_html == last_draft_text:
                    continue

                try:
                    LOGGER.info(
                        "Telegram draft flush attempt chat_id=%s draft_id=%s raw_chars=%s",
                        source_message.chat_id,
                        draft_id,
                        len(raw_text),
                    )
                    await self._send_draft(
                        chat_id=source_message.chat_id,
                        thread_id=source_message.message_thread_id,
                        draft_id=draft_id,
                        text=rendered_html,
                    )
                    last_flush_at = now
                    last_sent_length = len(raw_text)
                    last_draft_text = rendered_html
                except TelegramDraftRateLimitError as error:
                    retry_after = max(error.retry_after or 1, 1)
                    backoff_until = now + retry_after
                    last_flush_at = now
                    LOGGER.warning(
                        "Telegram draft streaming rate limited; backing off for %ss chat_id=%s draft_id=%s",
                        retry_after,
                        source_message.chat_id,
                        draft_id,
                    )
                except Exception:
                    drafts_enabled = False
                    LOGGER.warning("Telegram native draft streaming failed; continuing with final-only response", exc_info=True)
                continue

            if event.kind == "final" and event.response is not None:
                final_response = event.response

        final_raw_text = final_response.assistant_message if final_response is not None else raw_text
        rendered_final = self._render_stream_html(final_raw_text)
        if rendered_final:
            LOGGER.info(
                "Telegram final message send chat_id=%s draft_id=%s raw_chars=%s",
                source_message.chat_id,
                draft_id,
                len(final_raw_text),
            )
            await self._send_final_message(source_message, rendered_final)

        return final_response
