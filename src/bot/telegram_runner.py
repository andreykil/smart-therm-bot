"""Long polling runner и transport-routing для Telegram-бота SmartTherm."""

from __future__ import annotations

import asyncio
import argparse
import logging
import re
from dataclasses import dataclass
from typing import Any

from telegram import BotCommand, Message, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.chat.application.command_service import CommandParser
from src.chat.composition import build_dialog_registry
from src.config import Config

from .telegram_draft_sender import TelegramDraftSender, TelegramNativeStreamingSettings
from .telegram_transport import TelegramTransport, TelegramTransportRequest, TelegramTransportResponse

LOGGER = logging.getLogger(__name__)

TRANSPORT_KEY = "telegram_transport"
BOT_USERNAME_KEY = "telegram_bot_username"
STREAMING_SETTINGS_KEY = "telegram_streaming_settings"
DRAFT_SENDER_FACTORY_KEY = "telegram_draft_sender_factory"
GROUP_CHAT_TYPES = {"group", "supergroup"}


@dataclass(slots=True, frozen=True)
class IncomingTelegramText:
    """Нормализованное входящее сообщение до передачи в transport."""

    chat_id: int
    chat_type: str
    text: str
    user_id: int | None = None
    thread_id: int | None = None
    is_reply_to_bot: bool = False


def build_bot_parser() -> argparse.ArgumentParser:
    """Создать parser запуска Telegram-бота."""
    parser = argparse.ArgumentParser(description="Telegram long polling bot для SmartTherm chat application")
    parser.add_argument("--model", "-m", type=str, default=None, help="Название модели в Ollama")
    parser.add_argument("--url", "-u", type=str, default=None, help="URL Ollama сервера")
    parser.add_argument("--max-tokens", "-t", type=int, default=None, help="Максимум токенов на ответ")
    parser.add_argument("--system", "-s", type=str, default=None, help="Переопределение системного промпта")
    parser.add_argument("--verbose", "-v", action="store_true", help="Включить подробное логирование LLM")
    parser.add_argument("--debug", action="store_true", help="Передавать debug-флаг в chat runtime")
    parser.add_argument("--top-k", "-k", type=int, default=None, help="Количество результатов RAG")
    parser.add_argument("--vector-weight", type=float, default=0.5, help="Вес FAISS в гибридном поиске")
    parser.add_argument("--bm25-weight", type=float, default=0.5, help="Вес BM25 в гибридном поиске")
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=None,
        help="Путь к JSONL файлу чанков для переиндексации перед запуском RAG",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Использовать тестовые чанки (data/processed/chat/test/chunks_rag_test.jsonl)",
    )
    parser.add_argument("--no-rag", action="store_true", help="Запустить бота без RAG")
    return parser


def _normalize_bot_username(bot_username: str | None) -> str:
    return (bot_username or "").lstrip("@").strip().lower()


def is_command_for_bot(text: str, bot_username: str | None) -> bool:
    """Проверить, что slash-команда адресована этому боту или без явного суффикса."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return False

    token = stripped.split(maxsplit=1)[0]
    if "@" not in token:
        return True

    _, target = token.split("@", 1)
    normalized_username = _normalize_bot_username(bot_username)
    return bool(normalized_username) and target.lower() == normalized_username


def normalize_command_text(text: str, bot_username: str | None) -> str:
    """Убрать telegram-суффикс @bot_username из slash-команды."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return stripped

    parts = stripped.split(maxsplit=1)
    command_token = parts[0]
    argument = parts[1] if len(parts) > 1 else ""
    normalized_username = _normalize_bot_username(bot_username)

    if "@" in command_token:
        command_name, target = command_token.split("@", 1)
        if normalized_username and target.lower() == normalized_username:
            command_token = command_name

    return f"{command_token} {argument}".strip()


def is_direct_mention(text: str, bot_username: str | None) -> bool:
    """Проверить, что текст начинается с обращения к боту."""
    normalized_username = _normalize_bot_username(bot_username)
    if not normalized_username:
        return False

    pattern = rf"^\s*@{re.escape(normalized_username)}(?:\s+|\s*[:,]\s*|$)"
    return re.match(pattern, text, flags=re.IGNORECASE) is not None


def strip_leading_mention(text: str, bot_username: str | None) -> str:
    """Убрать ведущее обращение @bot_username из текста."""
    normalized_username = _normalize_bot_username(bot_username)
    if not normalized_username:
        return text.strip()

    pattern = rf"^\s*@{re.escape(normalized_username)}(?:\s+|\s*[:,]\s*|$)"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()


def normalize_incoming_text(text: str, bot_username: str | None) -> str:
    """Нормализовать Telegram text к формату transport/chat-core."""
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("/"):
        return normalize_command_text(stripped, bot_username)
    if is_direct_mention(stripped, bot_username):
        return strip_leading_mention(stripped, bot_username)
    return stripped


def should_process_message(message: IncomingTelegramText, *, bot_username: str | None) -> bool:
    """Определить, должен ли бот отвечать на сообщение."""
    stripped = message.text.strip()
    if not stripped:
        return False

    if CommandParser.is_command(stripped):
        return is_command_for_bot(stripped, bot_username)

    if message.chat_type not in GROUP_CHAT_TYPES:
        return True

    return message.is_reply_to_bot or is_direct_mention(stripped, bot_username)


def route_transport_message(
    transport: Any,
    message: IncomingTelegramText,
    *,
    bot_username: str | None,
) -> TelegramTransportResponse | None:
    """Применить Telegram routing policy и при необходимости вызвать transport."""
    request = build_transport_request(message, bot_username=bot_username)
    if request is None:
        return None

    return transport.handle_request(request)


def build_transport_request(
    message: IncomingTelegramText,
    *,
    bot_username: str | None,
) -> TelegramTransportRequest | None:
    """Собрать нормализованный transport request, если сообщение адресовано боту."""
    if not should_process_message(message, bot_username=bot_username):
        return None

    normalized_text = normalize_incoming_text(message.text, bot_username)
    if not normalized_text:
        return None

    return TelegramTransportRequest(
        chat_id=message.chat_id,
        user_id=message.user_id,
        thread_id=message.thread_id,
        text=normalized_text,
    )


def build_start_text(transport: Any, *, chat_id: int) -> str:
    """Собрать приветственное сообщение без вызова chat turn."""
    lease = transport.registry.acquire(f"chat:{chat_id}")
    try:
        with lease.lock:
            lines = [
                "<b>Привет! Я бот поддержки SmartTherm.</b>",
                "В личке отвечаю на любой текст. В группе — на команды, mention и reply на мои сообщения.",
                "",
                lease.session.command_help_html(),
            ]
            return "\n".join(lines)
    finally:
        transport.registry.release(lease)


def build_application(
    *,
    config: Config,
    model_name: str | None = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
    debug: bool = False,
    verbose: bool = False,
    top_k: int | None = None,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    chunks_file: str | None = None,
    test_mode: bool = False,
    use_rag: bool | None = None,
) -> Application:
    """Собрать telegram Application поверх общего chat composition flow."""
    if not config.bot.token:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан. Укажите его в окружении или config")

    registry = build_dialog_registry(
        config=config,
        model_name=model_name or config.llm.model,
        base_url=base_url or config.llm.base_url,
        max_tokens=max_tokens if max_tokens is not None else config.llm.max_tokens,
        temperature=config.llm.temperature,
        use_rag=config.bot.use_rag if use_rag is None else use_rag,
        system_prompt_override=system_prompt_override,
        debug=debug,
        verbose=verbose,
        think=config.llm.think,
        top_k=top_k if top_k is not None else config.rag.top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        chunks_file=chunks_file,
        test_mode=test_mode,
    )
    transport = TelegramTransport(registry)

    application = Application.builder().token(config.bot.token).post_init(_post_init).build()
    application.bot_data[TRANSPORT_KEY] = transport
    application.bot_data[STREAMING_SETTINGS_KEY] = TelegramNativeStreamingSettings(
        enabled=config.bot.streaming.enabled,
        private_native_drafts=config.bot.streaming.private_native_drafts,
        flush_interval_ms=config.bot.streaming.flush_interval_ms,
        min_chars_delta=config.bot.streaming.min_chars_delta,
        max_draft_chars=config.bot.streaming.max_draft_chars,
        max_draft_seconds=config.bot.streaming.max_draft_seconds,
    )
    application.bot_data[DRAFT_SENDER_FACTORY_KEY] = TelegramDraftSender
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(MessageHandler(filters.COMMAND, handle_command_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(handle_error)
    return application


def _get_transport(application: Application) -> TelegramTransport:
    return application.bot_data[TRANSPORT_KEY]


def _is_reply_to_current_bot(message: Message | None, *, bot_id: int | None) -> bool:
    if message is None or bot_id is None:
        return False
    reply = message.reply_to_message
    if reply is None or reply.from_user is None:
        return False
    return reply.from_user.id == bot_id


def _build_request_metadata(request: TelegramTransportRequest) -> dict[str, int | str | None]:
    return {
        "chat_id": request.chat_id,
        "user_id": request.user_id,
        "thread_id": request.thread_id,
        "dialog_key": request.dialog_key,
    }


async def _handle_private_stream_request(
    transport: TelegramTransport,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    message: Message,
    request: TelegramTransportRequest,
) -> None:
    lease = transport.registry.acquire(request.dialog_key)
    try:
        with lease.lock:
            session = lease.session
            sender_factory = context.application.bot_data[DRAFT_SENDER_FACTORY_KEY]
            streaming_settings = context.application.bot_data[STREAMING_SETTINGS_KEY]
            sender = sender_factory(
                bot=context.application.bot,
                bot_token=context.application.bot.token,
                settings=streaming_settings,
            )
            await sender.send_stream(
                source_message=message,
                events=session.stream_text(
                    request.text,
                    metadata=_build_request_metadata(request),
                ),
            )
    finally:
        transport.registry.release(lease)


def _build_incoming_message(message: Message, *, bot_id: int | None) -> IncomingTelegramText:
    return IncomingTelegramText(
        chat_id=message.chat_id,
        chat_type=message.chat.type,
        text=message.text or "",
        user_id=message.from_user.id if message.from_user else None,
        thread_id=message.message_thread_id,
        is_reply_to_bot=_is_reply_to_current_bot(message, bot_id=bot_id),
    )


async def _post_init(application: Application) -> None:
    me = await application.bot.get_me()
    application.bot_data[BOT_USERNAME_KEY] = me.username or ""
    await application.bot.set_my_commands(
        [
            BotCommand("start", "показать приветствие"),
            BotCommand("help", "показать команды"),
            BotCommand("clear", "очистить историю"),
            BotCommand("memory", "показать память"),
            BotCommand("remember", "сохранить факт"),
            BotCommand("forget", "удалить факт"),
            BotCommand("stats", "показать статистику"),
            BotCommand("rag", "переключить RAG"),
        ]
    )
    LOGGER.info("Telegram bot initialized as @%s", me.username or "unknown")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    transport = _get_transport(context.application)
    await message.reply_text(build_start_text(transport, chat_id=message.chat_id), parse_mode="HTML")


async def handle_command_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_transport_message(update, context)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_transport_message(update, context)


async def _handle_transport_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return

    transport = _get_transport(context.application)
    bot_username = context.application.bot_data.get(BOT_USERNAME_KEY)
    bot_id = context.application.bot.id if context.application.bot is not None else None
    incoming = _build_incoming_message(message, bot_id=bot_id)
    request = build_transport_request(incoming, bot_username=bot_username)

    try:
        if request is None:
            return

        streaming_settings: TelegramNativeStreamingSettings = context.application.bot_data[STREAMING_SETTINGS_KEY]
        if (
            streaming_settings.enabled
            and incoming.chat_type == "private"
            and not CommandParser.is_command(request.text)
        ):
            await _handle_private_stream_request(transport, context, message=message, request=request)
            return

        response = transport.handle_request(request)
        await message.reply_text(response.text, parse_mode=response.parse_mode)
    except Exception:
        LOGGER.exception("Telegram message handling failed")
        await message.reply_text("⚠️ Не удалось обработать сообщение. Попробуйте ещё раз.")


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Unhandled Telegram error for update %r", update, exc_info=context.error)


def run_telegram_bot(argv: list[str] | None = None) -> None:
    """Запустить Telegram-бота через long polling."""
    args = build_bot_parser().parse_args(argv)
    config = Config.load()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    application = build_application(
        config=config,
        model_name=args.model,
        base_url=args.url,
        max_tokens=args.max_tokens,
        system_prompt_override=args.system,
        debug=args.debug,
        verbose=args.verbose,
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        chunks_file=args.chunks_file,
        test_mode=args.test,
        use_rag=False if args.no_rag else None,
    )

    LOGGER.info("Starting Telegram long polling")
    application.run_polling()
