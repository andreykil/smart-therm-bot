"""Общий command-layer для transport-независимых slash-команд."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .runtime import ChatRuntime
from .service import ChatService


@dataclass(slots=True)
class ParsedCommand:
    """Нормализованное представление slash-команды."""

    name: str
    argument: str = ""


@dataclass(slots=True)
class CommandResult:
    """Результат выполнения transport-команды."""

    lines: list[str]


@dataclass(slots=True)
class CommandContext:
    """Контекст для выполнения общих slash-команд."""

    service: ChatService
    runtime: ChatRuntime


class CommandParser:
    """Parser slash-команд без привязки к transport-реализации."""

    @staticmethod
    def is_command(text: str) -> bool:
        """Проверить, является ли строка slash-командой."""
        return text.strip().startswith("/")

    @staticmethod
    def parse(text: str) -> ParsedCommand | None:
        """Разобрать строку в структуру команды."""
        stripped = text.strip()
        if not stripped or not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        return ParsedCommand(
            name=parts[0].lower(),
            argument=parts[1].strip() if len(parts) > 1 else "",
        )


class CommandDispatcher:
    """Dispatcher общих slash-команд поверх chat application context."""

    def __init__(self, context: CommandContext):
        self.context = context

    def command_lines(self) -> list[str]:
        """Вернуть список поддерживаемых команд."""
        return [
            "Команды:",
            "  /clear            — очистить историю",
            "  /stats            — показать статистику",
            "  /rag              — переключить RAG вкл/выкл",
            "  /help             — показать команды",
        ]

    def _stats_lines(self) -> list[str]:
        stats = self.context.service.get_stats()
        client_stats = cast(dict[str, object], stats["client"])
        session_stats = cast(dict[str, object], stats["session"])
        rag_stats = cast(dict[str, object] | None, stats["rag"])
        lines = ["\n📊 Статистика клиента:"]
        for key, value in client_stats.items():
            lines.append(f"  {key}: {value}")

        lines.append("\n🧠 Статистика сессии:")
        for key, value in session_stats.items():
            lines.append(f"  {key}: {value}")

        lines.append("\n⚙️ Runtime:")
        for key, value in self.context.runtime.stats().items():
            lines.append(f"  {key}: {value}")

        if rag_stats is not None:
            lines.append("\n📚 Статистика RAG:")
            for key, value in rag_stats.items():
                lines.append(f"  {key}: {value}")

        lines.append("")
        return lines

    def execute(self, raw: str) -> CommandResult:
        """Выполнить slash-команду и вернуть transport-ready результат."""
        parsed = CommandParser.parse(raw)
        if parsed is None:
            return CommandResult(lines=["❓ Пустая команда", *self.command_lines()])

        if parsed.name == "/clear":
            self.context.service.clear_history()
            return CommandResult(lines=["🗑️  История очищена"])

        if parsed.name == "/stats":
            return CommandResult(lines=self._stats_lines())

        if parsed.name == "/rag":
            if self.context.service.rag_pipeline is None:
                return CommandResult(
                    lines=["⚠️ RAG недоступен: индексы не загружены или pipeline не настроен"]
                )
            enabled = self.context.runtime.toggle_rag()
            return CommandResult(lines=[f"🔄 RAG {'включен' if enabled else 'выключен'}"])

        if parsed.name == "/help":
            return CommandResult(lines=self.command_lines())

        return CommandResult(lines=[f"❓ Неизвестная команда: {parsed.name}", *self.command_lines()])
