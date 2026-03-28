"""Transport-независимая обработка slash-команд."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .chat_service import ChatService
from .runtime import ChatRuntime


@dataclass(slots=True)
class ParsedCommand:
    name: str
    argument: str = ""


@dataclass(slots=True)
class CommandResult:
    lines: list[str]


@dataclass(slots=True)
class CommandContext:
    service: ChatService
    runtime: ChatRuntime


class CommandParser:
    @staticmethod
    def is_command(text: str) -> bool:
        return text.strip().startswith("/")

    @staticmethod
    def parse(text: str) -> ParsedCommand | None:
        stripped = text.strip()
        if not stripped or not stripped.startswith("/"):
            return None
        parts = stripped.split(maxsplit=1)
        return ParsedCommand(name=parts[0].lower(), argument=parts[1].strip() if len(parts) > 1 else "")


class CommandService:
    """Dispatcher общих slash-команд поверх chat application context."""

    def __init__(self, context: CommandContext):
        self.context = context

    def command_lines(self) -> list[str]:
        return [
            "Команды:",
            "  /clear            — очистить историю",
            "  /memory           — показать сохранённые факты",
            "  /remember k=v     — сохранить факт",
            "  /forget key       — удалить факт",
            "  /stats            — показать статистику",
            "  /rag              — переключить RAG вкл/выкл",
            "  /help             — показать команды",
        ]

    @staticmethod
    def _parse_memory_argument(argument: str) -> tuple[str, str] | None:
        stripped = argument.strip()
        if not stripped:
            return None
        if "=" in stripped:
            key, value = stripped.split("=", 1)
        else:
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                return None
            key, value = parts
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return None
        return key, value

    def _memory_lines(self) -> list[str]:
        facts = self.context.service.list_memory_facts()
        if not facts:
            return ["🧠 Память пуста"]
        lines = ["🧠 Сохранённые факты:"]
        for fact in facts:
            lines.append(f"  {fact.key}: {fact.value}")
        return lines

    def _stats_lines(self) -> list[str]:
        stats = self.context.service.get_stats()
        client_stats = cast(dict[str, object], stats["client"])
        dialog_stats = cast(dict[str, object], stats["dialog"])
        rag_stats = cast(dict[str, object] | None, stats["rag"])
        lines = ["\n📊 Статистика клиента:"]
        for key, value in client_stats.items():
            lines.append(f"  {key}: {value}")

        lines.append("\n🧠 Состояние диалога:")
        for key, value in dialog_stats.items():
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
        parsed = CommandParser.parse(raw)
        if parsed is None:
            return CommandResult(lines=["❓ Пустая команда", *self.command_lines()])

        if parsed.name == "/clear":
            self.context.service.clear_history()
            return CommandResult(lines=["🗑️  История и память очищены"])

        if parsed.name == "/memory":
            return CommandResult(lines=self._memory_lines())

        if parsed.name == "/remember":
            memory_argument = self._parse_memory_argument(parsed.argument)
            if memory_argument is None:
                return CommandResult(lines=["⚠️ Формат: /remember key=value"])
            key, value = memory_argument
            saved_fact = self.context.service.remember_fact(key, value)
            return CommandResult(lines=[f"💾 Сохранено: {saved_fact.key}"])

        if parsed.name == "/forget":
            if not parsed.argument.strip():
                return CommandResult(lines=["⚠️ Формат: /forget key"])
            deleted = self.context.service.forget_fact(parsed.argument)
            display_key = parsed.argument.strip()
            if deleted:
                return CommandResult(lines=[f"🧹 Удалено: {display_key}"])
            return CommandResult(lines=[f"ℹ️ Факт не найден: {display_key}"])

        if parsed.name == "/stats":
            return CommandResult(lines=self._stats_lines())

        if parsed.name == "/rag":
            if self.context.service.retriever is None:
                return CommandResult(lines=["⚠️ RAG недоступен: индексы не загружены или pipeline не настроен"])
            enabled = self.context.runtime.toggle_rag()
            return CommandResult(lines=[f"🔄 RAG {'включен' if enabled else 'выключен'}"])

        if parsed.name == "/help":
            return CommandResult(lines=self.command_lines())

        return CommandResult(lines=[f"❓ Неизвестная команда: {parsed.name}", *self.command_lines()])
