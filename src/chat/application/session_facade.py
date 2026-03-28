"""Transport-facing facade для одного диалога."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Any

from .chat_service import ChatService
from .command_service import CommandParser, CommandResult, CommandService
from .dto import ChatStreamEvent, ChatTurnRequest, ChatTurnResponse, PreparedChatTurn
from .runtime import ChatRuntime


@dataclass(slots=True)
class SessionFacade:
    """Единая transport-facing сессия одного диалога."""

    service: ChatService
    runtime: ChatRuntime
    commands: CommandService
    dialog_key: str
    rag_error: str | None = None

    @property
    def model_name(self) -> str:
        return self.service.model_name

    @property
    def history(self):
        return self.service.history

    @property
    def rag_enabled(self) -> bool:
        return self.runtime.use_rag

    def get_stats(self) -> dict[str, object]:
        return self.service.get_stats()

    def system_prompt(self) -> str:
        return self.service.prompting.get_system_prompt(
            use_rag=self.runtime.use_rag,
            system_prompt_override=self.runtime.system_prompt_override,
        )

    def command_lines(self) -> list[str]:
        return self.commands.command_lines()

    def try_execute_command(self, raw: str) -> CommandResult | None:
        if not CommandParser.is_command(raw):
            return None
        return self.commands.execute(raw)

    def build_request(
        self,
        user_message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatTurnRequest:
        request = ChatTurnRequest(
            user_message=user_message,
            max_tokens=self.runtime.max_tokens,
            temperature=self.runtime.temperature,
            use_rag=self.runtime.use_rag,
            system_prompt_override=self.runtime.system_prompt_override,
        )
        if metadata:
            request.metadata.update(dict(metadata))
        return request

    def prepare_request(
        self,
        user_message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ChatTurnRequest, PreparedChatTurn]:
        request = self.build_request(user_message, metadata=metadata)
        return request, self.service.prepare_turn(request)

    def run_request(
        self,
        request: ChatTurnRequest,
        *,
        prepared: PreparedChatTurn | None = None,
    ) -> ChatTurnResponse:
        return self.service.run_turn(request, prepared=prepared)

    def run_text(
        self,
        user_message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        prepared: PreparedChatTurn | None = None,
    ) -> ChatTurnResponse:
        request = self.build_request(user_message, metadata=metadata)
        return self.run_request(request, prepared=prepared)

    def stream_request(
        self,
        request: ChatTurnRequest,
        *,
        prepared: PreparedChatTurn | None = None,
    ) -> Generator[ChatStreamEvent, None, None]:
        return self.service.stream_turn(request, prepared=prepared)

    def stream_text(
        self,
        user_message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        prepared: PreparedChatTurn | None = None,
    ) -> Generator[ChatStreamEvent, None, None]:
        request = self.build_request(user_message, metadata=metadata)
        return self.stream_request(request, prepared=prepared)
