"""Chat domain package."""

from .bootstrap import ChatApp, build_chat_app, create_client, create_rag_pipeline
from .commands import CommandContext, CommandDispatcher, CommandParser, CommandResult, ParsedCommand
from .contracts import ChatModelClient, ChatRAGPipeline
from .models import ChatMessage, ChatStreamEvent, ChatTurnRequest, ChatTurnResponse, PreparedChatTurn, RetrievedContext
from .prompting import ChatPrompting
from .runtime import ChatRuntime
from .service import ChatService
from .session import ChatSession

__all__ = [
    "build_chat_app",
    "create_client",
    "create_rag_pipeline",
    "ChatApp",
    "ChatModelClient",
    "ChatRAGPipeline",
    "ChatMessage",
    "ChatPrompting",
    "ChatRuntime",
    "PreparedChatTurn",
    "ChatService",
    "ChatSession",
    "ChatStreamEvent",
    "ChatTurnRequest",
    "ChatTurnResponse",
    "CommandContext",
    "CommandDispatcher",
    "CommandParser",
    "CommandResult",
    "ParsedCommand",
    "RetrievedContext",
]
