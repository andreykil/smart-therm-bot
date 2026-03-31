import scripts.cli_chat as chat_cli
from pathlib import Path
import tempfile
from typing import cast

from src.bot import TelegramTransport, TelegramTransportRequest
from src.chat.application.chat_service import ChatService
from src.chat.application.command_service import CommandContext, CommandService
from src.chat.application.dto import ChatTurnRequest
from src.chat.application.runtime import ChatRuntime
from src.chat.application.session_facade import SessionFacade
from src.chat.composition import ChatRuntimeDefaults, SharedChatDependencies, create_chat_session
from src.chat.prompting import ChatPrompting
from src.chat.registry import DialogRegistry
from src.config import Config
from src.memory.sqlite_repository import SQLiteMemoryRepository
from src.rag.composition import RAGInitializationResult, initialize_retrieval_service
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.retrieval_service import RetrievalService
from src.utils.prompt_manager import PromptManager

from tests.helpers.in_memory_dialog_state import InMemoryDialogState


def build_prompting() -> ChatPrompting:
    return ChatPrompting(
        prompt_manager=PromptManager(
            prompts={
                "chat_system_base": "SYSTEM BASE",
                "chat_with_rag_policy": "RAG POLICY",
                "chat_without_rag_policy": "NO RAG POLICY",
                "chat_memory_block": "ПАМЯТЬ:\n{memory_context}",
                "chat_context_block": "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{rag_context}",
                "chat_question_block": "ВОПРОС:\n{user_question}",
            }
        )
    )


class FakeClient:
    def __init__(self, model: str, base_url: str, verbose: bool = False, think: bool | None = None):
        del verbose, think
        self.model = model
        self.base_url = base_url
        self.loaded = False

    def load(self, strict: bool = True) -> None:
        del strict
        self.loaded = True

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        think: bool | None = None,
    ) -> str:
        del messages, max_tokens, temperature, top_p, stop, think
        return "single-response"

    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        think: bool | None = None,
    ):
        del messages, max_tokens, temperature, top_p, stop, think
        yield "от"
        yield "вет"

    def get_stats(self) -> dict:
        return {"model": self.model, "loaded": self.loaded}


class FakeRAG:
    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False):
        del query, top_k, use_reranker
        raise AssertionError("search should not be called in this smoke test")

    def get_stats(self) -> dict:
        return {"total_chunks": 1, "weights": {"vector": 0.5, "bm25": 0.5}}


class FakeStore:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class FakeHybridRetrieverForStats:
    def __init__(self, *, total_chunks: int, bm25_documents: int, top_k: int = 3):
        self.vector_store = FakeStore(total_chunks)
        self.bm25_store = FakeStore(bm25_documents)
        self.vector_weight = 0.6
        self.bm25_weight = 0.4
        self.top_k = top_k
        self.reranker = type("FakeReranker", (), {"name": "no-op"})()

    def search(self, query: str, top_k: int, use_reranker: bool = False):
        del query, top_k, use_reranker
        raise AssertionError("search is not expected in stats-only test")


def build_memory_repository() -> SQLiteMemoryRepository:
    db_dir = Path(tempfile.mkdtemp())
    return SQLiteMemoryRepository(db_dir / "memory.sqlite3")


def build_registry(*, retriever=None, use_rag: bool = False, debug: bool = False) -> DialogRegistry:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    shared_dependencies = SharedChatDependencies(
        client=client,
        prompting=build_prompting(),
        retriever=retriever,
        memory_repository=build_memory_repository(),
        dialog_history_limit=12,
        registry_max_contexts=1000,
        registry_idle_ttl_seconds=3600,
        top_k=3,
        runtime_defaults=ChatRuntimeDefaults(
            max_tokens=64,
            temperature=0.3,
            use_rag=use_rag,
            debug=debug,
        ),
    )
    return DialogRegistry(
        session_factory=lambda dialog_key: create_chat_session(
            shared_dependencies=shared_dependencies,
            dialog_key=dialog_key,
        ),
        max_contexts=shared_dependencies.registry_max_contexts,
        idle_ttl_seconds=shared_dependencies.registry_idle_ttl_seconds,
    )


def test_main_prompt_mode_uses_unified_flow(monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.chat.composition.OllamaClient", FakeClient)

    chat_cli.main(["--prompt", "Привет"])
    captured = capsys.readouterr()

    assert "single-response" in captured.out
    assert "📝 Запрос:" in captured.out


def test_main_reports_rag_initialization_reason(monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.chat.composition.OllamaClient", FakeClient)

    def broken_rag_setup(*args, **kwargs):
        del args, kwargs
        return RAGInitializationResult(retrieval_service=None, index_manager=None, error="ValueError: broken faiss index")

    monkeypatch.setattr("src.chat.composition.initialize_retrieval_service", broken_rag_setup)

    chat_cli.main(["--rag", "--prompt", "Привет"])
    captured = capsys.readouterr()

    assert "RAG недоступен" in captured.out
    assert "ValueError: broken faiss index" in captured.out


def test_retrieval_service_stats_preserve_chunk_counters_for_cli() -> None:
    service = RetrievalService(
        hybrid_retriever=cast(
            HybridRetriever,
            FakeHybridRetrieverForStats(total_chunks=7, bm25_documents=7),
        ),
        default_top_k=3,
    )

    stats = service.get_stats()

    assert stats["total_chunks"] == 7
    assert stats["faiss_vectors"] == 7
    assert stats["bm25_documents"] == 7
    assert stats["weights"] == {"vector": 0.6, "bm25": 0.4}


def test_initialize_retrieval_service_resolves_relative_chunks_path(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeIndexManager:
        def index_from_file(self, chunks_file: str, *, save: bool = True) -> None:
            del save
            captured["chunks_file"] = chunks_file

    class FakeRuntime:
        def __init__(self) -> None:
            self.retrieval_service = object()
            self.index_manager = FakeIndexManager()

    monkeypatch.setattr("src.rag.composition.build_rag_runtime", lambda **_: FakeRuntime())

    config = Config()
    result = initialize_retrieval_service(
        config=config,
        base_url=config.llm.base_url,
        top_k=config.rag.top_k,
        vector_weight=0.5,
        bm25_weight=0.5,
        chunks_file="data/processed/chat/test/chunks_rag_test.jsonl",
    )

    assert result.retrieval_service is not None
    assert Path(captured["chunks_file"]) == config.project_root / "data" / "processed" / "chat" / "test" / "chunks_rag_test.jsonl"


def test_initialize_retrieval_service_without_chunks_file_only_loads_existing_indices(monkeypatch) -> None:
    events: list[str] = []

    class FakeIndexManager:
        def index_from_file(self, chunks_file: str, *, save: bool = True) -> None:
            del chunks_file, save
            events.append("index_from_file")
            raise AssertionError("index_from_file should not be called without chunks_file")

        def ensure_loaded(self) -> bool:
            events.append("ensure_loaded")
            return True

    class FakeRuntime:
        def __init__(self) -> None:
            self.retrieval_service = object()
            self.index_manager = FakeIndexManager()

    monkeypatch.setattr("src.rag.composition.build_rag_runtime", lambda **_: FakeRuntime())

    config = Config()
    result = initialize_retrieval_service(
        config=config,
        base_url=config.llm.base_url,
        top_k=config.rag.top_k,
        vector_weight=0.5,
        bm25_weight=0.5,
    )

    assert result.retrieval_service is not None
    assert events == ["ensure_loaded"]


def test_initialize_retrieval_service_test_mode_overrides_explicit_chunks_file(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeIndexManager:
        def index_from_file(self, chunks_file: str, *, save: bool = True) -> None:
            del save
            captured["chunks_file"] = chunks_file

    class FakeRuntime:
        def __init__(self) -> None:
            self.retrieval_service = object()
            self.index_manager = FakeIndexManager()

    monkeypatch.setattr("src.rag.composition.build_rag_runtime", lambda **_: FakeRuntime())

    config = Config()
    result = initialize_retrieval_service(
        config=config,
        base_url=config.llm.base_url,
        top_k=config.rag.top_k,
        vector_weight=0.5,
        bm25_weight=0.5,
        chunks_file="data/processed/chat/custom_chunks.jsonl",
        test_mode=True,
    )

    assert result.retrieval_service is not None
    assert Path(captured["chunks_file"]) == config.project_root / "data" / "processed" / "chat" / "test" / "chunks_rag_test.jsonl"


def test_command_dispatcher_and_stream() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    service = ChatService(
        client,
        state=InMemoryDialogState(),
        prompting=build_prompting(),
        retriever=FakeRAG(),
        top_k=3,
    )
    runtime = ChatRuntime(max_tokens=64, temperature=0.3, use_rag=False, debug=True)
    dispatcher = CommandService(
        CommandContext(
            service=service,
            runtime=runtime,
        )
    )

    result = dispatcher.execute("/rag")
    assert "RAG включен" in "\n".join(result.lines)
    assert runtime.use_rag is True

    request = ChatTurnRequest(
        user_message="Привет",
        max_tokens=runtime.max_tokens,
        temperature=runtime.temperature,
        use_rag=runtime.use_rag,
        system_prompt_override=runtime.system_prompt_override,
    )
    prepared = service.prepare_turn(request)
    events = list(service.stream_turn(request, prepared=prepared))
    assert "".join(event.text for event in events if event.kind == "token") == "ответ"


def test_command_dispatcher_does_not_enable_rag_without_pipeline() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    service = ChatService(
        client,
        state=InMemoryDialogState(),
        prompting=build_prompting(),
        retriever=None,
        top_k=3,
    )
    runtime = ChatRuntime(max_tokens=64, temperature=0.3, use_rag=False, debug=False)
    dispatcher = CommandService(CommandContext(service=service, runtime=runtime))

    result = dispatcher.execute("/rag")

    assert "RAG недоступен" in "\n".join(result.lines)
    assert runtime.use_rag is False


def test_run_interactive_chat_handles_exit_locally(monkeypatch, capsys) -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    service = ChatService(
        client,
        state=InMemoryDialogState(),
        prompting=build_prompting(),
        retriever=None,
        top_k=3,
    )
    runtime = ChatRuntime(max_tokens=64, temperature=0.3, use_rag=False, debug=False)
    dispatcher = CommandService(CommandContext(service=service, runtime=runtime))
    session = SessionFacade(service=service, runtime=runtime, commands=dispatcher, dialog_key="cli")

    monkeypatch.setattr("builtins.input", lambda _: "/exit")

    chat_cli.run_interactive_chat(session)
    captured = capsys.readouterr()

    assert "👋 До свидания!" in captured.out
    assert "/exit             — выйти" in captured.out


def test_telegram_transport_routes_commands_and_regular_messages(monkeypatch) -> None:
    def markdown_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        think: bool | None = None,
    ) -> str:
        del self, messages, max_tokens, temperature, top_p, stop, think
        return "# Ответ\n\n- пункт\n- второй"

    monkeypatch.setattr(FakeClient, "chat", markdown_chat)

    registry = build_registry()
    transport = TelegramTransport(registry)

    command_response = transport.handle_request(TelegramTransportRequest(chat_id=101, user_id=10, text="/help"))
    assert command_response.is_command is True
    assert "<b>Команды:</b>" in command_response.text
    assert command_response.parse_mode == "HTML"
    assert "/exit" not in command_response.text
    assert command_response.dialog_key == "chat:101"

    message_response = transport.handle_text("Привет", chat_id=101, user_id=10)
    assert message_response.is_command is False
    assert message_response.parse_mode == "HTML"
    assert "<b>Ответ</b>" in message_response.text
    assert "• пункт" in message_response.text
    assert "• второй" in message_response.text
    assert message_response.dialog_key == "chat:101"


def test_telegram_transport_isolates_state_between_dialogs() -> None:
    registry = build_registry()
    transport = TelegramTransport(registry)

    transport.handle_text("Привет из первого чата", chat_id=101, user_id=10)
    transport.handle_text("Привет из второго чата", chat_id=202, user_id=20)

    first_context = registry.acquire("chat:101")
    second_context = registry.acquire("chat:202")

    assert [message.content for message in first_context.session.history] == [
        "Привет из первого чата",
        "single-response",
    ]
    assert [message.content for message in second_context.session.history] == [
        "Привет из второго чата",
        "single-response",
    ]

    registry.release(first_context)
    registry.release(second_context)


def test_telegram_transport_scopes_commands_per_dialog() -> None:
    registry = build_registry(retriever=FakeRAG(), use_rag=False)
    transport = TelegramTransport(registry)

    first_response = transport.handle_text("/rag", chat_id=101, user_id=10)
    second_response = transport.handle_text("/stats", chat_id=202, user_id=20)

    first_context = registry.acquire("chat:101")
    second_context = registry.acquire("chat:202")

    assert "RAG включен" in first_response.text
    assert "use_rag: False" in second_response.text
    assert first_context.session.runtime.use_rag is True
    assert second_context.session.runtime.use_rag is False

    registry.release(first_context)
    registry.release(second_context)


def test_telegram_transport_ignores_thread_id_in_dialog_key() -> None:
    request = TelegramTransportRequest(chat_id=-100123, user_id=10, thread_id=777, text="Привет")

    assert request.dialog_key == "chat:-100123"


def test_telegram_transport_persists_manual_memory_commands() -> None:
    registry = build_registry()
    transport = TelegramTransport(registry)

    remember_response = transport.handle_text("/remember name=Андрей", chat_id=101, user_id=10)
    memory_response = transport.handle_text("/memory", chat_id=101, user_id=10)

    assert "Сохранено: name" in remember_response.text
    assert "name: Андрей" in memory_response.text


def test_dialog_registry_evicts_lru_context_when_limit_reached() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    shared_dependencies = SharedChatDependencies(
        client=client,
        prompting=build_prompting(),
        retriever=None,
        memory_repository=build_memory_repository(),
        dialog_history_limit=12,
        registry_max_contexts=2,
        registry_idle_ttl_seconds=None,
        top_k=3,
        runtime_defaults=ChatRuntimeDefaults(max_tokens=64, temperature=0.3),
    )
    registry = DialogRegistry(
        session_factory=lambda dialog_key: create_chat_session(
            shared_dependencies=shared_dependencies,
            dialog_key=dialog_key,
        ),
        max_contexts=2,
        idle_ttl_seconds=None,
    )

    first = registry.acquire("chat:1")
    registry.release(first)
    second = registry.acquire("chat:2")
    registry.release(second)
    first = registry.acquire("chat:1")
    registry.release(first)
    third = registry.acquire("chat:3")
    registry.release(third)

    assert set(registry._contexts.keys()) == {"chat:1", "chat:3"}


def test_dialog_registry_evicts_idle_contexts_before_creating_new_one() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    shared_dependencies = SharedChatDependencies(
        client=client,
        prompting=build_prompting(),
        retriever=None,
        memory_repository=build_memory_repository(),
        dialog_history_limit=12,
        registry_max_contexts=10,
        registry_idle_ttl_seconds=1,
        top_k=3,
        runtime_defaults=ChatRuntimeDefaults(max_tokens=64, temperature=0.3),
    )
    registry = DialogRegistry(
        session_factory=lambda dialog_key: create_chat_session(
            shared_dependencies=shared_dependencies,
            dialog_key=dialog_key,
        ),
        max_contexts=10,
        idle_ttl_seconds=1,
    )

    idle_context = registry.acquire("chat:1")
    registry.release(idle_context)
    idle_context.last_released_at -= 5

    second = registry.acquire("chat:2")
    registry.release(second)

    assert set(registry._contexts.keys()) == {"chat:2"}


def test_dialog_registry_does_not_evict_context_while_it_is_in_use() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    shared_dependencies = SharedChatDependencies(
        client=client,
        prompting=build_prompting(),
        retriever=None,
        memory_repository=build_memory_repository(),
        dialog_history_limit=12,
        registry_max_contexts=1,
        registry_idle_ttl_seconds=1,
        top_k=3,
        runtime_defaults=ChatRuntimeDefaults(max_tokens=64, temperature=0.3),
    )
    registry = DialogRegistry(
        session_factory=lambda dialog_key: create_chat_session(
            shared_dependencies=shared_dependencies,
            dialog_key=dialog_key,
        ),
        max_contexts=1,
        idle_ttl_seconds=1,
    )

    active_context = registry.acquire("chat:1")
    active_context.last_released_at -= 5

    second_context = registry.acquire("chat:2")

    assert set(registry._contexts.keys()) == {"chat:1", "chat:2"}

    registry.release(second_context)

    assert set(registry._contexts.keys()) == {"chat:1"}

    registry.release(active_context)
    active_context.last_released_at -= 5

    third_context = registry.acquire("chat:3")
    registry.release(third_context)

    assert set(registry._contexts.keys()) == {"chat:3"}


def test_dialog_registry_evicts_overflow_on_release_without_extra_acquire() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    shared_dependencies = SharedChatDependencies(
        client=client,
        prompting=build_prompting(),
        retriever=None,
        memory_repository=build_memory_repository(),
        dialog_history_limit=12,
        registry_max_contexts=1,
        registry_idle_ttl_seconds=None,
        top_k=3,
        runtime_defaults=ChatRuntimeDefaults(max_tokens=64, temperature=0.3),
    )
    registry = DialogRegistry(
        session_factory=lambda dialog_key: create_chat_session(
            shared_dependencies=shared_dependencies,
            dialog_key=dialog_key,
        ),
        max_contexts=1,
        idle_ttl_seconds=None,
    )

    active_context = registry.acquire("chat:1")
    overflow_context = registry.acquire("chat:2")

    assert set(registry._contexts.keys()) == {"chat:1", "chat:2"}

    registry.release(active_context)

    assert set(registry._contexts.keys()) == {"chat:2"}

    registry.release(overflow_context)
