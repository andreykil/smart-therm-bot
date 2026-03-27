import scripts.cli_chat as chat_cli
from src.bot import TelegramTransport
from src.chat import (
    ChatApp,
    ChatPrompting,
    ChatRuntime,
    ChatService,
    ChatSession,
    CommandContext,
    CommandDispatcher,
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


def test_main_prompt_mode_uses_unified_flow(monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.chat.bootstrap.OllamaClient", FakeClient)

    chat_cli.main(["--prompt", "Привет"])  # единый ход через ChatService.run_turn()
    captured = capsys.readouterr()

    assert "single-response" in captured.out
    assert "📝 Запрос:" in captured.out


def test_main_reports_rag_initialization_reason(monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.chat.bootstrap.OllamaClient", FakeClient)

    def broken_from_config(*args, **kwargs):
        del args, kwargs
        raise ValueError("broken faiss index")

    monkeypatch.setattr("src.chat.bootstrap.RAGPipeline.from_config", broken_from_config)

    chat_cli.main(["--rag", "--prompt", "Привет"])
    captured = capsys.readouterr()

    assert "RAG недоступен" in captured.out
    assert "ValueError: broken faiss index" in captured.out


def test_command_dispatcher_and_stream() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    session = ChatSession()
    service = ChatService(client, session=session, prompting=ChatPrompting(), rag_pipeline=FakeRAG(), top_k=3)
    runtime = ChatRuntime(max_tokens=64, temperature=0.3, use_rag=False, debug=True)
    dispatcher = CommandDispatcher(
        CommandContext(
            service=service,
            runtime=runtime,
        )
    )

    result = dispatcher.execute("/rag")
    assert "RAG включен" in "\n".join(result.lines)
    assert runtime.use_rag is True

    request = runtime.build_request("Привет")
    prepared = service.prepare_turn(request)
    events = list(service.stream_turn(request, prepared=prepared))
    assert "".join(event.text for event in events if event.kind == "token") == "ответ"


def test_command_dispatcher_does_not_enable_rag_without_pipeline() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    service = ChatService(client, session=ChatSession(), prompting=ChatPrompting(), rag_pipeline=None, top_k=3)
    runtime = ChatRuntime(max_tokens=64, temperature=0.3, use_rag=False, debug=False)
    dispatcher = CommandDispatcher(CommandContext(service=service, runtime=runtime))

    result = dispatcher.execute("/rag")

    assert "RAG недоступен" in "\n".join(result.lines)
    assert runtime.use_rag is False


def test_run_interactive_chat_handles_exit_locally(monkeypatch, capsys) -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    service = ChatService(client, session=ChatSession(), prompting=ChatPrompting(), rag_pipeline=None, top_k=3)
    runtime = ChatRuntime(max_tokens=64, temperature=0.3, use_rag=False, debug=False)
    dispatcher = CommandDispatcher(CommandContext(service=service, runtime=runtime))
    app = ChatApp(service=service, runtime=runtime, commands=dispatcher)

    monkeypatch.setattr("builtins.input", lambda _: "/exit")

    chat_cli.run_interactive_chat(app)
    captured = capsys.readouterr()

    assert "👋 До свидания!" in captured.out
    assert "/exit             — выйти" in captured.out


def test_telegram_transport_routes_commands_and_regular_messages() -> None:
    client = FakeClient(model="fake-model", base_url="http://localhost:11434")
    service = ChatService(client, session=ChatSession(), prompting=ChatPrompting(), rag_pipeline=None, top_k=3)
    runtime = ChatRuntime(max_tokens=64, temperature=0.3, use_rag=False, debug=False)
    dispatcher = CommandDispatcher(CommandContext(service=service, runtime=runtime))
    transport = TelegramTransport(ChatApp(service=service, runtime=runtime, commands=dispatcher))

    command_response = transport.handle_text("/help")
    assert command_response.is_command is True
    assert "Команды:" in command_response.text
    assert "/exit" not in command_response.text

    message_response = transport.handle_text("Привет")
    assert message_response.is_command is False
    assert message_response.text == "single-response"
