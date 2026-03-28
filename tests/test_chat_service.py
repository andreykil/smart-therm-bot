from collections.abc import Iterable
from pathlib import Path
import tempfile

from src.chat.application.chat_service import ChatService
from src.chat.application.dto import ChatTurnRequest
from src.chat.domain.models import RetrievalResult, RetrievedChunk
from src.chat.prompting import ChatPrompting
from src.memory.sqlite_dialog_state import SQLiteDialogState
from src.memory.sqlite_repository import SQLiteMemoryRepository
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
    def __init__(self) -> None:
        self.model = "fake-model"
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def load(self, strict: bool = True) -> None:
        del strict

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        think: bool | None = None,
    ) -> str:
        del top_p, stop, think
        self.chat_calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return "Ответ"

    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        think: bool | None = None,
    ) -> Iterable[str]:
        del top_p, stop, think
        self.stream_calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        yield "По"
        yield "ток"

    def get_stats(self) -> dict:
        return {"model": self.model}


class FakeRAG:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False) -> RetrievalResult:
        del use_reranker
        self.queries.append((query, top_k))
        chunk = RetrievedChunk(
            text="Полезный факт",
            source="telegram chat",
            tags=("wifi",),
            confidence=0.9,
        )
        return RetrievalResult(chunks=(chunk,), query=query, total_found=1, reranked=False)

    def get_stats(self) -> dict:
        return {"total_chunks": 1, "weights": {"vector": 0.5, "bm25": 0.5}}


class MisconfiguredRAG:
    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False) -> RetrievalResult:
        del query, top_k, use_reranker
        raise ValueError("missing FAISS index path")

    def get_stats(self) -> dict:
        return {}


class BrokenRAG:
    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False) -> RetrievalResult:
        del query, top_k, use_reranker
        raise RuntimeError("faiss search crashed")

    def get_stats(self) -> dict:
        return {}


def build_dialog_state() -> SQLiteDialogState:
    db_dir = Path(tempfile.mkdtemp())
    return SQLiteDialogState(
        SQLiteMemoryRepository(db_dir / "memory.sqlite3"),
        dialog_key="chat:42",
        history_window=12,
    )


def test_chat_service_runs_non_stream_turn_with_rag() -> None:
    client = FakeClient()
    rag = FakeRAG()
    service = ChatService(client, state=InMemoryDialogState(), prompting=build_prompting(), retriever=rag, top_k=3)

    response = service.run_turn(
        ChatTurnRequest(
            user_message="Как подключить WiFi?",
            max_tokens=128,
            temperature=0.3,
            use_rag=True,
        )
    )

    assert response.assistant_message == "Ответ"
    assert response.retrieved_context.used is True
    assert rag.queries == [("Как подключить WiFi?", 3)]
    assert client.chat_calls[0]["messages"][0]["role"] == "system"
    assert "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ" in client.chat_calls[0]["messages"][-1]["content"]
    assert len(service.history) == 2


def test_chat_service_stream_turn_updates_session_and_emits_final_event() -> None:
    client = FakeClient()
    service = ChatService(client, state=InMemoryDialogState(), prompting=build_prompting())

    events = list(
        service.stream_turn(
            ChatTurnRequest(
                user_message="Привет",
                max_tokens=32,
                temperature=0.5,
                use_rag=False,
            )
        )
    )

    assert [event.kind for event in events] == ["token", "token", "final"]
    assert [event.text for event in events if event.kind == "token"] == ["По", "ток"]
    assert events[-1].kind == "final"
    assert events[-1].response is not None
    assert events[-1].response.assistant_message == "Поток"
    assert service.history[1].content == "Поток"


def test_prepare_turn_is_reused_without_double_rag_query() -> None:
    client = FakeClient()
    rag = FakeRAG()
    service = ChatService(client, state=InMemoryDialogState(), prompting=build_prompting(), retriever=rag, top_k=4)

    request = ChatTurnRequest(
        user_message="Нужен совет по WiFi",
        max_tokens=64,
        temperature=0.2,
        use_rag=True,
    )

    prepared = service.prepare_turn(request)
    events = list(service.stream_turn(request, prepared=prepared))

    assert rag.queries == [("Нужен совет по WiFi", 4)]
    assert events[-1].kind == "final"
    assert events[-1].response is not None
    assert events[-1].response.retrieved_context.used is True


def test_prepare_turn_marks_retrieval_unavailable_when_rag_requested_without_pipeline() -> None:
    service = ChatService(FakeClient(), state=InMemoryDialogState(), prompting=build_prompting(), retriever=None)

    prepared = service.prepare_turn(ChatTurnRequest(user_message="Где логи?", use_rag=True))

    assert prepared.retrieved_context.enabled is False
    assert prepared.retrieved_context.failed is True
    assert prepared.retrieved_context.error_kind == "retrieval_unavailable"
    assert prepared.retrieved_context.error == "RAG retrieval unavailable: pipeline is not configured"


def test_prepare_turn_marks_rag_misconfiguration() -> None:
    service = ChatService(
        FakeClient(),
        state=InMemoryDialogState(),
        prompting=build_prompting(),
        retriever=MisconfiguredRAG(),
    )

    prepared = service.prepare_turn(ChatTurnRequest(user_message="Проверь индексы", use_rag=True))

    assert prepared.retrieved_context.enabled is True
    assert prepared.retrieved_context.failed is True
    assert prepared.retrieved_context.error_kind == "misconfiguration"
    assert prepared.retrieved_context.error == "ValueError: missing FAISS index path"


def test_prepare_turn_marks_rag_runtime_failure() -> None:
    service = ChatService(FakeClient(), state=InMemoryDialogState(), prompting=build_prompting(), retriever=BrokenRAG())

    prepared = service.prepare_turn(ChatTurnRequest(user_message="Проверь runtime", use_rag=True))

    assert prepared.retrieved_context.enabled is True
    assert prepared.retrieved_context.failed is True
    assert prepared.retrieved_context.error_kind == "runtime_failure"
    assert prepared.retrieved_context.error == "RuntimeError: faiss search crashed"


def test_chat_service_persists_turn_and_manual_facts() -> None:
    state = build_dialog_state()
    service = ChatService(
        FakeClient(),
        state=state,
        prompting=build_prompting(),
    )

    saved_fact = service.remember_fact(" Name ", " Андрей ")
    response = service.run_turn(ChatTurnRequest(user_message="Привет", metadata={"chat_id": 42}))

    assert response.assistant_message == "Ответ"
    assert saved_fact.key == "name"
    assert saved_fact.value == "Андрей"
    assert state.stats()["messages_persisted"] == 2
    assert [(fact.key, fact.value) for fact in state.list_facts()] == [("name", "Андрей")]
    assert "name: Андрей" in response.llm_messages[-1]["content"]


def test_chat_service_clear_history_clears_persistent_memory() -> None:
    state = build_dialog_state()
    service = ChatService(
        FakeClient(),
        state=state,
        prompting=build_prompting(),
    )

    service.remember_fact("name", "Андрей")
    service.run_turn(ChatTurnRequest(user_message="Привет"))
    service.clear_history()

    assert service.history == []
    assert state.stats() == {
        "messages_persisted": 0,
        "cached_messages": 0,
        "facts": 0,
        "history_window": 12,
    }
