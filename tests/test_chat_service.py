from collections.abc import Iterable

from src.chat import ChatService, ChatSession, ChatTurnRequest
from src.rag.models import ChunkContent, ChunkMetadata, RAGChunk, SearchResult


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

    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False) -> SearchResult:
        del use_reranker
        self.queries.append((query, top_k))
        chunk = RAGChunk(
            content=ChunkContent(text="Полезный факт", code=""),
            metadata=ChunkMetadata(date="2024-01-01", tags=["wifi"], version=None, confidence=0.9),
        )
        return SearchResult(chunks=[chunk], query=query, total_found=1, reranked=False)

    def get_stats(self) -> dict:
        return {"total_chunks": 1, "weights": {"vector": 0.5, "bm25": 0.5}}


class MisconfiguredRAG:
    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False) -> SearchResult:
        del query, top_k, use_reranker
        raise ValueError("missing FAISS index path")

    def get_stats(self) -> dict:
        return {}


class BrokenRAG:
    def search(self, query: str, top_k: int | None = None, use_reranker: bool = False) -> SearchResult:
        del query, top_k, use_reranker
        raise RuntimeError("faiss search crashed")

    def get_stats(self) -> dict:
        return {}


def test_chat_service_runs_non_stream_turn_with_rag() -> None:
    client = FakeClient()
    rag = FakeRAG()
    service = ChatService(client, session=ChatSession(), rag_pipeline=rag, top_k=3)

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
    assert len(service.session.messages) == 2


def test_chat_service_stream_turn_updates_session_and_emits_final_event() -> None:
    client = FakeClient()
    service = ChatService(client, session=ChatSession())

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
    assert service.session.messages[1].content == "Поток"


def test_prepare_turn_is_reused_without_double_rag_query() -> None:
    client = FakeClient()
    rag = FakeRAG()
    service = ChatService(client, session=ChatSession(), rag_pipeline=rag, top_k=4)

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
    service = ChatService(FakeClient(), session=ChatSession(), rag_pipeline=None)

    prepared = service.prepare_turn(ChatTurnRequest(user_message="Где логи?", use_rag=True))

    assert prepared.retrieved_context.enabled is False
    assert prepared.retrieved_context.failed is True
    assert prepared.retrieved_context.error_kind == "retrieval_unavailable"
    assert prepared.retrieved_context.error == "RAG retrieval unavailable: pipeline is not configured"


def test_prepare_turn_marks_rag_misconfiguration() -> None:
    service = ChatService(FakeClient(), session=ChatSession(), rag_pipeline=MisconfiguredRAG())

    prepared = service.prepare_turn(ChatTurnRequest(user_message="Проверь индексы", use_rag=True))

    assert prepared.retrieved_context.enabled is True
    assert prepared.retrieved_context.failed is True
    assert prepared.retrieved_context.error_kind == "misconfiguration"
    assert prepared.retrieved_context.error == "ValueError: missing FAISS index path"


def test_prepare_turn_marks_rag_runtime_failure() -> None:
    service = ChatService(FakeClient(), session=ChatSession(), rag_pipeline=BrokenRAG())

    prepared = service.prepare_turn(ChatTurnRequest(user_message="Проверь runtime", use_rag=True))

    assert prepared.retrieved_context.enabled is True
    assert prepared.retrieved_context.failed is True
    assert prepared.retrieved_context.error_kind == "runtime_failure"
    assert prepared.retrieved_context.error == "RuntimeError: faiss search crashed"
