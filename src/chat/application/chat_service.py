"""Orchestration одного хода чата."""

from __future__ import annotations

from collections.abc import Generator

from src.chat.domain.models import DialogMemoryFact, DialogMessage
from src.chat.domain.ports import ChatContextRetriever, ChatModelClient, DialogState
from src.utils.text_utils import clean_response_text

from ..prompting import ChatPrompting
from .dto import ChatStreamEvent, ChatTurnRequest, ChatTurnResponse, PreparedChatTurn, RetrievedContext


class ChatService:
    """Координирует один ход: history → retrieval → prompting → LLM → state."""

    def __init__(
        self,
        client: ChatModelClient,
        *,
        state: DialogState,
        prompting: ChatPrompting,
        retriever: ChatContextRetriever | None = None,
        top_k: int = 5,
    ):
        self.client = client
        self.state = state
        self.prompting = prompting
        self.retriever = retriever
        self.top_k = top_k

    @property
    def history(self) -> list[DialogMessage]:
        return self.state.recent_messages()

    @property
    def model_name(self) -> str:
        return self.client.model

    def clear_history(self) -> None:
        self.state.clear()

    def list_memory_facts(self) -> list[DialogMemoryFact]:
        return self.state.list_facts()

    def remember_fact(self, key: str, value: str) -> DialogMemoryFact:
        return self.state.remember_fact(key, value)

    def forget_fact(self, key: str) -> bool:
        return self.state.forget_fact(key)

    def get_stats(self) -> dict[str, object]:
        return {
            "client": self.client.get_stats(),
            "dialog": self.state.stats(),
            "rag": self.retriever.get_stats() if self.retriever is not None else None,
        }

    @staticmethod
    def _format_retrieval_error(error: Exception) -> str:
        return f"{type(error).__name__}: {error}"

    def _retrieve_context(self, user_message: str, use_rag: bool) -> RetrievedContext:
        if not use_rag:
            return RetrievedContext(enabled=False, query=user_message)

        if self.retriever is None:
            return RetrievedContext(
                enabled=False,
                query=user_message,
                error="RAG retrieval unavailable: pipeline is not configured",
                error_kind="retrieval_unavailable",
            )

        try:
            result = self.retriever.search(user_message, top_k=self.top_k)
            context_text = result.to_context_string() if result.total_found > 0 else ""
            return RetrievedContext(
                enabled=True,
                query=user_message,
                result=result,
                context_text=context_text,
            )
        except (AttributeError, TypeError, ValueError) as error:
            return RetrievedContext(
                enabled=True,
                query=user_message,
                error=self._format_retrieval_error(error),
                error_kind="misconfiguration",
            )
        except Exception as error:
            return RetrievedContext(
                enabled=True,
                query=user_message,
                error=self._format_retrieval_error(error),
                error_kind="runtime_failure",
            )

    def _build_llm_messages(self, request: ChatTurnRequest, context: RetrievedContext) -> list[dict[str, str]]:
        history = [message.to_llm_message() for message in self.state.recent_messages()]
        facts = self.state.list_facts()
        return self.prompting.build_chat_messages(
            user_question=request.user_message,
            history=history,
            rag_context=context.context_text,
            facts=facts,
            use_rag=request.use_rag,
            system_prompt_override=request.system_prompt_override,
        )

    def _store_turn(self, request: ChatTurnRequest, assistant_message: str, context: RetrievedContext) -> None:
        self.state.append_turn(
            user_message=request.user_message,
            assistant_message=assistant_message,
            rag_enabled=request.use_rag,
            rag_query=context.query,
            rag_total_found=context.total_found,
            user_metadata=request.metadata,
        )

    def prepare_turn(self, request: ChatTurnRequest) -> PreparedChatTurn:
        context = self._retrieve_context(request.user_message, request.use_rag)
        llm_messages = self._build_llm_messages(request, context)
        return PreparedChatTurn(request=request, llm_messages=llm_messages, retrieved_context=context)

    def run_turn(self, request: ChatTurnRequest, *, prepared: PreparedChatTurn | None = None) -> ChatTurnResponse:
        prepared_turn = prepared or self.prepare_turn(request)
        response_text = self.client.chat(
            messages=prepared_turn.llm_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        response_text = clean_response_text(response_text, strip_spaces=True)

        self._store_turn(request, response_text, prepared_turn.retrieved_context)
        return ChatTurnResponse(
            user_message=request.user_message,
            assistant_message=response_text,
            llm_messages=prepared_turn.llm_messages,
            retrieved_context=prepared_turn.retrieved_context,
            streamed=False,
        )

    def stream_turn(
        self,
        request: ChatTurnRequest,
        *,
        prepared: PreparedChatTurn | None = None,
    ) -> Generator[ChatStreamEvent, None, None]:
        prepared_turn = prepared or self.prepare_turn(request)

        response_parts: list[str] = []
        first_token = True
        for raw_token in self.client.chat_stream(
            messages=prepared_turn.llm_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            token = clean_response_text(raw_token, strip_spaces=first_token)
            first_token = False
            if not token:
                continue
            response_parts.append(token)
            yield ChatStreamEvent(kind="token", text=token)

        assistant_message = "".join(response_parts)
        self._store_turn(request, assistant_message, prepared_turn.retrieved_context)
        yield ChatStreamEvent(
            kind="final",
            response=ChatTurnResponse(
                user_message=request.user_message,
                assistant_message=assistant_message,
                llm_messages=prepared_turn.llm_messages,
                retrieved_context=prepared_turn.retrieved_context,
                streamed=True,
            ),
        )
