"""Composition root для сборки универсального chat application."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from src.llm import OllamaClient
from src.rag import RAGPipeline
from src.utils.config import Config

from .commands import CommandContext, CommandDispatcher
from .contracts import ChatModelClient
from .prompting import ChatPrompting
from .runtime import ChatRuntime
from .service import ChatService
from .session import ChatSession

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatApp:
    """Собранное приложение для transport-слоя."""

    service: ChatService
    runtime: ChatRuntime
    commands: CommandDispatcher
    rag_error: str | None = None


@dataclass(slots=True)
class RAGSetupResult:
    """Результат инициализации RAG для transport-слоя."""

    pipeline: RAGPipeline | None
    error: str | None = None


def create_client(
    *,
    model_name: str,
    base_url: str,
    verbose: bool = False,
    think: bool | None = None,
) -> ChatModelClient:
    """Создать и загрузить LLM client."""
    client = OllamaClient(
        model=model_name,
        base_url=base_url,
        verbose=verbose,
        think=think,
    )
    client.load(strict=True)
    return client


def create_rag_pipeline(
    *,
    config: Config,
    base_url: str,
    use_rag: bool,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    chunks_file: str | None = None,
    test_mode: bool = False,
) -> RAGSetupResult:
    """Создать RAG pipeline при необходимости."""
    if not use_rag:
        return RAGSetupResult(pipeline=None)

    try:
        rag_pipeline = RAGPipeline.from_config(
            config=config.model_dump(),
            data_dir=str(config.data_dir_path),
            ollama_base_url=base_url,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=top_k,
        )

        effective_chunks_file = chunks_file
        if test_mode:
            effective_chunks_file = "data/processed/chat/test/chunks_rag_test.jsonl"

        if effective_chunks_file:
            rag_pipeline.index_from_file(effective_chunks_file, save=True)
            return RAGSetupResult(pipeline=rag_pipeline)

        if rag_pipeline.ensure_indices_loaded():
            return RAGSetupResult(pipeline=rag_pipeline)

        return RAGSetupResult(
            pipeline=None,
            error="RAG индексы не найдены или не удалось загрузить с диска",
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        logger.warning("RAG initialization failed: %s", message)
        return RAGSetupResult(pipeline=None, error=message)

    return RAGSetupResult(pipeline=None, error="Не удалось инициализировать RAG")


def build_chat_app(
    *,
    config: Config,
    model_name: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    use_rag: bool,
    system_prompt_override: str | None = None,
    debug: bool = False,
    verbose: bool = False,
    think: bool | None = None,
    top_k: int = 5,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    chunks_file: str | None = None,
    test_mode: bool = False,
) -> ChatApp:
    """Собрать transport-ready chat application из конфигурации."""
    client = create_client(
        model_name=model_name,
        base_url=base_url,
        verbose=verbose,
        think=think,
    )
    rag_setup = create_rag_pipeline(
        config=config,
        base_url=base_url,
        use_rag=use_rag,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        chunks_file=chunks_file,
        test_mode=test_mode,
    )
    rag_pipeline = rag_setup.pipeline

    service = ChatService(
        client,
        session=ChatSession(),
        prompting=ChatPrompting(),
        rag_pipeline=rag_pipeline,
        top_k=top_k,
    )
    runtime = ChatRuntime(
        max_tokens=max_tokens,
        temperature=temperature,
        use_rag=use_rag and rag_pipeline is not None,
        system_prompt_override=system_prompt_override,
        debug=debug,
    )
    commands = CommandDispatcher(CommandContext(service=service, runtime=runtime))
    return ChatApp(
        service=service,
        runtime=runtime,
        commands=commands,
        rag_error=rag_setup.error,
    )
