"""Composition root для chat-подсистемы."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.llm import OllamaClient
from src.memory.sqlite_dialog_state import SQLiteDialogState
from src.memory.sqlite_repository import SQLiteMemoryRepository
from src.rag.composition import RAGInitializationResult, initialize_retrieval_service
from src.utils.prompt_manager import PromptManager

from .application.chat_service import ChatService
from .application.command_service import CommandContext, CommandService
from .application.runtime import ChatRuntime
from .application.session_facade import SessionFacade
from .domain.ports import ChatContextRetriever, ChatModelClient
from .prompting import ChatPrompting
from .registry import DialogRegistry


@dataclass(slots=True, frozen=True)
class ChatRuntimeDefaults:
    max_tokens: int = 1024
    temperature: float = 0.7
    use_rag: bool = False
    system_prompt_override: str | None = None
    debug: bool = False


@dataclass(slots=True)
class SharedChatDependencies:
    client: ChatModelClient
    prompting: ChatPrompting
    retriever: ChatContextRetriever | None
    memory_repository: SQLiteMemoryRepository
    dialog_history_limit: int | None
    registry_max_contexts: int | None
    registry_idle_ttl_seconds: int | None
    top_k: int
    runtime_defaults: ChatRuntimeDefaults
    rag_error: str | None = None


def create_client(
    *,
    model_name: str,
    base_url: str,
    verbose: bool = False,
    think: bool | None = None,
) -> ChatModelClient:
    client = OllamaClient(model=model_name, base_url=base_url, verbose=verbose, think=think)
    client.load(strict=True)
    return client


def build_chat_shared_dependencies(
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
) -> SharedChatDependencies:
    client = create_client(model_name=model_name, base_url=base_url, verbose=verbose, think=think)
    prompt_manager = PromptManager(prompts_path=config.project_root / "configs" / "prompts.yaml")

    rag_result = RAGInitializationResult(retrieval_service=None, index_manager=None, error=None)
    if use_rag:
        rag_result = initialize_retrieval_service(
            config=config,
            base_url=base_url,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            chunks_file=chunks_file,
            test_mode=test_mode,
        )

    sqlite_path = Path(config.memory.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = config.project_root / sqlite_path

    dialog_history_limit = max(config.memory.session_cache_limit, 0)
    registry_max_contexts = max(config.memory.registry_max_contexts, 0)
    registry_idle_ttl_seconds = max(config.memory.registry_idle_ttl_seconds, 0)

    memory_repository = SQLiteMemoryRepository(sqlite_path)
    runtime_defaults = ChatRuntimeDefaults(
        max_tokens=max_tokens,
        temperature=temperature,
        use_rag=use_rag and rag_result.retrieval_service is not None,
        system_prompt_override=system_prompt_override,
        debug=debug,
    )
    return SharedChatDependencies(
        client=client,
        prompting=ChatPrompting(prompt_manager=prompt_manager),
        retriever=rag_result.retrieval_service,
        memory_repository=memory_repository,
        dialog_history_limit=dialog_history_limit or None,
        registry_max_contexts=registry_max_contexts or None,
        registry_idle_ttl_seconds=registry_idle_ttl_seconds or None,
        top_k=top_k,
        runtime_defaults=runtime_defaults,
        rag_error=rag_result.error,
    )


def create_chat_session(
    *,
    shared_dependencies: SharedChatDependencies,
    dialog_key: str,
) -> SessionFacade:
    state = SQLiteDialogState(
        shared_dependencies.memory_repository,
        dialog_key=dialog_key,
        history_window=shared_dependencies.dialog_history_limit,
    )
    runtime = ChatRuntime(
        max_tokens=shared_dependencies.runtime_defaults.max_tokens,
        temperature=shared_dependencies.runtime_defaults.temperature,
        use_rag=shared_dependencies.runtime_defaults.use_rag,
        system_prompt_override=shared_dependencies.runtime_defaults.system_prompt_override,
        debug=shared_dependencies.runtime_defaults.debug,
    )
    service = ChatService(
        shared_dependencies.client,
        state=state,
        prompting=shared_dependencies.prompting,
        retriever=shared_dependencies.retriever,
        top_k=shared_dependencies.top_k,
    )
    commands = CommandService(CommandContext(service=service, runtime=runtime))
    return SessionFacade(
        service=service,
        runtime=runtime,
        commands=commands,
        dialog_key=dialog_key,
        rag_error=shared_dependencies.rag_error,
    )


def build_dialog_registry(
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
) -> DialogRegistry:
    shared_dependencies = build_chat_shared_dependencies(
        config=config,
        model_name=model_name,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        use_rag=use_rag,
        system_prompt_override=system_prompt_override,
        debug=debug,
        verbose=verbose,
        think=think,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        chunks_file=chunks_file,
        test_mode=test_mode,
    )
    return DialogRegistry(
        session_factory=lambda dialog_key: create_chat_session(
            shared_dependencies=shared_dependencies,
            dialog_key=dialog_key,
        ),
        max_contexts=shared_dependencies.registry_max_contexts,
        idle_ttl_seconds=shared_dependencies.registry_idle_ttl_seconds,
    )


def build_chat_session(
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
) -> SessionFacade:
    shared_dependencies = build_chat_shared_dependencies(
        config=config,
        model_name=model_name,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        use_rag=use_rag,
        system_prompt_override=system_prompt_override,
        debug=debug,
        verbose=verbose,
        think=think,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        chunks_file=chunks_file,
        test_mode=test_mode,
    )
    return create_chat_session(shared_dependencies=shared_dependencies, dialog_key="cli")
