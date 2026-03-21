"""
Фабрика LLM движков

Создаёт нужный backend в зависимости от конфигурации:
- local (llama-cpp-python)
- ollama
- vllm
"""

from typing import Optional

from .base import LLMEngine
from .llama_cpp_engine import LlamaCppEngine
from ..utils.config import Config


def create_llm_engine(
    model_id: Optional[str] = None,
    quantization: Optional[str] = None,
    n_ctx: int = 8192,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = -1,  # Все слои на GPU для M2 Max
    verbose: bool = False,
    provider: str = "local"
) -> LLMEngine:
    """
    Создать LLM движок

    Args:
        model_id: ID модели из реестра (None = использовать дефолтную из конфига)
        quantization: Уровень квантования (None = использовать дефолтное из конфига)
        n_ctx: Размер контекста
        n_threads: Количество потоков
        n_gpu_layers: Количество слоёв на GPU (-1 = все слои для M2 Max)
        verbose: Включить логирование
        provider: Тип провайдера ("local", "ollama", "vllm")

    Returns:
        LLMEngine экземпляр

    Raises:
        ValueError: Если provider не поддерживается
    """
    # Загрузить дефолтную конфигурацию если model_id=None
    if model_id is None:
        config = Config.load()
        model_id = config.llm.get("model")
    
    if quantization is None:
        config = Config.load()
        quantization = config.llm.get("quantization")
    
    if provider == "local":
        assert model_id is not None, "model_id must be set"
        assert quantization is not None, "quantization must be set"

        return LlamaCppEngine(
            model_id=model_id,
            quantization=quantization,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )

    elif provider == "ollama":
        # TODO: реализовать OllamaEngine
        raise NotImplementedError("Ollama provider пока не реализован")

    elif provider == "vllm":
        # TODO: реализовать VLLMEngine
        raise NotImplementedError("vLLM provider пока не реализован")

    else:
        raise ValueError(f"Неподдерживаемый provider: {provider}")


def create_engine_from_config(config: Optional[Config] = None, verbose: bool = False) -> LLMEngine:
    """
    Создать LLM движок из конфигурации

    Args:
        config: Конфигурация (если None, загружается default)
        verbose: Включить логирование

    Returns:
        LLMEngine экземпляр
    """
    if config is None:
        config = Config.load()

    llm_config = config.llm

    return create_llm_engine(
        model_id=llm_config.get("model"),
        quantization=llm_config.get("quantization"),
        n_ctx=llm_config.get("context_size", 8192),
        n_gpu_layers=llm_config.get("n_gpu_layers", -1),  # Все слои на GPU для M2 Max
        verbose=verbose
    )
