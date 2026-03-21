"""
LLM модуль для SmartTherm

Предоставляет:
- Реестр моделей (registry)
- Базовый класс (base)
- Фабрика движков (factory)
- llama-cpp-python backend (llama_cpp_engine)
"""

# Реестр — всегда доступен
from .registry import (
    MODEL_REGISTRY,
    get_model,
    list_models,
    get_model_file_path,
    get_model_url,
    get_recommended_quantization,
    get_model_size,
)

# Базовый класс — всегда доступен
from .base import LLMEngine

# Фабрика и движки — ленивый импорт (требуют llama-cpp-python)
__all__ = [
    # Реестр
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
    "get_model_file_path",
    "get_model_url",
    "get_recommended_quantization",
    "get_model_size",
    
    # Базовый класс
    "LLMEngine",
    
    # Фабрика (ленивый импорт)
    "create_llm_engine",
    "create_engine_from_config",
    
    # Backend'ы (ленивый импорт)
    "LlamaCppEngine",
]


def __getattr__(name):
    """Ленивый импорт для factory и engine"""
    if name in ("create_llm_engine", "create_engine_from_config"):
        from .factory import create_llm_engine, create_engine_from_config
        if name == "create_llm_engine":
            return create_llm_engine
        return create_engine_from_config
    
    if name == "LlamaCppEngine":
        from .llama_cpp_engine import LlamaCppEngine
        return LlamaCppEngine
    
    raise AttributeError(f"module {__name__!r} has no attribute {__name__!r}")
