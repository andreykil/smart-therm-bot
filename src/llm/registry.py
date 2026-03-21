"""
Реестр LLM моделей для SmartTherm

Каждая модель имеет:
- display_name: человекочитаемое имя
- huggingface_repo: репозиторий на HF
- file_pattern: шаблон имени файла GGUF
- sizes: словари с размерами для каждого квантования
- context_window: максимальный контекст
- recommended_quantization: квантование по умолчанию
"""

from typing import Dict, Optional


# Реестр доступных моделей
MODEL_REGISTRY: Dict[str, dict] = {
    # === Llama 3.1 семейство ===
    "llama-3.1-8b-instruct": {
        "display_name": "Llama 3.1 8B Instruct",
        "huggingface_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file_pattern": "Meta-Llama-3.1-8B-Instruct-{quantization}.gguf",
        "context_window": 128000,
        "recommended_quantization": "Q4_K_M",
        "sizes": {
            "Q4_K_M": 5.5,
            "Q5_K_M": 6.5,
            "Q6_K": 7.5,
            "Q8_0": 9.0,
        },
        "description": "Оригинальная Llama 3.1 8B от Meta. Хороший баланс скорости и качества.",
    },
    
    # === Vikhr семейство (улучшенные для русского) ===
    "vikhr-llama-3.1-8b-instruct-r": {
        "display_name": "Vikhr-Llama-3.1-8B-Instruct-R",
        "huggingface_repo": "featherless-ai-quants/Vikhrmodels-Vikhr-Llama3.1-8B-Instruct-R-21-09-24-GGUF",
        "file_pattern": "Vikhrmodels-Vikhr-Llama3.1-8B-Instruct-R-21-09-24-{quantization}.gguf",
        "context_window": 128000,
        "recommended_quantization": "Q4_K_M",
        "sizes": {
            "Q4_K_M": 5.5,
            "Q5_K_M": 6.5,
            "Q6_K": 7.5,
            "Q8_0": 9.0,
        },
        "description": "Улучшенная Llama 3.1 для русского языка. Адаптирована для RAG.",
    },
    
    "vikhr-nemo-12b-instruct-r": {
        "display_name": "Vikhr-Nemo-12B-Instruct-R",
        "huggingface_repo": "bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF",
        "file_pattern": "Vikhr-Nemo-12B-Instruct-R-21-09-24-{quantization}.gguf",
        "context_window": 128000,
        "recommended_quantization": "Q4_K_M",
        "sizes": {
            "Q4_K_M": 7.5,
            "Q5_K_M": 8.5,
            "Q6_K": 10.0,
            "Q8_0": 12.0,
        },
        "description": "Vikhr-Nemo 12B — лучшее понимание русского. Рекомендована для обработки чата.",
    },
    
    # === Qwen семейство ===
    "qwen-2.5-7b-instruct": {
        "display_name": "Qwen 2.5 7B Instruct",
        "huggingface_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file_pattern": "Qwen2.5-7B-Instruct-{quantization}.gguf",
        "context_window": 131072,
        "recommended_quantization": "Q4_K_M",
        "sizes": {
            "Q4_K_M": 5.0,
            "Q5_K_M": 6.0,
            "Q6_K": 7.0,
            "Q8_0": 8.5,
        },
        "description": "Qwen 2.5 с отличной поддержкой 29+ языков включая русский.",
    },
}


def get_model(model_id: str) -> Optional[dict]:
    """Получить информацию о модели по ID"""
    return MODEL_REGISTRY.get(model_id)


def list_models() -> list:
    """Вернуть список всех доступных моделей"""
    return list(MODEL_REGISTRY.keys())


def get_model_file_path(model_id: str, quantization: str) -> str:
    """Получить имя файла модели"""
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Модель '{model_id}' не найдена в реестре")
    
    return model["file_pattern"].format(quantization=quantization)


def get_model_url(model_id: str, quantization: str) -> str:
    """Получить URL для скачивания модели"""
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Модель '{model_id}' не найдена в реестре")
    
    filename = get_model_file_path(model_id, quantization)
    return f"https://huggingface.co/{model['huggingface_repo']}/resolve/main/{filename}"


def get_recommended_quantization(model_id: str) -> str:
    """Получить рекомендуемое квантование для модели"""
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Модель '{model_id}' не найдена в реестре")
    
    return model["recommended_quantization"]


def get_model_size(model_id: str, quantization: str) -> float:
    """Получить размер модели в GB"""
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Модель '{model_id}' не найдена в реестре")
    
    return model["sizes"].get(quantization, 0.0)
