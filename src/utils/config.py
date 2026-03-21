"""
Конфигурация проекта
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Конфигурация проекта"""
    
    # Базовые пути
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    # Пути к данным (будут конвертированы в Path)
    data_dir: str = "data"
    models_dir: str = "data/models"
    
    # RAG настройки
    rag: dict = Field(default_factory=lambda: {
        "embedding_model": "BAAI/bge-m3",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5,
        "reranker_model": "BAAI/bge-reranker-base",
    })
    
    # LLM настройки
    llm: dict = Field(default_factory=lambda: {
        "provider": "local",  # local, ollama, vllm
        "model": "vikhr-nemo-12b-instruct-r",  # ДЕФОЛТНАЯ LLM
        "quantization": "Q8_0",
        "temperature": 0.3,
        "max_tokens": 2048,
        "context_size": 8192,
        "stage1": {"temperature": 0.5, "max_tokens": 1000},
        "stage2": {"temperature": 0.1, "max_tokens": 50},
        "stage3": {"temperature": 0.5, "max_tokens": 2048},
    })
    
    # Параметры обработки чата
    chat_processing: dict = Field(default_factory=lambda: {
        "group_size": 50,
        "overlap_size": 5,
        "min_message_length": 10,
        "stop_words": [
            "спасибо", "ок", "понял", "благодарю", "+", "да",
            "спс", "благодарка", "круто", "супер", "хорошо"
        ],
    })
    
    # Bot настройки
    bot: dict = Field(default_factory=lambda: {
        "token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "admin_ids": [],
    })
    
    # Server настройки
    server: dict = Field(default_factory=lambda: {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
    })
    
    # LoRA настройки
    lora: dict = Field(default_factory=lambda: {
        "r": 8,
        "alpha": 16,
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-4,
    })
    
    @field_validator('data_dir', 'models_dir', mode='before')
    @classmethod
    def convert_to_string(cls, v):
        """Конвертировать Path в строку для валидации"""
        if isinstance(v, Path):
            return str(v)
        return v
    
    @property
    def data_dir_path(self) -> Path:
        """Получить путь к data_dir"""
        return self.project_root / self.data_dir
    
    @property
    def models_dir_path(self) -> Path:
        """Получить путь к models_dir"""
        return self.project_root / self.models_dir
    
    @property
    def processed_dir(self) -> Path:
        """Директория обработанных данных"""
        return self.data_dir_path / "processed"
    
    @property
    def indices_dir(self) -> Path:
        """Директория индексов"""
        return self.data_dir_path / "indices"
    
    @property
    def raw_dir(self) -> Path:
        """Директория сырых данных"""
        return self.data_dir_path / "raw"
    
    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "Config":
        """
        Загрузить конфигурацию из файла
        
        Args:
            config_file: Путь к YAML файлу (если None, используется default)
        """
        if config_file is None:
            # Попробовать загрузить из configs/default.yaml
            default_config = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
            if default_config.exists():
                return cls._load_from_yaml(default_config)
        
        return cls()
    
    @classmethod
    def _load_from_yaml(cls, filepath: Path) -> "Config":
        """Загрузить из YAML файла"""
        try:
            import yaml
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except ImportError:
            # YAML не установлен, вернуть default
            return cls()
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}")
            return cls()
    
    def save(self, filepath: Path) -> None:
        """Сохранить конфигурацию в YAML файл"""
        try:
            import yaml
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            print("YAML не установлен, невозможно сохранить конфиг")


# Глобальный экземпляр
_config: Optional[Config] = None


def get_config() -> Config:
    """Получить глобальную конфигурацию"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config
