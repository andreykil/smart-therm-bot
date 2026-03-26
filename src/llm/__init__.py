"""
LLM модуль для работы с Ollama

Использование:
    from src.llm import OllamaClient
    
    client = OllamaClient(model="llama3.1")
    client.load()
    response = client.generate(prompt="Привет!", max_tokens=100)
"""

from .ollama import OllamaClient

__all__ = ["OllamaClient"]
