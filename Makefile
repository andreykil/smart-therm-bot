# Makefile для SmartTherm-помощник

.PHONY: help install download-model chat test clean

help: ## Показать справку
	@echo "SmartTherm-помощник — Makefile команды"
	@echo ""
	@echo "Использование:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-15s %s\n", $$1, $$2}'
	@echo ""

install: ## Установить зависимости
	@echo "📦 Установка зависимостей..."
	pip install -r requirements.txt
	@echo "✅ Готово"

install-dev: ## Установить зависимости для разработки
	@echo "📦 Установка dev зависимостей..."
	pip install -r requirements.txt -r requirements-dev.txt
	@echo "✅ Готово"

download-model-q4: ## Скачать модель Q4_K_M (5.5 GB, хорошее качество)
	@echo "📥 Загрузка модели Q4_K_M..."
	python scripts/download_model.py --quantization Q4_K_M

download-model-q5: ## Скачать модель Q5_K_M (6.5 GB, лучшее качество)
	@echo "📥 Загрузка модели Q5_K_M..."
	python scripts/download_model.py --quantization Q5_K_M

download-model-q6: ## Скачать модель Q6_K (7.5 GB, ещё лучше)
	@echo "📥 Загрузка модели Q6_K..."
	python scripts/download_model.py --quantization Q6_K

download-model-q8: ## Скачать модель Q8_0 (9 GB, наилучшее качество)
	@echo "📥 Загрузка модели Q8_0..."
	python scripts/download_model.py --quantization Q8_0

download-model: download-model-q5 ## Скачать модель (по умолчанию Q5_K_M)

list-models: ## Показать доступные модели
	python scripts/download_model.py --list

chat: ## Запустить интерактивный чат с Llama 3.1 8B
	@echo "🦙 Запуск чата с Llama 3.1 8B..."
	python scripts/chat_llama.py

chat-q4: ## Запустить чат с моделью Q4_K_M
	python scripts/chat_llama.py --quantization Q4_K_M

chat-q5: ## Запустить чат с моделью Q5_K_M
	python scripts/chat_llama.py --quantization Q5_K_M

chat-q6: ## Запустить чат с моделью Q6_K
	python scripts/chat_llama.py --quantization Q6_K

chat-q8: ## Запустить чат с моделью Q8_0
	python scripts/chat_llama.py --quantization Q8_0

test: ## Запустить тесты
	@echo "🧪 Запуск тестов..."
	pytest tests/ -v

test-llm: ## Тестировать LLM клиент
	@echo "🧪 Тест LLM клиента..."
	python -c "from src.llm.llama_client import LlamaClient; c = LlamaClient(); print('Model exists:', c.model_exists())"

process-data: ## Обработать данные (чат, инструкция, репо)
	@echo "🔄 Обработка данных..."
	@echo "⚠️  Пока не реализовано"

build-index: ## Построить векторный индекс
	@echo "🔨 Построение индекса..."
	@echo "⚠️  Пока не реализовано"

train-lora: ## Обучить LoRA адаптер
	@echo "🎯 Обучение LoRA..."
	@echo "⚠️  Пока не реализовано"

run-bot: ## Запустить Telegram-бота
	@echo "🤖 Запуск бота..."
	@echo "⚠️  Пока не реализовано"

eval: ## Оценка качества RAG/LoRA
	@echo "📊 Оценка качества..."
	@echo "⚠️  Пока не реализовано"

clean: ## Очистить обработанные данные и индексы
	@echo "🧹 Очистка..."
	rm -rf data/processed/chat/*
	rm -rf data/processed/instruction/*
	rm -rf data/processed/repo/*
	rm -rf data/processed/versions/*
	rm -rf data/indices/faiss/*
	rm -rf data/indices/bm25/*
	@echo "✅ Очищено"

clean-models: ## Очистить модели (освободить место)
	@echo "🧹 Очистка моделей..."
	rm -rf data/models/*
	@echo "✅ Очищено"

clean-all: clean clean-models ## Очистить всё
	@echo "✅ Полная очистка завершена"

# Быстрые команды для разработки
dev: install-dev ## Настроить среду разработки
	@echo "✅ Среда разработки готова"

setup: install download-model ## Полная настройка проекта
	@echo "✅ Проект готов к работе"
