# Makefile для SmartTherm-помощник

.PHONY: help install download-model chat test clean

help: ## Показать справку
	@echo "SmartTherm-помощник — Makefile команды"
	@echo ""
	@echo "Использование:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-20s %s\n", $$1, $$2}'
	@echo ""

# ==============================================================================
# КОМАНДЫ
# ==============================================================================

install: ## Установка зависимостей
	@echo "📦 Установка зависимостей..."
	pip install -r requirements.txt
	@echo "✅ Готово"

install-dev: ## Установить зависимости для разработки
	@echo "Установка dev зависимостей..."
	pip install -r requirements.txt -r requirements-dev.txt
	@echo "Готово"

list-models: ## Показать все доступные модели
	python3 scripts/download_model.py --list

download-model: ## Скачать дефолтную модель (из configs/default.yaml)
	python3 scripts/download_model.py

chat: ## Запустить интерактивный чат (модель из конфига)
	python3 scripts/chat.py

process-chat: process-all ## Запустить обработку чата (все этапы)

process-all: ## Запустить все этапы обработки
	@echo "Обработка чата (все этапы)..."
	python scripts/process_chat_cli.py all

process-stage1: ## Этап 1: Фильтрация шума
	@echo "🔄 Этап 1: Фильтрация..."
	python scripts/process_chat_cli.py stage1

process-stage2: ## Этап 2: Выделение веток
	@echo "🔄 Этап 2: Ветки..."
	python scripts/process_chat_cli.py stage2

process-stage3: ## Этап 3: Создание чанков
	@echo "🔄 Этап 3: Чанки..."
	python scripts/process_chat_cli.py stage3

# ==============================================================================
# ТЕСТОВЫЕ ДАННЫЕ (обрезанные)
# ==============================================================================

truncate: ## Обрезать сообщения до n слов
	@echo "✂️  Обрезка сообщений..."
	python scripts/truncate_messages.py --limit 95

truncate-n: ## Обрезать сообщения до N (make truncate-n N=100)
	@echo "✂️  Обрезка сообщений до $(N)..."
	python scripts/truncate_messages.py --limit $(N)

test-stage2: ## Тест Этапа 2 на обрезанных данных (с debug)
	@echo "🧪 Тест Этапа 2 (обрезанные данные, debug)..."
	python scripts/process_chat_cli.py stage2 \
		--input-path data/processed/chat/test/messages_test.json \
		--output-path data/processed/chat/test/threads_test.json \
		--debug

test-stage2-nodebug: ## Тест Этапа 2 на обрезанных данных (без debug)
	@echo "🧪 Тест Этапа 2 (обрезанные данные)..."
	python scripts/process_chat_cli.py stage2 \
		--input-path data/processed/chat/test/messages_test.json \
		--output-path data/processed/chat/test/threads_test.json

test-stage3: ## Тест Этапа 3 на готовых ветках (threads_test.json)
	@echo "🧪 Тест Этапа 3 (готовые ветки)..."
	python scripts/process_chat_cli.py stage3 \
		--threads-path data/processed/chat/test/threads_test.json \
		--messages-path data/processed/chat/test/messages_test.json \
		--output-path data/processed/chat/test/chunks_rag_test.jsonl \
		--sample-size 5

test-stage3-debug: ## Тест Этапа 3 с debug (сырой вывод LLM)
	@echo "🧪 Тест Этапа 3 (debug)..."
	python scripts/process_chat_cli.py stage3 \
		--threads-path data/processed/chat/test/threads_test.json \
		--messages-path data/processed/chat/test/messages_test.json \
		--output-path data/processed/chat/test/chunks_rag_test.jsonl \
		--debug

test-stage3-full: test-stage2-nodebug test-stage3 ## Тест Этапа 3 на обрезанных данных (stage2 + stage3)
	@echo "✅ Тест Этапа 3 завершён"

test-all: truncate test-stage2-nodebug test-stage3 ## Полный тест на обрезанных данных (truncate + stage2 + stage3)
	@echo "✅ Полный тест завершён"
	@echo "Результаты в: data/processed/chat/test/"

validate: ## Валидировать результаты обработки
	@echo "Валидация..."
	python scripts/validate_stages.py

test-stage2-debug: ## Тест Этапа 2 (debug, 50 сообщений)
	@echo "Тест Этапа 2 (debug)..."
	python scripts/test_stage1.py 

# ==============================================================================
# НЕ РЕАЛИЗОВАНО
# ==============================================================================

process-data: process-all ## Обработать данные (чат, инструкция, репо)

build-index: ## Построить векторный индекс
	@echo "Построение индекса..."
	@echo "Пока не реализовано"

train-lora: ## Обучить LoRA адаптер
	@echo "Обучение LoRA..."
	@echo "Пока не реализовано"

test: ## Запустить тесты
	@echo "Запуск тестов..."
	pytest tests/ -v

test-llm: ## Тестировать LLM клиент (registry)
	@echo "Тест LLM registry..."
	python3 -c "from src.llm.registry import list_models, get_model; models = list_models(); print(f'✅ Доступно моделей: {len(models)}'); print('Модели:', models)"

run-bot: ## Запустить Telegram-бота
	@echo "Запуск бота..."
	@echo "Пока не реализовано"

eval: ## Оценка качества RAG/LoRA
	@echo "Оценка качества..."
	@echo "Пока не реализовано"

# ==============================================================================
# ПРОЧЕЕ
# ==============================================================================

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
