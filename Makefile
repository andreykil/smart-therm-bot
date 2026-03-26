PYTHON := .venv/bin/python3

.PHONY: help install install-dev chat process-chat chat-filter chat-chunks chunks-debug reindex truncate truncate-n test-chunks test clean clean-models clean-all

help: ## Показать справку
	@echo "SmartTherm-помощник — Makefile команды"
	@echo ""
	@echo "Использование:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-20s %s\n", $$1, $$2}'
	@echo ""

install: ## Установка зависимостей
	@echo "📦 Установка зависимостей..."
	pip install -r requirements.txt
	@echo "✅ Готово"

install-dev: ## Установить зависимости для разработки
	@echo "Установка зависимостей (единый requirements.txt)..."
	pip install -r requirements.txt
	@echo "Готово"

chat: ## Запустить интерактивный чат (модель из конфига)
	$(PYTHON) scripts/chat.py

process-chat: ## Запустить обработку чата (фильтрация + чанки)
	@echo "🔄 Фильтрация..."
	$(PYTHON) scripts/process_chat.py filter
	@echo "🔄 Создание чанков..."
	$(PYTHON) scripts/process_chat.py chunks

chat-filter: ## Фильтрация сообщений
	@echo "🔄 Фильтрация..."
	$(PYTHON) scripts/process_chat.py filter

chat-chunks: ## Создание RAG чанков
	@echo "🔄 Создание чанков..."
	$(PYTHON) scripts/process_chat.py chunks

chunks-debug: ## Создание чанков с сохранением групп
	@echo "🔄 Создание чанков (debug)..."
	$(PYTHON) scripts/process_chat.py chunks --save-groups

reindex: ## Переиндексация RAG (make reindex chunks_file=...)
	@echo "🔄 Переиндексация RAG..."
	$(PYTHON) scripts/reindex_rag.py $(if $(chunks_file),--chunks-file $(chunks_file),)

truncate: ## Обрезать сообщения (default из конфига)
	@echo "✂️  Обрезка сообщений..."
	$(PYTHON) scripts/truncate_messages.py

truncate-n: ## Обрезать сообщения до N (make truncate-n N=100)
	@echo "✂️  Обрезка сообщений до $(N)..."
	$(PYTHON) scripts/truncate_messages.py --limit $(N)

test-chunks: ## Тест чанков (обрезанные данные)
	@echo "🧪 Тест чанков..."
	$(PYTHON) scripts/process_chat.py chunks \
		--input-path data/processed/chat/test/messages_filtered_test.json \
		--output-path data/processed/chat/test/chunks_rag_test.jsonl

test: ## Запустить тесты
	@echo "Запуск тестов..."
	pytest tests/ -v

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
