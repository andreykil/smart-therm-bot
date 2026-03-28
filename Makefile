PYTHON := .venv/bin/python3
DOCKER_ENV_FILE ?= .env

.PHONY: help install install-dev chat telegram-bot process-chat chat-filter chat-chunks chunks-debug reindex truncate truncate-n test-chunks test docker-build docker-up docker-down docker-remove docker-logs docker-reindex clean clean-models clean-all

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
	$(PYTHON) -m scripts.cli_chat

telegram-bot: ## Запустить Telegram-бота через long polling
	$(PYTHON) -m scripts.run_telegram_bot

docker-build: ## Собрать Docker-образ бота
	docker compose --env-file $(DOCKER_ENV_FILE) build bot

docker-up: ## Поднять bot + ollama через Docker Compose
	docker compose --env-file $(DOCKER_ENV_FILE) up -d

docker-down: ## Остановить Docker Compose стек
	docker compose --env-file $(DOCKER_ENV_FILE) down

docker-remove: ## Удалить контейнеры, сеть и volumes Docker Compose
	docker compose --env-file $(DOCKER_ENV_FILE) down --volumes --remove-orphans

docker-logs: ## Показать логи docker compose
	docker compose --env-file $(DOCKER_ENV_FILE) logs -f bot

docker-reindex: ## Переиндексация RAG внутри Docker-контейнера бота
	docker compose --env-file $(DOCKER_ENV_FILE) run --rm bot python -m scripts.reindex_rag

process-chat: ## Запустить обработку чата (фильтрация + чанки)
	@echo "🔄 Фильтрация..."
	$(PYTHON) -m scripts.process_chat filter
	@echo "🔄 Создание чанков..."
	$(PYTHON) -m scripts.process_chat chunks

chat-filter: ## Фильтрация сообщений
	@echo "🔄 Фильтрация..."
	$(PYTHON) -m scripts.process_chat filter

chat-chunks: ## Создание RAG чанков
	@echo "🔄 Создание чанков..."
	$(PYTHON) -m scripts.process_chat chunks

chunks-debug: ## Создание чанков с сохранением групп
	@echo "🔄 Создание чанков (debug)..."
	$(PYTHON) -m scripts.process_chat chunks --save-groups

reindex: ## Переиндексация RAG (make reindex chunks_file=...)
	@echo "🔄 Переиндексация RAG..."
	$(PYTHON) -m scripts.reindex_rag $(if $(chunks_file),--chunks-file $(chunks_file),)

truncate: ## Обрезать сообщения (default из конфига)
	@echo "✂️  Обрезка сообщений..."
	$(PYTHON) -m scripts.truncate_messages

truncate-n: ## Обрезать сообщения до N (make truncate-n N=100)
	@echo "✂️  Обрезка сообщений до $(N)..."
	$(PYTHON) -m scripts.truncate_messages --limit $(N)

test-chunks: ## Тест чанков (обрезанные данные)
	@echo "🧪 Тест чанков..."
	$(PYTHON) -m scripts.process_chat chunks \
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
