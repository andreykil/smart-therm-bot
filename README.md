# Smart Therm Bot

Telegram-бот для поддержки пользователей SmartTherm с локальной LLM через Ollama, отвечающий на основе истории чата SmartTherm.

### Быстрый старт в Docker

1. Скопировать шаблон окружения:

```bash
cp .env.example .env
```

2. Заполнить `TELEGRAM_BOT_TOKEN` в [`.env`](.env).

3. Убедиться, что локальная Ollama уже запущена на хосте на стандартном порту `11434`.

4. Собрать и поднять нужный сервис:

```bash
make bot-build-docker
make bot-run-docker

# или веб-чат
make web-build-docker
make web-run-docker
```

5. Подтянуть модель в локальную Ollama и при необходимости собрать индексы:

```bash
ollama pull qwen3.5:9b
make bot-reindex-docker

# для веб-контейнера отдельно
make web-reindex-docker
```

## Что сейчас делает проект

- Отвечает на технические вопросы по SmartTherm через Telegram бота.
- Использует релевантный контекст из истории чата SmartTherm через RAG.
- Использует локальную модель через Ollama.
- Хранит историю ходов и вручную вводимые факты по `dialog_key` в SQLite.
- Поддерживает полный pipeline подготовки данных: фильтрация → чанки → индексация.
- Поддерживает CLI чат для тестов.

По умолчанию бот стартует с RAG, если индексы доступны. В личке он отвечает на любой текст; в группах — только на slash-команды, обращения `@smart_therm_bot` и ответы на сообщение бота.

### Где хранятся параметры

- В [`configs/default.yaml`](configs/default.yaml): все постоянные параметры приложения — модель, RAG, memory, общие пути.
- В [`configs/docker.bot.yaml`](configs/docker.bot.yaml) и [`configs/docker.web.yaml`](configs/docker.web.yaml): per-service docker override, сейчас это адрес локальной Ollama на хосте и web server settings.
- В [`.env`](.env): только environment-specific и секретные значения, прежде всего `TELEGRAM_BOT_TOKEN`.

## Полезные команды

- `make bot-build-docker` / `make web-build-docker` — пересобрать Docker-образ нужного сервиса.
- `make bot-run-docker` / `make web-run-docker` — поднять Telegram-бота или веб-чат.
- `make bot-stop-docker` / `make web-stop-docker` — остановить нужный контейнер.
- `make bot-remove-docker` / `make web-remove-docker` — удалить контейнер и его volumes.
- `make bot-reindex-docker` / `make web-reindex-docker` — собрать RAG-индексы внутри нужного docker-контейнера.

## Структура проекта

- `scripts/` — CLI-скрипты для чата, обработки и индексации.
- `src/chat/` — структура чата и composition root.
- `src/data_processing/` — фильтрация сообщений и формирование RAG чанков.
- `src/rag/` — индексация и retrieval implementation.
- `src/memory/` — память диалогов.
- `src/llm/` — клиент работы с Ollama.
- `src/utils/` — prompt manager и текстовые утилиты.
- `data/` — сырые/обработанные данные и индексы.
