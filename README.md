# Smart Therm Bot

Telegram-бот для поддержки пользователей SmartTherm с локальной LLM через Ollama, отвечающий на основе истории чата SmartTherm.

### Быстрый старт в Docker

1. Скопировать шаблон окружения:

```bash
cp .env.example .env
```

2. Заполнить `TELEGRAM_BOT_TOKEN` в [`.env`](.env).

3. Убедиться, что локальная Ollama уже запущена на хосте на стандартном порту `11434`.

4. Собрать и поднять стек:

```bash
make docker-build
make docker-up
```

5. Подтянуть модель в локальную Ollama и при необходимости собрать индексы:

```bash
ollama pull qwen3.5:9b
make docker-reindex
```

## Что сейчас делает проект

- Отвечает на технические вопросы по SmartTherm через Telegram бота.
- Использует релевантный контекст из истории чата SmartTherm через RAG.
- Использует локальную модель через Ollama.
- Хранит историю ходов и вручную вводимые факты по `dialog_key` в SQLite.
- Поддерживает полный pipeline подготовки данных: фильтрация → чанки → индексация.
- Поддерживает CLI чат для тестов.
- Единый composition flow: `telegram -> transport -> composition -> application -> domain`.

По умолчанию бот стартует с RAG, если индексы доступны. В личке он отвечает на любой текст; в группах — только на slash-команды, `@mention` и reply на сообщение бота.

### Где хранятся параметры

- В [`configs/default.yaml`](configs/default.yaml): все постоянные параметры приложения — модель, RAG, memory, общие пути.
- В [`configs/docker.yaml`](configs/docker.yaml): только docker-specific override, сейчас это адрес локальной Ollama на хосте.
- В [`.env`](.env): только environment-specific и секретные значения, прежде всего `TELEGRAM_BOT_TOKEN`.

## Полезные команды

- `make docker-build` — пересобрать Docker-образ бота.
- `make docker-up` / `make docker-down` — поднять или остановить контейнер бота.
- `make docker-remove` — удалить контейнер, сеть и volumes compose-стека.
- `make docker-reindex` — собрать RAG-индексы внутри docker-контейнера.

## Структура проекта

- `scripts/` — CLI-скрипты для чата, обработки и индексации.
- `src/chat/` — структура чата и composition root.
- `src/data_processing/` — фильтрация сообщений и формирование RAG чанков.
- `src/rag/` — индексация и retrieval implementation.
- `src/memory/` — память диалогов.
- `src/llm/` — клиент работы с Ollama.
- `src/utils/` — prompt manager и текстовые утилиты.
- `data/` — сырые/обработанные данные и индексы.
