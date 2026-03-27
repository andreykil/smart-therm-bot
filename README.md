# Smart Therm Bot

Telegram-бот для поддержки пользователей SmartTherm с локальной LLM (Ollama) и RAG-поиском по истории чата. Telegram bot пока не реализован.

## Что сейчас делает проект

- Отвечает на технические вопросы по SmartTherm в интерактивном CLI-чате.
- Использует локальную модель через Ollama.
- Подключает RAG, чтобы опираться на факты из истории чата.
- Поддерживает полный pipeline подготовки данных: фильтрация → чанки → индексация.

## Старт

1. Создать и активировать виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Подготовить Ollama и модель из конфига:

```bash
ollama serve
ollama pull qwen3.5:9b
```

4. Посмотреть доступные команды:

```bash
make help
```

## Сценарий работы

```bash
make chat-filter
make chat-chunks
make reindex
make chat
```

Быстрый запуск чата с RAG:

```bash
python scripts/cli_chat.py --rag
```

## Полезные команды

- `make process-chat` — фильтрация и создание чанков одним шагом.
- `make reindex` — переиндексация RAG.
- `make truncate` / `make truncate-n N=100` — подготовка укороченного датасета для тестов.
- `make test` — запуск тестов.
- `make clean` — очистка обработанных данных и индексов.

## Конфигурация

- `configs/default.yaml` — основные параметры проекта (LLM, RAG, обработка, bot/server).
- `configs/prompts.yaml` — централизованные шаблоны промптов.

Изменяйте параметры через конфиг, без хардкода.

## Структура проекта

- `scripts/` — CLI-скрипты для чата, обработки и индексации.
- `src/data_processing/` — фильтрация сообщений и формирование чанков.
- `src/rag/` — индексация и гибридный retrieval.
- `src/llm/` — клиент работы с Ollama.
- `src/utils/` — конфигурация и сборка промптов.
- `data/` — сырые/обработанные данные и индексы.
