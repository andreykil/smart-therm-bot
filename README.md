# Smart Therm Bot

Telegram-бот для поддержки пользователей SmartTherm с локальной LLM через Ollama, retrieval по истории чата и persistent memory диалогов на SQLite.

## Что сейчас делает проект

- Отвечает на технические вопросы по SmartTherm в CLI-чате и через Telegram long polling бота.
- Использует локальную модель через Ollama.
- Подключает RAG как внешний retrieval adapter к chat-core.
- Хранит историю ходов и ручные facts по `dialog_key` в SQLite.
- Поддерживает полный pipeline подготовки данных: фильтрация → чанки → индексация.
- Использует единый composition flow для CLI и Telegram: `transport/scripts -> composition -> application -> domain`.

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
python -m scripts.cli_chat --rag
```

Запуск Telegram-бота:

```bash
python -m scripts.run_telegram_bot
# или
make telegram-bot
```

По умолчанию бот стартует с RAG, если индексы доступны. В личке он отвечает на любой текст; в группах — только на slash-команды, `@mention` и reply на сообщение бота.

## Полезные команды

- `make process-chat` — фильтрация и создание чанков одним шагом.
- `make reindex` — переиндексация RAG.
- `make telegram-bot` — запуск Telegram-бота через long polling.
- `make truncate` / `make truncate-n N=100` — подготовка укороченного датасета для тестов.
- `make test` — запуск тестов.
- `source .venv/bin/activate && pyright` — обязательная проверка типов.

## Архитектура чата

```text
transport / scripts -> composition -> application -> domain
                              ↘ memory / rag / llm / config / prompting
```

- [`src/chat/domain/`](src/chat/domain) — только chat-owned модели и порты. Все доменные контракты находятся в [`src/chat/domain/ports.py`](src/chat/domain/ports.py).
- [`src/chat/application/`](src/chat/application) — orchestration одного хода, session facade, slash-команды и runtime одной session в [`src/chat/application/runtime.py`](src/chat/application/runtime.py).
- [`src/chat/composition.py`](src/chat/composition.py) — единственный production composition root: создаёт LLM client, [`PromptManager`](src/utils/prompt_manager.py), SQLite state adapter и retrieval adapter, затем собирает [`SessionFacade`](src/chat/application/session_facade.py).
- [`src/memory/sqlite_dialog_state.py`](src/memory/sqlite_dialog_state.py) и [`src/memory/sqlite_repository.py`](src/memory/sqlite_repository.py) — внешний persistent adapter к порту состояния диалога.
- [`src/rag/retrieval_service.py`](src/rag/retrieval_service.py) — внешний retrieval adapter: преобразует внутренний RAG-результат в chat-owned [`RetrievalResult`](src/chat/domain/models.py).
- [`src/chat/prompting.py`](src/chat/prompting.py) — prompt adapter без self-composition: [`ChatPrompting`](src/chat/prompting.py) получает [`PromptManager`](src/utils/prompt_manager.py) извне.
- [`scripts/cli_chat.py`](scripts/cli_chat.py), [`scripts/run_telegram_bot.py`](scripts/run_telegram_bot.py), [`src/bot/telegram_transport.py`](src/bot/telegram_transport.py) и [`src/bot/telegram_runner.py`](src/bot/telegram_runner.py) — transport entrypoints и Telegram runner поверх одного composition flow.

## Конфигурация

- [`configs/default.yaml`](configs/default.yaml) — основные параметры проекта.
- [`configs/prompts.yaml`](configs/prompts.yaml) — централизованные шаблоны промптов.

Изменяйте параметры через [`Config.load()`](src/config/models.py), без хардкода.

## Структура проекта

- `scripts/` — CLI-скрипты для чата, обработки и индексации.
- `src/chat/` — chat core и composition root.
- `src/data_processing/` — фильтрация сообщений и формирование чанков.
- `src/rag/` — индексация и retrieval implementation.
- `src/memory/` — persistent dialog state adapter.
- `src/llm/` — клиент работы с Ollama.
- `src/utils/` — prompt manager и текстовые утилиты.
- `data/` — сырые/обработанные данные и индексы.
