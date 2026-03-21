# Текущая архитектура проекта SmartTherm-помощник

## Оглавление

1. [Текущая архитектура](#текущая-архитектура)
2. [Будущая архитектура](#будущая-архитектура)
3. [Правила проекта](#правила-проекта)
4. [Команды Makefile](#команды-makefile)
5. [Структура данных](#структура-данных)

---

## Текущая архитектура

```
smart-therm-bot/
│
├── configs/                        # ВСЯ конфигурация здесь
│   └── default.yaml                # Дефолтные параметры
│
├── data/                           # Данные (не коммитить!)
│   ├── raw/                        # НЕ МЕНЯТЬ! Исходные данные
│   │   ├── chat_history.json       # Telegram чат (36K сообщений)
│   │   ├── instruction/            # PDF инструкции
│   │   └── repo/                   # SmartTherm репозиторий
│   │
│   ├── models/                     # GGUF модели (скачанные)
│   │   └── *.gguf
│   │
│   ├── processed/                  # Обработанные данные
│   │   └── chat/
│   │       ├── messages_filtered.json    # Этап 0
│   │       ├── threads.json              # Этап 1
│   │       ├── threads_deduped.json      # Этап 2
│   │       ├── chunks_rag.jsonl          # Этап 3
│   │       └── chunks_sample.json        # Выборка для валидации
│   │
│   └── indices/                    # Векторные индексы
│       ├── faiss/
│       └── bm25/
│
├── src/                            # Исходный код
│   │
│   ├── utils/                      # Общие утилиты
│   │   ├── __init__.py
│   │   ├── config.py               # Загрузка Config из default.yaml
│   │   ├── json_utils.py           # extract_json_from_text()
│   │   └── chat_format.py          # format_llama_chat_prompt()
│   │
│   ├── llm/                        # LLM движки
│   │   ├── __init__.py             
│   │   ├── base.py                 # Абстрактный LLMEngine
│   │   ├── factory.py              # create_llm_engine()
│   │   ├── llama_cpp_engine.py     # llama-cpp-python backend
│   │   └── registry.py             # Реестр моделей (URL, размеры)
│   │
│   └── chat_processor/             # Обработка чата
│       ├── __init__.py
│       ├── models.py               # Pydantic модели данных
│       ├── stage0_filter.py        # Фильтрация шума
│       ├── stage1_threads.py       # Поиск веток
│       ├── stage2_dedup.py         # Удаление дубликатов
│       └── stage3_chunks.py        # Создание RAG чанков
│   
│
├── scripts/                        # ← Скрипты
│   ├── download_model.py           # Скачивание моделей
│   ├── chat.py                     # Интерактивный чат с LLM
│   ├── process_chat_cli.py         # CLI для всех этапов обработки
│   ├── test_stage1.py              # Тест Этапа 1 (отладка)
│   └── validate_stages.py          # Валидация результатов
│
├── docs/                           # Документация
│   ├── plan.md                     # План проекта
│   └── architecture.md             # Этот файл
│
├── tests/                          # Тесты
│   └── __init__.py
│
├── evaluation/                     # Оценка качества
│   └── __init__.py
│
├── docker/                         # Docker конфигурации
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── Makefile                        # Команды для разработки
├── requirements.txt                # Зависимости
├── requirements-dev.txt            # Dev зависимости
├── .env.example                    # Пример переменных окружения
└── .gitignore                      # Git ignore
```

---

## Будущая архитектура

```
smart-therm-bot/
│
├── src/
│   ├── utils/                      # Вспомогательные функции
│   ├── llm/                        # Добавить Ollama
│   ├── chat_processor/             # Почти готово
│   │
│   ├── rag/                        
│   │   ├── __init__.py
│   │   ├── embeddings.py           # bge-m3 эмбеддинги
│   │   ├── retrieval.py            # Гибридный поиск (FAISS + BM25)
│   │   ├── reranker.py             # bge-reranker
│   │   └── faiss_index.py          # FAISS индекс
│   │
│   ├── lora/                       
│   │   ├── __init__.py
│   │   ├── dataset.py              # Подготовка датасета
│   │   ├── train.py                # Обучение (MLX)
│   │   └── adapters/               # Сохранённые адаптеры
│   │
│   ├── bot/                        
│   │   ├── __init__.py
│   │   ├── main.py                 # aiogram entry point
│   │   └── handlers.py             # Обработчики сообщений
│   │
│   └── api/                        
│       ├── __init__.py
│       ├── main.py                 # FastAPI app
│       └── endpoints.py            # API роуты
│
└── data/
    └── indices/                    # После RAG
        ├── faiss/
        │   ├── index.bin           # Векторный индекс
        │   └── metadata.json       # Метаданные чанков
        └── bm25/
            └── index.pkl
```

---

## Правила разработки

- **ВСЕ параметры в `configs/default.yaml`** | Модель, температура, размер группы, и т.д.

- **Никакого хардкода**: В коде читать из `Config.load()`

- **CLI переопределяет дефолты**: `--model`, `--quantization`, и т.д.

- `data/raw/`: **НЕ МЕНЯТЬ!** Только чтение

- **`src/llm/` — общие функции для llm**. Без специфики для обработки данных/чата.

- **`src/utils/` — утилиты**. Вспомогательные функции должны быть здесь

- **`src/chat_processor/` — обработка сообщений чата**

- **Stop sequences** явно передавать в `generate(stop=...)` для остановки генерации

- **JSON извлечение в utils**: `extract_json_from_text()`

- **Pyright: 0 errors** Перед коммитом

---

## Команды Makefile (для дефолтной модели из default.yaml)

### Модели

```bash
# Скачать модель
make download-model

# Показать доступные
make list-models

# Интерактивный чат
make chat
```

### Обработка чата

```bash
# Все этапы (с дефолтной моделью из default.yaml)
make process-all

# Отдельные этапы
make process-stage0      # Фильтрация шума
make process-stage1      # Поиск веток
make process-stage2      # Удаление дубликатов
make process-stage3      # Создание RAG чанков

# Тест Этапа 1 (50 сообщений)
make test-stage1

# Валидация
make validate            # Проверка результатов
```

### Прочее

```bash

# Установка зависимостей
make install         # Основные
make install-dev     # + dev

# Очистка
make clean           # Обработанные данные
make clean-models    # Модели
make clean-all       # Всё

# Тест LLM (registry)
make test-llm

# Помощь
make help
```

---

## Структура данных

### Этапы обработки чата

```
Этап 0: Фильтрация
  Вход:  data/raw/chat_history.json (36K сообщений)
  Выход: data/processed/chat/messages_filtered.json
  Логика: Удаление сервисных, эмодзи, флуда, дубликатов

Этап 1: Поиск веток(из перекрывающихся наборов сообщений)
  Вход:  messages_filtered.json
  Выход: data/processed/chat/threads.json
  Логика: LLM выделяет независимые ветки обсуждения

Этап 2: Дедупликация(опционально)
  Вход:  threads.json
  Выход: data/processed/chat/threads_deduped.json
  Логика: Объединение дубликатов на границах групп

Этап 3: RAG чанки
  Вход:  threads_deduped.json
  Выход: data/processed/chat/chunks_rag.jsonl
  Логика: Создание чанков {topic, knowledge, metadata}
```

### Формат чанка (JSONL)

```json
{
  "chunk_id": "tg_t001_0",
  "source": {
    "type": "telegram",
    "message_ids": [12345, 12346],
    "date_range": "2024-01-15 — 2024-01-17"
  },
  "content": {
    "summary": "Подключение контроллера к котлу Navien",
    "text": "При подключении котла Navien через OpenTherm..."
  },
  "metadata": {
    "tags": ["opentherm", "navien", "connection"],
    "version": "0.73",
    "confidence": 0.9,
  }
}
```

---

## Конфигурация (configs/default.yaml)

```yaml
# LLM настройки
llm:
  model: "..."  # ДЕФОЛТНАЯ LLM
  quantization: "..." # дефолтное квантование
  temperature: 0.3
  max_tokens: 2048
  context_size: 8192
  
  # Параметры для этапов
  stage1:
    temperature: 0.5
    max_tokens: 1000
  stage2:
    temperature: 0.1
    max_tokens: 50
  stage3:
    temperature: 0.5
    max_tokens: 2048

# Параметры обработки чата
chat_processing:
  group_size: 50
  overlap_size: 5
  min_message_length: 10
  stop_words: [...]
```
