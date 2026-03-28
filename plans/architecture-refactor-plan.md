# План доведения архитектуры до чистого разделения слоёв

## Цель

Довести текущую архитектуру до чистого и проверяемого разделения ролей без полной переделки проекта.

Финальное состояние:

- `chat domain` знает только свои модели и порты
- `application` управляет use case и runtime-состоянием, но не создаёт инфраструктуру
- `memory` и `rag` являются внешними адаптерами к портам `chat`
- `composition root` является единственным местом wiring
- transport-слой остаётся тонким и не знает деталей storage / retrieval реализации
- package-root API узкий и поддерживаемый
- compatibility shim и случайные re-export отсутствуют
- документация описывает достигнутое состояние, а не промежуточную схему

## Главный принцип

В проекте должен существовать **один направленный граф зависимостей**:

`transport / scripts -> composition -> application -> domain`

Инфраструктурные адаптеры подключаются только сбоку через `composition`:

- `composition -> memory`
- `composition -> rag`
- `composition -> llm`
- `composition -> config`
- `composition -> prompting`

Обратные зависимости запрещены.

## Жёсткие архитектурные границы

Ниже зафиксированы **не пожелания, а правила**. Если модуль нарушает правило, архитектура считается незавершённой.

### 1. `src/chat/domain/`

Роль:

- предметные модели чата
- доменные контракты и порты
- только те абстракции, которые нужны ядру для ведения диалога

Допустимо хранить:

- модели сообщений, фактов, результатов хода
- `Protocol` / `ABC`-порты для LLM, retrieval и состояния диалога
- чистые value objects и доменные типы

Разрешённые зависимости:

- stdlib
- `typing`, `dataclasses`, `abc`
- другие модули внутри `src/chat/domain/`

Запрещено:

- импорты из `src/chat/application/*`
- импорты из `src/memory/*`
- импорты из `src/rag/*`
- импорты из `src/bot/*`
- импорты из `scripts/*`
- concrete implementation любого порта
- runtime state транспорта или сессии
- DTO инфраструктуры, retrieval-модели индекса, storage-модели БД

Следствие:

- `domain` не знает, как именно устроены SQLite, BM25, vector store, Telegram, CLI, Ollama

### 2. `src/chat/application/`

Роль:

- orchestration одного хода
- session facade
- slash-команды
- runtime state одной сессии
- application DTO, которые принадлежат chat use case

Допустимо хранить:

- `ChatService`
- `SessionFacade`
- `CommandService`
- `ChatRuntime`
- request / response DTO
- mapping инфраструктурных данных в application-friendly DTO, если это нужно use case

Разрешённые зависимости:

- stdlib
- `src/chat/domain/*`
- локальные модули внутри `src/chat/application/*`

Запрещено:

- создание `PromptManager`, `DialogState`, `Retriever`, `LLM client` по умолчанию
- fallback-composition
- прямые импорты из `src/memory/*`
- прямые импорты из `src/rag/*`
- прямые импорты из `src/bot/*`
- работа с storage-моделями или индексными моделями

Следствие:

- `application` оперирует только зависимостями, пришедшими извне через конструкторы / фабрики / composition

### 3. `src/chat/composition.py`

Роль:

- единственная точка runtime wiring для chat
- создание shared dependencies
- связывание `application` с `memory`, `rag`, `llm`, `config`, `prompting`
- сборка `SessionFacade` и `DialogRegistry`

Разрешённые зависимости:

- `src/chat/application/*`
- `src/chat/domain/*`
- `src/memory/*`
- `src/rag/*`
- `src/llm/*`
- `src/config/*`
- `src/utils/prompt_manager.py`
- `src/chat/prompting.py`

Запрещено:

- переносить сюда бизнес-логику хода диалога
- дублировать application-слой через альтернативные builders и полу-оркестраторы
- плодить несколько равноправных путей сборки одной и той же сессии

Следствие:

- сборка сессии существует ровно в одном поддерживаемом сценарии

### 4. `src/chat/prompting.py`

Роль:

- адаптер подготовки промпта для chat use case
- преобразование domain/application данных в вход для LLM

Разрешённые зависимости:

- `src/chat/domain/*`
- при необходимости `src/chat/application/*` только для application DTO
- `src/utils/prompt_manager.py`

Запрещено:

- самостоятельное создание `PromptManager`
- доступ к `memory` / `rag` / `telegram` / `sqlite`

Следствие:

- prompting получает все зависимости явно и не занимается composition

### 5. `src/memory/`

Роль:

- persistent adapter к порту состояния диалога
- storage-модели и репозиторий хранения

Разрешённые зависимости:

- `src/chat/domain/*` только для реализации доменного порта
- внутренние модули `src/memory/*`

Запрещено:

- быть источником domain-моделей
- экспортировать compatibility shim
- влиять на application orchestration

Следствие:

- `memory` реализует порт, но не определяет ядро

### 6. `src/rag/`

Роль:

- retrieval implementation
- индексный lifecycle
- retrieval-side internal models

Разрешённые зависимости:

- `src/chat/domain/*` только для реализации порта retriever, если это действительно требуется сигнатурой порта
- внутренние модули `src/rag/*`

Запрещено:

- протаскивать свои internal DTO в `domain` или `application`
- делать `chat` зависимым от `src/rag/models.py`
- навязывать package layout ядру чата

Следствие:

- любые `rag`-специфичные модели остаются внутри `src/rag/`

### 7. `scripts/` и `src/bot/`

Роль:

- entrypoints
- transport adapters

Разрешённые зависимости:

- `src/chat/composition.py`
- `src/chat/application/session_facade.py`
- `src/chat/registry.py`
- transport-specific utility code

Запрещено:

- прямые импорты `src/memory/*`
- прямые импорты `src/rag/*`
- ручная сборка зависимостей вне composition root

Следствие:

- transport знает только публичный session API и builders

## Гарантированное отсутствие утечек

Чтобы не осталось двусмысленности, ниже зафиксированы конкретные правила отсутствия утечек.

### Правило A. Отсутствие инфраструктурных импортов в ядре

Считается утечкой, если любой файл в `src/chat/domain/` или `src/chat/application/` импортирует:

- `src/memory/*`
- `src/rag/*`
- `src/bot/*`
- `scripts/*`

Исключений нет.

### Правило B. Отсутствие инфраструктурных DTO в chat core

Считается утечкой, если сигнатуры, поля dataclass, возвращаемые значения или публичные методы в `src/chat/domain/` или `src/chat/application/` используют типы из:

- `src.rag.models`
- `src.memory.models`
- любых storage/index-specific модулей

Исключений нет.

### Правило C. Единственная точка преобразования retrieval-результата

Если retrieval implementation возвращает свои внутренние структуры, они должны быть преобразованы **до входа в `application`**.

Допустимы только два варианта:

1. retriever сразу возвращает chat-friendly тип, определённый в `src/chat/domain/` или `src/chat/application/`
2. преобразование происходит внутри адаптера `src/rag/`, а наружу отдаётся только тип chat-слоя

Недопустимо:

- импортировать `src/rag/models.py` в `chat`
- делать `application` местом знания формата индекса / retrieval store

### Правило D. Единственная точка создания зависимостей

Считается нарушением, если `ChatService`, `SessionFacade`, `CommandService`, `ChatPrompting` или transport code создают зависимости по умолчанию.

Разрешено создавать concrete dependency только в:

- `src/chat/composition.py`
- тестовых фабриках / test helpers

### Правило E. Временные shim не допускаются как постоянное состояние

Если shim нужен для миграции, он должен:

- существовать не более одного этапа внедрения
- быть помечен как временный в плане и в коде
- быть удалён в следующем же этапе

По умолчанию стратегия такая:

- если можно удалить shim сразу без каскадных поломок, удалить сразу
- если нужен переходный слой, заранее указать точку удаления

### Правило F. Package root не является слоем

`__init__.py` не должен:

- скрывать реальную архитектуру
- служить совместимым входом для legacy-импортов без явной причины
- реэкспортировать всё подряд

Package-root экспорт допустим только для устойчивых публичных точек входа.

## Главные проблемы, которые нужно устранить

1. `domain` зависит от package layout `rag`
2. `domain` содержит concrete implementation состояния диалога
3. session runtime лежит в `domain`, хотя это application concern
4. `application` умеет собирать часть зависимостей самостоятельно
5. в проекте нет одного жёстко закреплённого места преобразования retrieval-результата
6. composition root размазан через несколько builders с повторяющимися сигнатурами
7. в проекте остались shim и широкие package-root exports
8. документация частично отстала от целевой схемы

## Целевая раскладка по файлам

### Оставить как базовые узлы

- `src/chat/application/chat_service.py`
- `src/chat/application/session_facade.py`
- `src/chat/application/command_service.py`
- `src/chat/composition.py`
- `src/chat/prompting.py`
- `src/chat/registry.py`
- `src/chat/domain/models.py`
- `src/memory/sqlite_repository.py`
- `src/memory/sqlite_dialog_state.py`
- `src/rag/retrieval_service.py`
- `src/rag/index_manager.py`

### Добавить

- `src/chat/domain/ports.py` — единственное место объявления доменных портов
- `src/chat/application/runtime.py` — mutable runtime state одной сессии
- `tests/helpers/in_memory_dialog_state.py` — test helper вместо production `InMemoryDialogState`

### Опционально добавить

- `src/chat/application/retrieval_models.py` **только если** chat use case действительно требует собственного retrieval DTO
- `src/chat/application/builders.py` **только если** после упрощения `src/chat/composition.py` останется доказуемое дублирование

### Удалить

- `src/memory/models.py`
- `src/chat/domain/runtime.py`
- production `InMemoryDialogState`, если он сейчас живёт в `domain`

### Сузить

- `src/chat/__init__.py`
- `src/rag/__init__.py`
- `src/utils/__init__.py`

## Архитектурные решения

### Решение 1. Очистить `domain` до моделей и портов

Изменения:

- вынести порты из `src/chat/domain/contracts.py` в `src/chat/domain/ports.py`
- зафиксировать `src/chat/domain/ports.py` как единственный источник портов
- убрать любые импорты `src.rag.models` из `chat domain`
- убрать concrete implementation состояния из `domain`

Явная граница:

- `domain` объявляет только контракты
- `domain` не содержит ни одного concrete adapter

Результат:

- `domain` действительно независим
- `rag` и `memory` становятся внешними реализациями, а не частью ядра

### Решение 2. Перенести session runtime в `application`

Изменения:

- перенести `ChatRuntime` из `src/chat/domain/runtime.py` в `src/chat/application/runtime.py`
- обновить импорты в `SessionFacade`, `CommandService`, `composition`, тестах
- удалить `src/chat/domain/runtime.py` без постоянного shim

Явная граница:

- состояние session lifecycle принадлежит только `application`

Результат:

- `domain` больше не содержит transport/session concern

### Решение 3. Зафиксировать точку преобразования retrieval-результата

Изменения:

- выбрать один chat-friendly тип retrieval-результата
- запретить импорт `src/rag/models.py` в `chat`
- если нужен mapping, выполнять его внутри `src/rag/` адаптера до передачи результата в chat core

Явная граница:

- `rag` знает свой внутренний формат
- `chat` знает только свой публичный retrieval-контракт

Результат:

- отсутствуют типовые утечки `rag` в ядро

### Решение 4. Запретить self-composition вне composition root

Изменения:

- `ChatPrompting` больше не создаёт `PromptManager` сам
- `ChatService` больше не создаёт `DialogState`, `ChatPrompting` и другие зависимости по умолчанию
- все зависимости приходят только из `src/chat/composition.py` и тестовых фабрик

Явная граница:

- business orchestration не создаёт инфраструктуру

Результат:

- исчезает скрытая магия и альтернативные пути сборки

### Решение 5. Свести wiring к одному builder flow

Изменения:

- выделить одну поддерживаемую схему сборки shared runtime dependencies
- сократить число повторяющихся сигнатур в `build_chat_shared_dependencies`, `build_dialog_registry`, `build_chat_session`
- при необходимости добавить компактный internal settings object
- не вводить `builders.py`, если цель достигается упрощением `src/chat/composition.py`

Явная граница:

- существует один основной путь сборки runtime

Результат:

- поддержка CLI и Telegram не требует параллельных схем wiring

### Решение 6. Удалить compatibility и случайные re-export

Изменения:

- удалить `src/memory/models.py`
- убрать re-export `Config` из `src/utils/__init__.py`
- сузить `src/rag/__init__.py` до реального runtime API
- перевести entrypoints и тесты на прямые, предсказуемые импорты

Явная граница:

- package root не скрывает архитектуру и не поддерживает legacy surface без необходимости

Результат:

- архитектурная поверхность становится узкой и проверяемой

## Порядок внедрения

### Этап 1. Зафиксировать целевую карту слоёв

Сделать:

- утвердить этот план как единственный источник целевой схемы
- явно отметить, какие модули относятся к `domain`, `application`, `composition`, `infrastructure`, `transport`
- отдельно зафиксировать правило: `chat` не импортирует `rag` и `memory`

Готово, когда:

- нет двусмысленности, какой модуль за что отвечает
- зафиксирована единственная точка преобразования retrieval-результата

Проверка этапа:

- ручная проверка плана на отсутствие противоречий

### Этап 2. Очистить `chat domain`

Сделать:

- создать `src/chat/domain/ports.py`
- перенести порты из `src/chat/domain/contracts.py`
- либо удалить `src/chat/domain/contracts.py`, либо оставить временный shim ровно на один этап с последующим удалением на этом же цикле работ
- убрать прямые зависимости `chat` от `src.rag.models`
- удалить concrete implementation из domain-модуля состояния

Затронутые файлы:

- `src/chat/domain/contracts.py`
- `src/chat/domain/ports.py`
- `src/chat/domain/dialog_state.py`
- `src/chat/application/dto.py`
- `src/chat/application/chat_service.py`
- `src/memory/sqlite_dialog_state.py`
- `src/rag/retrieval_service.py`
- тесты

Готово, когда:

- ни один файл в `src/chat/domain/` не импортирует `src/rag/*` или `src/memory/*`
- concrete `DialogState` не находится в `domain`

Проверка этапа:

- `source .venv/bin/activate && pyright`
- тесты для domain/application
- grep/поиск по импортам `src.rag` и `src.memory` внутри `src/chat/domain/`

### Этап 3. Перенести runtime в `application`

Сделать:

- добавить `src/chat/application/runtime.py`
- перенести туда `ChatRuntime`
- обновить `SessionFacade`, `CommandService`, `composition`, тесты
- удалить `src/chat/domain/runtime.py` без сохранения постоянного shim

Затронутые файлы:

- `src/chat/domain/runtime.py`
- `src/chat/application/runtime.py`
- `src/chat/application/session_facade.py`
- `src/chat/application/command_service.py`
- `src/chat/composition.py`
- тесты

Готово, когда:

- `ChatRuntime` существует только в `application`
- `domain` не содержит runtime/session state

Проверка этапа:

- `source .venv/bin/activate && pyright`
- релевантные тесты application

### Этап 4. Убрать утечки retrieval DTO

Сделать:

- определить конечный chat-friendly retrieval type
- удалить импорт `src.rag.models` из `chat`
- выполнить mapping внутри `src/rag/` адаптера или сразу возвращать chat-friendly тип из retriever-порта
- если нужен `src/chat/application/retrieval_models.py`, создать его как chat-owned DTO, а не как зеркало `rag`

Затронутые файлы:

- `src/chat/domain/ports.py`
- `src/chat/application/dto.py`
- опционально `src/chat/application/retrieval_models.py`
- `src/chat/application/chat_service.py`
- `src/rag/models.py`
- `src/rag/retrieval_service.py`
- тесты

Готово, когда:

- `src/chat/domain/*` и `src/chat/application/*` не импортируют `src/rag/models.py`
- сигнатуры chat core используют только chat-owned типы

Проверка этапа:

- `source .venv/bin/activate && pyright`
- поиск по `src.rag.models` в `src/chat/`
- тесты retrieval integration и chat service

### Этап 5. Убрать self-composition

Сделать:

- запретить `ChatPrompting()` без переданного `PromptManager`
- запретить `ChatService()` без явно переданных `state` и `prompting`
- в тестах заменить implicit construction на явные фабрики / helpers

Затронутые файлы:

- `src/chat/prompting.py`
- `src/chat/application/chat_service.py`
- `src/chat/composition.py`
- `tests/test_chat_service.py`
- `tests/test_cli_smoke.py`
- `tests/test_prompt_manager.py`

Готово, когда:

- вне `composition` и тестовых фабрик не создаются concrete dependency

Проверка этапа:

- `source .venv/bin/activate && pyright`
- unit tests
- smoke для `scripts/cli_chat.py`

### Этап 6. Свести wiring к одному builder flow

Сделать:

- упростить сигнатуры `build_dialog_registry` и `build_chat_session`
- оставить один поддерживаемый способ сборки shared dependencies
- при необходимости добавить внутренний settings object или `builders.py`
- проверить, что CLI и Telegram используют один и тот же composition-поток

Затронутые файлы:

- `src/chat/composition.py`
- опционально `src/chat/application/builders.py`
- `scripts/cli_chat.py`
- `src/bot/telegram_transport.py`
- тесты

Готово, когда:

- нет двух конкурирующих путей сборки runtime
- transport code не собирает зависимости сам

Проверка этапа:

- `source .venv/bin/activate && pyright`
- smoke CLI
- transport-level regression checks

### Этап 7. Удалить shim и сузить package-root API

Сделать:

- удалить `src/memory/models.py`
- сузить `src/chat/__init__.py`
- сузить `src/rag/__init__.py`
- убрать лишнее из `src/utils/__init__.py`
- обновить entrypoints и тесты на прямые импорты

Затронутые файлы:

- `src/memory/models.py`
- `src/chat/__init__.py`
- `src/rag/__init__.py`
- `src/utils/__init__.py`
- `scripts/cli_chat.py`
- `scripts/reindex_rag.py`
- тесты

Готово, когда:

- package root экспортирует только поддерживаемые точки входа
- legacy re-export больше не нужен

Проверка этапа:

- `source .venv/bin/activate && pyright`
- поиск package-root импортов в `scripts/` и `tests/`

### Этап 8. Синхронизировать документацию по факту

Сделать:

- обновить `README.md`
- синхронизировать `AGENTS.md` и `.kilocode/`, если изменятся структура или команды
- удалить из документации устаревшие описания промежуточной архитектуры

Готово, когда:

- документация описывает уже достигнутое состояние

Проверка этапа:

- сверка документации с реальным кодом и импортными границами

## Финальные инварианты

После завершения работ должны выполняться все условия одновременно:

1. ни один файл в `src/chat/domain/` не импортирует `src/chat/application/*`, `src/memory/*`, `src/rag/*`, `src/bot/*`, `scripts/*`
2. ни один файл в `src/chat/application/` не импортирует `src/memory/*`, `src/rag/*`, `src/bot/*`, `scripts/*`
3. `ChatRuntime` находится только в `src/chat/application/runtime.py`
4. `ChatService` и `ChatPrompting` не создают зависимости сами
5. `src/chat/` не импортирует `src.rag.models`
6. concrete retrieval/storage DTO не выходят в публичные типы `chat`
7. `src/memory/models.py` удалён
8. package-root экспорты сведены к поддерживаемым точкам входа
9. `scripts/` и transports используют только composition / session API
10. документация описывает уже достигнутое состояние

## Definition of Done

- целевой граф зависимостей выполняется без исключений
- нет инфраструктурных импортов в `domain` и `application`
- нет типовых утечек `rag` и `memory` в `chat`
- нет self-composition вне `src/chat/composition.py` и тестовых helpers
- нет постоянных compatibility shim
- CLI и Telegram проходят через один composition flow
- `source .venv/bin/activate && pyright` даёт `0 errors`
- релевантные тесты и smoke-проверки проходят
- документация синхронизирована с кодом
