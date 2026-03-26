#!/usr/bin/env python3
"""
Интерактивный чат с Ollama и RAG

Примеры:
    python scripts/chat.py
    python scripts/chat.py --model llama3.1 --prompt "Как подключить WiFi?"
    python scripts/chat.py --rag --top-k 3
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import OllamaClient
from src.rag import RAGPipeline
from src.utils.prompt_builder import PromptBuilder
from src.utils.text_utils import clean_response_text

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChatSession:
    """Сессия чата с моделью и опциональным RAG"""

    def __init__(
        self,
        client: OllamaClient,
        prompt_builder: PromptBuilder,
        rag_pipeline: Optional[RAGPipeline] = None,
        top_k: int = 5,
        debug: bool = False,
        system_prompt_override: Optional[str] = None,
    ):
        self.client = client
        self.prompt_builder = prompt_builder
        self.rag_pipeline = rag_pipeline
        self.top_k = top_k
        self.debug = debug
        self.system_prompt_override = system_prompt_override
        self.messages: list[dict[str, str]] = []
        self.history: list[dict] = []

    def _search_rag(self, user_message: str, use_rag: bool) -> str:
        """Получить контекст из RAG, если включен режим RAG"""
        if not use_rag or not self.rag_pipeline:
            return ""

        try:
            results = self.rag_pipeline.search(user_message, top_k=self.top_k)
            if results.total_found > 0:
                logger.info(f"📚 RAG нашёл {results.total_found} релевантных источников")
                return results.to_context_string()
        except Exception as e:
            logger.warning(f"RAG ошибка: {e}")

        return ""

    def send(
        self,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        use_rag: bool = False,
    ) -> str:
        """Отправить сообщение и получить ответ (нестримовый режим)"""
        rag_context = self._search_rag(user_message, use_rag)
        prompt = self.prompt_builder.build_chat_prompt(
            user_question=user_message,
            history=self.messages,
            rag_context=rag_context,
            use_rag=use_rag,
            system_prompt_override=self.system_prompt_override,
        )

        if self.debug:
            print("\n" + "=" * 70)
            print("🐛 DEBUG: Промпт, передаваемый в LLM:")
            print("=" * 70)
            print(prompt)
            print("=" * 70 + "\n")

        response = self.client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>"],
        )
        response = clean_response_text(response, strip_spaces=True)

        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": response})
        self.history.append({"user": user_message, "assistant": response, "rag_used": bool(rag_context)})

        return response

    def clear(self):
        """Очистить историю чата"""
        self.messages = []
        self.history = []

    def save(self, filepath: str):
        """Сохранить историю чата"""
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"💾 История сохранена: {filepath}")


def interactive_chat(
    client: OllamaClient,
    prompt_builder: PromptBuilder,
    max_tokens: int = 1024,
    rag_pipeline: Optional[RAGPipeline] = None,
    top_k: int = 5,
    use_rag: bool = False,
    debug: bool = False,
    system_prompt_override: Optional[str] = None,
    temperature: float = 0.7,
):
    """Интерактивный режим чата"""
    session = ChatSession(
        client=client,
        prompt_builder=prompt_builder,
        rag_pipeline=rag_pipeline,
        top_k=top_k,
        debug=debug,
        system_prompt_override=system_prompt_override,
    )

    rag_status = "✅ RAG" if use_rag else "❌ RAG"
    print("\n" + "=" * 70)
    print(f"💬 {client.model} — Интерактивный чат (Ollama) [{rag_status}]")
    print("=" * 70)
    system_preview = prompt_builder.get_system_prompt(
        use_rag=use_rag,
        system_prompt_override=system_prompt_override,
    )
    print(f"Системный промпт: {system_preview[:70]}...")
    if use_rag and rag_pipeline:
        stats = rag_pipeline.get_stats()
        print(
            f"📚 RAG: {stats['total_chunks']} чанков, "
            f"веса FAISS={stats['weights']['vector']:.2f}, BM25={stats['weights']['bm25']:.2f}"
        )
    print()
    print("Команды:")
    print("  /clear     — очистить историю")
    print("  /save      — сохранить историю")
    print("  /stats     — показать статистику")
    print("  /rag       — переключить RAG вкл/выкл")
    print("  /exit      — выйти")
    print("=" * 70)
    print()

    while True:
        try:
            user_input = input("👤 Вы: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print("\n👋 До свидания!")
                break

            if user_input.lower() == "/clear":
                session.clear()
                print("🗑️  История очищена")
                continue

            if user_input.lower() == "/save":
                filepath = input("Путь для сохранения [chat_history.json]: ").strip()
                if not filepath:
                    filepath = "chat_history.json"
                session.save(filepath)
                continue

            if user_input.lower() == "/stats":
                stats = client.get_stats()
                print("\n📊 Статистика клиента:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                if session.rag_pipeline:
                    rag_stats = session.rag_pipeline.get_stats()
                    print("\n📚 Статистика RAG:")
                    for key, value in rag_stats.items():
                        print(f"  {key}: {value}")
                print()
                continue

            if user_input.lower() == "/rag":
                use_rag = not use_rag
                print(f"🔄 RAG {'включен' if use_rag else 'выключен'}")
                continue

            rag_context = session._search_rag(user_input, use_rag)
            prompt = prompt_builder.build_chat_prompt(
                user_question=user_input,
                history=session.messages,
                rag_context=rag_context,
                use_rag=use_rag,
                system_prompt_override=system_prompt_override,
            )

            if session.debug:
                print("\n" + "=" * 70)
                print("🐛 DEBUG: Промпт, передаваемый в LLM:")
                print("=" * 70)
                print(prompt)
                print("=" * 70 + "\n")

            print("\n🤖 Модель: ", end="", flush=True)
            response = ""
            first_token = True
            try:
                for token in client.generate_stream(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["<|eot_id|>"],
                ):
                    if first_token:
                        token = clean_response_text(token, strip_spaces=True)
                        first_token = False
                    else:
                        token = clean_response_text(token, strip_spaces=False)

                    if token:
                        print(token, end="", flush=True)
                        response += token
            except Exception as e:
                print(f"\n❌ Ошибка генерации: {e}")
                continue

            session.messages.append({"role": "user", "content": user_input})
            session.messages.append({"role": "assistant", "content": response})
            session.history.append({"user": user_input, "assistant": response, "rag_used": bool(rag_context)})

            print()
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Прервано. Для выхода используйте /exit")
            continue
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            continue


def single_prompt_mode(
    client: OllamaClient,
    prompt: str,
    output_file: Optional[str] = None,
    max_tokens: int = 1024,
):
    """Одиночный промпт (не интерактивный)"""
    response = client.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.7)

    print("\n" + "=" * 70)
    print("📝 Запрос:")
    print(prompt)
    print()
    print("🤖 Ответ:")
    print(response)
    print("=" * 70)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Запрос:\n{prompt}\n\nОтвет:\n{response}\n")
        print(f"💾 Сохранено: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Интерактивный чат с LLM моделями через Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s
  %(prog)s --model llama3.1 --prompt "Как прошить?"
        """,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Название модели в Ollama (по умолчанию из конфига)",
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default=None,
        help="URL Ollama сервера",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=1024,
        help="Максимум токенов на ответ (по умолчанию: 1024)",
    )
    parser.add_argument("--prompt", "-p", type=str, help="Одиночный промпт (не интерактивный режим)")
    parser.add_argument("--output", "-o", type=str, help="Файл для сохранения ответа")
    parser.add_argument(
        "--system",
        "-s",
        type=str,
        default=None,
        help="Переопределение системного промпта",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Включить подробное логирование")
    parser.add_argument("--rag", action="store_true", help="Включить RAG поиск")
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Количество результатов RAG (по умолчанию из конфига)",
    )
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.5,
        help="Вес FAISS в гибридном поиске (0-1, по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="Вес BM25 в гибридном поиске (0-1, по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=None,
        help="Путь к JSONL файлу чанков для переиндексации перед запуском RAG",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Использовать тестовые чанки (data/processed/chat/test/chunks_rag_test.jsonl)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки: выводить промпт, передаваемый в LLM",
    )

    args = parser.parse_args()

    # Загрузить конфиг для получения модели и URL
    from src.utils.config import Config

    config = Config.load()

    # Использовать модель из конфига если не передана в CLI
    model_name = args.model or config.llm.get("model") or "llama3.1"
    base_url = args.url or config.llm.get("base_url", "http://localhost:11434")
    think = config.llm.get("think")
    temperature = config.llm.get("temperature", 0.7)

    print(f"🔄 Подключение к Ollama: {base_url}")
    print(f"📦 Модель: {model_name}")
    if think is not None:
        print(f"🧠 Think mode: {think}")

    try:
        client = OllamaClient(
            model=model_name,
            base_url=base_url,
            verbose=args.verbose,
            think=think,
        )
        client.load(strict=True)
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print()
        print("Убедитесь, что Ollama запущен:")
        print("   ollama serve")
        print()
        print("Или установите модель:")
        print(f"   ollama pull {model_name}")
        sys.exit(1)

    print("✅ Подключение успешно")
    print()

    # Берем top_k из конфига если не передан через CLI
    top_k = args.top_k if args.top_k is not None else config.rag.get("top_k", 5)
    
    rag_pipeline = None
    if args.rag:
        try:
            print("📚 Инициализация RAG...")
            rag_pipeline = RAGPipeline.from_config(
                config=config.model_dump(),
                data_dir=str(config.data_dir_path),
                ollama_base_url=base_url,
                vector_weight=args.vector_weight,
                bm25_weight=args.bm25_weight,
                top_k=top_k,
            )

            chunks_file = None
            if args.test:
                chunks_file = "data/processed/chat/test/chunks_rag_test.jsonl"
                print(f"🧪 Тестовый режим: используется {chunks_file}")
            elif args.chunks_file:
                chunks_file = args.chunks_file

            if chunks_file:
                print(f"🧩 Индексация чанков из файла: {chunks_file}")
                rag_pipeline.index_from_file(chunks_file, save=True)
                stats = rag_pipeline.get_stats()
                print(f"✅ RAG проиндексирован: {stats['total_chunks']} чанков")
            else:
                try:
                    loaded = rag_pipeline._load_indices()
                    if not loaded:
                        print("⚠️  RAG индексы не найдены. Сначала запустите индексацию или передайте --chunks-file.")
                        rag_pipeline = None
                    else:
                        stats = rag_pipeline.get_stats()
                        print(f"✅ RAG загружен: {stats['total_chunks']} чанков")
                except FileNotFoundError:
                    print("⚠️  RAG индексы не найдены. Сначала запустите индексацию или передайте --chunks-file.")
                    rag_pipeline = None
        except Exception as e:
            print(f"⚠️  RAG недоступен: {e}")
            rag_pipeline = None

    prompt_builder = PromptBuilder()

    if args.prompt:
        single_prompt_mode(client, args.prompt, args.output, args.max_tokens)
    else:
        interactive_chat(
            client=client,
            prompt_builder=prompt_builder,
            max_tokens=args.max_tokens,
            rag_pipeline=rag_pipeline,
            top_k=top_k,
            use_rag=args.rag,
            debug=args.debug,
            system_prompt_override=args.system,
            temperature=temperature,
        )


if __name__ == "__main__":
    main()
