#!/usr/bin/env python3
"""CLI transport для универсального chat application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.chat.application.command_service import CommandParser
from src.chat.application.session_facade import SessionFacade
from src.chat.composition import build_chat_session
from src.config import Config


def build_cli_parser() -> argparse.ArgumentParser:
    """Создать parser для CLI transport."""
    parser = argparse.ArgumentParser(
        description="Интерактивный чат с LLM моделями через Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s
  %(prog)s --model llama3.1 --prompt "Как прошить?"
        """,
    )
    parser.add_argument("--model", "-m", type=str, default=None, help="Название модели в Ollama")
    parser.add_argument("--url", "-u", type=str, default=None, help="URL Ollama сервера")
    parser.add_argument("--max-tokens", "-t", type=int, default=1024, help="Максимум токенов на ответ")
    parser.add_argument("--prompt", "-p", type=str, help="Один ход без интерактивного режима")
    parser.add_argument("--output", "-o", type=str, help="Файл для сохранения ответа")
    parser.add_argument("--system", "-s", type=str, default=None, help="Переопределение системного промпта")
    parser.add_argument("--verbose", "-v", action="store_true", help="Включить подробное логирование")
    parser.add_argument("--rag", action="store_true", help="Включить RAG поиск")
    parser.add_argument("--top-k", "-k", type=int, default=None, help="Количество результатов RAG")
    parser.add_argument("--vector-weight", type=float, default=0.5, help="Вес FAISS в гибридном поиске")
    parser.add_argument("--bm25-weight", type=float, default=0.5, help="Вес BM25 в гибридном поиске")
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
    parser.add_argument("--debug", action="store_true", help="Выводить payload, передаваемый в LLM")
    return parser


def _print_debug_payload(payload: object) -> None:
    print("\n" + "=" * 70)
    print("🐛 DEBUG: Messages payload, передаваемый в LLM:")
    print("=" * 70)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("=" * 70)


def _banner_lines(session: SessionFacade) -> list[str]:
    rag_status = "✅ RAG" if session.rag_enabled else "❌ RAG"
    system_prompt = session.system_prompt()
    lines = [
        "\n" + "=" * 70,
        f"💬 {session.model_name} — Интерактивный чат (Ollama) [{rag_status}]",
        "=" * 70,
        f"Системный промпт: {system_prompt[:70]}...",
    ]

    rag_stats = session.get_stats().get("rag")
    if session.rag_enabled and isinstance(rag_stats, dict):
        weights = rag_stats.get("weights", {})
        lines.append(
            f"📚 RAG: {rag_stats.get('total_chunks', 0)} чанков, "
            f"веса FAISS={weights.get('vector', 0.0):.2f}, BM25={weights.get('bm25', 0.0):.2f}"
        )

    lines.append("")
    lines.extend(session.command_lines())
    lines.append("  /exit             — выйти")
    lines.append("=" * 70)
    lines.append("")
    return lines


def _run_prompt_mode(session: SessionFacade, prompt: str, output_file: str | None = None) -> None:
    prepared = None
    request = session.build_request(prompt)
    if session.runtime.debug:
        request, prepared = session.prepare_request(prompt)
    if prepared is not None:
        _print_debug_payload(prepared.llm_messages)
    response = session.run_request(request, prepared=prepared)

    print("\n" + "=" * 70)
    print("📝 Запрос:")
    print(prompt)
    print()
    print("🤖 Ответ:")
    print(response.assistant_message)
    print("=" * 70)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(f"Запрос:\n{prompt}\n\nОтвет:\n{response.assistant_message}\n")
        print(f"💾 Сохранено: {output_file}")


def run_interactive_chat(session: SessionFacade) -> None:
    for line in _banner_lines(session):
        print(line)

    while True:
        try:
            user_input = input("👤 Вы: ").strip()
            if not user_input:
                continue

            if user_input == "/exit":
                print("👋 До свидания!")
                break

            if CommandParser.is_command(user_input):
                result = session.try_execute_command(user_input)
                if result is None:
                    continue
                for line in result.lines:
                    print(line)
                continue

            prepared = None
            request = session.build_request(user_input)
            if session.runtime.debug:
                request, prepared = session.prepare_request(user_input)
            if prepared is not None:
                _print_debug_payload(prepared.llm_messages)

            print("\n🤖 Модель: ", end="", flush=True)
            for event in session.stream_request(request, prepared=prepared):
                if event.kind == "token" and event.text:
                    print(event.text, end="", flush=True)

            print()
            print()
        except KeyboardInterrupt:
            print("\n\n👋 Прервано. Для выхода используйте /exit")
        except Exception as error:
            print(f"\n❌ Ошибка: {error}")


def run_cli(argv: list[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)
    config = Config.load()

    model_name = args.model or config.llm.model
    base_url = args.url or config.llm.base_url
    temperature = config.llm.temperature
    think = config.llm.think
    top_k = args.top_k if args.top_k is not None else config.rag.top_k

    print(f"🔄 Подключение к Ollama: {base_url}")
    print(f"📦 Модель: {model_name}")
    if think is not None:
        print(f"🧠 Think mode: {think}")

    try:
        session = build_chat_session(
            config=config,
            model_name=model_name,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=temperature,
            use_rag=args.rag,
            system_prompt_override=args.system,
            debug=args.debug,
            verbose=args.verbose,
            think=think,
            top_k=top_k,
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            chunks_file=args.chunks_file,
            test_mode=args.test,
        )
    except Exception as error:
        print(f"❌ Ошибка подключения: {error}")
        print()
        print("Убедитесь, что Ollama запущен:")
        print("   ollama serve")
        print()
        print("Или установите модель:")
        print(f"   ollama pull {model_name}")
        sys.exit(1)

    print("✅ Подключение успешно")
    if args.rag and not session.rag_enabled:
        print("⚠️  RAG недоступен. Чат запущен без RAG.")
        if session.rag_error:
            print(f"   Причина: {session.rag_error}")
    print()

    if args.prompt:
        _run_prompt_mode(session, args.prompt, args.output)
        return

    run_interactive_chat(session)


def main(argv: list[str] | None = None) -> None:
    run_cli(argv)


if __name__ == "__main__":
    main()
