#!/usr/bin/env python3
"""Run the minimal web UI for the SmartTherm chat application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.chat.composition import build_dialog_registry
from src.config import Config
from src.web import WebTransport, run_web_server


def build_web_parser() -> argparse.ArgumentParser:
    """Create parser for the minimal web chat server."""
    parser = argparse.ArgumentParser(description="Minimal web UI for SmartTherm chat application")
    parser.add_argument("--model", "-m", type=str, default=None, help="Название модели в Ollama")
    parser.add_argument("--url", "-u", type=str, default=None, help="URL Ollama сервера")
    parser.add_argument("--max-tokens", "-t", type=int, default=None, help="Максимум токенов на ответ")
    parser.add_argument("--system", "-s", type=str, default=None, help="Переопределение системного промпта")
    parser.add_argument("--verbose", "-v", action="store_true", help="Включить подробное логирование LLM")
    parser.add_argument("--debug", action="store_true", help="Передавать debug-флаг в chat runtime")
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
    parser.add_argument("--no-rag", action="store_true", help="Запустить веб-чат без RAG")
    parser.add_argument("--host", type=str, default=None, help="HTTP host override")
    parser.add_argument("--port", type=int, default=None, help="HTTP port override")
    return parser


def run_web_chat(argv: list[str] | None = None) -> None:
    args = build_web_parser().parse_args(argv)
    config = Config.load()

    registry = build_dialog_registry(
        config=config,
        model_name=args.model or config.llm.model,
        base_url=args.url or config.llm.base_url,
        max_tokens=args.max_tokens if args.max_tokens is not None else config.llm.max_tokens,
        temperature=config.llm.temperature,
        use_rag=False if args.no_rag else config.bot.use_rag,
        system_prompt_override=args.system,
        debug=args.debug,
        verbose=args.verbose,
        think=config.llm.think,
        top_k=args.top_k if args.top_k is not None else config.rag.top_k,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        chunks_file=args.chunks_file,
        test_mode=args.test,
    )
    transport = WebTransport(registry)
    host = args.host or config.server.host
    port = args.port or config.server.port

    print(f"🌐 Web UI: http://{host}:{port}")
    print(f"📦 Модель: {args.model or config.llm.model}")
    print(f"🔄 Ollama: {args.url or config.llm.base_url}")
    print(f"📚 RAG: {'on' if (False if args.no_rag else config.bot.use_rag) else 'off'}")

    run_web_server(transport=transport, host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    run_web_chat(argv)


if __name__ == "__main__":
    main()
