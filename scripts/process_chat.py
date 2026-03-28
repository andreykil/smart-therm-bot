"""
CLI для обработки чата

Команды:
    filter   - Фильтрация сообщений
    chunks   - Создание чанков из групп
"""

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import Config
from src.data_processing import run_filtering, run_chunks
from src.llm import OllamaClient


def setup_logging(verbose: bool = False):
    """Настроить логирование"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def cmd_filter(args):
    """Фильтрация сообщений"""
    config = Config.load()

    input_path = Path(args.input_path) if args.input_path else None
    output_path = Path(args.output_path) if args.output_path else None

    run_filtering(config, input_path=input_path, output_path=output_path)


def cmd_chunks(args):
    """Создание RAG чанков"""
    config = Config.load()

    model_name = config.llm.model
    llm = OllamaClient(
        model=model_name,
        base_url=config.llm.base_url,
        verbose=args.verbose,
        think=config.llm.think,
    )

    if not llm.model_exists():
        logging.error(f"Модель не найдена в Ollama: {model_name}")
        logging.info(f"Скачайте модель: ollama pull {model_name}")
        sys.exit(1)

    llm.load()

    input_path = Path(args.input_path) if args.input_path else None
    output_path = Path(args.output_path) if args.output_path else None
    save_groups = args.save_groups
    groups_dir = Path(args.groups_dir) if args.groups_dir else None

    run_chunks(
        config,
        llm,
        input_path=input_path,
        output_path=output_path,
        save_groups=save_groups,
        groups_dir=groups_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Обработка чата SmartTherm")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")

    subparsers = parser.add_subparsers(dest="command", help="Команда")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-path", type=str, default=None, help="Входной файл")
    common.add_argument("--output-path", type=str, default=None, help="Выходной файл")

    p_filter = subparsers.add_parser("filter", help="Фильтрация сообщений", parents=[common])
    p_filter.set_defaults(func=cmd_filter)

    p_chunks = subparsers.add_parser("chunks", help="Создание чанков", parents=[common])
    p_chunks.add_argument("--save-groups", action="store_true", help="Сохранять группы в файлы")
    p_chunks.add_argument("--groups-dir", type=str, default=None, help="Директория для групп")
    p_chunks.set_defaults(func=cmd_chunks)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
