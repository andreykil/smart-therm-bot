"""
CLI утилиты для обработки чата

Отдельные команды для каждого этапа:
- stage0: Фильтрация шума
- stage1: Выделение веток
- stage2: Дедупликация веток
- stage3: Создание RAG чанков
- all: Запуск всех этапов
"""

import argparse
import logging
import sys
from pathlib import Path

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat_processor import (
    run_stage0,
    run_stage1,
    run_stage2,
    run_stage3
)
from src.llm.factory import create_llm_engine
from src.utils.config import Config


def setup_logging(verbose: bool = False):
    """Настроить логирование"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_config(args) -> Config:
    """Создать конфигурацию из аргументов"""
    config = Config.load()

    # Применение аргументов командной строки
    if hasattr(args, 'model') and args.model:
        config.llm["model"] = args.model
    if hasattr(args, 'quantization') and args.quantization:
        config.llm["quantization"] = args.quantization
    if hasattr(args, 'context_size') and args.context_size:
        config.llm["context_size"] = args.context_size
    if hasattr(args, 'max_tokens') and args.max_tokens:
        config.llm["max_tokens"] = args.max_tokens
    if hasattr(args, 'group_size') and args.group_size:
        config.chat_processing["group_size"] = args.group_size
    if hasattr(args, 'overlap_size') and args.overlap_size is not None:
        config.chat_processing["overlap_size"] = args.overlap_size

    return config


def cmd_stage0(args):
    """Этап 0: Фильтрация шума"""
    config = get_config(args)
    
    input_path = Path(args.input_path) if hasattr(args, 'input_path') and args.input_path else None
    output_path = Path(args.output_path) if hasattr(args, 'output_path') and args.output_path else None
    
    run_stage0(config, input_path=input_path, output_path=output_path)


def cmd_stage1(args):
    """Этап 1: Выделение веток"""
    config = get_config(args)

    # Инициализация LLM
    llm = create_llm_engine(
        model_id=config.llm.get("model"),
        quantization=config.llm.get("quantization"),
        n_ctx=config.llm.get("context_size") or 8192,
        verbose=args.verbose if hasattr(args, 'verbose') else False
    )

    if not llm.model_exists():
        logging.error(f"Модель не найдена: {llm.get_model_path()}")
        sys.exit(1)

    llm.load()
    
    input_path = Path(args.input_path) if hasattr(args, 'input_path') and args.input_path else None
    output_path = Path(args.output_path) if hasattr(args, 'output_path') and args.output_path else None
    debug = args.debug if hasattr(args, 'debug') else False
    
    run_stage1(config, llm, input_path=input_path, output_path=output_path, debug=debug)


def cmd_stage2(args):
    """Этап 2: Дедупликация веток"""
    config = get_config(args)

    # Инициализация LLM
    llm = create_llm_engine(
        model_id=config.llm.get("model"),
        quantization=config.llm.get("quantization"),
        n_ctx=config.llm.get("context_size") or 8192,
        verbose=args.verbose if hasattr(args, 'verbose') else False
    )

    if not llm.model_exists():
        logging.error(f"Модель не найдена: {llm.get_model_path()}")
        sys.exit(1)

    llm.load()
    
    input_path = Path(args.input_path) if hasattr(args, 'input_path') and args.input_path else None
    output_path = Path(args.output_path) if hasattr(args, 'output_path') and args.output_path else None
    
    run_stage2(config, llm, input_path=input_path, output_path=output_path)


def cmd_stage3(args):
    """Этап 3: Создание RAG чанков"""
    config = get_config(args)

    # Инициализация LLM
    llm = create_llm_engine(
        model_id=config.llm.get("model"),
        quantization=config.llm.get("quantization"),
        n_ctx=config.llm.get("context_size") or 8192,
        verbose=args.verbose if hasattr(args, 'verbose') else False
    )

    if not llm.model_exists():
        logging.error(f"Модель не найдена: {llm.get_model_path()}")
        sys.exit(1)

    llm.load()
    
    threads_path = Path(args.threads_path) if hasattr(args, 'threads_path') and args.threads_path else None
    messages_path = Path(args.messages_path) if hasattr(args, 'messages_path') and args.messages_path else None
    output_path = Path(args.output_path) if hasattr(args, 'output_path') and args.output_path else None
    
    run_stage3(
        config, 
        llm, 
        threads_path=threads_path, 
        messages_path=messages_path, 
        output_path=output_path,
        sample_size=args.sample_size if hasattr(args, 'sample_size') else 5
    )


def cmd_all(args):
    """Запуск всех этапов"""
    config = get_config(args)

    # Инициализация LLM (один раз для всех этапов)
    llm = create_llm_engine(
        model_id=config.llm.get("model"),
        quantization=config.llm.get("quantization"),
        n_ctx=config.llm.get("context_size") or 8192,
        verbose=args.verbose if hasattr(args, 'verbose') else False
    )

    if not llm.model_exists():
        logging.error(f"Модель не найдена: {llm.get_model_path()}")
        sys.exit(1)

    llm.load()

    # Этап 0
    print("\n" + "=" * 70)
    print("ЭТАП 0: Фильтрация шума")
    print("=" * 70)
    stats0 = run_stage0(config)

    # Этап 1
    print("\n" + "=" * 70)
    print("ЭТАП 1: Выделение веток")
    print("=" * 70)
    stats1 = run_stage1(config, llm)

    # Этап 2
    print("\n" + "=" * 70)
    print("ЭТАП 2: Дедупликация веток")
    print("=" * 70)
    stats2 = run_stage2(config, llm)

    # Этап 3
    print("\n" + "=" * 70)
    print("ЭТАП 3: Создание RAG чанков")
    print("=" * 70)
    stats3 = run_stage3(config, llm, sample_size=args.sample_size if hasattr(args, 'sample_size') else 5)

    # Итоговая статистика
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"Этап 0: {stats0['total_filtered']}/{stats0['total_original']} сообщений")
    print(f"Этап 1: Найдено {stats1['threads_found']} веток")
    print(f"Этап 2: {stats2['threads_deduped']} веток после дедупликации")
    print(f"Этап 3: Создано {stats3['chunks_created']} чанков")


def main():
    parser = argparse.ArgumentParser(
        description="Обработка чата SmartTherm",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )

    subparsers = parser.add_subparsers(dest="command", help="Команда")

    # Общие аргументы
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--model", "-m", type=str, default=None, help="ID модели (по умолчанию из конфига)")
    common_args.add_argument("--quantization", "-q", type=str, default=None, help="Квантование (по умолчанию из конфига)")
    common_args.add_argument("--context-size", type=int, default=None, help="Размер контекста")
    common_args.add_argument("--max-tokens", type=int, default=None, help="Максимум токенов")
    common_args.add_argument("--group-size", type=int, default=50, help="Размер группы")
    common_args.add_argument("--overlap-size", type=int, default=5, help="Перекрытие")

    # Этап 0
    parser_stage0 = subparsers.add_parser("stage0", help="Фильтрация шума", parents=[common_args])
    parser_stage0.add_argument("--input-path", type=str, default=None, help="Входной файл")
    parser_stage0.add_argument("--output-path", type=str, default=None, help="Выходной файл")
    parser_stage0.set_defaults(func=cmd_stage0)

    # Этап 1
    parser_stage1 = subparsers.add_parser("stage1", help="Выделение веток", parents=[common_args])
    parser_stage1.add_argument("--input-path", type=str, default=None, help="Входной файл")
    parser_stage1.add_argument("--output-path", type=str, default=None, help="Выходной файл")
    parser_stage1.add_argument("--debug", action="store_true", help="Debug режим: сырой ответ LLM")
    parser_stage1.set_defaults(func=cmd_stage1)

    # Этап 2
    parser_stage2 = subparsers.add_parser("stage2", help="Дедупликация веток", parents=[common_args])
    parser_stage2.add_argument("--input-path", type=str, default=None, help="Входной файл")
    parser_stage2.add_argument("--output-path", type=str, default=None, help="Выходной файл")
    parser_stage2.set_defaults(func=cmd_stage2)

    # Этап 3
    parser_stage3 = subparsers.add_parser("stage3", help="Создание RAG чанков", parents=[common_args])
    parser_stage3.add_argument("--threads-path", type=str, default=None, help="Входной файл с ветками")
    parser_stage3.add_argument("--messages-path", type=str, default=None, help="Входной файл с сообщениями")
    parser_stage3.add_argument("--output-path", type=str, default=None, help="Выходной файл")
    parser_stage3.add_argument("--sample-size", type=int, default=5, help="Размер выборки")
    parser_stage3.set_defaults(func=cmd_stage3)

    # Все этапы
    parser_all = subparsers.add_parser("all", help="Все этапы", parents=[common_args])
    parser_all.add_argument("--sample-size", type=int, default=5, help="Размер выборки")
    parser_all.set_defaults(func=cmd_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
