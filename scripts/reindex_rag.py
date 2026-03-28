"""
Переиндексация RAG чанков

Создаёт векторные индексы (FAISS + BM25) из JSONL файла с чанками.
Используется после обновления чанков или для первичной индексации.
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
from src.rag.composition import build_rag_runtime


def setup_logging(verbose: bool = False):
    """Настроить логирование"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Переиндексация RAG чанков",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python scripts/reindex_rag.py
  python scripts/reindex_rag.py --chunks-file data/processed/chat/test/chunks_rag_test.jsonl
  python scripts/reindex_rag.py --verbose
        """
    )

    parser.add_argument(
        "--chunks-file",
        type=str,
        default="data/processed/chat/chunks_rag.jsonl",
        help="Путь к JSONL файлу с чанками (по умолчанию: data/processed/chat/chunks_rag.jsonl)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    config = Config.load()

    chunks_path = Path(args.chunks_file)
    if not chunks_path.exists():
        logger.error(f"❌ Файл не найден: {args.chunks_file}")
        logger.info("Сначала создайте чанки: make chat-chunks")
        sys.exit(1)

    try:
        logger.info("🔧 Инициализация RAG runtime...")
        runtime = build_rag_runtime(
            config=config,
            base_url=config.llm.base_url,
            top_k=config.rag.top_k,
        )

        logger.info(f"📦 Переиндексация из {args.chunks_file}...")
        stats = runtime.index_manager.index_from_file(args.chunks_file, save=True)

        logger.info("✅ Переиндексация завершена!")
        logger.info(f"   Всего чанков: {stats.total_chunks}")
        logger.info(f"   FAISS векторов: {stats.faiss_vectors}")
        logger.info(f"   BM25 документов: {stats.bm25_documents}")
        logger.info(f"   Размерность эмбеддингов: {stats.embedding_dim}")

    except FileNotFoundError as error:
        logger.error(f"❌ Ошибка: {error}")
        sys.exit(1)
    except ConnectionError as error:
        logger.error(f"❌ Ollama недоступен: {error}")
        logger.info("Убедитесь что Ollama запущен: ollama serve")
        sys.exit(1)
    except Exception as error:
        logger.error(f"❌ Ошибка переиндексации: {error}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
