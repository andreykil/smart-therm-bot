#!/usr/bin/env python3
"""CLI entrypoint для Telegram long polling бота."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.bot.telegram_runner import run_telegram_bot


def main(argv: list[str] | None = None) -> None:
    run_telegram_bot(argv)


if __name__ == "__main__":
    main()
