from __future__ import annotations

import sys
import types


if "rank_bm25" not in sys.modules:
    rank_bm25 = types.ModuleType("rank_bm25")

    class BM25Okapi:
        def __init__(self, corpus, k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b

        def get_scores(self, query_tokens):
            del query_tokens
            return [0.0 for _ in self.corpus]

    setattr(rank_bm25, "BM25Okapi", BM25Okapi)
    sys.modules["rank_bm25"] = rank_bm25


if "telegram" not in sys.modules:
    telegram = types.ModuleType("telegram")

    class Bot:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class Message:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class Update:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class BotCommand:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    setattr(telegram, "Bot", Bot)
    setattr(telegram, "Message", Message)
    setattr(telegram, "Update", Update)
    setattr(telegram, "BotCommand", BotCommand)
    sys.modules["telegram"] = telegram

    telegram_ext = types.ModuleType("telegram.ext")

    class Application:
        @classmethod
        def builder(cls):
            return cls()

        def token(self, *args, **kwargs):
            del args, kwargs
            return self

        def post_init(self, *args, **kwargs):
            del args, kwargs
            return self

        def build(self):
            return self

    class CommandHandler:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class ContextTypes:
        DEFAULT_TYPE = object

    class MessageHandler:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class _FilterExpr:
        def __and__(self, other):
            del other
            return self

        def __rand__(self, other):
            del other
            return self

        def __invert__(self):
            return self

    class _Filters:
        COMMAND = _FilterExpr()
        TEXT = _FilterExpr()

    setattr(telegram_ext, "Application", Application)
    setattr(telegram_ext, "CommandHandler", CommandHandler)
    setattr(telegram_ext, "ContextTypes", ContextTypes)
    setattr(telegram_ext, "MessageHandler", MessageHandler)
    setattr(telegram_ext, "filters", _Filters())
    sys.modules["telegram.ext"] = telegram_ext
