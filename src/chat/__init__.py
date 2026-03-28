"""Публичный API chat-подсистемы без eager-import composition root."""

from __future__ import annotations

from typing import Any

from .application.session_facade import SessionFacade
from .registry import DialogRegistry


def build_chat_session(*args: Any, **kwargs: Any) -> SessionFacade:
    from .composition import build_chat_session as _build_chat_session

    return _build_chat_session(*args, **kwargs)


def build_dialog_registry(*args: Any, **kwargs: Any) -> DialogRegistry:
    from .composition import build_dialog_registry as _build_dialog_registry

    return _build_dialog_registry(*args, **kwargs)


__all__ = ["build_chat_session", "build_dialog_registry", "SessionFacade", "DialogRegistry"]
