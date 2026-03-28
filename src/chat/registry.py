"""Registry and lifecycle management for dialog sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any, Callable

from .application.session_facade import SessionFacade


@dataclass(slots=True)
class DialogSessionLease:
    """Acquired dialog session with lifecycle metadata."""

    dialog_key: str
    session: SessionFacade
    lock: Any = field(default_factory=RLock)
    in_use_count: int = 0
    last_released_at: float = field(default_factory=time.monotonic)


class DialogRegistry:
    """In-memory registry dialog sessions with lazy creation and eviction."""

    def __init__(
        self,
        *,
        session_factory: Callable[[str], SessionFacade],
        max_contexts: int | None = None,
        idle_ttl_seconds: int | None = None,
    ):
        self._session_factory = session_factory
        self._max_contexts = max_contexts
        self._idle_ttl_seconds = idle_ttl_seconds
        self._contexts: dict[str, DialogSessionLease] = {}
        self._registry_lock = RLock()

    def _acquire(self, lease: DialogSessionLease) -> DialogSessionLease:
        lease.in_use_count += 1
        return lease

    def release(self, lease: DialogSessionLease) -> None:
        with self._registry_lock:
            existing = self._contexts.get(lease.dialog_key)
            if existing is None or existing is not lease or existing.in_use_count <= 0:
                return
            existing.in_use_count -= 1
            if existing.in_use_count == 0:
                existing.last_released_at = time.monotonic()
                self._sweep_contexts(existing.last_released_at)

    def _evict_expired_contexts(self, now: float) -> None:
        idle_ttl_seconds = self._idle_ttl_seconds
        if idle_ttl_seconds is None or idle_ttl_seconds <= 0:
            return

        expired_keys = [
            key
            for key, lease in self._contexts.items()
            if lease.in_use_count == 0 and now - lease.last_released_at >= idle_ttl_seconds
        ]
        for key in expired_keys:
            self._contexts.pop(key, None)

    def _evict_lru_contexts(self, *, reserve_slots: int = 0) -> None:
        max_contexts = self._max_contexts
        if max_contexts is None or max_contexts <= 0:
            return

        allowed_contexts = max_contexts - reserve_slots
        if allowed_contexts < 0:
            allowed_contexts = 0

        overflow = len(self._contexts) - allowed_contexts
        if overflow <= 0:
            return

        inactive_contexts = sorted(
            (
                (key, lease)
                for key, lease in self._contexts.items()
                if lease.in_use_count == 0
            ),
            key=lambda item: item[1].last_released_at,
        )
        for key, _lease in inactive_contexts[:overflow]:
            self._contexts.pop(key, None)

    def _sweep_contexts(self, now: float) -> None:
        self._evict_expired_contexts(now)
        self._evict_lru_contexts()

    def acquire(self, dialog_key: str) -> DialogSessionLease:
        """Acquire existing dialog session or create a new one."""
        with self._registry_lock:
            now = time.monotonic()
            self._sweep_contexts(now)
            lease = self._contexts.get(dialog_key)
            if lease is not None:
                return self._acquire(lease)

            self._evict_lru_contexts(reserve_slots=1)
            lease = DialogSessionLease(
                dialog_key=dialog_key,
                session=self._session_factory(dialog_key),
            )
            lease.last_released_at = now
            self._contexts[dialog_key] = lease
            return self._acquire(lease)
