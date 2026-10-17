"""Thread-safe bounded TTL cache for completed RAG responses."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

Value = TypeVar("Value")


@dataclass(frozen=True)
class _Entry(Generic[Value]):
    value: Value
    expires_at: float


class TTLCache(Generic[Value]):
    """Least-recently-used cache with monotonic TTL expiry."""

    def __init__(
        self,
        max_entries: int = 256,
        ttl_seconds: float = 300,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries: OrderedDict[str, _Entry[Value]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Value | None:
        now = self.clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value

    def set(self, key: str, value: Value) -> None:
        with self._lock:
            self._entries[key] = _Entry(
                value=value,
                expires_at=self.clock() + self.ttl_seconds,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


@dataclass(frozen=True)
class FlightClaim(Generic[Value]):
    """One caller's role in a shared computation for a cache key."""

    future: Future[Value]
    leader: bool


class SingleFlight(Generic[Value]):
    """Coalesce concurrent work for the same key into one computation."""

    def __init__(self) -> None:
        self._flights: dict[str, Future[Value]] = {}
        self._lock = RLock()

    def claim(self, key: str) -> FlightClaim[Value]:
        with self._lock:
            future = self._flights.get(key)
            if future is not None:
                return FlightClaim(future=future, leader=False)
            future = Future()
            self._flights[key] = future
            return FlightClaim(future=future, leader=True)

    def succeed(self, key: str, value: Value) -> None:
        self._finish(key, value=value)

    def fail(self, key: str, error: BaseException) -> None:
        self._finish(key, error=error)

    def _finish(
        self,
        key: str,
        *,
        value: Value | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            try:
                future = self._flights[key]
            except KeyError as exc:
                raise RuntimeError(
                    f"No in-flight computation exists for key {key!r}"
                ) from exc
            if error is None:
                future.set_result(value)
            else:
                future.set_exception(error)
            del self._flights[key]
