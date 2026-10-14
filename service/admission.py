"""Bounded admission control for expensive RAG inference requests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import BoundedSemaphore


@dataclass(frozen=True)
class AdmissionDecision:
    """Result of waiting for one inference-capacity slot."""

    admitted: bool
    wait_seconds: float


class RequestAdmissionController:
    """Bound concurrent inference and reject requests after a short wait."""

    def __init__(
        self,
        max_concurrent_requests: int,
        queue_timeout_seconds: float,
    ) -> None:
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be at least 1")
        if queue_timeout_seconds < 0:
            raise ValueError("queue_timeout_seconds must not be negative")

        self.max_concurrent_requests = max_concurrent_requests
        self.queue_timeout_seconds = queue_timeout_seconds
        self._slots = BoundedSemaphore(max_concurrent_requests)

    def acquire(self) -> AdmissionDecision:
        """Wait up to the configured timeout for a request slot."""
        started = time.perf_counter()
        admitted = self._slots.acquire(timeout=self.queue_timeout_seconds)
        return AdmissionDecision(
            admitted=admitted,
            wait_seconds=time.perf_counter() - started,
        )

    def release(self) -> None:
        """Return one previously acquired request slot."""
        self._slots.release()
