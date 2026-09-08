"""Polite request pacing.

Every source gets a :class:`Pacer` with two independent brakes: a minimum gap between
consecutive requests, and a cap on requests per sliding minute. Both are configurable from
the CLI. A little jitter is added so that retries from several machines do not synchronise.
"""
from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Pacer:
    min_interval: float = 1.0  # seconds between the start of two requests
    max_per_minute: int | None = None  # None means no per-minute cap
    jitter: float = 0.15  # fraction of min_interval added at random
    _last: float = field(default=0.0, init=False, repr=False)
    _window: deque = field(default_factory=deque, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def wait(self) -> None:
        """Block until a request may be sent, then record it."""
        with self._lock:
            now = time.monotonic()
            gap = self.min_interval * (1 + random.random() * self.jitter)
            sleep_for = max(0.0, self._last + gap - now)
            if self.max_per_minute:
                while self._window and now - self._window[0] >= 60:
                    self._window.popleft()
                if len(self._window) >= self.max_per_minute:
                    sleep_for = max(sleep_for, 60 - (now - self._window[0]) + 0.05)
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self._last = now
            if self.max_per_minute:
                self._window.append(now)


def backoff_seconds(attempt: int, base: float = 2.0, cap: float = 300.0) -> float:
    """Exponential backoff with full jitter: attempt 0 -> up to 2s, 1 -> up to 4s, ... capped."""
    return random.uniform(0, min(cap, base * 2**attempt))
