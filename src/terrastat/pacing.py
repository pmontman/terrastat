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
    """Two fixed brakes, plus one that reacts to what the server says.

    ``min_interval`` and ``max_per_minute`` are the polite defaults chosen per source. The third
    brake exists because those defaults are a guess: OECD throttles in long bursts, and a client
    that treats each 429 as an isolated event forgets, the instant it moves to the next dataset,
    everything the server just told it. ``pushback()`` doubles the gap and it stays doubled until
    a run of clean responses earns it back, so a burst slows the whole crawl rather than being
    re-discovered dataset by dataset.
    """

    min_interval: float = 1.0  # seconds between the start of two requests
    max_per_minute: int | None = None  # None means no per-minute cap
    jitter: float = 0.15  # fraction of min_interval added at random
    max_penalty: float = 16.0  # the gap never grows past this multiple of min_interval
    recover_after: int = 20  # clean responses needed before the gap halves back down
    _last: float = field(default=0.0, init=False, repr=False)
    _window: deque = field(default_factory=deque, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _penalty: float = field(default=1.0, init=False, repr=False)
    _clean: int = field(default=0, init=False, repr=False)
    _not_before: float = field(default=0.0, init=False, repr=False)

    @property
    def penalty(self) -> float:
        return self._penalty

    @property
    def interval(self) -> float:
        """The gap actually being applied, after any penalty."""
        return self.min_interval * self._penalty

    def wait(self) -> None:
        """Block until a request may be sent, then record it."""
        with self._lock:
            now = time.monotonic()
            gap = self.interval * (1 + random.random() * self.jitter)
            sleep_for = max(0.0, self._last + gap - now, self._not_before - now)
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

    def pushback(self, cooldown: float = 0.0) -> float:
        """The server asked us to slow down. Widen the gap, and optionally hold off entirely.

        Returns the new gap, for logging. ``cooldown`` is a hard "send nothing before" deadline —
        what to use when a request has exhausted its retries, so the next dataset does not open by
        walking straight back into the same wall.
        """
        with self._lock:
            self._penalty = min(self.max_penalty, self._penalty * 2)
            self._clean = 0
            if cooldown > 0:
                self._not_before = max(self._not_before, time.monotonic() + cooldown)
            return self.interval

    def ok(self) -> None:
        """A clean response. Earn the gap back gradually — resetting it outright just re-triggers
        the throttle a few requests later."""
        with self._lock:
            if self._penalty <= 1.0:
                return
            self._clean += 1
            if self._clean >= self.recover_after:
                self._penalty = max(1.0, self._penalty / 2)
                self._clean = 0

    def holding_off(self) -> float:
        """Seconds still to wait on a cooldown, or 0."""
        with self._lock:
            return max(0.0, self._not_before - time.monotonic())


def backoff_seconds(attempt: int, base: float = 2.0, cap: float = 300.0) -> float:
    """Exponential backoff with full jitter: attempt 0 -> up to 2s, 1 -> up to 4s, ... capped."""
    return random.uniform(0, min(cap, base * 2**attempt))
