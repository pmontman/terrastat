"""A small HTTP client that is polite by construction: paced, retrying, streaming to disk."""
from __future__ import annotations

import datetime as dt
import logging
import shutil
import time
from email.utils import parsedate_to_datetime
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from terrastat.config import USER_AGENT
from terrastat.pacing import Pacer, backoff_seconds

log = logging.getLogger(__name__)

# 520-530 are Cloudflare's own codes, which sdmx.oecd.org sits behind; 524 is its origin
# timeout and fires on the largest dataflows, so it must be retried like any other 5xx.
RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 527, 530}
# Statuses that mean "you are asking too often", as opposed to "something broke". These widen
# the pacer's gap for every later request, not only for this one.
PUSHBACK_STATUSES = {429, 503, 509}
PROGRESS_EVERY = 50_000_000  # report every 50 MB, so a long download visibly moves


class GaveUp(Exception):
    """Raised when a request keeps failing after all retries."""


class Throttled(GaveUp):
    """Gave up while the server was still rate-limiting us.

    Worth separating from a plain failure: nothing is wrong with the dataset, and a later run
    is likely to get it. The pipeline records it under its own status, so it is picked up next
    run without needing --retry-failed.
    """


class StopRequested(Exception):
    """Raised by a source between checkpointed steps when the run asked to stop (budget, Ctrl+C)."""


class TooLarge(Exception):
    """A response exceeded the size cap and was abandoned mid-stream (see ``max_bytes``)."""


@dataclass
class Stats:
    requests: int = 0
    retries: int = 0
    bytes: int = 0
    seconds: float = 0.0
    by_status: dict = field(default_factory=dict)

    def summary(self) -> str:
        mb = self.bytes / 1e6
        return (
            f"{self.requests} requests, {self.retries} retries, {mb:,.1f} MB, "
            f"{self.seconds:,.0f}s in flight, statuses={self.by_status}"
        )


class PoliteClient:
    """GET-only client. Every call goes through the pacer; transient errors are retried with backoff."""

    def __init__(self, pacer: Pacer, headers: dict | None = None, timeout: float = 900.0,
                 max_retries: int = 6, throttle_floor: float = 30.0, cooldown: float = 300.0):
        # httpx applies `timeout` per socket operation, not to the whole transfer: it is the
        # longest we will wait for the *next* chunk. So a multi-GB download that keeps streaming
        # never trips it however long it runs (the 6.6 GB OECD dataflow took 38 minutes with no
        # retry), while a server that stalls mid-response still fails in a bounded time.
        self.pacer = pacer
        self.max_retries = max_retries
        # A pushback waits at least `throttle_floor` seconds, then twice that, and so on.
        # `cooldown` is how long to stop sending altogether once a request has spent all its
        # retries on pushback.
        self.throttle_floor = throttle_floor
        self.cooldown = cooldown
        self._pushbacks = 0  # consecutive pushbacks, counted across requests
        self.stats = Stats()
        self.stop_check = None  # optional callable -> bool; sources poll it between checkpointed steps
        self.on_progress = None  # optional callable(bytes_so_far) -> None, for long downloads
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            timeout=httpx.Timeout(timeout, connect=30.0),
            follow_redirects=True,
        )

    def stop_requested(self) -> bool:
        return bool(self.stop_check and self.stop_check())

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float:
        """`Retry-After`, which is either a count of seconds or an HTTP date."""
        raw = resp.headers.get("Retry-After", "").strip()
        if not raw:
            return 0.0
        if raw.isdigit():
            return float(raw)
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return 0.0
        if when is None:
            return 0.0
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        return max(0.0, (when - dt.datetime.now(dt.timezone.utc)).total_seconds())

    def _note(self, resp: httpx.Response | None) -> bool:
        """Record what the server said. True if it was pushback rather than breakage.

        A 5xx arriving in the middle of a throttling burst counts as pushback too. OECD emits both
        while it is unhappy, and treating the 500 as an unrelated glitch resets the backoff to a
        few seconds and puts us straight back into the 429s, which is what the logs showed.
        """
        pushed = resp is not None and (
            resp.status_code in PUSHBACK_STATUSES
            or (self._pushbacks and resp.status_code in RETRY_STATUSES)
            or self._retry_after(resp) > 0
        )
        if pushed:
            self._pushbacks += 1
            self.pacer.pushback()
        return pushed

    def _retry_wait(self, resp: httpx.Response | None, attempt: int) -> float:
        wait = backoff_seconds(attempt)
        if resp is not None:
            wait = max(wait, self._retry_after(resp))
        if self._pushbacks:
            # Escalate on the running streak, not on this request's attempt count: a burst that
            # spans several datasets should keep widening rather than restart at the floor each
            # time a new dataset begins.
            steps = max(attempt + 1, self._pushbacks)
            wait = max(wait, min(600.0, self.throttle_floor * steps))
        return wait

    def _gave_up(self, url: str) -> GaveUp:
        """Give up, and if we were being throttled, stop sending for a while.

        Without the cooldown the next dataset opens by firing straight into the same wall, which
        is how one throttling burst became a run of consecutive failures.
        """
        if self._pushbacks:
            host = url.split("/")[2] if "//" in url else url
            self.pacer.pushback(cooldown=self.cooldown)
            log.warning("throttled by %s; pausing this source for %.0fs (gap now %.0fs)",
                        host, self.cooldown, self.pacer.interval)
            return Throttled(f"still rate-limited after {self.max_retries + 1} attempts on {url}")
        return GaveUp(f"gave up on {url} after {self.max_retries + 1} attempts")

    def _succeeded(self) -> None:
        self._pushbacks = 0
        self.pacer.ok()

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        """GET with pacing and retries; returns the response (not raised for 4xx other than retryables)."""
        for attempt in range(self.max_retries + 1):
            self.pacer.wait()
            t0 = time.monotonic()
            try:
                resp = self._client.get(url, params=params, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                self.stats.retries += 1
                wait = self._retry_wait(None, attempt)
                log.warning("network error on %s (%s); retry %d in %.0fs", url, exc, attempt + 1, wait)
                time.sleep(wait)
                continue
            self.stats.requests += 1
            self.stats.seconds += time.monotonic() - t0
            self.stats.by_status[resp.status_code] = self.stats.by_status.get(resp.status_code, 0) + 1
            if resp.status_code in RETRY_STATUSES:
                self.stats.retries += 1
                self._note(resp)
                wait = self._retry_wait(resp, attempt)
                log.warning("HTTP %d on %s; retry %d in %.0fs", resp.status_code, url, attempt + 1, wait)
                time.sleep(wait)
                continue
            self._succeeded()
            self.stats.bytes += len(resp.content)
            return resp
        raise self._gave_up(url)

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        resp = self.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def download(self, url: str, dest: Path, params: dict | None = None, headers: dict | None = None, max_bytes: int | None = None) -> Path:
        """Stream a response body to ``dest`` atomically (temp file first, then rename).

        ``max_bytes`` abandons the transfer as soon as it is exceeded, so one runaway dataset
        cannot fill the disk. The partial file is removed and :class:`TooLarge` is raised.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        for attempt in range(self.max_retries + 1):
            self.pacer.wait()
            t0 = time.monotonic()
            try:
                with self._client.stream("GET", url, params=params, headers=headers) as resp:
                    self.stats.requests += 1
                    self.stats.by_status[resp.status_code] = self.stats.by_status.get(resp.status_code, 0) + 1
                    if resp.status_code in RETRY_STATUSES:
                        self.stats.retries += 1
                        self._note(resp)
                        wait = self._retry_wait(resp, attempt)
                        log.warning("HTTP %d on %s; retry %d in %.0fs", resp.status_code, url, attempt + 1, wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    self._succeeded()
                    n = 0
                    next_report = PROGRESS_EVERY
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 16):
                            fh.write(chunk)
                            n += len(chunk)
                            if n >= next_report:
                                next_report += PROGRESS_EVERY
                                if self.on_progress:
                                    self.on_progress(n)
                                log.info("downloading %s: %.0f MB so far", dest.name, n / 1e6)
                            if max_bytes is not None and n > max_bytes:
                                raise TooLarge(f"{dest.name} exceeded the {max_bytes / 1e9:.1f} GB cap (abandoned at {n / 1e9:.1f} GB)")
                            if self.stop_requested():
                                raise StopRequested(f"download of {dest.name} abandoned after {n / 1e6:.1f} MB; it restarts next run")
                    self.stats.bytes += n
                    self.stats.seconds += time.monotonic() - t0
                shutil.move(tmp, dest)
                if self.on_progress:
                    self.on_progress(0)  # clear the indicator
                return dest
            except (StopRequested, TooLarge):
                tmp.unlink(missing_ok=True)
                raise
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                self.stats.retries += 1
                wait = self._retry_wait(None, attempt)
                log.warning("network error on %s (%s); retry %d in %.0fs", url, exc, attempt + 1, wait)
                time.sleep(wait)
        raise self._gave_up(url)
