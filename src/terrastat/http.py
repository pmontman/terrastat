"""A small HTTP client that is polite by construction: paced, retrying, streaming to disk."""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from terrastat.config import USER_AGENT
from terrastat.pacing import Pacer, backoff_seconds

log = logging.getLogger(__name__)

# 520-530 are Cloudflare's own codes, which sdmx.oecd.org sits behind; 524 is its origin
# timeout and fires on the largest dataflows, so it must be retried like any other 5xx.
RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 527, 530}
PROGRESS_EVERY = 50_000_000  # report every 50 MB, so a long download visibly moves


class GaveUp(Exception):
    """Raised when a request keeps failing after all retries."""


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

    def __init__(self, pacer: Pacer, headers: dict | None = None, timeout: float = 900.0, max_retries: int = 6):
        # httpx applies `timeout` per socket operation, not to the whole transfer: it is the
        # longest we will wait for the *next* chunk. So a multi-GB download that keeps streaming
        # never trips it however long it runs (the 6.6 GB OECD dataflow took 38 minutes with no
        # retry), while a server that stalls mid-response still fails in a bounded time.
        self.pacer = pacer
        self.max_retries = max_retries
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

    def _retry_wait(self, resp: httpx.Response | None, attempt: int) -> float:
        wait = backoff_seconds(attempt)
        if resp is not None:
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, float(retry_after))
            if resp.status_code == 429:
                # OECD throttles in bursts and a flat 30 s was not always enough to clear one,
                # so escalate: 30 s, 60 s, 90 s ... up to five minutes.
                wait = max(wait, min(300.0, 30.0 * (attempt + 1)))
        return wait

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
                wait = self._retry_wait(resp, attempt)
                log.warning("HTTP %d on %s; retry %d in %.0fs", resp.status_code, url, attempt + 1, wait)
                time.sleep(wait)
                continue
            self.stats.bytes += len(resp.content)
            return resp
        raise GaveUp(f"gave up on {url} after {self.max_retries + 1} attempts")

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
                        wait = self._retry_wait(resp, attempt)
                        log.warning("HTTP %d on %s; retry %d in %.0fs", resp.status_code, url, attempt + 1, wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
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
        raise GaveUp(f"gave up on {url} after {self.max_retries + 1} attempts")
