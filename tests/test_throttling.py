"""What the client does when a source says "slow down", against a server that actually says it.

The behaviour these pin down came out of a real OECD crawl: seven 429s on one dataflow over
fourteen minutes, give up, mark it failed, then open the next dataflow and take a 429 on the very
first request. Every retry was correct in isolation and the run as a whole learned nothing.
"""
import http.server
import threading
import time

import pytest

from terrastat.http import GaveUp, PoliteClient, Throttled
from terrastat.pacing import Pacer


class _Throttling(http.server.BaseHTTPRequestHandler):
    """Answers 429 for the first `n_429` requests, then 200. Configured per test."""

    n_429 = 0
    retry_after = None
    status_after = 200
    seen = []

    def do_GET(self):                                    # noqa: N802 (stdlib naming)
        cls = type(self)
        cls.seen.append(time.monotonic())
        if len(cls.seen) <= cls.n_429:
            self.send_response(429)
            if cls.retry_after is not None:
                self.send_header("Retry-After", str(cls.retry_after))
        else:
            self.send_response(cls.status_after)
        body = b"ok"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                           # keep pytest output readable
        pass


@pytest.fixture
def server():
    _Throttling.seen = []
    _Throttling.n_429 = 0
    _Throttling.retry_after = None
    _Throttling.status_after = 200
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Throttling)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/", _Throttling
    srv.shutdown()


def _client(**kw):
    """A client whose waits are milliseconds rather than minutes, so a test can run."""
    kw.setdefault("throttle_floor", 0.01)
    kw.setdefault("cooldown", 0.05)
    kw.setdefault("max_retries", 2)
    return PoliteClient(Pacer(min_interval=0.0, jitter=0.0, max_per_minute=None), **kw)


# -- the pacer's own behaviour ---------------------------------------------------------------------

def test_pushback_widens_the_gap_and_recovery_earns_it_back():
    p = Pacer(min_interval=2.0, recover_after=5)
    assert p.interval == 2.0
    p.pushback()
    p.pushback()
    assert p.interval == 8.0, "two pushbacks should double twice"
    for _ in range(4):
        p.ok()
    assert p.interval == 8.0, "recovery must not be instant, or the throttle just re-triggers"
    p.ok()
    assert p.interval == 4.0
    for _ in range(5):
        p.ok()
    assert p.interval == 2.0


def test_the_gap_is_bounded():
    p = Pacer(min_interval=3.0, max_penalty=4.0)
    for _ in range(20):
        p.pushback()
    assert p.interval == 12.0


def test_a_cooldown_holds_off_the_next_request():
    p = Pacer(min_interval=0.0, jitter=0.0)
    p.pushback(cooldown=0.25)
    assert p.holding_off() > 0.1
    t0 = time.monotonic()
    p.wait()
    assert time.monotonic() - t0 >= 0.2, "wait() ignored the cooldown"
    assert p.holding_off() == 0


def test_ok_on_an_unpenalised_pacer_does_nothing():
    p = Pacer(min_interval=1.0)
    for _ in range(100):
        p.ok()
    assert p.interval == 1.0


# -- the client, against a server that throttles ----------------------------------------------------

def test_a_burst_is_survived_and_the_gap_stays_wide_afterwards(server):
    url, handler = server
    handler.n_429 = 2
    c = _client()
    resp = c.get(url)
    assert resp.status_code == 200
    assert len(handler.seen) == 3
    # the pacer keeps what it learned: one success does not undo two pushbacks
    assert c.pacer.penalty == 4.0


def test_giving_up_while_throttled_is_not_an_ordinary_failure(server):
    """A dataset that was only ever answered with 429 is not broken, and the distinction is what
    lets the pipeline retry it next run instead of burying it among real failures."""
    url, handler = server
    handler.n_429 = 999
    c = _client()
    with pytest.raises(Throttled) as e:
        c.get(url)
    assert isinstance(e.value, GaveUp)          # existing handlers still catch it
    assert "rate-limited" in str(e.value)
    assert c.pacer.holding_off() > 0, "the next dataset would walk straight back into the wall"


def test_giving_up_on_a_plain_5xx_is_still_a_plain_failure(server):
    """Only pushback earns the softer treatment. A server that is simply broken should not put
    the whole source to sleep."""
    url, handler = server
    handler.n_429 = 0
    handler.status_after = 500
    c = _client()
    with pytest.raises(GaveUp) as e:
        c.get(url)
    assert not isinstance(e.value, Throttled)
    assert c.pacer.holding_off() == 0


def test_a_5xx_inside_a_burst_does_not_reset_the_backoff(server):
    """The bug from the logs: after waiting 120s for a 429, a 500 came back and the client waited
    20s, then 28s, hammering a server that was plainly still unhappy."""
    c = _client(throttle_floor=10.0)

    class _Resp:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}

    # a fresh client treats a 500 as breakage: short, jittered backoff
    assert c._retry_wait(_Resp(500), attempt=0) < 10.0
    # but once pushback has started, the 500 keeps the escalation
    c._note(_Resp(429))
    c._note(_Resp(500))
    assert c._pushbacks == 2
    assert c._retry_wait(_Resp(500), attempt=1) >= 20.0


def test_the_streak_escalates_across_requests_not_within_one(server):
    c = _client(throttle_floor=10.0)

    class _Resp:
        status_code = 429
        headers = {}

    for _ in range(5):
        c._note(_Resp())
    # attempt 0 of a *new* request, but the fifth consecutive pushback
    assert c._retry_wait(_Resp(), attempt=0) >= 50.0


def test_a_success_clears_the_streak(server):
    url, handler = server
    handler.n_429 = 1
    c = _client()
    c.get(url)
    assert c._pushbacks == 0


# -- Retry-After -----------------------------------------------------------------------------------

def test_retry_after_in_seconds_is_honoured(server):
    c = _client()

    class _Resp:
        status_code = 429
        headers = {"Retry-After": "45"}

    assert c._retry_wait(_Resp(), attempt=0) >= 45.0


def test_retry_after_as_an_http_date_is_honoured(server):
    """The header is defined as either a count of seconds or a date, and only the first form was
    being read; a date silently became no delay at all."""
    import datetime as dt
    from email.utils import format_datetime

    c = _client()
    soon = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=90)

    class _Resp:
        status_code = 429
        headers = {"Retry-After": format_datetime(soon)}

    assert 80 <= c._retry_wait(_Resp(), attempt=0) <= 100


def test_a_date_in_the_past_does_not_produce_a_negative_wait(server):
    import datetime as dt
    from email.utils import format_datetime

    c = _client()
    past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)

    class _Resp:
        status_code = 429
        headers = {"Retry-After": format_datetime(past)}

    assert c._retry_wait(_Resp(), attempt=0) >= 0


def test_an_unparseable_retry_after_is_ignored_rather_than_fatal(server):
    c = _client()

    class _Resp:
        status_code = 429
        headers = {"Retry-After": "whenever you like"}

    assert c._retry_wait(_Resp(), attempt=0) >= 0


def test_a_retry_after_on_any_status_counts_as_pushback(server):
    """Some hosts send Retry-After with a 503 or even a 200-adjacent code. If the server is
    telling us when to come back, it is telling us to slow down."""
    c = _client()

    class _Resp:
        status_code = 502
        headers = {"Retry-After": "30"}

    assert c._note(_Resp()) is True
    assert c.pacer.penalty == 2.0
