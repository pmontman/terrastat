"""Fetching a published snapshot: caching, resuming, verifying — against a real local server."""
import functools
import http.server
import json
import threading

import os

import httpx
import polars as pl
import pytest

from terrastat import datasets


@pytest.fixture
def published(tmp_path):
    """A snapshot laid out exactly as a published one, served over HTTP."""
    src = tmp_path / "site" / "starter"
    src.mkdir(parents=True)
    df = pl.DataFrame({
        "series_uid": [f"s{i}" for i in range(50)],
        "frequency": ["M"] * 25 + ["A"] * 25,
        "n_obs": list(range(10, 60)),
        "values": [[float(j) for j in range(5)] for _ in range(50)],
    })
    df.write_parquet(src / "shard-00000.parquet")
    df.head(10).write_parquet(src / "index.parquet")
    manifest = {
        "name": "starter", "rows": 50,
        "shards": [{"file": "shard-00000.parquet",
                    "bytes": (src / "shard-00000.parquet").stat().st_size,
                    "sha256": datasets.sha256(src / "shard-00000.parquet")}],
        "index": {"file": "index.parquet",
                  "bytes": (src / "index.parquet").stat().st_size,
                  "sha256": datasets.sha256(src / "index.parquet")},
    }
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (src / "README.md").write_text("# starter\n", encoding="utf-8")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(tmp_path / "site"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/starter/", tmp_path / "cache", manifest
    srv.shutdown()


def test_it_downloads_and_verifies(published):
    url, cache, manifest = published
    out = datasets.fetch(url, cache=cache)
    assert (out / "shard-00000.parquet").exists()
    assert (out / "manifest.json").exists()
    assert datasets.verify(out)["complete"]
    assert not list(out.glob("*.part"))          # nothing half-written is left behind


def test_a_second_call_downloads_nothing(published, caplog):
    url, cache, _ = published
    datasets.fetch(url, cache=cache)
    with caplog.at_level("INFO"):
        datasets.fetch(url, cache=cache)
    assert "already complete" in caplog.text


def test_a_server_that_ignores_range_still_produces_the_right_file(published):
    """Plenty of static hosts ignore `Range` and answer 200 with the whole body. The client must
    notice (206 vs 200) and rewrite from the start rather than appending to what it already had,
    which would corrupt the file. `SimpleHTTPRequestHandler` is exactly such a host, which is why
    the real resume path needs its own server below."""
    url, cache, manifest = published
    out = cache / "starter"
    out.mkdir(parents=True)
    full = manifest["shards"][0]
    # simulate a transfer that died a third of the way through
    import httpx
    whole = httpx.get(url + full["file"]).content
    part = out / (full["file"] + ".part")
    part.write_bytes(whole[: len(whole) // 3])
    assert part.stat().st_size < full["bytes"]

    datasets.fetch(url, cache=cache)
    assert (out / full["file"]).read_bytes() == whole
    assert datasets.verify(out)["complete"]


def test_a_stale_partial_larger_than_the_file_is_discarded(published):
    url, cache, manifest = published
    out = cache / "starter"
    out.mkdir(parents=True)
    full = manifest["shards"][0]
    (out / (full["file"] + ".part")).write_bytes(b"x" * (full["bytes"] + 500))
    datasets.fetch(url, cache=cache)
    assert datasets.verify(out)["complete"]


def test_a_corrupted_file_is_re_fetched_not_trusted(published):
    url, cache, manifest = published
    out = datasets.fetch(url, cache=cache)
    target = out / manifest["shards"][0]["file"]
    good = target.read_bytes()
    target.write_bytes(b"\0" * len(good))        # same size, wrong content
    assert datasets.verify(out)["corrupt"] == [target.name]
    datasets.fetch(url, cache=cache)             # size matches but the checksum does not
    assert target.read_bytes() == good


def test_verify_reports_what_is_wrong_without_the_network(published):
    url, cache, manifest = published
    out = datasets.fetch(url, cache=cache)
    (out / manifest["shards"][0]["file"]).unlink()
    rep = datasets.verify(out)
    assert rep["missing"] == ["shard-00000.parquet"] and not rep["complete"]


def test_load_returns_a_frame_and_honours_filters(published):
    url, cache, _ = published
    lf = datasets.load(url, cache=cache, frequencies=["M"], min_obs=20)
    df = lf.collect()
    assert df.height and set(df.get_column("frequency").unique()) == {"M"}
    assert df.get_column("n_obs").min() >= 20


def test_offline_mode_refuses_to_reach_the_network(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=False"):
        datasets.load("http://127.0.0.1:1/nothing/", cache=tmp_path, download=False)


def test_an_unknown_name_says_what_to_do_instead():
    with pytest.raises(KeyError) as e:
        datasets.fetch("does-not-exist")
    assert "terrastat export" in str(e.value)


def test_the_cache_location_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("TERRASTAT_CACHE", str(tmp_path / "elsewhere"))
    assert datasets.cache_dir() == tmp_path / "elsewhere"


def test_clear_removes_a_cached_snapshot(published):
    url, cache, _ = published
    out = datasets.fetch(url, cache=cache)
    assert out.exists()
    datasets.clear("starter", cache=cache)
    assert not out.exists()


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """A server that honours `Range`, which `SimpleHTTPRequestHandler` does not.

    Without this the resume test passes for the wrong reason: the stock handler ignores the header
    and returns 200 with the whole body, so the client silently rewrites from scratch and the file
    still ends up correct. That exercises the fallback, not the resume.
    """

    served = []          # bytes actually sent per request, so a test can prove what moved

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().send_head()
        path = self.translate_path(self.path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        start = int(rng.split("=", 1)[1].split("-", 1)[0])
        if start >= size:
            f.close()
            self.send_error(416)
            return None
        f.seek(start)
        length = size - start
        type(self).served.append(length)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.end_headers()
        return f


@pytest.fixture
def published_ranged(tmp_path):
    """The same layout, served by something that actually supports resuming."""
    src = tmp_path / "site" / "starter"
    src.mkdir(parents=True)
    df = pl.DataFrame({"series_uid": [f"s{i}" for i in range(4000)],
                       "frequency": ["M"] * 4000, "n_obs": [30] * 4000,
                       "values": [[float(j) for j in range(20)] for _ in range(4000)]})
    df.write_parquet(src / "shard-00000.parquet")
    manifest = {"name": "starter", "rows": 4000,
                "shards": [{"file": "shard-00000.parquet",
                            "bytes": (src / "shard-00000.parquet").stat().st_size,
                            "sha256": datasets.sha256(src / "shard-00000.parquet")}]}
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _RangeHandler.served = []
    handler = functools.partial(_RangeHandler, directory=str(tmp_path / "site"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/starter/", tmp_path / "cache", manifest
    srv.shutdown()


def test_resume_transfers_only_the_missing_bytes(published_ranged):
    """The point of resuming: a dropped transfer costs the remainder, not the whole file."""
    url, cache, manifest = published_ranged
    full = manifest["shards"][0]
    out = cache / "starter"
    out.mkdir(parents=True)
    whole = httpx.get(url + full["file"]).content
    cut = len(whole) * 3 // 4
    (out / (full["file"] + ".part")).write_bytes(whole[:cut])

    _RangeHandler.served = []
    datasets.fetch(url, cache=cache)
    assert (out / full["file"]).read_bytes() == whole
    # a ranged response was served, and it carried only the tail
    assert _RangeHandler.served, "no 206 was served: the resume path did not run"
    assert sum(_RangeHandler.served) == len(whole) - cut
    assert sum(_RangeHandler.served) < len(whole) / 2
