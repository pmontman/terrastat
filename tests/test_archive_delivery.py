"""Checksummed archive delivery through the same resumable HTTP path as shards."""
import hashlib

import httpx
import pytest

from terrastat import archive, datasets
from terrastat.cli import main


@pytest.fixture
def release(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Keep this attribution.\n", encoding="utf-8")
    (source / "values.bin").write_bytes(bytes(range(256)) * 100)
    packed = tmp_path / "release.tar.zst"
    archive.pack(source, packed)
    body = packed.read_bytes()
    return body, hashlib.sha256(body).hexdigest()


def serve(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(datasets.httpx, "Client", lambda **kwargs: client(
        transport=httpx.MockTransport(handler), **kwargs))


def test_url_resume_and_verified_cache_reuse(tmp_path, monkeypatch, release):
    body, digest = release
    cache = tmp_path / "cache"
    cache.mkdir()
    offset = len(body) // 3
    (cache / f"{digest}.tar.zst.part").write_bytes(body[:offset])
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["range"] == f"bytes={offset}-"
        return httpx.Response(206, content=body[offset:], headers={
            "Content-Range": f"bytes {offset}-{len(body)-1}/{len(body)}"})

    serve(monkeypatch, handle)
    for name in ("first", "second"):
        destination = tmp_path / name
        assert datasets.deploy("https://archive.example/release.tar.zst", destination,
                               sha256=digest, cache=cache) == destination
        assert (destination / "values.bin").read_bytes() == bytes(range(256)) * 100
    assert len(requests) == 1
    assert not list(cache.glob("*.part"))


def test_wrong_download_never_publishes_destination(tmp_path, monkeypatch, release):
    _, digest = release
    serve(monkeypatch, lambda request: httpx.Response(200, content=b"corrupted download"))
    destination = tmp_path / "deployed"
    with pytest.raises(OSError, match="checksum"):
        datasets.deploy("https://archive.example/release.tar.zst", destination,
                        sha256=digest, cache=tmp_path / "cache")
    assert not destination.exists()
    assert not list((tmp_path / "cache").glob("*.tar.zst"))


@pytest.mark.parametrize("digest", [None, "bad", "z" * 64])
def test_remote_digest_required_before_network(tmp_path, monkeypatch, digest):
    def forbidden(**kwargs):
        pytest.fail("should validate before reaching the network")
    monkeypatch.setattr(datasets.httpx, "Client", forbidden)
    with pytest.raises(ValueError, match="SHA-256"):
        datasets.deploy("https://archive.example/release.tar.zst", tmp_path / "new", sha256=digest)


def test_cli_pack_and_deploy_do_not_log_into_source(tmp_path, monkeypatch, capsys):
    source = tmp_path / "data"
    source.mkdir()
    (source / "metadata.json").write_text('{"version": 1}', encoding="utf-8")
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(source))
    packed = tmp_path / "release.tar.zst"
    destination = tmp_path / "restored"
    assert main(["pack", str(source), str(packed)]) == 0
    assert main(["deploy", str(packed), str(destination)]) == 0
    assert (destination / "metadata.json").read_bytes() == (source / "metadata.json").read_bytes()
    assert not (source / "logs").exists()
    assert "Deployed and verified" in capsys.readouterr().out


@pytest.mark.parametrize("bound", ["nan", "inf", "-1"])
def test_cli_rejects_invalid_expansion_bound(bound, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["deploy", "missing.tar.zst", str(tmp_path / "new"), "--max-gb", bound])
    assert exc.value.code == 1
