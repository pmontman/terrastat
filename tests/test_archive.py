"""A portable archive must preserve data and fail without publishing a partial tree."""
import datetime as dt
import hashlib
import io
import json
import math
import tarfile

import polars as pl
import pyarrow as pa
import pytest

from terrastat import archive, dataset


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def _record(path, data):
    return {"path": path, "bytes": len(data), "sha256": _digest(data)}


def _write_archive(path, files, *, records=None, version=1, extra=()):
    """Build independent format fixtures, including intentionally invalid tar members."""
    manifest = {"format": "terrastat-archive", "version": version,
                "files": records if records is not None else
                [_record(name, data) for name, data in files]}
    members = [("archive.json", json.dumps(manifest).encode()),
               *[(f"payload/{name}", data) for name, data in files], *extra]
    with pa.OSFile(str(path), "wb") as raw:
        with pa.CompressedOutputStream(raw, "zstd") as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tf:
                for name, data in members:
                    member = name if isinstance(name, tarfile.TarInfo) else tarfile.TarInfo(name)
                    member.size = len(data)
                    tf.addfile(member, io.BytesIO(data))
    return path


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    frame = pl.DataFrame({
        "series_uid": ["quarterly:one", "quarterly:empty"],
        "frequency": ["Q", "Q"], "n_obs": [2, 0],
        "dates": [[dt.date(2020, m, 1) for m in (1, 4, 7, 10)], []],
        "values": [[1.25, None, float("nan"), -0.0], []],
        "license_id": ["eurostat-reuse", "eurostat-reuse"],
    }, schema_overrides={"dates": pl.List(pl.Date), "values": pl.List(pl.Float32)})
    frame.write_parquet(source / "shard-00000.parquet", compression="zstd")
    frame.select("series_uid").write_parquet(source / "index.parquet")
    # Existing snapshots predate archive checksums. Their manifests remain valid as-is.
    (source / "manifest.json").write_bytes(b'{"rows": 2, "n_shards": 1}\r\n')
    (source / "ATTRIBUTIONS.md").write_text("Eurostat: source terms and attribution.\n", encoding="utf-8")
    (source / "metadata").mkdir()
    (source / "metadata" / "notes.json").write_bytes(b'{"notes": "unchanged"}\n')
    (source / "metadata" / "empty.txt").write_bytes(b"")
    return source


def test_snapshot_round_trip_preserves_every_byte_and_existing_reader(snapshot, tmp_path):
    original = _tree(snapshot)
    packed = tmp_path / "snapshot.tar.zst"
    result = archive.pack(snapshot, packed)

    assert result["bytes"] == packed.stat().st_size
    assert result["original_bytes"] == sum(map(len, original.values()))
    assert result["files"] == len(original)
    assert result["sha256"] == _digest(packed.read_bytes())
    assert _tree(snapshot) == original

    destination = tmp_path / "deployed"
    assert archive.deploy(packed, destination, sha256=result["sha256"]) == destination
    assert _tree(destination) == original
    assert _tree(snapshot) == original

    loaded = dataset.load(destination).collect()
    assert loaded["series_uid"].to_list() == ["quarterly:one", "quarterly:empty"]
    assert loaded.schema["values"] == pl.List(pl.Float32)
    assert loaded.schema["dates"] == pl.List(pl.Date)
    values = loaded["values"][0].to_list()
    assert values[0] == 1.25 and values[1] is None and math.isnan(values[2])
    assert math.copysign(1, values[3]) == -1
    assert loaded["dates"][0].to_list() == [dt.date(2020, m, 1) for m in (1, 4, 7, 10)]


def test_written_format_is_a_standard_zstd_tar_with_verifiable_inventory(snapshot, tmp_path):
    packed = tmp_path / "snapshot.tar.zst"
    archive.pack(snapshot, packed)
    with pa.OSFile(str(packed), "rb") as raw:
        with pa.CompressedInputStream(raw, "zstd") as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as tf:
                first = next(iter(tf))
                assert first.name == "archive.json" and first.isfile()
                manifest = json.load(tf.extractfile(first))
                assert manifest["format"] == "terrastat-archive" and manifest["version"] == 1
                inventory = {f["path"]: f for f in manifest["files"]}
                payload = {}
                for member in tf:
                    if member.name == "archive.json":
                        continue  # TarFile iteration includes the already-read first member.
                    assert member.isfile() and member.name.startswith("payload/")
                    data = tf.extractfile(member).read()
                    name = member.name.removeprefix("payload/")
                    assert inventory[name] == _record(name, data)
                    payload[name] = data
                assert payload == _tree(snapshot)


def test_expected_archive_checksum_is_checked_without_publishing(snapshot, tmp_path):
    packed = tmp_path / "snapshot.tar.zst"
    archive.pack(snapshot, packed)
    before = _tree(tmp_path)
    with pytest.raises(ValueError):
        archive.deploy(packed, tmp_path / "deployed", sha256="0" * 64)
    assert _tree(tmp_path) == before
    assert not (tmp_path / "deployed").exists()


@pytest.mark.parametrize("remaining", [0, 8, -1])
def test_truncated_archive_is_rejected_without_partial_deployment(snapshot, tmp_path, remaining):
    packed = tmp_path / "snapshot.tar.zst"
    archive.pack(snapshot, packed)
    data = packed.read_bytes()
    packed.write_bytes(data[:remaining])
    before = set(tmp_path.iterdir())
    with pytest.raises((ValueError, OSError, EOFError, tarfile.TarError, pa.ArrowException)):
        archive.deploy(packed, tmp_path / "deployed")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("path", [
    "../escaped.txt", "a/../../escaped.txt", "/escaped.txt",
    "C:/escaped.txt", "C:escaped.txt", "\\\\host\\share\\escaped.txt",
    "a\\..\\escaped.txt", "a//b.txt", "./a.txt", "a/./b.txt",
    "CON", "notes.txt:stream", "trailing.", "trailing ",
])
def test_unsafe_paths_are_rejected_on_every_platform(tmp_path, path):
    packed = _write_archive(tmp_path / "unsafe.tar.zst", [(path, b"unsafe")])
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        archive.deploy(packed, tmp_path / "deployed")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE])
def test_payload_links_and_special_members_are_rejected(tmp_path, kind):
    link = tarfile.TarInfo("payload/link")
    link.type = kind
    link.linkname = "../../outside"
    packed = _write_archive(tmp_path / "link.tar.zst", [],
                            records=[_record("link", b"")], extra=[(link, b"")])
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        archive.deploy(packed, tmp_path / "deployed")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("case", [
    "duplicate_inventory", "duplicate_payload", "extra_payload", "missing_payload",
    "wrong_size", "wrong_checksum", "file_directory_conflict", "case_collision",
    "second_manifest", "non_payload_member",
])
def test_archive_inventory_must_match_exactly_and_failure_cleans_staging(tmp_path, case):
    files = [("first.txt", b"first verified payload"), ("second.txt", b"second payload")]
    records = [_record(name, data) for name, data in files]
    extra = []
    if case == "duplicate_inventory":
        records.append(records[-1].copy())
    elif case == "duplicate_payload":
        files.append(files[-1])
    elif case == "extra_payload":
        files.append(("extra.txt", b"unexpected"))
    elif case == "missing_payload":
        files.pop()
    elif case == "wrong_size":
        records[-1]["bytes"] += 1
    elif case == "wrong_checksum":
        records[-1]["sha256"] = "0" * 64
    elif case == "file_directory_conflict":
        files.append(("first.txt/child", b"child"))
        records.append(_record(*files[-1]))
    elif case == "case_collision":
        files.append(("FIRST.TXT", b"another"))
        records.append(_record(*files[-1]))
    elif case == "second_manifest":
        extra = [("archive.json", b"{}")]
    elif case == "non_payload_member":
        extra = [("elsewhere.txt", b"outside payload")]
    packed = _write_archive(tmp_path / "invalid.tar.zst", files, records=records, extra=extra)
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        archive.deploy(packed, tmp_path / "deployed")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("version", [0, 2, "1", True])
def test_unsupported_manifest_version_is_rejected(tmp_path, version):
    packed = _write_archive(tmp_path / "future.tar.zst", [("a", b"data")], version=version)
    with pytest.raises(ValueError):
        archive.deploy(packed, tmp_path / "deployed")
    assert not (tmp_path / "deployed").exists()


def test_declared_payload_limit_is_enforced_before_publication(tmp_path):
    packed = _write_archive(tmp_path / "bounded.tar.zst", [("a", b"1234"), ("b", b"5678")])
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        archive.deploy(packed, tmp_path / "too-large", max_bytes=7)
    assert set(tmp_path.iterdir()) == before
    archive.deploy(packed, tmp_path / "fits", max_bytes=8)
    assert _tree(tmp_path / "fits") == {"a": b"1234", "b": b"5678"}


@pytest.mark.parametrize("occupied_by", ["file", "empty_directory", "populated_directory"])
def test_existing_destination_is_never_replaced(snapshot, tmp_path, occupied_by):
    packed = tmp_path / "snapshot.tar.zst"
    archive.pack(snapshot, packed)
    destination = tmp_path / "deployed"
    if occupied_by == "file":
        destination.write_bytes(b"existing file")
    else:
        destination.mkdir()
        if occupied_by == "populated_directory":
            (destination / "keep.txt").write_bytes(b"existing data")
    before = _tree(tmp_path)
    with pytest.raises(FileExistsError):
        archive.deploy(packed, destination)
    assert destination.exists() and _tree(tmp_path) == before


def test_existing_archive_is_never_replaced(snapshot, tmp_path):
    packed = tmp_path / "snapshot.tar.zst"
    packed.write_bytes(b"previous archive")
    before = _tree(tmp_path)
    with pytest.raises(FileExistsError):
        archive.pack(snapshot, packed)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("relative", ["snapshot.tar.zst", "nested/snapshot.tar.zst"])
def test_archive_cannot_be_created_inside_its_source(snapshot, relative):
    before = _tree(snapshot)
    with pytest.raises(ValueError):
        archive.pack(snapshot, snapshot / relative)
    assert _tree(snapshot) == before


@pytest.mark.parametrize("relative", [".env", ".git/config", "incomplete.part", "incomplete.tmp"])
def test_pack_refuses_credentials_repository_metadata_and_incomplete_files(snapshot, tmp_path, relative):
    unexpected = snapshot / relative
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_bytes(b"must remain local")
    before = _tree(snapshot)
    packed = tmp_path / "snapshot.tar.zst"
    with pytest.raises(ValueError):
        archive.pack(snapshot, packed)
    assert not packed.exists() and _tree(snapshot) == before


def test_pack_refuses_source_symlinks_without_following_them(snapshot, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside the selected snapshot")
    try:
        (snapshot / "linked.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks requires privileges on this platform")
    packed = tmp_path / "snapshot.tar.zst"
    with pytest.raises(ValueError):
        archive.pack(snapshot, packed)
    assert not packed.exists() and outside.read_bytes() == b"outside the selected snapshot"
