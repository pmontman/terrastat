import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pyarrow.parquet as pq
import pytest

from terrastat.series import SERIES_SCHEMA


def _fake_series(n: int, source: str, dataset: str, license_id: str, us_fed: bool | None) -> pl.DataFrame:
    rows = []
    for i in range(n):
        k = 30 + i % 10
        rows.append(
            {
                **{c: None for c in SERIES_SCHEMA},
                "series_uid": f"{source}:{dataset}:{i}",
                "source": source,
                "source_id": f"{dataset}:{i}",
                "dataset_id": dataset,
                "dataset_title": "t",
                "title": f"series {i}",
                "description": "d",
                "frequency": "M",
                "license_id": license_id,
                "origin_us_federal": us_fed,
                "origin_agencies": ["x"],
                "tags": [],
                "dimensions": [],
                "n_points": k,
                "n_obs": k,
                "n_flagged": 0,
                "start_date": dt.date(2000, 1, 1),
                "end_date": dt.date(2000, 1, 1),
                "dates": [dt.date(2000, 1, 1)] * k,
                "values": [float(j) for j in range(k)],
                "flags": [None] * k,
            }
        )
    return pl.DataFrame(rows, schema=SERIES_SCHEMA)


def test_export_shards_mix_and_index(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    from terrastat.export import export_shards, iter_shard_batches

    (tmp_path / "series" / "fred" / "freq=M").mkdir(parents=True)
    (tmp_path / "series" / "eurostat" / "freq=M").mkdir(parents=True)
    _fake_series(300, "fred", "10", "fred-public-domain-citation-requested", True).write_parquet(tmp_path / "series/fred/freq=M/10.parquet")
    _fake_series(300, "fred", "11", "fred-copyrighted-citation-required", False).write_parquet(tmp_path / "series/fred/freq=M/11.parquet")
    _fake_series(300, "eurostat", "ds", "eurostat-reuse", None).write_parquet(tmp_path / "series/eurostat/freq=M/ds.parquet")

    m = export_shards("t", public_only=True, min_obs=1, target_mb=0.02, seed=1)
    out = tmp_path / "snapshot" / "t"
    assert m["rows"] == 600  # the copyrighted FRED file is excluded
    assert m["n_shards"] >= 2
    idx = pl.read_parquet(out / "index.parquet")
    assert idx.height == 600 and idx["series_uid"].n_unique() == 600
    # every shard mixes both sources
    per_shard = idx.group_by("shard").agg(pl.col("source").n_unique().alias("s"))
    assert (per_shard["s"] == 2).all()
    shard0 = pl.read_parquet(out / "shard-00000.parquet")
    assert shard0.schema["values"] == pl.List(pl.Float32)
    assert set(shard0.columns) == set(SERIES_SCHEMA)
    n = sum(b.height for b in iter_shard_batches(out, batch_rows=100, columns=["series_uid", "values"]))
    assert n == 600
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["license_id"] == {"fred-public-domain-citation-requested": 300, "eurostat-reuse": 300}


def test_the_licensing_notice_is_loud_when_no_filter_was_applied():
    """The terms already travel in the column, the card, ATTRIBUTIONS.md and the manifest, and a
    person can still publish without reading any of them. The notice is printed by the command
    that made the files, which is the one place it cannot be scrolled past."""
    from terrastat.export import licensing_notice

    mixed = {"selection": {"public_only": False},
             "counts": {"license_id": {"fred-copyrighted": 500, "eurostat-reuse": 100}}}
    text = licensing_notice(mixed)
    assert "NO LICENCE FILTER" in text
    assert "--public-only" in text
    assert "must NOT be redistributed" in text
    assert "500" in text and "100" in text          # the composition is spelled out

    clean = {"selection": {"public_only": True},
             "counts": {"license_id": {"eurostat-reuse": 100}}}
    text = licensing_notice(clean)
    assert "NO LICENCE FILTER" not in text
    assert "REDISTRIBUTABLE SUBSET" in text
    # even the shareable subset still requires attribution, and says so
    assert "attribution" in text.lower()


def test_storage_layout_changes_preserve_shuffled_rows_index_and_missing_values(tmp_path, monkeypatch):
    from terrastat.export import export_shards
    from terrastat.datasets import verify

    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    source = tmp_path / "series/eurostat/freq=M/ds.parquet"
    source.parent.mkdir(parents=True)
    frame = _fake_series(120, "eurostat", "ds", "eurostat-reuse", None)
    frame = frame.with_columns(pl.lit([1., None, float("nan"), -0.]).alias("values"))
    frame.write_parquet(source)
    for name, rows, level in [("small", 16, 3), ("compact", 128, 9)]:
        result = export_shards(name, seed=7, row_group_rows=rows, compression_level=level)
        assert result["parquet"]["row_group_rows"] == rows
        assert verify(tmp_path / "snapshot" / name)["complete"]
    a, b = [tmp_path / "snapshot" / name for name in ("small", "compact")]
    assert pl.read_parquet(a / "shard-00000.parquet").equals(pl.read_parquet(b / "shard-00000.parquet"))
    assert pl.read_parquet(a / "index.parquet").equals(pl.read_parquet(b / "index.parquet"))
    assert pq.ParquetFile(a / "shard-00000.parquet").num_row_groups > pq.ParquetFile(b / "shard-00000.parquet").num_row_groups


@pytest.mark.parametrize("options", [{"row_group_rows": 0}, {"row_group_rows": -1},
                                    {"compression_level": 0}, {"compression_level": 23}])
def test_invalid_storage_settings_fail_before_export_changes_any_files(tmp_path, monkeypatch, options):
    from terrastat.export import export_shards

    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    target = tmp_path / "snapshot/existing/shard-00000.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original")
    with pytest.raises(ValueError):
        export_shards("existing", **options)
    assert target.read_bytes() == b"original"


@pytest.fixture
def existing_snapshot(tmp_path, monkeypatch):
    from terrastat.export import export_shards

    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    source = tmp_path / "series/eurostat/freq=M/ds.parquet"
    source.parent.mkdir(parents=True)
    _fake_series(30, "eurostat", "ds", "eurostat-reuse", None).write_parquet(source)
    export_shards("existing", target_mb=0.004)
    return tmp_path / "snapshot/existing"


def _snapshot_bytes(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("failure", ["empty", "notes", "checksum", "index_rows"])
def test_failed_replacement_preserves_every_original_file(existing_snapshot, monkeypatch, failure):
    from terrastat import export, croissant

    before = _snapshot_bytes(existing_snapshot)
    options = {}
    if failure == "empty":
        options["min_obs"] = 1000
    elif failure == "notes":
        def fail_notes(*args):
            raise OSError("injected documentation write failure")
        monkeypatch.setattr(export, "_write_release_notes", fail_notes)
    elif failure == "checksum":
        original = croissant.write
        def corrupt_after_writing(out, manifest):
            original(out, manifest)
            with (out / manifest["shards"][0]["file"]).open("ab") as stream:
                stream.write(b"corrupt")
        monkeypatch.setattr(croissant, "write", corrupt_after_writing)
    else:
        original = export.write_parquet
        monkeypatch.setattr(export, "write_parquet", lambda df, path: original(df.head(0), path))
    with pytest.raises((OSError, ValueError, RuntimeError)):
        export.export_shards("existing", **options)
    assert _snapshot_bytes(existing_snapshot) == before
    assert not (existing_snapshot.parent / ".existing.export-staging").exists()
    assert not (existing_snapshot.parent / ".existing.export-backup").exists()


def test_successful_replacement_is_complete_and_removes_obsolete_shards(existing_snapshot, monkeypatch):
    from terrastat import export, datasets

    before = _snapshot_bytes(existing_snapshot)
    original = export._verify_export
    verified = []
    def verify_while_original_is_still_published(staging, expected_rows=None):
        assert _snapshot_bytes(existing_snapshot) == before
        original(staging, expected_rows)
        verified.append(True)
    monkeypatch.setattr(export, "_verify_export", verify_while_original_is_still_published)
    result = export.export_shards("existing", per_file_cap=7)
    assert verified and result["rows"] == 7
    assert datasets.verify(existing_snapshot)["complete"]
    assert len(list(existing_snapshot.glob("shard-*.parquet"))) == result["n_shards"] == 1
    assert pl.read_parquet(existing_snapshot / "index.parquet").height == 7
    assert not (existing_snapshot.parent / ".existing.export-backup").exists()


def test_publish_rename_failure_rolls_back_original(existing_snapshot, monkeypatch):
    from terrastat.export import export_shards

    before = _snapshot_bytes(existing_snapshot)
    original = Path.rename
    def fail_publish(path, target):
        if path.name == ".existing.export-staging":
            raise PermissionError("injected locked destination")
        return original(path, target)
    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(PermissionError, match="injected"):
        export_shards("existing", per_file_cap=7)
    assert _snapshot_bytes(existing_snapshot) == before
    assert not (existing_snapshot.parent / ".existing.export-backup").exists()


@pytest.mark.parametrize("phase", ["build", "after_backup", "after_publish"])
def test_killed_export_recovers_on_next_attempt(existing_snapshot, phase):
    from terrastat import export, datasets

    before = _snapshot_bytes(existing_snapshot)
    script = r'''
import os
from pathlib import Path
import sys
from terrastat import export
phase = sys.argv[1]
original = Path.rename
def interrupted_rename(path, target):
    result = original(path, target)
    if ((phase == "after_backup" and Path(target).name == ".existing.export-backup")
            or (phase == "after_publish" and path.name == ".existing.export-staging")):
        os._exit(73)
    return result
Path.rename = interrupted_rename
if phase == "build":
    export._write_release_notes = lambda *args: os._exit(73)
export.export_shards("existing", per_file_cap=7)
'''
    child = subprocess.run([sys.executable, "-c", script, phase], capture_output=True, text=True, timeout=30)
    assert child.returncode == 73, child.stderr
    backup = existing_snapshot.parent / ".existing.export-backup"
    if phase == "after_backup":
        assert not existing_snapshot.exists()
        assert _snapshot_bytes(backup) == before
    elif phase == "build":
        assert _snapshot_bytes(existing_snapshot) == before
    else:
        assert _snapshot_bytes(backup) == before
        assert datasets.verify(existing_snapshot)["complete"]
    published = _snapshot_bytes(existing_snapshot) if existing_snapshot.exists() else before
    # Recovery happens before this intentionally empty new build; no third release is published.
    with pytest.raises(RuntimeError, match="filters leave no series"):
        export.export_shards("existing", min_obs=1000)
    assert _snapshot_bytes(existing_snapshot) == published
    assert not backup.exists()
    assert not (existing_snapshot.parent / ".existing.export-staging").exists()


def test_damaged_published_copy_keeps_recovery_backup(existing_snapshot):
    from terrastat import export
    import shutil

    before = _snapshot_bytes(existing_snapshot)
    backup = existing_snapshot.parent / ".existing.export-backup"
    shutil.copytree(existing_snapshot, backup)
    (existing_snapshot / "index.parquet").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="previous copy retained"):
        export.export_shards("existing")
    assert _snapshot_bytes(backup) == before


def test_second_export_cannot_touch_active_writer_files(existing_snapshot):
    from terrastat.export import export_shards
    from terrastat.locking import exclusive_lock

    before = _snapshot_bytes(existing_snapshot)
    staging = existing_snapshot.parent / ".existing.export-staging"
    staging.mkdir()
    (staging / "in-progress").write_bytes(b"active")
    with exclusive_lock(existing_snapshot.parent / ".existing.export.lock"):
        with pytest.raises(RuntimeError, match="writer lock"):
            export_shards("existing")
    assert (staging / "in-progress").read_bytes() == b"active"
    assert _snapshot_bytes(existing_snapshot) == before


def test_unmanaged_files_are_not_deleted_by_replacement(existing_snapshot):
    from terrastat.export import export_shards

    (existing_snapshot / "research-notes.md").write_text("Keep these notes", encoding="utf-8")
    before = _snapshot_bytes(existing_snapshot)
    with pytest.raises(ValueError, match="unmanaged"):
        export_shards("existing")
    assert _snapshot_bytes(existing_snapshot) == before


@pytest.mark.parametrize("name", ["../outside", "a/b", r"a\b", ".", "..", "CON", "x.", "a:b"])
def test_export_name_cannot_escape_snapshot_root(tmp_path, monkeypatch, name):
    from terrastat.export import export_shards

    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="single portable"):
        export_shards(name)
    assert not (tmp_path / "snapshot").exists()
