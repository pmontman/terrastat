import datetime as dt
import json

import polars as pl

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
