import datetime as dt
import json

import polars as pl


def _fake_fred_dataset(root, rid="10"):
    d = root / "datasets" / "fred" / rid
    d.mkdir(parents=True)
    obs = pl.DataFrame(
        {
            "series_key": ["CPIAUCSL"] * 3 + ["XYZ"] * 2,
            "period": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-01", "2020-02-01"],
            "date": [dt.date(2020, 1, 1), dt.date(2020, 2, 1), dt.date(2020, 3, 1), dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "value": [1.0, 2.0, 3.0, 5.0, 6.0],
            "flag": [None] * 5,
        },
        schema={"series_key": pl.String, "period": pl.String, "date": pl.Date, "value": pl.Float64, "flag": pl.String},
    )
    obs.write_parquet(d / "observations.parquet")
    pl.DataFrame(
        {
            "series_id": ["CPIAUCSL", "XYZ"],
            "title": ["Consumer Price Index", "Other"],
            "frequency_raw": ["Monthly", "Annual"],
            "frequency": ["M", "A"],
            "units": ["Index", "Percent"],
            "seasonal_adjustment": ["Seasonally Adjusted", "Not Seasonally Adjusted"],
            "last_updated": ["2026-01-01T00:00:00Z"] * 2,
            "copyright_id": ["public domain: citation requested", "copyrighted: citation required"],
            "notes": [None, None],
            "n_obs": [3, 2],
            "n_missing": [0, 0],
            "obs_start": ["2020-01-01", "2020-01-01"],
            "obs_end": ["2020-03-01", "2020-02-01"],
        }
    ).write_parquet(d / "series.parquet")
    meta = {
        "source": "fred", "dataset_id": rid, "title": "Consumer Price Index", "license": {}, "origin": {"agencies": ["U.S. Bureau of Labor Statistics"], "us_federal": True},
        "frequencies": ["M", "A"], "dimensions": [], "attributes": [], "retrieved_at": "2026-09-04T00:00:00+00:00", "vintage": "latest",
    }
    (d / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")


def test_fred_tags_attached_and_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    from terrastat import series as series_mod
    from terrastat.tags import series_tags_path, tags_path

    _fake_fred_dataset(tmp_path)
    tags_path().parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name": ["cpi", "usa", "inflation"], "group_id": ["gen", "geo", "gen"], "notes": ["Consumer Price Index", "United States of America", ""],
                  "created": [""] * 3, "popularity": [0] * 3, "series_count": [1] * 3}).write_parquet(tags_path())
    pl.DataFrame({"series_id": ["CPIAUCSL"], "tags": [["cpi", "inflation", "usa"]]}).write_parquet(series_tags_path())
    series_mod._TAGS_CACHE.update(mtime=None, table=None, notes=None)

    files = series_mod.build_dataset_series("fred", "10", force=True)
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("series_uid")
    cpi = df.filter(pl.col("source_id") == "CPIAUCSL").row(0, named=True)
    assert cpi["tags"] == ["cpi", "inflation", "usa", "public domain: citation requested", "sa", "monthly"]
    assert "It is associated with the tags: Consumer Price Index, inflation, United States of America, public domain: citation requested, sa, monthly." in cpi["description"]
    other = df.filter(pl.col("source_id") == "XYZ").row(0, named=True)
    assert other["tags"] == ["copyrighted: citation required", "nsa", "annual"]  # derived only, nothing fetched for it
    assert other["description"].startswith("Time series about Other. It is associated with the tags: copyrighted: citation required, nsa, annual. Frequency: Annual.")
