"""Refreshing cannot interpret malformed or incomplete payloads as removals."""
import gzip
import json

import polars as pl
import pytest

from terrastat.http import TooLarge
from terrastat.sources.base import DatasetRef
from terrastat.sources.eurostat import EurostatSource, _tidy_tsv
from terrastat.sources.fred import FredSource
from terrastat.sources.oecd import OecdSource, _tidy_csv
from tests.test_eurostat_tidy import TSV
from tests.test_oecd_split import HEADER, FakeClient, _rows, _seed_structure


def _gz(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(text)
    return path


@pytest.mark.parametrize("content", [TSV.replace("1.5 ", "invalid "),
                                     TSV.replace("1.5 \t: \t2.5 p", "1.5 \t:"),
                                     TSV.replace("1.5 \t: \t2.5 p", "1.5 \t: \t2.5 p\textra")])
def test_eurostat_rejects_invalid_number_and_ragged_rows(tmp_path, content):
    raw = _gz(tmp_path / "data.gz", content)
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        _tidy_tsv(raw, tmp_path / "out.parquet", strict=True)
    assert not (tmp_path / "out.parquet").exists()


def test_eurostat_strict_preserves_sparse_missingness_and_flags(tmp_path):
    raw = _gz(tmp_path / "data.gz", TSV)
    _tidy_tsv(raw, tmp_path / "strict.parquet", strict=True)
    _tidy_tsv(raw, tmp_path / "normal.parquet")
    assert pl.read_parquet(tmp_path / "strict.parquet").equals(pl.read_parquet(tmp_path / "normal.parquet"))


@pytest.mark.parametrize("failure", ["structure", "labels"])
def test_eurostat_refresh_does_not_silently_discard_metadata(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    source = EurostatSource()
    source.strict_refresh = True
    raw = _gz(tmp_path / "raw/eurostat/demo/data.gz", TSV)
    monkeypatch.setattr(source, "fetch_raw", lambda *a: [raw])
    def fail(*a):
        raise OSError("metadata unavailable")
    monkeypatch.setattr(source, "_dsd", fail if failure == "structure" else
                        lambda *a: {"dimensions": [{"id": "geo", "codelist": "ESTAT/GEO/1"}], "attributes": []})
    monkeypatch.setattr(source, "_codelist", fail)
    with pytest.raises(OSError, match="metadata unavailable"):
        source.fetch_and_tidy(DatasetRef("eurostat", "demo", "Demo"), None)
    assert not (tmp_path / "datasets/eurostat/demo/dataset.json").exists()


@pytest.mark.parametrize("body", [_rows("AT").replace("1.5", "invalid"), _rows("AT") + "bad,row\n"])
def test_oecd_rejects_invalid_or_malformed_rows(tmp_path, body):
    raw = _gz(tmp_path / "data.gz", HEADER + body)
    with pytest.raises(Exception):
        _tidy_csv(raw, tmp_path / "out.parquet", strict=True)
    assert not (tmp_path / "out.parquet").exists()


def test_oecd_refuses_to_skip_an_oversized_slice(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    _seed_structure(tmp_path, ["AT", "DE"])
    source = OecdSource()
    source.strict_refresh = True
    ref = DatasetRef("oecd", "X__Y__1.0", "Demo", extra={"agency": "X", "flow": "Y", "version": "1.0"})
    with pytest.raises(TooLarge, match="partial refresh"):
        source.fetch_raw(ref, FakeClient(["AT", "DE"], oversized=["DE"]))


@pytest.mark.parametrize("value", [None, ".", "invalid"])
def test_fred_distinguishes_known_missing_tokens_from_bad_numbers(tmp_path, monkeypatch, value):
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    source = FredSource()
    source.strict_refresh = True
    page = {"series": [{"series_id": "TEST", "frequency": "Quarterly", "copyright_id": "Public Domain",
                         "observations": [{"date": "2020-01-01", "value": "1.0"},
                                          {"date": "2020-04-01", "value": value}]}]}
    raw = _gz(tmp_path / "raw/fred/1/page.gz", json.dumps(page))
    monkeypatch.setattr(source, "_download_pages", lambda *a: [raw])
    if value == "invalid":
        with pytest.raises(ValueError, match="Invalid FRED"):
            source.fetch_and_tidy(DatasetRef("fred", "1", "Demo"), None)
    else:
        source.fetch_and_tidy(DatasetRef("fred", "1", "Demo"), None)
        assert pl.read_parquet(tmp_path / "datasets/fred/1/observations.parquet").height == 1
