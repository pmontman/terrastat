"""Croissant metadata: the structural guarantees a platform relies on when it reads the file."""
import json

import pytest

from terrastat import croissant
from terrastat.series import SERIES_SCHEMA

MANIFEST = {
    "name": "starter",
    "created_at": "2026-09-08T10:00:00+00:00",
    "terrastat_version": "0.1.0",
    "rows": 120_000,
    "bytes": 480_000_000,
    "n_shards": 4,
    "columns": list(SERIES_SCHEMA),
    "selection": {"public_only": True, "min_obs": 60},
    "counts": {
        "source": {"eurostat": 80_000, "oecd": 30_000, "fred": 10_000},
        "frequency": {"A": 60_000, "M": 40_000, "Q": 20_000},
        "license_id": {"eurostat-reuse": 80_000, "oecd-terms": 30_000,
                       "fred-public-domain-citation-requested": 10_000},
    },
}


@pytest.fixture
def rec():
    return croissant.build(MANIFEST)


def test_it_declares_the_version_platforms_match_on(rec):
    assert rec["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert rec["@type"] == "sc:Dataset"
    assert rec["@context"]["cr"] == "http://mlcommons.org/croissant/"
    assert rec["@context"]["rai"] == "http://mlcommons.org/croissant/RAI/"


def test_every_schema_column_is_described(rec):
    """Thirty-six columns is a lot to meet cold; a viewer should be able to explain each one at
    the point the reader sees it."""
    fields = rec["recordSet"][0]["field"]
    assert {f["name"] for f in fields} == set(SERIES_SCHEMA)
    undocumented = [f["name"] for f in fields if not f["description"].strip()]
    assert not undocumented, f"fields with no description: {undocumented}"


def test_every_field_points_at_the_shards_and_names_its_column(rec):
    for f in rec["recordSet"][0]["field"]:
        assert f["source"]["fileSet"]["@id"] == "shards"
        assert f["source"]["extract"]["column"] == f["name"]
        assert f["dataType"].startswith("sc:")


def test_list_columns_are_marked_repeated(rec):
    by = {f["name"]: f for f in rec["recordSet"][0]["field"]}
    for c in ("dates", "values", "flags", "tags", "origin_agencies", "dimensions"):
        assert by[c].get("repeated") is True, c
    assert "repeated" not in by["series_uid"]


def test_the_record_key_is_the_unique_identifier(rec):
    assert rec["recordSet"][0]["key"] == {"@id": "series/series_uid"}


def test_a_mixed_licence_corpus_says_so_rather_than_picking_one(rec):
    """schema.org wants a single licence value. Naming one of three would be a false claim, so
    the field enumerates them and says the filter was applied."""
    lic = rec["license"]
    assert lic.startswith("Mixed, per series.")
    for who in ("Eurostat", "OECD", "FRED"):
        assert who in lic
    assert "redistributed with acknowledgement" in lic


def test_an_unfiltered_snapshot_carries_the_warning_into_the_metadata():
    m = dict(MANIFEST, selection={"public_only": False, "min_obs": 60})
    lic = croissant.build(m)["license"]
    assert "NO LICENCE FILTER WAS APPLIED" in lic
    assert "license_id" in lic


def test_the_two_traps_are_in_the_machine_readable_limitations(rec):
    """Vintages and duplication are the failures that bite quietly. They belong here, not only in
    a document nobody opens."""
    lim = rec["rai:dataLimitations"]
    assert "revision" in lim and "real-time" in lim
    assert "random" in lim and "contaminated" in lim
    assert "80,274" in lim                      # the concrete FRED/OECD overlap


def test_the_whole_record_is_json_serialisable(rec):
    assert json.loads(json.dumps(rec))["name"] == "starter"


def test_it_is_written_beside_the_shards(tmp_path):
    p = croissant.write(tmp_path, MANIFEST)
    assert p.name == "croissant.json"
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["recordSet"][0]["field"][0]["name"] == "series_uid"


def test_a_file_object_carries_a_checksum(tmp_path):
    """The specification requires md5 or sha256 on every cr:FileObject, and the reference
    validator rejects the record without one — a rule my own structural checks missed until
    mlcroissant was run against a real snapshot."""
    idx = tmp_path / "index.parquet"
    idx.write_bytes(b"not really parquet, but hashable")
    rec = json.loads(croissant.write(tmp_path, MANIFEST).read_text(encoding="utf-8"))
    obj = next(d for d in rec["distribution"] if d["@type"] == "cr:FileObject")
    assert obj["sha256"] == croissant.sha256(idx)
    assert len(obj["sha256"]) == 64


def test_the_context_matches_the_official_one(rec):
    """mlcroissant compares the @context key-for-key and warns on any difference; two keys were
    missing until it said so."""
    for key in ("samplingRate", "equivalentProperty", "isLiveDataset", "recordSet", "dataType"):
        assert key in rec["@context"], key
