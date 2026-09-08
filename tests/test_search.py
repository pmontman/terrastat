"""Finding datasets by subject, including by the code labels buried inside their dimensions."""
import json

import polars as pl
import pytest

from terrastat import search


def _dataset(root, source, did, title, dims, n_series=100, description="", freqs=("A",)):
    d = root / "datasets" / source / did
    d.mkdir(parents=True)
    (d / "dataset.json").write_text(json.dumps({
        "source": source, "dataset_id": did, "title": title, "description": description,
        "dimensions": [{"id": k, "name": n, "codes": c} for k, n, c in dims],
        "n_series": n_series, "n_obs": n_series * 20, "frequencies": list(freqs),
        "time_start": "1990-01-01", "time_end": "2024-01-01", "source_url": f"http://x/{did}",
    }), encoding="utf-8")


@pytest.fixture
def catalogue(tmp_path):
    """Three datasets that between them cover every way a subject search can go wrong."""
    # the one you want, whose title never says "chicken" -- it is a code label
    _dataset(tmp_path, "eurostat", "apro_ec_poulm", "Poultry farming",
             [("animals", "Animals", {"A1": "Laying hens", "A2": "Broiler chickens"})],
             n_series=500, freqs=("M",))
    # a huge unrelated table that merely mentions the word in one of its thousands of codes
    _dataset(tmp_path, "oecd", "BIG__SBS__1.0", "Structural business statistics by size class",
             [("activity", "Activity", {"C1": "Processing and preserving of poultry meat",
                                        "C2": "Manufacture of cement"})],
             n_series=2_000_000)
    # species-level detail, the sort of thing only a label search finds
    _dataset(tmp_path, "oecd", "FISH__INLAND__1.0", "Inland fisheries",
             [("SPECIES", "Species", {"S1": "Tilapias and other cichlids", "S2": "Carps"})],
             n_series=1_665, description="Inland fisheries production in tonnes")
    return tmp_path


def test_a_code_label_finds_a_dataset_its_title_never_mentions(catalogue):
    """`apro_ec_poulm` is called "Poultry farming". Nothing in its title says chicken; the word is
    a code inside its animals dimension, which is exactly the case a title search misses."""
    hits = search.search("chicken", root=catalogue)
    assert hits.get_column("dataset_id").to_list() == ["apro_ec_poulm"]
    r = hits.row(0, named=True)
    assert r["where"] == "labels"
    assert "Broiler chickens" in r["matched_labels"]


def test_relevance_beats_size(catalogue):
    """Searching "poultry" matches both the poultry dataset and a two-million-series business
    table with one poultry NACE code. Ranking by series count puts the wrong one first."""
    hits = search.search("poultry", root=catalogue)
    assert hits.get_column("dataset_id").to_list()[0] == "apro_ec_poulm"
    by = {r["dataset_id"]: r for r in hits.iter_rows(named=True)}
    assert by["apro_ec_poulm"]["relevance"] > by["BIG__SBS__1.0"]["relevance"]
    assert by["BIG__SBS__1.0"]["n_series"] > by["apro_ec_poulm"]["n_series"]   # and it is far bigger


def test_titles_only_gives_up_the_label_matches(catalogue):
    assert search.search("chicken", root=catalogue, in_labels=False).height == 0
    assert search.search("tilapia", root=catalogue, in_labels=False).height == 0
    assert search.search("fisheries", root=catalogue, in_labels=False).height == 1


def test_the_query_is_a_regex_so_alternatives_work(catalogue):
    hits = search.search("tilapia|carps", root=catalogue)
    assert hits.height == 1
    assert "Tilapias and other cichlids" in hits.row(0, named=True)["matched_labels"]


def test_filters_narrow_by_source_frequency_and_size(catalogue):
    assert search.search("poultry", root=catalogue, sources=["oecd"]).height == 1
    assert search.search("poultry", root=catalogue, frequencies=["M"]).height == 1
    assert search.search("poultry", root=catalogue, min_series=1_000_000).height == 1


def test_the_index_is_rebuilt_when_asked_and_reused_otherwise(catalogue):
    p = search.build_index(catalogue)
    assert p.exists()
    before = p.stat().st_mtime_ns
    search.build_index(catalogue)                       # no force: left alone
    assert p.stat().st_mtime_ns == before
    _dataset(catalogue, "eurostat", "new_one", "Sheep and goats", [("a", "A", {"x": "Lambs"})])
    assert search.search("lambs", root=catalogue).height == 0      # stale index, as expected
    search.build_index(catalogue, force=True)
    assert search.search("lambs", root=catalogue).height == 1
