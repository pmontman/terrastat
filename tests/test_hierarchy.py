"""Aggregation structure: series that are exact sums of other series, and when they are not."""
import datetime as dt
import json

import polars as pl
import pytest

from terrastat import hierarchy

SCHEMA = {
    "series_uid": pl.String, "source": pl.String, "dataset_id": pl.String, "frequency": pl.String,
    "units": pl.String, "start_date": pl.Date, "end_date": pl.Date,
    "n_points": pl.Int64, "n_obs": pl.Int64, "n_flagged": pl.Int64,
    "dimensions": pl.List(pl.Struct({"id": pl.String, "name": pl.String,
                                     "code": pl.String, "label": pl.String})),
    "dates": pl.List(pl.Date), "values": pl.List(pl.Float64), "flags": pl.List(pl.String),
}
DATES = [dt.date(2020 + i, 1, 1) for i in range(4)]


def _row(uid, geo, sex, values, units="Number", dataset="d1"):
    return {
        "series_uid": uid, "source": "eurostat", "dataset_id": dataset, "frequency": "A",
        "units": units, "start_date": DATES[0], "end_date": DATES[-1],
        "n_points": len(values), "n_obs": len(values), "n_flagged": 0,
        "dimensions": [
            {"id": "geo", "name": "Geo", "code": geo, "label": geo},
            {"id": "sex", "name": "Sex", "code": sex,
             "label": {"M": "Males", "F": "Females", "T": "Total"}[sex]},
        ],
        "dates": DATES, "values": values, "flags": [None] * len(values),
    }


def test_total_codes_are_recognised_by_code_or_by_label():
    assert hierarchy.is_total("_T") and hierarchy.is_total("TOTAL") and hierarchy.is_total("T")
    assert hierarchy.is_total("XX", "Total")
    assert hierarchy.is_total("ZZ", "All items")
    assert not hierarchy.is_total("M", "Males")
    assert not hierarchy.is_total("TR", "Türkiye")        # a country code that merely starts with T


def test_additivity_is_judged_conservatively():
    for u in ("Number", "Person", "Thousand tonnes", "Million euro", "Head (animal)"):
        assert hierarchy.is_additive(u) is True
    for u in ("Percentage", "Index, 2015=100", "Rate", "Average", "Ratio",
              "Percentage of GDP", "Purchasing power standard (PPS)"):
        assert hierarchy.is_additive(u) is False, u
    assert hierarchy.is_additive(None) is None


def test_code_levels_finds_nesting_without_a_lookup_table():
    """NUTS, NACE and COICOP encode depth in the code itself, so nesting is recoverable from the
    codelist alone. Counting proper prefixes is linear per code; the pairwise version does not
    finish on a geo dimension holding thousands of codes."""
    lv = hierarchy.code_levels(["DE", "DE1", "DE11", "DE12", "FR", "_T"])
    assert lv["DE"] == 0 and lv["DE1"] == 1 and lv["DE11"] == 2
    assert lv["FR"] == 0
    assert "_T" not in lv                                  # the total is not part of the tree


@pytest.fixture
def layer(tmp_path):
    """Two datasets with identical structure and different units: one summable, one not.

    counts     males + females = total, exactly
    shares     the three are percentages, so summing the children roughly doubles the parent
    """
    rows = []
    for geo in ("AT", "BE"):
        rows += [
            _row(f"c:{geo}:M", geo, "M", [10.0, 11.0, 12.0, 13.0]),
            _row(f"c:{geo}:F", geo, "F", [20.0, 21.0, 22.0, 23.0]),
            _row(f"c:{geo}:T", geo, "T", [30.0, 32.0, 34.0, 36.0]),
            _row(f"s:{geo}:M", geo, "M", [40.0, 41.0, 42.0, 43.0], "Percentage", "shares"),
            _row(f"s:{geo}:F", geo, "F", [60.0, 59.0, 58.0, 57.0], "Percentage", "shares"),
            _row(f"s:{geo}:T", geo, "T", [50.0, 50.0, 50.0, 50.0], "Percentage", "shares"),
        ]
    d = tmp_path / "series" / "eurostat" / "freq=A"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(d / "x.parquet")
    for did, unit in (("d1", "Number"), ("shares", "Percentage")):
        dd = tmp_path / "datasets" / "eurostat" / did
        dd.mkdir(parents=True)
        (dd / "dataset.json").write_text(json.dumps({
            "source": "eurostat", "dataset_id": did, "title": did, "n_series": 6,
            "dimensions": [
                {"id": "geo", "name": "Geo", "codes": {"AT": "Austria", "BE": "Belgium"}},
                {"id": "sex", "name": "Sex", "codes": {"M": "Males", "F": "Females", "T": "Total"}},
                {"id": "unit", "name": "Unit", "codes": {"U": unit}},
            ],
        }), encoding="utf-8")
    return tmp_path


def test_aggregation_sets_pairs_each_total_with_its_own_siblings(layer):
    """Grouping is by every dimension *except* the one holding the total, so Austria's total is
    matched with Austria's males and females and not with Belgium's."""
    s = hierarchy.aggregation_sets("d1", "sex", root=layer / "series")
    assert s.height == 2                                   # one group per geo
    r = s.filter(pl.col("parent_uid") == "c:AT:T").row(0, named=True)
    assert r["n_children"] == 2
    assert sorted(r["child_uids"]) == ["c:AT:F", "c:AT:M"]
    assert r["additive"] is True


def test_the_identity_is_verified_not_assumed(layer):
    r = hierarchy.check_identity(hierarchy.aggregation_sets("d1", "sex", root=layer / "series"),
                                 root=layer / "series")
    assert r.height == 2
    assert r.get_column("holds").all()
    assert r.get_column("max_rel_error").max() == pytest.approx(0.0)


def test_percentages_have_the_structure_but_not_the_arithmetic(layer):
    """The most common unit in the corpus is Percentage. Those datasets carry totals that look
    exactly like sums and are not: adding the children roughly doubles the parent."""
    s = hierarchy.aggregation_sets("shares", "sex", root=layer / "series")
    assert s.height == 2
    assert s.get_column("additive").to_list() == [False, False]
    r = hierarchy.check_identity(s, root=layer / "series")
    assert not r.get_column("holds").any()
    assert r.get_column("max_rel_error").min() > 0.5       # 40 + 60 against a parent of 50


def test_hierarchy_table_separates_structure_from_summability(layer):
    t = hierarchy.hierarchy_table(root=layer)
    by = {r["dataset_id"]: r for r in t.iter_rows(named=True)}
    assert by["d1"]["total_dims"] == "sex" and by["d1"]["additive"] is True
    assert by["shares"]["total_dims"] == "sex" and by["shares"]["additive"] is False


def _dataset_with_unit_codes(tmp_path, dataset_id, unit_codes):
    dd = tmp_path / "datasets" / "eurostat" / dataset_id
    dd.mkdir(parents=True)
    (dd / "dataset.json").write_text(json.dumps({
        "source": "eurostat", "dataset_id": dataset_id, "title": dataset_id, "n_series": 2,
        "dimensions": [
            {"id": "sex", "name": "Sex", "codes": {"M": "Males", "F": "Females", "T": "Total"}},
            {"id": "unit", "name": "Unit", "codes": unit_codes},
        ],
    }), encoding="utf-8")


def test_additive_is_unknown_not_true_when_every_unit_is_unlabelled(tmp_path):
    """A codelist entry can be blank for a deprecated or untranslated code without the unit
    dimension going missing. If every unit found is like that, the honest answer is 'we don't
    know', not 'yes, sum away' — ``all([])`` being vacuously true was silently flipping this."""
    _dataset_with_unit_codes(tmp_path, "blank_units", {"X1": "", "X2": ""})
    t = hierarchy.hierarchy_table(root=tmp_path)
    row = t.filter(pl.col("dataset_id") == "blank_units").row(0, named=True)
    assert row["additive"] is None


def test_additive_stays_false_when_known_and_unknown_units_mix(tmp_path):
    """One additive unit next to an unlabelled one is not enough to call the dataset additive.
    This is the existing, intentionally unchanged behaviour — locked in so it isn't 'fixed' by
    accident alongside the all-unknown case above."""
    _dataset_with_unit_codes(tmp_path, "mixed_units", {"NR": "Number", "X2": ""})
    t = hierarchy.hierarchy_table(root=tmp_path)
    row = t.filter(pl.col("dataset_id") == "mixed_units").row(0, named=True)
    assert row["additive"] is False


def _row3(uid, geo, code, label, values, dataset="multi"):
    return {
        "series_uid": uid, "source": "eurostat", "dataset_id": dataset, "frequency": "A",
        "units": "Number", "start_date": DATES[0], "end_date": DATES[-1],
        "n_points": len(values), "n_obs": len(values), "n_flagged": 0,
        "dimensions": [
            {"id": "geo", "name": "Geo", "code": geo, "label": geo},
            {"id": "prod", "name": "Product", "code": code, "label": label},
        ],
        "dates": DATES, "values": values, "flags": [None] * len(values),
    }


@pytest.fixture
def nested_totals(tmp_path):
    """A dimension carrying several codes that all read as totals, which is what 8.8% of
    total-carrying dimensions actually look like. Eurostat's `apri_pi00_outm` offers "Total crop
    including fruit", "Total crop excluding fruit", "Total animal" and "Total agricultural goods"
    in one dimension."""
    rows = [
        _row3("p:apples", "AT", "0111", "Apples", [1.0, 1.0, 1.0, 1.0]),
        _row3("p:pears", "AT", "0112", "Pears", [2.0, 2.0, 2.0, 2.0]),
        _row3("p:tot_crop_inc", "AT", "100000", "Total crop, including fruit", [3.0, 3.0, 3.0, 3.0]),
        _row3("p:tot_crop_exc", "AT", "101000", "Total crop, excluding fruit", [9.0, 9.0, 9.0, 9.0]),
        _row3("p:tot_agri", "AT", "140000", "Total agricultural goods", [99.0, 99.0, 99.0, 99.0]),
    ]
    d = tmp_path / "series" / "eurostat" / "freq=A"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(d / "x.parquet")
    return tmp_path


def test_several_total_looking_codes_are_declined_rather_than_guessed(nested_totals):
    """Keeping whichever came last would name a wrong parent *and* drop the other totals from the
    children, so the group would be scored against an incomplete sum."""
    s = hierarchy.aggregation_sets("multi", "prod", root=nested_totals / "series")
    assert s.height == 0


def test_a_reserved_code_wins_over_free_text_totals(tmp_path):
    """`_T` is the SDMX convention and unambiguous, so it resolves a group that free-text labels
    alone could not."""
    rows = [
        _row3("q:a", "AT", "A1", "Apples", [1.0] * 4, dataset="mixed"),
        _row3("q:b", "AT", "A2", "Pears", [2.0] * 4, dataset="mixed"),
        _row3("q:t", "AT", "_T", "Total", [3.0] * 4, dataset="mixed"),
        _row3("q:x", "AT", "X9", "Total of something else", [7.0] * 4, dataset="mixed"),
    ]
    d = tmp_path / "series" / "eurostat" / "freq=A"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(d / "x.parquet")
    s = hierarchy.aggregation_sets("mixed", "prod", root=tmp_path / "series")
    assert s.height == 1
    assert s.row(0, named=True)["parent_uid"] == "q:t"
