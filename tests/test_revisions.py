"""Revision reports compare source periods without dropping missing observations."""
import datetime as dt

import polars as pl
import pytest

from terrastat.revisions import compare_observations

_SCHEMA = {"series_key": pl.String, "period": pl.String, "date": pl.Date,
           "value": pl.Float64, "flag": pl.String}


def _frame(rows):
    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def _row(key="one", period="2020-Q1", value=1.0, flag=None, date=dt.date(2020, 1, 1)):
    return key, period, date, value, flag


def _compare(tmp_path, old, new):
    old_path, new_path = tmp_path / "old.parquet", tmp_path / "new.parquet"
    old.write_parquet(old_path)
    new.write_parquet(new_path)
    return compare_observations(old_path, new_path)


def test_reordering_does_not_create_revisions_and_files_are_unchanged(tmp_path):
    old = _frame([_row("one"), _row("two", value=None, flag="c"), _row("three", value=float("nan"))])
    old_path, new_path = tmp_path / "old.parquet", tmp_path / "new.parquet"
    old.write_parquet(old_path)
    old.reverse().write_parquet(new_path)
    before = old_path.read_bytes(), new_path.read_bytes()
    result = compare_observations(old_path, new_path)
    assert result == {
        "old_rows": 3, "new_rows": 3, "added": 0, "removed": 0, "revised": 0,
        "value_revisions": 0, "flag_revisions": 0, "date_revisions": 0,
        "dimension_revisions": 0, "schema_changes": {"added": [], "removed": [], "types": {}},
    }
    assert (old_path.read_bytes(), new_path.read_bytes()) == before


def test_historical_values_and_flags_are_counted_once_per_revised_observation(tmp_path):
    old = _frame([_row("one"), _row("two", value=None), _row("three", value=float("nan")),
                  _row("four", value=4), _row("removed", value=5)])
    new = _frame([_row("added", period="2025-Q4", value=12), _row("four", value=4, flag="c"),
                  _row("three", value=float("nan")), _row("two", value=7), _row("one", value=2, flag="p")])
    result = _compare(tmp_path, old, new)
    assert {k: result[k] for k in ("old_rows", "new_rows", "added", "removed", "revised",
                                   "value_revisions", "flag_revisions")} == {
        "old_rows": 5, "new_rows": 5, "added": 1, "removed": 1,
        "revised": 3, "value_revisions": 2, "flag_revisions": 2,
    }


@pytest.mark.parametrize("old,new,changed", [
    (None, None, 0), (float("nan"), float("nan"), 0), (None, float("nan"), 1),
    (float("nan"), None, 1), (None, 3.0, 1), (3.0, None, 1),
    (float("inf"), float("inf"), 0), (float("inf"), float("-inf"), 1),
])
def test_missing_and_nonfinite_values_are_compared_without_dropping_rows(tmp_path, old, new, changed):
    result = _compare(tmp_path, _frame([_row(value=old)]), _frame([_row(value=new)]))
    assert result["old_rows"] == result["new_rows"] == 1
    assert result["added"] == result["removed"] == 0
    assert result["value_revisions"] == result["revised"] == changed


def test_original_period_labels_distinguish_calendar_date_collisions(tmp_path):
    old = _frame([_row(period="2020", value=1), _row(period="2020-Q1", value=2)])
    new = _frame([_row(period="2020", value=3), _row(period="2020-01", value=2)])
    result = _compare(tmp_path, old, new)
    assert result["added"] == result["removed"] == result["value_revisions"] == 1
    assert result["old_rows"] == result["new_rows"] == 2


def test_date_corrections_and_dimensions_are_separate_from_value_revisions(tmp_path):
    old = _frame([_row("one", date=None), _row("two")]).with_columns(
        pl.Series("geo", ["AT", "BE"]), pl.Series("unit", ["EUR", "EUR"]),
    )
    new = _frame([_row("one"), _row("two")]).with_columns(
        pl.Series("geo", ["FR", "DE"]), pl.Series("unit", ["USD", "EUR"]),
    )
    result = _compare(tmp_path, old, new)
    assert result["date_revisions"] == 1
    assert result["dimension_revisions"] == 2  # Two fields changed in row one, counted once.
    assert result["revised"] == result["value_revisions"] == result["flag_revisions"] == 0


def test_schema_changes_are_reported_without_inventing_observation_revisions(tmp_path):
    old = _frame([_row()]).with_columns(pl.lit("old").alias("old_dimension"))
    new = _frame([_row()]).with_columns(
        pl.lit("new").alias("new_dimension"), pl.col("value").cast(pl.Float32),
    )
    result = _compare(tmp_path, old, new)
    assert result["schema_changes"] == {
        "added": ["new_dimension"], "removed": ["old_dimension"],
        "types": {"value": {"old": "Float64", "new": "Float32"}},
    }
    assert result["revised"] == result["dimension_revisions"] == 0


def test_incompatible_dimension_types_count_as_changes_without_coercing_codes(tmp_path):
    old = _frame([_row()]).with_columns(pl.lit("01").alias("geo"))
    new = _frame([_row()]).with_columns(pl.lit(1, dtype=pl.Int64).alias("geo"))
    result = _compare(tmp_path, old, new)
    assert result["dimension_revisions"] == 1 and result["revised"] == 0
    assert result["schema_changes"]["types"] == {"geo": {"old": "String", "new": "Int64"}}


@pytest.mark.parametrize("old_rows,new_rows", [(0, 0), (0, 2), (2, 0)])
def test_empty_inputs_and_all_missing_observations_are_accounted_for(tmp_path, old_rows, new_rows):
    data = [_row("one", value=None), _row("two", value=None)]
    result = _compare(tmp_path, _frame(data[:old_rows]), _frame(data[:new_rows]))
    assert result["old_rows"] == old_rows and result["new_rows"] == new_rows
    assert result["added"] == new_rows and result["removed"] == old_rows
    assert result["revised"] == 0
    assert all(type(result[k]) is int for k in result if k != "schema_changes")


def test_null_typed_columns_are_supported_for_empty_or_entirely_missing_data(tmp_path):
    old = _frame([_row(value=None, flag=None, date=None)]).with_columns(
        pl.lit(None).alias("value"), pl.lit(None).alias("flag"), pl.lit(None).alias("date"),
    )
    result = _compare(tmp_path, old, _frame([_row(value=None, date=None)]))
    assert result["revised"] == result["date_revisions"] == 0
    assert set(result["schema_changes"]["types"]) == {"date", "value", "flag"}


@pytest.mark.parametrize("side", ["old", "new"])
@pytest.mark.parametrize("missing", list(_SCHEMA))
def test_missing_required_columns_fail_before_comparison(tmp_path, side, missing):
    good = _frame([_row()])
    bad = good.drop(missing)
    with pytest.raises(ValueError, match=f"{side} observations are missing required columns: {missing}"):
        _compare(tmp_path, bad if side == "old" else good, bad if side == "new" else good)


@pytest.mark.parametrize("side", ["old", "new"])
def test_duplicate_keys_are_rejected_even_if_values_are_identical(tmp_path, side):
    good = _frame([_row()])
    bad = pl.concat([good, good])
    with pytest.raises(ValueError, match=f"{side} observations contain duplicate"):
        _compare(tmp_path, bad if side == "old" else good, bad if side == "new" else good)


@pytest.mark.parametrize("side", ["old", "new"])
@pytest.mark.parametrize("key", ["series_key", "period"])
def test_null_keys_are_rejected(tmp_path, side, key):
    good = _frame([_row()])
    bad = good.with_columns(pl.lit(None, dtype=pl.String).alias(key))
    with pytest.raises(ValueError, match=f"{side} observations contain 1 null"):
        _compare(tmp_path, bad if side == "old" else good, bad if side == "new" else good)


@pytest.mark.parametrize("column,value", [("series_key", 1), ("period", 2020),
                                         ("value", "unknown"), ("flag", 1), ("date", "2020-01-01")])
def test_invalid_normalized_column_types_are_rejected(tmp_path, column, value):
    good = _frame([_row()])
    bad = good.with_columns(pl.lit(value).alias(column))
    with pytest.raises(ValueError, match="new observations"):
        _compare(tmp_path, good, bad)


def test_dimension_names_cannot_collide_with_internal_join_columns(tmp_path):
    old = _frame([_row()]).with_columns(
        pl.lit("unchanged").alias("_key"), pl.lit("unchanged").alias("_old_0"),
        pl.lit("old").alias("_new_present"), pl.lit("source").alias("value_right"),
    )
    new = old.with_columns(pl.lit("new").alias("_new_present"))
    result = _compare(tmp_path, old, new)
    assert result["dimension_revisions"] == 1 and result["revised"] == 0


def test_comparison_collects_only_scalar_results_for_many_series(tmp_path, monkeypatch):
    old = pl.DataFrame({"series_key": [f"series-{i}" for i in range(20_000)]}).with_columns(
        pl.lit("2020-Q1").alias("period"), pl.lit(dt.date(2020, 1, 1)).alias("date"),
        pl.lit(1.0).alias("value"), pl.lit(None, dtype=pl.String).alias("flag"),
    )
    new = old.with_columns(pl.when(pl.col("series_key") == "series-1")
                           .then(pl.lit(2.0)).otherwise(pl.col("value")).alias("value"))
    original_collect = pl.LazyFrame.collect
    collected = []

    def scalar_collect(frame, *args, **kwargs):
        result = original_collect(frame, *args, **kwargs)
        assert result.height <= 1, "observation frames must remain inside the lazy query"
        collected.append(result.shape)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", scalar_collect)
    result = _compare(tmp_path, old, new)
    assert collected and result["old_rows"] == result["new_rows"] == 20_000
    assert result["revised"] == result["value_revisions"] == 1
