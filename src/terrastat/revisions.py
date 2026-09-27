"""Read-only comparison of two normalized observation files before a refresh.

Observation identity is the source's exact ``(series_key, period)`` pair. Calendar
dates are not keys: different period labels can map to the same start date.
Comparison preserves missing observations and never rewrites either input.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

_KEYS = ("series_key", "period")
_REQUIRED = (*_KEYS, "date", "value", "flag")


def _scan(path: Path, side: str) -> tuple[pl.LazyFrame, pl.Schema, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{side} observations file does not exist: {path}")
    frame = pl.scan_parquet(path, hive_partitioning=False)
    schema = frame.collect_schema()
    missing = sorted(set(_REQUIRED) - set(schema))
    if missing:
        raise ValueError(f"{side} observations are missing required columns: {', '.join(missing)}")
    for key in _KEYS:
        if schema[key] not in (pl.String, pl.Null):
            raise ValueError(f"{side} observations key {key!r} must contain strings")
    if not schema["value"].is_numeric() and schema["value"] != pl.Null:
        raise ValueError(f"{side} observations value must be numeric or null")
    if schema["flag"] not in (pl.String, pl.Null):
        raise ValueError(f"{side} observations flag must contain strings or nulls")
    if schema["date"] not in (pl.Date, pl.Null) and not isinstance(schema["date"], pl.Datetime):
        raise ValueError(f"{side} observations date must contain dates or nulls")
    counts = frame.select(
        pl.len().alias("rows"),
        pl.any_horizontal(pl.col(key).is_null() for key in _KEYS).sum().alias("null_keys"),
        pl.struct(_KEYS).n_unique().alias("unique_keys"),
    ).collect(engine="streaming").row(0, named=True)
    if counts["null_keys"]:
        raise ValueError(f"{side} observations contain {counts['null_keys']} null (series_key, period) keys")
    if counts["rows"] != counts["unique_keys"]:
        raise ValueError(f"{side} observations contain duplicate (series_key, period) keys")
    return frame, schema, int(counts["rows"])


def compare_observations(old: Path, new: Path) -> dict:
    """Count added, removed and revised observations in normalized Parquet files.

    ``revised`` counts matched keys whose value **or** flag changed, once per row.
    Its two component counts can overlap. Null-to-value changes count; nulls
    remain distinct from NaN, while NaN compares equal to NaN. ``date_revisions``
    and ``dimension_revisions`` count other changes among matched keys separately.
    Columns added/removed or changing type are listed in ``schema_changes``;
    added/removed dimension columns do not turn every row into a revision.

    The query collects only scalar summaries using Polars' streaming engine.
    Exact key validation and the full join still require hash state proportional
    to the number of distinct observation keys; this is not a fixed-memory sort.
    Call with immutable/quiescent inputs, such as the old file and a staged refresh.
    Neither input is modified, and invalid keys/schemas raise ``ValueError``.
    """
    old_frame, old_schema, old_rows = _scan(old, "old")
    new_frame, new_schema, new_rows = _scan(new, "new")
    shared = sorted(set(old_schema) & set(new_schema))
    columns = [name for name in shared if name not in _KEYS]
    positions = {name: i for i, name in enumerate(columns)}
    dimensions = [name for name in columns if name not in ("date", "value", "flag")]
    schema_changes = {
        "added": sorted(set(new_schema) - set(old_schema)),
        "removed": sorted(set(old_schema) - set(new_schema)),
        "types": {name: {"old": str(old_schema[name]), "new": str(new_schema[name])}
                  for name in shared if old_schema[name] != new_schema[name]},
    }

    def project(frame: pl.LazyFrame, side: str) -> pl.LazyFrame:
        # Rename every projected column so source dimension names cannot collide
        # with join suffixes, keys or internal presence markers.
        return frame.select(
            pl.col("series_key").cast(pl.String).alias("_key"),
            pl.col("period").cast(pl.String).alias("_period"),
            *[pl.col(name).alias(f"_{side}_{i}") for i, name in enumerate(columns)],
            pl.lit(True).alias(f"_{side}_present"),
        )

    def changed(name: str) -> pl.Expr:
        left = pl.col(f"_old_{positions[name]}")
        right = pl.col(f"_new_{positions[name]}")
        left_type, right_type = old_schema[name], new_schema[name]
        if name in dimensions and left_type != right_type:
            if left_type == pl.Null:
                return right.is_not_null()
            if right_type == pl.Null:
                return left.is_not_null()
            if not (left_type.is_numeric() and right_type.is_numeric()):
                # A dimension's incompatible type is itself a change; do not
                # coerce string codes like "01" to the numeric code 1.
                return pl.lit(True)
        return left.ne_missing(right)

    joined = project(old_frame, "old").join(
        project(new_frame, "new"), on=["_key", "_period"], how="full", coalesce=True,
    )
    both = pl.col("_old_present").is_not_null() & pl.col("_new_present").is_not_null()
    value_changed, flag_changed = changed("value"), changed("flag")
    dimension_changed = pl.any_horizontal([changed(name) for name in dimensions]) if dimensions else pl.lit(False)
    counts = joined.select(
        pl.col("_old_present").is_null().sum().alias("added"),
        pl.col("_new_present").is_null().sum().alias("removed"),
        (both & (value_changed | flag_changed)).sum().alias("revised"),
        (both & value_changed).sum().alias("value_revisions"),
        (both & flag_changed).sum().alias("flag_revisions"),
        (both & changed("date")).sum().alias("date_revisions"),
        (both & dimension_changed).sum().alias("dimension_revisions"),
    ).collect(engine="streaming").row(0, named=True)
    return {"old_rows": old_rows, "new_rows": new_rows,
            **{name: int(value) for name, value in counts.items()}, "schema_changes": schema_changes}
