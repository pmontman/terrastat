"""Layer 2: one row per time series, metadata expanded into text, partitioned by frequency.

Input is a layer-1 dataset folder; output is ``series/<source>/freq=<F>/<dataset_id>.parquet``.
The observations file is read one row group at a time (it is grouped by series), so memory stays
bounded whatever the dataset size. Series that straddle two row groups are carried over.

Every output file has the same columns (SERIES_SCHEMA), whatever the source, so the whole
``series/`` tree can be opened as one dataset::

    pl.scan_parquet("data/series/**/*.parquet")            # polars
    duckdb.sql("select * from 'data/series/**/*.parquet'")  # duckdb
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from terrastat.sources import get_source
from terrastat.storage import State, dataset_dir, read_json, series_dir, write_parquet
from terrastat.timeparse import canonical_frequency, frequency_of_period

log = logging.getLogger(__name__)

# FRED tag names that follow from fields we already store (no request needed).
# The spellings are FRED's own, from the "freq" tag group, so a derived tag is indistinguishable
# from a fetched one.
FREQ_TAGS = {
    "D": "daily",
    "W": "weekly",
    "BW": "biweekly",
    "M": "monthly",
    "Q": "quarterly",
    "S": "semiannual",
    "A": "annual",
    "P": "5 year (freq)",
}
_TAGS_CACHE: dict = {"mtime": None, "table": None, "notes": None}


def _fred_tags():
    """(series_id -> tags table, tag -> spelled-out form), cached while the file is unchanged."""
    from terrastat.tags import load_series_tags, series_tags_path, tag_notes

    p = series_tags_path()
    mtime = p.stat().st_mtime if p.exists() else None
    if _TAGS_CACHE["mtime"] != mtime:
        _TAGS_CACHE.update(mtime=mtime, table=load_series_tags(), notes=tag_notes())
    return _TAGS_CACHE["table"], _TAGS_CACHE["notes"]

DIM_STRUCT = pl.Struct({"id": pl.String, "name": pl.String, "code": pl.String, "label": pl.String})

SERIES_SCHEMA = {
    "series_uid": pl.String,  # "<source>:<source_id>", globally unique
    "source": pl.String,
    "source_id": pl.String,  # FRED series id; "<dataset>:<sdmx key>" for Eurostat and OECD
    "dataset_id": pl.String,
    "dataset_title": pl.String,
    "title": pl.String,  # FRED analyst title; for SDMX sources the dataset title
    "description": pl.String,  # one readable paragraph built from all metadata (for text encoders)
    "frequency": pl.String,  # canonical D W BW M Q S A P OTHER
    "frequency_raw": pl.String,
    "units": pl.String,
    "seasonal_adjustment": pl.String,
    "geo": pl.String,  # code as given by the source (null for FRED)
    "geo_label": pl.String,
    "dimensions": pl.List(DIM_STRUCT),  # every SDMX dimension: id, human name, code, label
    "tags": pl.List(pl.String),  # FRED tags when fetched (not in this version); empty otherwise
    "notes": pl.String,
    "license_id": pl.String,
    "license_name": pl.String,
    "license_url": pl.String,
    "license_detail": pl.String,  # raw status string from the source (FRED copyright_id)
    "attribution": pl.String,
    "license_notes": pl.String,
    "origin_agencies": pl.List(pl.String),
    "origin_us_federal": pl.Boolean,  # heuristic, FRED only; null elsewhere
    "source_url": pl.String,
    "last_updated": pl.String,
    "retrieved_at": pl.String,
    "vintage": pl.String,
    "start_date": pl.Date,
    "end_date": pl.Date,
    "n_points": pl.Int64,  # length of the dates/values/flags lists
    "n_obs": pl.Int64,  # non-missing values (a flagged point can carry a null value)
    "n_flagged": pl.Int64,  # points carrying a status flag
    "dates": pl.List(pl.Date),
    "values": pl.List(pl.Float64),
    "flags": pl.List(pl.String),
}

DIM_NAME_UNITS = {"unit", "unit_measure", "units"}
DIM_NAME_SADJ = {"s_adj", "adjustment", "seasonal_adjustment"}
DIM_NAME_GEO = {"geo", "ref_area", "location", "country"}
DIM_NAME_FREQ = {"freq", "frequency"}
NON_DIM_COLS = {"unit_mult", "decimals", "conf_status"}


def build_dataset_series(source_name: str, dataset_id: str, min_obs: int = 1, force: bool = False) -> list[Path]:
    """Build the layer-2 files of one dataset. Returns the paths written."""
    d = dataset_dir(source_name, dataset_id)
    meta = read_json(d / "dataset.json")
    obs_path = d / "observations.parquet"
    state = State(f"{source_name}-series")
    if state.status(dataset_id) == "done" and not force:
        return []
    if not obs_path.exists():
        state.mark(dataset_id, "done", files=[], n_series=0)
        return []
    src = get_source(source_name)
    series_meta = _load_series_meta(d)
    dims_meta = {dm["id"]: dm for dm in meta.get("dimensions", [])}

    pf = pq.ParquetFile(obs_path)
    key_cols = [c for c in pf.schema_arrow.names if c not in ("series_key", "period", "date", "value", "flag")]
    parts: dict[str, list[pl.DataFrame]] = {}
    carry: pl.DataFrame | None = None
    n_written = 0
    for i in range(pf.num_row_groups):
        chunk = pl.from_arrow(pf.read_row_group(i))
        if carry is not None:
            chunk = pl.concat([carry, chunk], how="vertical_relaxed")
        if i < pf.num_row_groups - 1:
            last_key = chunk.get_column("series_key")[-1]
            carry = chunk.filter(pl.col("series_key") == last_key)
            chunk = chunk.filter(pl.col("series_key") != last_key)
        else:
            carry = None
        if chunk.is_empty():
            continue
        rows = _aggregate(chunk, key_cols).filter(pl.col("n_obs") >= min_obs)
        if rows.is_empty():
            continue
        out = _decorate(rows, meta, src, series_meta, dims_meta, key_cols)
        for key, g in out.partition_by("frequency", as_dict=True).items():
            f = key[0] if isinstance(key, tuple) else key
            parts.setdefault(f or "OTHER", []).append(g)
            n_written += g.height
    written = []
    for f, frames in parts.items():
        df = pl.concat(frames, how="vertical_relaxed").select(list(SERIES_SCHEMA))
        path = series_dir(source_name) / f"freq={f}" / f"{dataset_id}.parquet"
        write_parquet(df, path)
        written.append(path)
    for old in series_dir(source_name).glob(f"freq=*/{dataset_id}.parquet"):
        if old not in written:
            old.unlink()
    state.mark(dataset_id, "done", files=[str(p) for p in written], n_series=n_written)
    return written


def build_series(source_name: str, dataset_ids: list[str] | None = None, min_obs: int = 1, force: bool = False) -> dict:
    root = dataset_dir(source_name, "_").parent
    if dataset_ids:
        ids = list(dataset_ids)
    elif root.exists():
        ids = sorted(p.name for p in root.iterdir() if (p / "dataset.json").exists())
    else:
        ids = []
    summary = {"datasets": 0, "files": 0, "skipped": 0, "failed": 0}
    for did in tqdm(ids, unit="dataset", dynamic_ncols=True, desc=f"{source_name} series"):
        try:
            files = build_dataset_series(source_name, did, min_obs=min_obs, force=force)
        except Exception as exc:  # noqa: BLE001
            log.error("series build failed for %s: %s", did, exc)
            summary["failed"] += 1
            continue
        if files:
            summary["datasets"] += 1
            summary["files"] += len(files)
        else:
            summary["skipped"] += 1
    return summary


# -- helpers ---------------------------------------------------------------------------------------


def _load_series_meta(d: Path) -> pl.DataFrame | None:
    p = d / "series.parquet"
    return pl.read_parquet(p) if p.exists() else None


def _aggregate(chunk: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    chunk = chunk.sort(["series_key", "date"])
    aggs = [
        pl.col("date").alias("dates"),
        pl.col("value").alias("values"),
        pl.col("flag").alias("flags"),
        pl.len().cast(pl.Int64).alias("n_points"),
        pl.col("value").is_not_null().sum().cast(pl.Int64).alias("n_obs"),
        pl.col("flag").is_not_null().sum().cast(pl.Int64).alias("n_flagged"),
        pl.col("date").min().alias("start_date"),
        pl.col("date").max().alias("end_date"),
        pl.col("period").first().alias("_period0"),
    ]
    aggs += [pl.col(c).first().alias(c) for c in key_cols]
    return chunk.group_by("series_key", maintain_order=True).agg(aggs)


def _const_list(name: str, n: int, value: list, dtype) -> pl.Series:
    return pl.Series(name, [value] * n, dtype=pl.List(dtype))


def _decorate(rows: pl.DataFrame, meta: dict, src, series_meta, dims_meta: dict, key_cols: list[str]) -> pl.DataFrame:
    source = meta["source"]
    dataset_id = meta["dataset_id"]
    lic = meta.get("license", {})
    origin = meta.get("origin", {})
    agencies = [a for a in (origin.get("agencies") or []) if a]
    us_fed = origin.get("us_federal")
    n = rows.height

    if series_meta is not None:  # FRED: per-series metadata and licence
        sm = series_meta.select("series_id", "title", "frequency_raw", "frequency", "units", "seasonal_adjustment", "last_updated", "copyright_id", "notes").rename({"series_id": "series_key"})
        rows = rows.join(sm, on="series_key", how="left")
        lic_rows = [
            src.license_for_series(meta, r)
            for r in rows.select(pl.col("series_key").alias("series_id"), "title", "copyright_id").iter_rows(named=True)
        ]
        # tags: fetched ones (terrastat tags fred) plus the three groups derivable from the metadata
        tag_table, notes = _fred_tags()
        if tag_table is not None:
            rows = rows.join(tag_table.rename({"series_id": "series_key", "tags": "_tags_fetched"}), on="series_key", how="left")
        else:
            rows = rows.with_columns(pl.lit(None, pl.List(pl.String)).alias("_tags_fetched"))
        derived = pl.concat_list(
            [
                pl.col("copyright_id").str.to_lowercase(),
                pl.when(pl.col("seasonal_adjustment").str.contains("(?i)not seasonally")).then(pl.lit("nsa"))
                .when(pl.col("seasonal_adjustment").str.contains("(?i)seasonally adjusted")).then(pl.lit("sa"))
                .otherwise(pl.lit(None, pl.String)),
                pl.col("frequency").replace_strict(FREQ_TAGS, default=None, return_dtype=pl.String),
            ]
        ).list.drop_nulls()
        rows = rows.with_columns(pl.concat_list([pl.col("_tags_fetched").fill_null([]), derived]).alias("tags"))
        if notes:
            expanded = pl.col("tags").list.eval(pl.element().replace_strict(notes, default=pl.element(), return_dtype=pl.String)).list.join(", ")
        else:
            expanded = pl.col("tags").list.join(", ")
        tags_clause = pl.when(pl.col("tags").list.len() > 0).then(pl.format(" It is associated with the tags: {}.", expanded)).otherwise(pl.lit(""))
        out = rows.with_columns(
            pl.Series("license_id", [x["id"] for x in lic_rows], pl.String),
            pl.Series("license_name", [x["name"] for x in lic_rows], pl.String),
            pl.Series("license_url", [x["url"] for x in lic_rows], pl.String),
            pl.Series("license_detail", [x.get("detail") for x in lic_rows], pl.String),
            pl.Series("attribution", [x["attribution"] for x in lic_rows], pl.String),
            pl.Series("license_notes", [x["notes"] for x in lic_rows], pl.String),
            pl.col("series_key").alias("source_id"),
            pl.lit(None, pl.String).alias("geo"),
            pl.lit(None, pl.String).alias("geo_label"),
            pl.Series("dimensions", [[]] * n, dtype=pl.List(DIM_STRUCT)),
            (pl.lit("https://fred.stlouisfed.org/series/") + pl.col("series_key")).alias("source_url"),
            pl.format(
                "Time series about {}.{} Frequency: {}. Units: {}. {}. Source: {}.",
                pl.col("title").fill_null(""),
                tags_clause,
                pl.col("frequency_raw").fill_null(""),
                pl.col("units").fill_null(""),
                pl.col("seasonal_adjustment").fill_null(""),
                pl.lit("; ".join(agencies) if agencies else "FRED"),
            ).alias("description"),
        )
    else:  # SDMX sources: dimensions carry the metadata
        dim_ids = [c for c in key_cols if c in dims_meta or c.lower() not in NON_DIM_COLS]
        dim_names = {c: dims_meta.get(c, {}).get("name") or c for c in dim_ids}
        dim_codes = {c: dims_meta.get(c, {}).get("codes") or {} for c in dim_ids}
        out = rows
        for c in dim_ids:
            codes = {k: (v or k) for k, v in dim_codes[c].items()}
            expr = pl.col(c).replace_strict(codes, default=pl.col(c), return_dtype=pl.String) if codes else pl.col(c)
            out = out.with_columns(expr.fill_null("").alias(f"_lbl_{c}"))
        freq_col = next((c for c in dim_ids if c.lower() in DIM_NAME_FREQ), None)
        unit_col = next((c for c in dim_ids if c.lower() in DIM_NAME_UNITS), None)
        sadj_col = next((c for c in dim_ids if c.lower() in DIM_NAME_SADJ), None)
        geo_col = next((c for c in dim_ids if c.lower() in DIM_NAME_GEO), None)
        if dim_ids:
            dim_struct = pl.concat_list(
                [pl.struct(id=pl.lit(c), name=pl.lit(dim_names[c]), code=pl.col(c).fill_null(""), label=pl.col(f"_lbl_{c}")) for c in dim_ids]
            ).cast(pl.List(DIM_STRUCT))
        else:
            dim_struct = pl.Series("dimensions", [[]] * n, dtype=pl.List(DIM_STRUCT))
        desc_parts = [pl.lit(meta.get("title", "") + ".")]
        for c in dim_ids:
            if c == freq_col:
                continue
            desc_parts.append(pl.format(" {}: {}.", pl.lit(dim_names[c]), pl.col(f"_lbl_{c}")))
        if freq_col:
            fmap = {f: canonical_frequency(f, source) for f in out.get_column(freq_col).unique().to_list() if f is not None}
            freq_expr = pl.col(freq_col).replace_strict(fmap, default="OTHER", return_dtype=pl.String) if fmap else pl.lit("OTHER")
        else:
            pmap = {p: frequency_of_period(p) for p in out.get_column("_period0").unique().to_list() if p is not None}
            freq_expr = pl.col("_period0").replace_strict(pmap, default="OTHER", return_dtype=pl.String) if pmap else pl.lit("OTHER")
        out = out.with_columns(
            dim_struct.alias("dimensions"),
            (pl.lit(f"{dataset_id}:") + pl.col("series_key")).alias("source_id"),
            pl.lit(meta.get("title", "")).alias("title"),
            (pl.col(freq_col) if freq_col else pl.lit(None, pl.String)).alias("frequency_raw"),
            (pl.col(f"_lbl_{unit_col}") if unit_col else pl.lit(None, pl.String)).alias("units"),
            (pl.col(f"_lbl_{sadj_col}") if sadj_col else pl.lit(None, pl.String)).alias("seasonal_adjustment"),
            (pl.col(geo_col) if geo_col else pl.lit(None, pl.String)).alias("geo"),
            (pl.col(f"_lbl_{geo_col}") if geo_col else pl.lit(None, pl.String)).alias("geo_label"),
            pl.lit(None, pl.String).alias("notes"),
            pl.lit(meta.get("last_updated"), pl.String).alias("last_updated"),
            pl.lit(lic.get("id")).alias("license_id"),
            pl.lit(lic.get("name")).alias("license_name"),
            pl.lit(lic.get("url")).alias("license_url"),
            pl.lit(None, pl.String).alias("license_detail"),
            pl.lit(lic.get("attribution")).alias("attribution"),
            pl.lit(lic.get("notes")).alias("license_notes"),
            pl.lit(meta.get("source_url")).alias("source_url"),
            freq_expr.alias("frequency"),
            pl.concat_str(desc_parts, separator="").alias("description"),
        )
        freq_text = pl.col(f"_lbl_{freq_col}") if freq_col else pl.col("frequency")
        out = out.with_columns(pl.format("{} Frequency: {}.", pl.col("description"), freq_text).alias("description"))

    out = out.with_columns(
        (pl.lit(f"{source}:") + pl.col("source_id")).alias("series_uid"),
        pl.lit(source).alias("source"),
        pl.lit(dataset_id).alias("dataset_id"),
        pl.lit(meta.get("title", "")).alias("dataset_title"),
        (pl.col("tags") if "tags" in out.columns else _const_list("tags", n, [], pl.String)),
        _const_list("origin_agencies", n, agencies, pl.String),
        pl.lit(us_fed, pl.Boolean).alias("origin_us_federal"),
        pl.lit(meta.get("retrieved_at")).alias("retrieved_at"),
        pl.lit(meta.get("vintage", "latest")).alias("vintage"),
    )
    for col, dtype in SERIES_SCHEMA.items():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype).alias(col))
    return out.select([pl.col(c).cast(t) for c, t in SERIES_SCHEMA.items()])
