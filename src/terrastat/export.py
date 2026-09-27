"""Training snapshots: the series view repacked into fixed-size, globally shuffled Parquet shards.

Why shards: a model-training loop reads one shard at a time (a few hundred MB at most), samples
windows from it, and moves on, so memory stays flat on a laptop and any number of workers can
each take a different shard. Fixed size also makes the snapshot easy to publish: Hugging Face
``datasets`` loads a folder of Parquet shards directly, streaming or not.

Layout of ``data/snapshot/<name>/``::

    shard-00000.parquet ... shard-NNNNN.parquet   same columns as the series view (values as float32 by default)
    index.parquet      series_uid -> shard, row, source, dataset_id, frequency, license_id, n_points, dates
    manifest.json      filters, seed, counts by source / frequency / licence, per-shard rows and bytes, schema

Every shard is a uniform random sample of the whole selection (rows are assigned to shards at
random, then shuffled inside each shard), so a model that sees shards in any order sees a mix
of sources, frequencies and datasets from the first batch on.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import stat

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from terrastat import __version__
from terrastat.config import data_dir
from terrastat.locking import exclusive_lock
from terrastat.series import SERIES_SCHEMA
from terrastat.storage import series_dir, write_json, write_parquet

log = logging.getLogger(__name__)

PUBLIC_LICENSES = ("eurostat-reuse", "oecd-terms")  # plus FRED public domain from a US federal source
ROW_GROUP_ROWS = 2048


def _sha256(path: Path) -> str:
    """Checksum of a file, streamed so a multi-gigabyte shard never has to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_dir(name: str) -> Path:
    return data_dir() / "snapshot" / name


def _export_paths(name: str) -> tuple[Path, Path, Path, Path]:
    # All moves and recursive cleanup must remain inside one resolved snapshot root.
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
            or name.endswith(".") or name.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("snapshot name must be a single portable directory name")
    root = (data_dir() / "snapshot").resolve()
    return (root / name, root / f".{name}.export-staging",
            root / f".{name}.export-backup", root / f".{name}.export.lock")


def _check_export_directory(path: Path, parent: Path) -> None:
    if path.parent != parent or path.resolve().parent != parent:
        raise ValueError(f"Export path escapes snapshot directory: {path}")
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise ValueError(f"Export requires a plain directory, not a link or file: {path}")


def _remove_export_directory(path: Path, parent: Path) -> None:
    _check_export_directory(path, parent)
    if path.exists():
        shutil.rmtree(path)


def _check_replaceable_snapshot(out: Path) -> None:
    """Do not discard files the exporter does not own when retiring a snapshot."""
    if not out.exists():
        return
    metadata = {"index.parquet", "manifest.json", "README.md", "ATTRIBUTIONS.md", "croissant.json"}
    unexpected = []
    for path in out.iterdir():
        info = path.lstat()
        if ((path.name not in metadata and not re.fullmatch(r"shard-\d+\.(parquet|tmp)", path.name))
                or not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400):
            unexpected.append(path.name)
    if unexpected:
        raise ValueError(f"Snapshot contains unmanaged entries {unexpected}; use a new snapshot name: {out}")


def _sync_directory(path: Path) -> None:
    """Persist rename ordering where directory fsync is supported (not Windows)."""
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _verify_export(out: Path, expected_rows: int | None = None) -> None:
    """Read back the complete release before changing the published directory."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["shards"] + [manifest["index"]]
    names = [entry["file"] for entry in entries]
    if (len(names) != len(set(names)) or manifest["n_shards"] != len(manifest["shards"])
            or manifest["index"]["file"] != "index.parquet"
            or any(not re.fullmatch(r"shard-\d{5,}\.parquet", e["file"]) for e in manifest["shards"])):
        raise ValueError("Invalid snapshot file inventory")
    expected_files = set(names) | {"manifest.json", "README.md", "ATTRIBUTIONS.md", "croissant.json"}
    if {p.name for p in out.iterdir()} != expected_files:
        raise ValueError("Snapshot contains missing, unexpected or unfinished files")
    for path in out.iterdir():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Snapshot member must be a regular file: {path}")
        if info.st_size == 0:
            raise ValueError(f"Empty snapshot file: {path}")
    total_rows = 0
    for entry in entries:
        path = out / entry["file"]
        if path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise ValueError(f"Snapshot checksum or size mismatch: {path.name}")
        with pq.ParquetFile(path) as parquet:
            rows = parquet.metadata.num_rows
            if entry is manifest["index"]:
                if rows != manifest["rows"]:
                    raise ValueError("Snapshot index row count does not match manifest")
            else:
                if rows != entry["rows"] or parquet.schema_arrow.names != manifest["columns"]:
                    raise ValueError(f"Snapshot row count or columns mismatch: {path.name}")
                total_rows += rows
    if total_rows != manifest["rows"] or (expected_rows is not None and total_rows != expected_rows):
        raise ValueError("Snapshot row count does not match the selected input")
    if sum(e["bytes"] for e in manifest["shards"]) != manifest["bytes"]:
        raise ValueError("Snapshot byte count does not match manifest")
    json.loads((out / "croissant.json").read_text(encoding="utf-8"))


def _recover_export(out: Path, staging: Path, backup: Path) -> None:
    """Recover a killed publisher while holding this snapshot's writer lock."""
    for path in (out, staging, backup):
        _check_export_directory(path, out.parent)
    if backup.exists():
        if not out.exists():
            backup.rename(out)
            _sync_directory(out.parent)
            log.warning("Restored previous snapshot after interrupted export: %s", out)
        else:
            # A previous process published the new directory but stopped before cleanup.
            # If that release is damaged, retain both copies for explicit recovery.
            try:
                _verify_export(out)
            except (OSError, ValueError, KeyError) as exc:
                raise RuntimeError(f"Cannot verify published snapshot {out}; previous copy retained at {backup}") from exc
            _check_replaceable_snapshot(backup)
            _remove_export_directory(backup, out.parent)
    _remove_export_directory(staging, out.parent)


def _publish_export(out: Path, staging: Path, backup: Path) -> None:
    moved_old = False
    try:
        if out.exists():
            out.rename(backup)
            moved_old = True
            _sync_directory(out.parent)
        staging.rename(out)
    except BaseException:
        if moved_old and not out.exists():
            backup.rename(out)
            _sync_directory(out.parent)
        raise
    _sync_directory(out.parent)
    try:
        _remove_export_directory(backup, out.parent)
    except OSError:
        # Publication succeeded. Retain the backup and let the next export retry cleanup.
        log.warning("Snapshot published; previous copy retained at %s", backup, exc_info=True)


def _selection_filter(min_obs: int, license_ids: list[str] | None, public_only: bool) -> pl.Expr:
    expr = pl.col("n_obs") >= min_obs
    if license_ids:
        expr = expr & pl.col("license_id").is_in(license_ids)
    if public_only:
        expr = expr & (
            pl.col("license_id").is_in(list(PUBLIC_LICENSES))
            | ((pl.col("license_id") == "fred-public-domain-citation-requested") & (pl.col("origin_us_federal") == True))  # noqa: E712
        )
    return expr


def _input_files(sources: list[str] | None, freqs: list[str] | None) -> list[Path]:
    root = data_dir() / "series"
    if not root.exists():
        return []
    files = []
    for src_dir in sorted(root.iterdir()):
        if sources and src_dir.name not in sources:
            continue
        for part in sorted(src_dir.glob("freq=*")):
            if freqs and part.name.split("=", 1)[1] not in [f.upper() for f in freqs]:
                continue
            files.extend(sorted(part.glob("*.parquet")))
    return files


def export_shards(
    name: str,
    sources: list[str] | None = None,
    freqs: list[str] | None = None,
    min_obs: int = 1,
    license_ids: list[str] | None = None,
    public_only: bool = False,
    target_mb: float = 128.0,
    seed: int = 0,
    float32: bool = True,
    columns: list[str] | None = None,
    limit_rows: int | None = None,
    per_file_cap: int | None = None,
    row_group_rows: int = ROW_GROUP_ROWS,
    compression_level: int = 3,
) -> dict:
    """Build and verify a complete snapshot before replacing the existing directory.

    Replacement uses two directory renames on local filesystems, with rollback and
    next-run recovery. Use distinct release names for concurrent readers. Internal
    staging, backup and writer-lock paths are reserved beside the snapshot.
    """
    options = locals().copy()
    if type(row_group_rows) is not int or row_group_rows < 1:
        raise ValueError("row_group_rows must be a positive integer")
    if type(compression_level) is not int or not 1 <= compression_level <= 22:
        raise ValueError("compression_level must be an integer from 1 to 22")
    if not math.isfinite(target_mb) or target_mb <= 0:
        raise ValueError("target_mb must be positive and finite")
    out, staging, backup, lock = _export_paths(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(lock):
        _recover_export(out, staging, backup)
        _check_replaceable_snapshot(out)
        staging.mkdir()
        try:
            manifest = _build_shards(staging, **options)
            _verify_export(staging, expected_rows=manifest["rows"])
            for path in staging.iterdir():
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
            _sync_directory(staging)
            _publish_export(out, staging, backup)
            return manifest
        finally:
            _remove_export_directory(staging, out.parent)


def _build_shards(
    out: Path,
    name: str,
    sources: list[str] | None = None,
    freqs: list[str] | None = None,
    min_obs: int = 1,
    license_ids: list[str] | None = None,
    public_only: bool = False,
    target_mb: float = 128.0,
    seed: int = 0,
    float32: bool = True,
    columns: list[str] | None = None,
    limit_rows: int | None = None,
    per_file_cap: int | None = None,
    row_group_rows: int = ROW_GROUP_ROWS,
    compression_level: int = 3,
) -> dict:
    """Build ``data/snapshot/<name>/``. Returns the manifest.

    ``limit_rows`` truncates: it stops once the total is reached, so it takes whole datasets in
    catalogue order and nothing after them. That is right for a quick test and wrong for a sample
    — a 400,000-row limit produced a "starter" set that was entirely Eurostat annual series.
    ``per_file_cap`` samples instead, taking at most that many series from every dataset, which
    spreads the result across sources, frequencies and subjects. Use it for anything meant to be
    representative.
    """
    if type(row_group_rows) is not int or row_group_rows < 1:
        raise ValueError("row_group_rows must be a positive integer")
    if type(compression_level) is not int or not 1 <= compression_level <= 22:
        raise ValueError("compression_level must be an integer from 1 to 22")
    files = _input_files(sources, freqs)
    if not files:
        raise RuntimeError("no series files found; run `terrastat series <source>` first")
    flt = _selection_filter(min_obs, license_ids, public_only)
    cols = list(columns) if columns else list(SERIES_SCHEMA)
    for required in ("series_uid",):
        if required not in cols:
            cols.insert(0, required)

    # pass 1: how much survives the filter, to choose the number of shards
    est_bytes, total_rows = 0.0, 0
    per_file: list[tuple[Path, int]] = []
    for f in files:
        n_all = pq.ParquetFile(f).metadata.num_rows
        n_keep = pl.scan_parquet(f).filter(flt).select(pl.len()).collect().item()
        if per_file_cap is not None:
            n_keep = min(n_keep, per_file_cap)
        if limit_rows is not None:
            n_keep = min(n_keep, max(0, limit_rows - total_rows))
        if n_keep == 0:
            continue
        per_file.append((f, n_keep))
        total_rows += n_keep
        est_bytes += f.stat().st_size * n_keep / max(1, n_all)
    if total_rows == 0:
        raise RuntimeError("the filters leave no series to export")
    n_shards = max(1, math.ceil(est_bytes / (target_mb * 1e6)))
    log.info("export %s: %d series from %d files, ~%.0f MB -> %d shards of ~%.0f MB", name, total_rows, len(per_file), est_bytes / 1e6, n_shards, target_mb)

    # pass 2: stream every file once, scattering its rows over the shards
    rng = np.random.default_rng(seed)
    writers: dict[int, pq.ParquetWriter] = {}
    schema: pa.Schema | None = None
    remaining = limit_rows
    try:
        for f, n_keep in per_file:
            df = pl.read_parquet(f).filter(flt)
            if per_file_cap is not None and df.height > per_file_cap:
                # sampled, not the first N: series inside one file are sorted by key, so `head`
                # would take one corner of the dimension cross-product every time
                df = df.sample(n=per_file_cap, seed=seed, shuffle=True)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= df.height
            df = df.select(cols)
            if float32 and "values" in df.columns:
                df = df.with_columns(pl.col("values").cast(pl.List(pl.Float32)))
            df = df.with_columns(pl.Series("_shard", rng.integers(0, n_shards, df.height), dtype=pl.Int32))
            for key, part in df.partition_by("_shard", as_dict=True).items():
                sid = int(key[0] if isinstance(key, tuple) else key)
                table = part.drop("_shard").to_arrow()
                if schema is None:
                    schema = table.schema
                elif not table.schema.equals(schema):
                    table = table.cast(schema)
                if sid not in writers:
                    writers[sid] = pq.ParquetWriter(out / f"shard-{sid:05d}.tmp", schema, compression="zstd")
                writers[sid].write_table(table, row_group_size=ROW_GROUP_ROWS)
    finally:
        for w in writers.values():
            w.close()

    # pass 3: shuffle inside each shard, write the final file, collect the index
    shards_meta, index_parts = [], []
    counts = {"source": {}, "frequency": {}, "license_id": {}}
    for sid in range(n_shards):
        tmp = out / f"shard-{sid:05d}.tmp"
        if not tmp.exists():
            continue
        df = pl.read_parquet(tmp).sample(fraction=1.0, shuffle=True, seed=seed + sid)
        tmp.unlink()
        path = out / f"shard-{sid:05d}.parquet"
        df.write_parquet(path, compression="zstd", compression_level=compression_level,
                         row_group_size=row_group_rows, statistics=True)
        # sha256 per shard, so *any* client can verify a download -- not only ours. Without it a
        # truncated or corrupted transfer is indistinguishable from a short shard.
        shards_meta.append({"file": path.name, "rows": df.height,
                            "bytes": path.stat().st_size, "sha256": _sha256(path)})
        idx_cols = [c for c in ("series_uid", "source", "dataset_id", "frequency", "license_id", "n_points", "start_date", "end_date") if c in df.columns]
        index_parts.append(df.select(idx_cols).with_columns(pl.lit(sid, pl.Int32).alias("shard"), pl.int_range(0, df.height, dtype=pl.Int64).alias("row")))
        for c in counts:
            if c in df.columns:
                for k, v in df.group_by(c).len().iter_rows():
                    counts[c][str(k)] = counts[c].get(str(k), 0) + int(v)
    index = pl.concat(index_parts, how="vertical_relaxed")
    if index.height != total_rows:
        raise ValueError("Selected inputs changed during export; retry with a stable series layer")
    write_parquet(index, out / "index.parquet")

    manifest = {
        "name": name,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "terrastat_version": __version__,
        "selection": {"sources": sources, "frequencies": freqs, "min_obs": min_obs, "license_ids": license_ids, "public_only": public_only, "limit_rows": limit_rows, "per_file_cap": per_file_cap},
        "seed": seed,
        "float32_values": float32,
        "target_mb": target_mb,
        "parquet": {"compression": "zstd", "compression_level": compression_level,
                    "row_group_rows": row_group_rows},
        "n_shards": len(shards_meta),
        "rows": int(index.height),
        "bytes": int(sum(s["bytes"] for s in shards_meta)),
        "columns": cols,
        "counts": counts,
        "shards": shards_meta,
        "how_to_read": {
            "polars": "pl.scan_parquet('shard-*.parquet')",
            "pandas": "pd.read_parquet('shard-00000.parquet')",
            "huggingface": "load_dataset('parquet', data_files='shard-*.parquet', streaming=True)",
        },
    }
    manifest["index"] = {"file": "index.parquet", "bytes": (out / "index.parquet").stat().st_size,
                         "sha256": _sha256(out / "index.parquet")}
    write_json(out / "manifest.json", manifest)
    _write_release_notes(out, name, manifest, index)
    # machine-readable metadata, written from the manifest so it cannot disagree with the shards
    from terrastat.croissant import write as write_croissant

    write_croissant(out, manifest)
    log.info("export %s: %d shards, %d rows, %.0f MB", name, len(shards_meta), index.height, manifest["bytes"] / 1e6)
    manifest["licensing_notice"] = licensing_notice(manifest)
    return manifest


def licensing_notice(manifest: dict) -> str:
    """What the person who just built this snapshot needs to read before sharing it.

    The terms already travel in four places — a column on every row, the dataset card,
    ATTRIBUTIONS.md, and the manifest — and someone can still publish a snapshot without noticing
    any of them. This is the fourth line of defence, printed where it cannot be scrolled past:
    at the end of the command that produced the files.
    """
    lic = manifest["counts"].get("license_id", {})
    public = manifest["selection"].get("public_only")
    lines = ["", "=" * 72]
    if public:
        lines += [
            "REDISTRIBUTABLE SUBSET  (--public-only)",
            "",
            "Every series here may be shared onward, but attribution is still required:",
        ]
    else:
        lines += [
            "!! NO LICENCE FILTER WAS APPLIED !!",
            "",
            "This snapshot may contain series that must NOT be redistributed. FRED carries",
            "third-party series copyrighted by the agency that produced them; only those marked",
            "public domain from a US federal body are free of copyright.",
            "",
            "Re-run with --public-only to get a shareable subset. Licences present:",
        ]
    for lid, n in sorted(lic.items(), key=lambda kv: -kv[1]):
        who, terms, _ = LICENSE_SUMMARY.get(lid, (lid, "check the source's terms", ""))
        lines.append(f"  {n:>12,}  {who:<10} {terms[:52]}")
    lines += [
        "",
        "The exact citation each source asks for is in ATTRIBUTIONS.md and, per series, in the",
        "`attribution` column — so it survives any slicing of the data.",
        "=" * 72,
    ]
    return "\n".join(lines)


LICENSE_SUMMARY = {
    "eurostat-reuse": (
        "Eurostat", "Free re-use for commercial and non-commercial purposes with acknowledgement "
        "(Commission Decision 2011/833/EU).", "https://ec.europa.eu/eurostat/web/main/help/copyright-notice",
    ),
    "oecd-terms": (
        "OECD", "Extraction, adaptation and distribution for any purpose, including commercial, with citation.",
        "https://www.oecd.org/en/about/terms-conditions.html",
    ),
    "fred-public-domain-citation-requested": (
        "FRED", "Series FRED marks 'Public Domain: Citation Requested' whose originating body is a US federal "
        "one, so the underlying work carries no copyright of its own (17 U.S.C. 105). Citation requested.",
        "https://fred.stlouisfed.org/legal/",
    ),
}


def _size_category(n: int) -> str:
    """Hugging Face's own size buckets, so the Hub badge is right without anyone choosing it."""
    for hi, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                      (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M"),
                      (100_000_000, "10M<n<100M"), (1_000_000_000, "100M<n<1B")):
        if n < hi:
            return label
    return "n>1B"


def _front_matter(name: str, manifest: dict) -> str:
    """YAML the Hugging Face Hub reads: it drives the dataset viewer, the licence badge and tags.

    `configs.data_files` is the line that matters most -- without it the Hub cannot preview a
    folder of shards, and a dataset nobody can preview is a dataset nobody tries.
    """
    public = manifest["selection"].get("public_only")
    freqs = ", ".join(sorted(manifest["counts"].get("frequency", {})))
    return f"""---
pretty_name: {name}
license: other
license_name: mixed-source-terms
license_link: ATTRIBUTIONS.md
size_categories:
- {_size_category(manifest['rows'])}
task_categories:
- time-series-forecasting
tags:
- economics
- time-series
- FRED
- Eurostat
- OECD
{'- redistributable' if public else '- mixed-licence'}
configs:
- config_name: default
  data_files: shard-*.parquet
---

"""


def _write_release_notes(out: Path, name: str, manifest: dict, index: pl.DataFrame) -> None:
    """A dataset card and an attributions file, so the snapshot can be published as it stands."""
    lic_counts = manifest["counts"].get("license_id", {})
    src_counts = manifest["counts"].get("source", {})
    freq_counts = manifest["counts"].get("frequency", {})
    sel = manifest["selection"]

    rows = [
        "| licence | series | terms |",
        "|---|---:|---|",
    ]
    for lid, n in sorted(lic_counts.items(), key=lambda kv: -kv[1]):
        who, terms, url = LICENSE_SUMMARY.get(lid, (lid, "See the source's terms.", ""))
        rows.append(f"| {who} (`{lid}`) | {n:,} | {terms} [terms]({url}) |")

    mb = manifest["bytes"] / 1e6
    size_txt = (f"{mb / 1000:.2f} GB" if mb >= 1000
                else f"{mb:.0f} MB" if mb >= 1 else f"{manifest['bytes'] / 1e3:.0f} KB")
    shard_txt = "shard" if manifest["n_shards"] == 1 else "shards"

    card = _front_matter(name, manifest) + f"""# {name}

Public economic time series from FRED, Eurostat and the OECD, one row per series, gathered with
[terrastat](https://github.com/USER/terrastat).

| | |
|---|---|
| **series** | {manifest['rows']:,} |
| **size** | {size_txt} in {manifest['n_shards']} Parquet {shard_txt} |
| **sources** | {', '.join(f'{k} {v:,}' for k, v in sorted(src_counts.items(), key=lambda kv: -kv[1]))} |
| **frequencies** | {', '.join(f'{k} {v:,}' for k, v in sorted(freq_counts.items(), key=lambda kv: -kv[1]))} |
| **built** | {manifest['created_at'][:10]} with terrastat {manifest['terrastat_version']} |

```python
import polars as pl
df = pl.read_parquet("shard-*.parquet")      # or scan_parquet, for the larger snapshots
one = df.row(0, named=True)
print(one["title"], one["frequency"], one["units"])
print(list(zip(one["dates"], one["values"]))[:5])
```

## Licensing

{"Only series that may be redistributed were included (`--public-only`)." if sel.get("public_only") else "**No licence filter was applied.** Check `license_id` before redistributing."}
Every row carries `license_id`, `license_url`, `attribution` and `license_notes`, so the terms
travel with the data rather than living only in this file.

{chr(10).join(rows)}

See `ATTRIBUTIONS.md` for the citations the sources ask for. Where a source asks for citation,
cite it; the per-row `attribution` column gives the exact wording for each series.

## What is in a row

One row is one complete time series. `dates`, `values` and `flags` are equal-length lists in date
order; `values` may contain nulls where the source published a status flag without a number, and
`n_obs` counts the non-null values. The metadata is both structured (`units`, `frequency`, `geo`,
`dimensions`) and written out as one sentence in `description`, ready for a text encoder.

Selection: minimum {sel.get('min_obs')} observations per series{', sources ' + ', '.join(sel['sources']) if sel.get('sources') else ''}{', frequencies ' + ', '.join(sel['frequencies']) if sel.get('frequencies') else ''}.
Values are stored as {'float32' if manifest['float32_values'] else 'float64'}.

## Loading it

```python
import polars as pl
df = pl.scan_parquet("shard-*.parquet").filter(pl.col("frequency") == "M").collect()
```

```python
from datasets import load_dataset            # Hugging Face
ds = load_dataset("parquet", data_files="shard-*.parquet", streaming=True)
```

Every shard is a uniform random sample of the whole selection (seed {manifest['seed']}), so reading
shards in any order gives a mix of sources and frequencies from the first batch. `index.parquet`
maps every `series_uid` to its shard and row; `manifest.json` is the machine-readable version of
this page, and `croissant.json` is the same in [MLCommons Croissant](https://mlcommons.org/croissant/)
form, which Hugging Face, Kaggle and OpenML read directly.

## Provenance and caveats

Values are the **latest revision** as published at the retrieval date in each row's
`retrieved_at`; this is not a real-time/vintage dataset. Series lengths vary enormously, and short
cross-sectional series are common in the Eurostat and OECD catalogues, so filter on `n_obs` for
modelling. Nothing here has been rescaled, interpolated or seasonally adjusted by terrastat.
"""
    (out / "README.md").write_text(card, encoding="utf-8")

    # the distinct attributions, which is what a downstream user must reproduce
    attr = ["# Attributions", "", "This snapshot redistributes data from the sources below, under the terms in `README.md`.", ""]
    for lid, n in sorted(lic_counts.items(), key=lambda kv: -kv[1]):
        who, terms, url = LICENSE_SUMMARY.get(lid, (lid, "", ""))
        attr += [f"## {who} ({n:,} series)", "", terms, f"Terms: {url}", ""]
    attr += [
        "## Per-series citations", "",
        "Each row carries the exact wording its source asks for in the `attribution` column:", "",
        "```python",
        "import polars as pl",
        'pl.scan_parquet("shard-*.parquet").select("series_uid", "attribution").collect()',
        "```", "",
        "FRED series additionally record the originating agency in `origin_agencies`; the",
        "Federal Reserve Bank of St. Louis is the compiler, not the author, of most of them.",
    ]
    (out / "ATTRIBUTIONS.md").write_text("\n".join(attr), encoding="utf-8")


def iter_shard_batches(snapshot: str | Path, batch_rows: int = 256, columns: list[str] | None = None, seed: int | None = None):
    """Yield polars frames of ``batch_rows`` series, one shard at a time, in a (seeded) random shard order.

    This is the reading pattern a training loop wants: bounded memory, every batch a random mix.
    """
    root = Path(snapshot) if Path(snapshot).exists() else snapshot_dir(str(snapshot))
    files = sorted(root.glob("shard-*.parquet"))
    if seed is not None:
        files = list(np.random.default_rng(seed).permutation(files))
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            yield pl.from_arrow(batch)
