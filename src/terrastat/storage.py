"""Where things live on disk, and the small primitives that keep writes atomic and runs resumable.

Layout (under ``config.data_dir()``)::

    raw/<source>/<dataset_id>/...                 payloads exactly as downloaded
    datasets/<source>/<dataset_id>/dataset.json   all dataset-level metadata (see sources/base.py)
    datasets/<source>/<dataset_id>/observations.parquet   long table: series_key, date, value, flag, ...
    datasets/<source>/<dataset_id>/series.parquet         per-series metadata when the source has it (FRED)
    catalog/<source>.parquet                      what the source offers, with size hints and frequencies
    cache/<source>/...                            shared reference data (codelists), reused across datasets
    state/<source>.jsonl                          append-only log: one line per finished/failed dataset
    series/<source>/freq=<F>/<dataset_id>.parquet the model-ready view (one row per series)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from terrastat.config import data_dir

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(s: str) -> str:
    """Folder-safe dataset identifier (OECD ids contain ':', '@', '(' ...)."""
    return _SAFE.sub("_", s).strip("_")


def raw_dir(source: str, dataset_id: str) -> Path:
    return data_dir() / "raw" / source / dataset_id


def dataset_dir(source: str, dataset_id: str) -> Path:
    return data_dir() / "datasets" / source / dataset_id


def cache_dir(source: str) -> Path:
    return data_dir() / "cache" / source


def catalog_path(source: str) -> Path:
    return data_dir() / "catalog" / f"{source}.parquet"


def state_path(source: str) -> Path:
    return data_dir() / "state" / f"{source}.jsonl"


def series_dir(source: str) -> Path:
    return data_dir() / "series" / source


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# -- atomic writes --------------------------------------------------------------------------


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_parquet(df: pl.DataFrame, path: Path, compression: str = "zstd") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    df.write_parquet(tmp, compression=compression, statistics=True)
    os.replace(tmp, path)


class ParquetBatchWriter:
    """Append polars frames one batch at a time to a single parquet file, atomically.

    The schema is fixed by the first batch; later batches are cast to it.
    """

    def __init__(self, path: Path, compression: str = "zstd"):
        self.path = path
        self.tmp = path.with_name(path.name + ".part")
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self.rows = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            return
        table = df.to_arrow()
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.tmp, self._schema, compression=self.compression)
        elif not table.schema.equals(self._schema):
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self.rows += table.num_rows

    def close(self) -> Path | None:
        """Finish the file. Returns None (and writes nothing) if no rows were written."""
        if self._writer is None:
            return None
        self._writer.close()
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self.tmp.exists():
            self.tmp.unlink()


# -- resumable state ------------------------------------------------------------------------


class State:
    """One JSON line per event. The last event for a dataset id wins."""

    def __init__(self, source: str):
        self.path = state_path(source)
        self.events: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        ev = json.loads(line)
                        self.events[ev["dataset_id"]] = ev

    def mark(self, dataset_id: str, status: str, **info) -> None:
        ev = {"dataset_id": dataset_id, "status": status, "at": now_iso(), **info}
        self.events[dataset_id] = ev
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

    def status(self, dataset_id: str) -> str | None:
        ev = self.events.get(dataset_id)
        return ev["status"] if ev else None

    def ids_with(self, status: str) -> set[str]:
        return {k for k, v in self.events.items() if v["status"] == status}


def ensure_gzip(path: Path) -> Path:
    """Make sure ``path`` holds gzip bytes (servers sometimes hand us the decompressed payload)."""
    import gzip
    import shutil

    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return path
    tmp = path.with_name(path.name + ".plain")
    os.replace(path, tmp)
    with open(tmp, "rb") as src, gzip.open(path, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    tmp.unlink()
    return path


def is_fresh(path: Path, max_age_hours: float = 24.0) -> bool:
    """True if ``path`` exists and was modified less than ``max_age_hours`` ago."""
    import time

    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
