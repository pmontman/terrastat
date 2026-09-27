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
import logging
import os
import re
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from terrastat.config import data_dir, reference_cache_dir
from terrastat.locking import exclusive_lock

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
log = logging.getLogger(__name__)


def safe_id(s: str) -> str:
    """Folder-safe dataset identifier (OECD ids contain ':', '@', '(' ...)."""
    return _SAFE.sub("_", s).strip("_")


def raw_dir(source: str, dataset_id: str) -> Path:
    return data_dir() / "raw" / source / dataset_id


def dataset_dir(source: str, dataset_id: str) -> Path:
    return data_dir() / "datasets" / source / dataset_id


def cache_dir(source: str) -> Path:
    return reference_cache_dir() / source


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


def _incomplete_event(raw: bytes) -> bool:
    """Whether the final bytes are a prefix of a JSON object, rather than bad JSON.

    Let the standard decoder validate everything before its first error. Recovery
    is limited to EOF inside a string/container or a partially written token;
    invalid escapes, separators, extra data and complete invalid records stay errors.
    """
    partial_utf8 = False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(raw):
            return False
        text = raw[:exc.start].decode("utf-8")
        partial_utf8 = True
    if not text.lstrip().startswith("{"):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        if partial_utf8:
            # Non-ASCII bytes can occur only inside a JSON string.
            return exc.msg == "Unterminated string starting at"
        if exc.msg == "Unterminated string starting at" or exc.pos == len(text):
            return True
        tail = text[exc.pos:]
        if exc.msg == "Invalid \\uXXXX escape":
            # CPython also reports this when all four digits are present but
            # the containing string ends at EOF before its closing quote.
            return bool(re.fullmatch(r"u[0-9a-fA-F]{0,4}", tail))
        if exc.msg == "Expecting value":
            return any(token.startswith(tail) and token != tail
                       for token in ("true", "false", "null", "NaN", "Infinity", "-Infinity", "-0"))
        if exc.msg == "Expecting ',' delimiter":
            # A decoder accepts the complete numeric prefix of e.g. 1. or 1e+.
            before = text[:exc.pos]
            integer = r"-?(?:0|[1-9][0-9]*)"
            if tail == ".":
                number = integer
            elif re.fullmatch(r"[eE][+-]?", tail):
                number = integer + r"(?:\.[0-9]+)?"
            else:
                return False
            return bool(re.search(r"(?:^|[\s\[{: ,])" + number + r"\Z", before))
    return False


def _state_signature(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class State:
    """Durable JSONL events; the last event for a dataset id wins.

    A crash may leave an incomplete final line. Loading repairs only that tail,
    with a warning and under the same short-lived lock used by appends. Malformed
    terminated lines and complete invalid records require manual investigation.
    Instances cache reads; ``mark`` refreshes under the lock before appending.
    """

    def __init__(self, source: str):
        self.path = state_path(source)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.events: dict[str, dict] = {}
        self._checkpoint: tuple[int, int, int, int] | None = None
        self._needs_newline = False
        with exclusive_lock(self.lock_path):
            self.events, self._checkpoint, self._needs_newline = self._read_locked()

    def _read_locked(self) -> tuple[dict[str, dict], tuple[int, int, int, int] | None, bool]:
        """Read new records (or reload a replaced/truncated log) while holding its lock."""
        try:
            fh = self.path.open("r+b")
        except FileNotFoundError:
            return {}, None, False
        with fh:
            signature = _state_signature(os.fstat(fh.fileno()))
            events: dict[str, dict] = {}
            if self._checkpoint is not None and signature[:2] == self._checkpoint[:2]:
                if signature == self._checkpoint:
                    # The caller changes this cache only after its append is
                    # durable. Avoid copying a growing dictionary per event.
                    return self.events, signature, self._needs_newline
                if signature[2] > self._checkpoint[2] and not self._needs_newline:
                    # Most crawls append thousands of events. Re-reading their
                    # entire history for each append would be quadratic.
                    events = dict(self.events)
                    fh.seek(self._checkpoint[2])
            needs_newline = False
            while True:
                offset = fh.tell()
                raw = fh.readline()
                if not raw:
                    break
                needs_newline = not raw.endswith(b"\n")
                if not raw.strip():
                    continue  # Older logs may contain blank lines.
                try:
                    ev = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if needs_newline and _incomplete_event(raw):
                        fh.seek(offset)
                        fh.truncate()
                        fh.flush()
                        os.fsync(fh.fileno())
                        log.warning("Recovered incomplete final state record in %s; discarded %d tail bytes",
                                    self.path.name, len(raw))
                        needs_newline = False
                        break
                    raise ValueError(f"corrupt state record in {self.path.name} at byte {offset}") from exc
                if not isinstance(ev, dict) or not isinstance(ev.get("dataset_id"), str) or not isinstance(ev.get("status"), str):
                    raise ValueError(f"invalid state event in {self.path.name} at byte {offset}")
                events[ev["dataset_id"]] = ev
            return events, _state_signature(os.fstat(fh.fileno())), needs_newline

    def mark(self, dataset_id: str, status: str, **info) -> None:
        if not isinstance(dataset_id, str) or not isinstance(status, str):
            raise ValueError("state dataset_id and status must be strings")
        ev = {"dataset_id": dataset_id, "status": status, "at": now_iso(), **info}
        record = (json.dumps(ev, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with exclusive_lock(self.lock_path):
            events, _, needs_newline = self._read_locked()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as fh:
                payload = (b"\n" if needs_newline else b"") + record
                if fh.write(payload) != len(payload):
                    raise OSError("incomplete state log write")
                fh.flush()
                os.fsync(fh.fileno())
                checkpoint = _state_signature(os.fstat(fh.fileno()))
            # Never report a status in memory before its append has been flushed.
            events[dataset_id] = ev
            self.events, self._checkpoint, self._needs_newline = events, checkpoint, False

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
