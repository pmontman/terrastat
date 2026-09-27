"""Refresh existing datasets from fresh provider payloads without changing snapshots.

Provider adapters run in an isolated data workspace. Publication and recovery are
journaled by refresh_transaction; no network request writes into a live dataset.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import polars as pl
import pyarrow.parquet as pq

from terrastat.config import data_dir, data_workspace
from terrastat.http import StopRequested
from terrastat.locking import exclusive_lock
from terrastat.revisions import compare_observations
from terrastat import refresh_transaction as transaction
from terrastat.series import build_dataset_series
from terrastat.sources import SOURCES, get_source
from terrastat.sources.base import NoData, NotTimeSeries, ref_from_row
from terrastat.storage import State, now_iso, read_json

log = logging.getLogger(__name__)
_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


class RefreshValidationError(ValueError):
    """Controlled diagnostics safe to include in reports without request secrets."""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or value.endswith((".", " ")):
        raise ValueError("Refresh requires a portable, single-component dataset ID")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)]}:
        raise ValueError("Reserved dataset ID")
    return value


def _plain(path: Path, root: Path) -> None:
    """Reject redirected paths before reading or publishing local dataset files."""
    path.relative_to(root)
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Refresh paths must not contain links or reparse points")
        if part == root:
            return
    raise ValueError("Refresh path is outside the data root")


def _files(path: Path, root: Path) -> list[Path]:
    _plain(path, root)
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    paths = sorted(path.rglob("*"))
    for item in paths:
        _plain(item, root)
    return [item for item in paths if item.is_file()]


def _series_files(root: Path, source: str, did: str) -> list[Path]:
    parent = root / "series" / source
    _plain(parent, root)
    found = sorted(parent.glob(f"freq=*/{did}.parquet"))
    for path in found:
        _plain(path, root)
    return found


def _fingerprint(root: Path, source: str, did: str, *, include_raw: bool = True) -> dict:
    """Local change guard, not a cryptographic proof against filesystem tampering."""
    files = _files(root / "datasets" / source / did, root)
    if include_raw:
        files += _files(root / "raw" / source / did, root)
    files += _series_files(root, source, did)
    return {p.relative_to(root).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metadata(meta: dict) -> dict:
    # Retrieval time and payload partitioning can change without a provider revision.
    out = {k: v for k, v in meta.items() if k not in {"retrieved_at", "raw_files"}}
    if isinstance(out.get("extra"), dict):
        out["extra"] = {k: v for k, v in out["extra"].items() if k not in {"n_raw_parts"}}
    if isinstance(out.get("license"), dict):
        license_record = dict(out["license"])
        attribution = license_record.get("attribution", "")
        # Our rendered citations include acquisition dates. Do not classify
        # fetching the same data on another day as a changed licence.
        if meta.get("source") in {"eurostat", "oecd"}:
            attribution = re.sub(r"(accessed(?: on)? )\d{4}-\d{2}-\d{2}", r"\1<retrieval-date>", attribution)
        if meta.get("source") == "oecd":
            attribution = re.sub(r"^OECD \(\d{4}\)", "OECD (<retrieval-year>)", attribution)
        license_record["attribution"] = attribution
        out["license"] = license_record
    return out


def _series_metadata_changed(old: Path, new: Path) -> bool:
    if not old.exists() or not new.exists():
        return old.exists() != new.exists()
    left, right = pl.read_parquet(old), pl.read_parquet(new)
    if left.schema != right.schema:
        return True
    if "series_id" in left.columns:
        for frame in (left, right):
            if frame["series_id"].null_count() or frame["series_id"].n_unique() != frame.height:
                raise ValueError("Per-series metadata has invalid series IDs")
        left, right = left.sort("series_id"), right.sort("series_id")
    return not left.equals(right)


def _durable_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _error(exc: Exception, phase: str) -> str:
    # HTTP exceptions can embed API keys in request URLs. Reports and logs use
    # controlled diagnostics rather than str(exc) or a request traceback.
    if isinstance(exc, RefreshValidationError):
        detail = str(exc)
    elif isinstance(exc, NoData):
        detail = "Provider returned no data; dataset withdrawal requires manual review"
    elif isinstance(exc, NotTimeSeries):
        detail = "Provider response is no longer a time-series dataset"
    else:
        detail = f"{phase} failed ({type(exc).__name__})"
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            detail += f", HTTP {status}"
    return detail + "; existing dataset retained"


def _cache_updates(root: Path, stage: Path, cache: Path, source: str, installed: dict) -> list[str]:
    """Publish reference files actually obtained in this run; reuse them across datasets."""
    paths = []
    for path in _files(cache / source, cache):
        if path.name.endswith((".part", ".lock")):
            continue
        relative = (Path("cache") / path.relative_to(cache)).as_posix()
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        if installed.get(relative) == stamp:
            continue
        live = root / relative
        _plain(live, root)
        if not live.is_file() or _sha256(live) != _sha256(path):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            paths.append(relative)
    return paths


def _refresh_one(root, source, did, ref, src, client, cache, force, installed):
    result = {"source": source, "dataset_id": did, "method": "forced" if force else "download_compare"}
    txn = None
    phase = "local validation"
    try:
        old = root / "datasets" / source / did
        before = _fingerprint(root, source, did)
        old_meta = read_json(old / "dataset.json")
        if old_meta.get("source") != source or old_meta.get("dataset_id") != did:
            raise RefreshValidationError("Local metadata identity does not match its directory")
        ledger_path = root / "refresh" / "checks" / source / f"{did}.json"
        _plain(ledger_path, root)
        ledger = read_json(ledger_path) if ledger_path.exists() else {}
        token = src.refresh_token(ref)
        if not force and token and ledger.get("token") == token and ledger.get("fingerprint") == before:
            return {**result, "status": "unchanged", "method": "catalog_token"}

        txn = transaction.begin(root, source, did)
        stage = txn / "stage"
        # --no-raw leaves empty folders or paging checkpoints in some adapters.
        # Their presence must not silently switch raw retention back on.
        keep_raw = bool(old_meta.get("raw_files"))
        if "raw_files" not in old_meta:  # Compatibility with older/manual metadata.
            keep_raw = any(not p.name.startswith("_") for p in _files(root / "raw" / source / did, root))
        phase = "download and parsing"
        with data_workspace(stage, cache=cache):
            fetched = src.fetch_and_tidy(ref, client, keep_raw=keep_raw)
        if client.stop_requested():
            raise StopRequested()
        phase = "validation and comparison"
        new = stage / "datasets" / source / did
        new_meta = read_json(new / "dataset.json")
        if new_meta.get("source") != source or new_meta.get("dataset_id") != did or fetched.dataset_id != did:
            raise RefreshValidationError("Downloaded metadata identity does not match the requested dataset")
        comparison = compare_observations(old / "observations.parquet", new / "observations.parquet")
        if comparison["new_rows"] == 0:
            raise RefreshValidationError("An empty replacement needs manual withdrawal review")
        if new_meta.get("n_obs") != comparison["new_rows"] or fetched.n_obs != comparison["new_rows"]:
            raise RefreshValidationError("Downloaded observation counts do not match the normalized data")
        meta_changed = (_metadata(old_meta) != _metadata(new_meta)
                        or _series_metadata_changed(old / "series.parquet", new / "series.parquet"))
        result.update(comparison=comparison, metadata_changed=meta_changed)
        changed = (meta_changed or any(comparison[k] for k in ("added", "removed", "revised", "date_revisions", "dimension_revisions"))
                   or any(comparison["schema_changes"].values()))
        old_series = _series_files(root, source, did)
        rebuild = changed or not old_series or bool(ledger and ledger.get("fingerprint") != before)
        if rebuild:
            owned = {"dataset.json", "observations.parquet", "series.parquet"}
            if any(path.parent != old or path.name not in owned for path in _files(old, root)):
                raise RefreshValidationError("Local normalized directory contains unmanaged files; move them before refreshing")
            phase = "series build"
            with data_workspace(stage, cache=cache):
                built = build_dataset_series(source, did, min_obs=1, force=True)
            # The builder has read the observations; the transaction additionally
            # verifies exact file bytes. Check its returned paths and footers here.
            for path in built:
                path.relative_to(stage / "series" / source)
                pq.ParquetFile(path).metadata
            if built:
                counts = pl.scan_parquet(built, hive_partitioning=False).select(
                    pl.len().alias("rows"), pl.col("series_uid").n_unique().alias("unique"),
                    pl.col("series_uid").null_count().alias("nulls"),
                ).collect(engine="streaming").row(0, named=True)
                if counts["rows"] != counts["unique"] or counts["nulls"]:
                    raise RefreshValidationError("Rebuilt view contains duplicate or missing series identifiers")
        else:
            built = []
        if client.stop_requested():
            raise StopRequested()
        if before != _fingerprint(root, source, did):
            raise RefreshValidationError("Local data changed during refresh; stop other writers")
        if not rebuild:
            phase = "recording successful check"
            _durable_json(ledger_path, {"token": token, "checked_at": now_iso(), "fingerprint": before})
            transaction.abort(txn)
            txn = None
            return {**result, "status": "unchanged"}

        phase = "publication"
        relative_series = [p.relative_to(stage).as_posix() for p in built]
        paths = [f"datasets/{source}/{did}", *sorted(set(relative_series) | {p.relative_to(root).as_posix() for p in old_series})]
        if keep_raw:
            paths.append(f"raw/{source}/{did}")
        paths += _cache_updates(root, stage, cache, source, installed)
        ledger_relative = ledger_path.relative_to(root).as_posix()
        after = _fingerprint(stage, source, did, include_raw=keep_raw)
        if not keep_raw:
            after.update({path: stamp for path, stamp in before.items() if path.startswith(f"raw/{source}/{did}/")})
        _durable_json(stage / ledger_relative, {"token": token, "checked_at": now_iso(), "fingerprint": after})
        paths.append(ledger_relative)
        updates = [
            {"source": source, "dataset_id": did, "status": "done", "extra": {
                "n_series": fetched.n_series, "n_obs": fetched.n_obs, "frequencies": fetched.frequencies,
                "refreshed_at": now_iso()}},
            {"source": f"{source}-series", "dataset_id": did, "status": "done", "extra": {
                "files": [str(root / p) for p in relative_series],
                "n_series": sum(pq.ParquetFile(p).metadata.num_rows for p in built)}},
        ]
        cache_stamps = {}
        for path in _files(cache / source, cache):
            cache_stamps[(Path("cache") / path.relative_to(cache)).as_posix()] = (path.stat().st_size, path.stat().st_mtime_ns)
        transaction.publish(txn, paths, updates)
        txn = None
        installed.update(cache_stamps)
        return {**result, "status": "updated", "series_rebuilt": True}
    except transaction.CommittedRefreshError:
        txn = None  # Recovery must complete the already committed state update.
        return {**result, "status": "updated", "state_pending": True,
                "warning": "Data published; state finalization is pending. Run refresh again to recover."}
    except (KeyboardInterrupt, StopRequested, transaction.RecoveryRequiredError):
        raise
    except Exception as exc:
        return {**result, "status": "failed", "phase": phase, "error": _error(exc, phase)}
    finally:
        if txn is not None:
            # abort only removes building transactions; prepared/committed ones
            # remain available for journal recovery if rollback itself failed.
            transaction.abort(txn)


def run_refresh(sources: list[str] | None = None, ids: list[str] | None = None,
                force: bool = False, min_interval: float | None = None,
                max_per_minute: int | None = None, max_hours: float | None = None,
                limit: int | None = None) -> dict:
    """Check locally normalized datasets, then stage, compare and publish revisions.

    ``run`` retains its resumable acquisition semantics. Refresh is explicit:
    no new dataset IDs are acquired and no exported snapshot is rewritten.
    """
    root = data_dir()
    selected = list(dict.fromkeys(SOURCES if sources is None else sources))
    if any(source not in SOURCES for source in selected):
        raise ValueError("Unknown refresh source")
    if ids is not None and (sources is None or len(selected) != 1 or not ids):
        raise ValueError("Dataset IDs require exactly one explicitly selected source")
    for value, name in ((max_hours, "max_hours"), (max_per_minute, "max_per_minute"), (limit, "limit")):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite")
    if min_interval is not None and (not math.isfinite(min_interval) or min_interval < 0):
        raise ValueError("min_interval must be nonnegative and finite")
    for value in ids or []:
        _identifier(value)
    refresh_root = root / "refresh"
    _plain(refresh_root, root)
    refresh_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def stopped():
        return max_hours is not None and time.monotonic() - started >= max_hours * 3600
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
    report_path = refresh_root / "reports" / f"{run_id}.json"
    summary = {"checked": 0, "unchanged": 0, "updated": 0, "failed": 0, "stopped_early": False,
               "interrupted": False, "started_at": now_iso(), "report": str(report_path), "results": []}
    with exclusive_lock(refresh_root / "writer.lock"):
        transaction.recover(root)
        local = {}
        for source in selected:
            parent = root / "datasets" / source
            _plain(parent, root)
            existing = sorted(_identifier(p.name) for p in parent.iterdir() if p.is_dir() and (p / "dataset.json").is_file()) if parent.exists() else []
            if ids is not None:
                if set(ids) - set(existing):
                    raise ValueError("Requested dataset IDs are not locally collected: " + ", ".join(sorted(set(ids) - set(existing))))
                existing = list(dict.fromkeys(ids))
            if existing:
                local[source] = existing
        summary["selected"] = sum(len(values) for values in local.values())
        try:
            for source, dataset_ids in local.items():
                if stopped() or (limit is not None and summary["checked"] >= limit):
                    summary["stopped_early"] = True
                    break
                work_parent = refresh_root / "work"
                work_parent.mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=f"{source}-", dir=work_parent) as temporary:
                    workspace = Path(temporary)
                    cache = workspace / "cache"
                    installed = {}
                    client = None
                    try:
                        # Existing state corruption must be found before any publication.
                        State(source)
                        State(f"{source}-series")
                        src = get_source(source)
                        src.strict_refresh = True
                        client = src.make_client(min_interval=min_interval, max_per_minute=max_per_minute)
                        client.stop_check = stopped
                        if source == "fred":
                            for name in ("tags.parquet", "series_tags.parquet"):
                                path = root / "cache" / source / name
                                _plain(path, root)
                                if path.is_file():
                                    target = cache / source / name
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(path, target)
                        with data_workspace(workspace, cache=cache):
                            catalog = src.catalog(client)
                        if catalog["dataset_id"].n_unique() != catalog.height:
                            raise ValueError("Fresh catalogue contains duplicate dataset IDs")
                        refs = {row["dataset_id"]: ref_from_row(source, row) for row in catalog.iter_rows(named=True)
                                if row["dataset_id"] in dataset_ids}
                        source_error = None
                    except (KeyboardInterrupt, StopRequested):
                        if client is not None:
                            client.close()
                        raise
                    except Exception as exc:
                        refs = {}
                        source_error = _error(exc, "catalogue or source initialization")
                    try:
                        for did in dataset_ids:
                            if stopped() or (limit is not None and summary["checked"] >= limit):
                                summary["stopped_early"] = True
                                break
                            log.info("Refresh %s/%s", source, did)
                            if source_error or did not in refs:
                                item = {"source": source, "dataset_id": did, "status": "failed",
                                        "error": source_error or "Dataset absent from the fresh catalogue; existing dataset retained"}
                            else:
                                item = _refresh_one(root, source, did, refs[did], src, client, cache, force, installed)
                            summary["results"].append(item)
                            summary["checked"] += 1
                            summary[item["status"]] += 1
                            if item["status"] == "failed":
                                log.warning("%s/%s: %s", source, did, item["error"])
                            _durable_json(report_path, summary)
                            if item.get("state_pending"):
                                summary["stopped_early"] = True
                                return summary
                    finally:
                        if client is not None:
                            client.close()
        except KeyboardInterrupt:
            summary.update(stopped_early=True, interrupted=True)
        except StopRequested:
            summary["stopped_early"] = True
        finally:
            summary["finished_at"] = now_iso()
            _durable_json(report_path, summary)
    return summary
