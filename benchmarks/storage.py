"""Measure archive savings on local data without changing the source files.

    python benchmarks/storage.py --data-dir data --out reports/storage-benchmark
    python benchmarks/storage.py --data-dir data --out reports/storage-benchmark \
        --parquet snapshot/starter/shard-00000.parquet

The archive experiment samples the median and approximately 8 MB Parquet/raw file
per provider, plus complete snapshots when they fit the per-group byte budget.
It compares exact, SHA-256-checked restoration. The optional Parquet experiment
preserves logical values but changes file bytes and, in its sorted cases, row
order. Its results are NOT deployable snapshots: their row indexes would need
rebuilding. No original files are overwritten or removed.

Timings are single, warm-cache local measurements, not throughput guarantees.
Archive compression uses bounded whole-group buffers to compare codec levels;
production archiving should stream. Parquet experiments decode a complete file
and can use much more RAM than its compressed size. Choose a small input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and not p.is_symlink())


def inventory(data: Path) -> list[dict]:
    rows = []
    for layer in sorted(data.iterdir()):
        if layer.is_dir() and not layer.is_symlink():
            files = regular_files(layer)
            rows.append({"layer": layer.name, "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
    return rows


def choose_groups(data: Path, max_bytes: int) -> dict[str, list[Path]]:
    groups = {}
    for layer in ("datasets", "series", "raw"):
        selected = []
        for provider in ("eurostat", "oecd", "fred"):
            files = regular_files(data / layer / provider)
            if layer != "raw":
                files = [p for p in files if p.suffix == ".parquet"]
            files = sorted(files, key=lambda p: p.stat().st_size)
            if files:
                selected.extend([files[len(files) // 2], min(files, key=lambda p: abs(p.stat().st_size - 8_000_000))])
        groups[f"{layer}_sample"] = list(dict.fromkeys(selected))
    groups["snapshots"] = regular_files(data / "snapshot")
    for name, files in groups.items():
        kept = []
        used = 0
        for path in files:
            size = path.stat().st_size
            if used + size <= max_bytes:
                kept.append(path)
                used += size
        groups[name] = kept
    return {name: files for name, files in groups.items() if files}


def archive_benchmark(data: Path, out: Path, name: str, files: list[Path]) -> dict:
    expected = {p.relative_to(data).as_posix(): sha256(p) for p in files}
    source_bytes = sum(p.stat().st_size for p in files)
    started = time.perf_counter()
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            info = archive.gettarinfo(str(path), arcname=path.relative_to(data).as_posix())
            # Avoid differences from owner metadata and filesystem timestamps.
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as source:
                archive.addfile(info, source)
    tar_bytes = tar_buffer.getvalue()
    del tar_buffer
    tar_seconds = time.perf_counter() - started
    result = {"group": name, "files": list(expected), "source_bytes": source_bytes, "tar_bytes": len(tar_bytes), "codecs": []}
    for method in ("gzip6", "zstd3", "zstd9"):
        with tempfile.TemporaryDirectory(prefix="storage-benchmark-", dir=out) as work:
            work = Path(work)
            packed = work / "archive"
            started = time.perf_counter()
            compressed = gzip.compress(tar_bytes, compresslevel=6, mtime=0) if method == "gzip6" else pa.Codec("zstd", compression_level=int(method[4:])).compress(tar_bytes)
            packed.write_bytes(compressed)
            del compressed
            pack_seconds = tar_seconds + time.perf_counter() - started
            destination = work / "deployed"
            destination.mkdir()
            started = time.perf_counter()
            with pa.CompressedInputStream(str(packed), "gzip" if method == "gzip6" else "zstd") as stream:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for member in archive:
                        target = destination / member.name
                        if not member.isfile() or not target.resolve().is_relative_to(destination.resolve()):
                            raise ValueError("Unexpected archive entry")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
            deploy_seconds = time.perf_counter() - started
            if {p: sha256(destination / p) for p in expected} != expected:
                raise ValueError("Restored files differ")
            row = {"codec": method, "archive_bytes": packed.stat().st_size, "saving_percent": 100 * (1 - packed.stat().st_size / source_bytes), "pack_seconds": pack_seconds, "deploy_seconds": deploy_seconds, "sha256_identical": True}
            result["codecs"].append(row)
            print(json.dumps({"group": name, **row}), flush=True)
    return result


def parquet_benchmark(data: Path, relative_path: str, out: Path, max_bytes: int) -> dict:
    source = (data / relative_path).resolve()
    if not source.is_relative_to(data) or not source.is_file():
        raise ValueError("--parquet must name a file inside --data-dir")
    if source.stat().st_size > max_bytes:
        raise ValueError("Parquet sample exceeds --max-group-mb")
    before_sha256 = sha256(source)
    started = time.perf_counter()
    table = pq.ParquetFile(source).read()
    original = pl.from_arrow(table)
    result = {"source": source.relative_to(data).as_posix(), "source_bytes": source.stat().st_size, "arrow_bytes": table.nbytes, "rows": table.num_rows, "source_read_seconds": time.perf_counter() - started, "variants": []}
    sort_columns = [c for c in ("dataset_id", "frequency", "series_uid") if c in original.columns]
    for sort_rows in ([False, True] if sort_columns else [False]):
        started = time.perf_counter()
        expected = original.sort(sort_columns) if sort_rows else original
        source_table = expected.to_arrow() if sort_rows else table
        sort_seconds = time.perf_counter() - started
        for row_group_rows in (2048, 65536):
            for level in (3, 9):
                with tempfile.TemporaryDirectory(prefix="parquet-benchmark-", dir=out) as work:
                    target = Path(work) / "repacked.parquet"
                    started = time.perf_counter()
                    pq.write_table(source_table, target, compression="zstd", compression_level=level, row_group_size=row_group_rows)
                    write_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    actual = pl.read_parquet(target)
                    read_seconds = time.perf_counter() - started
                    if not actual.equals(expected, null_equal=True):
                        raise ValueError("Repacked values differ")
                    row = {"sorted": sort_rows, "row_group_rows": row_group_rows, "zstd_level": level, "bytes": target.stat().st_size, "saving_percent": 100 * (1 - target.stat().st_size / source.stat().st_size), "sort_seconds": sort_seconds, "write_seconds": write_seconds, "read_seconds": read_seconds, "logical_values_equal": True}
                    result["variants"].append(row)
                    print(json.dumps(row), flush=True)
    if sha256(source) != before_sha256:
        raise ValueError("Source file changed during benchmark")
    return result


def compact_snapshots_benchmark(data: Path, out: Path, max_bytes: int) -> list[dict]:
    """Repack without reordering, limiting each buffered row group to ~128 MiB.

    A single input batch can exceed this budget if individual series are huge;
    the benchmark does not artificially truncate long series to enforce it.
    """
    snapshots = []
    for snapshot in sorted((data / "snapshot").iterdir()):
        if not snapshot.is_dir():
            continue
        files = regular_files(snapshot)
        total_bytes = sum(p.stat().st_size for p in files)
        if total_bytes > max_bytes:
            continue
        result = {"snapshot": snapshot.relative_to(data).as_posix(), "source_bytes": total_bytes, "variants": []}
        for level in (3, 9):
            rows = []
            for source in files:
                if source.suffix != ".parquet":
                    continue
                parquet = pq.ParquetFile(source)
                with tempfile.TemporaryDirectory(prefix="compact-benchmark-", dir=out) as work:
                    target = Path(work) / "repacked.parquet"
                    pending, pending_rows, pending_bytes = [], 0, 0
                    started = time.perf_counter()
                    with pq.ParquetWriter(target, parquet.schema_arrow, compression="zstd", compression_level=level) as writer:
                        for batch in parquet.iter_batches(batch_size=2048):
                            if pending and (pending_rows + batch.num_rows > 65536 or pending_bytes + batch.nbytes > 128 * 1024 * 1024):
                                writer.write_table(pa.Table.from_batches(pending), row_group_size=pending_rows)
                                pending, pending_rows, pending_bytes = [], 0, 0
                            pending.append(batch)
                            pending_rows += batch.num_rows
                            pending_bytes += batch.nbytes
                        if pending:
                            writer.write_table(pa.Table.from_batches(pending), row_group_size=pending_rows)
                    del pending
                    repack_seconds = time.perf_counter() - started
                    with pq.ParquetFile(target) as restored:
                        if not parquet.schema_arrow.equals(restored.schema_arrow, check_metadata=True):
                            raise ValueError("Repacked schema differs")
                        for expected, actual in itertools.zip_longest(parquet.iter_batches(batch_size=2048), restored.iter_batches(batch_size=2048)):
                            if expected is None or actual is None or not pl.from_arrow(expected).equals(pl.from_arrow(actual), null_equal=True):
                                raise ValueError("Repacked rows or values differ")
                        started = time.perf_counter()
                        for batch in restored.iter_batches(batch_size=2048):
                            pass
                        read_seconds = time.perf_counter() - started
                        rows.append({"file": source.relative_to(data).as_posix(), "source_bytes": source.stat().st_size, "compacted_bytes": target.stat().st_size, "source_row_groups": parquet.metadata.num_row_groups, "compacted_row_groups": restored.metadata.num_row_groups, "repack_seconds": repack_seconds, "read_seconds": read_seconds, "schema_rows_values_equal": True})
                parquet.close()
            compacted_bytes = total_bytes - sum(r["source_bytes"] for r in rows) + sum(r["compacted_bytes"] for r in rows)
            variant = {"zstd_level": level, "compacted_bytes": compacted_bytes, "saving_percent": 100 * (1 - compacted_bytes / total_bytes), "repack_seconds": sum(r["repack_seconds"] for r in rows), "read_seconds": sum(r["read_seconds"] for r in rows), "files": rows}
            result["variants"].append(variant)
            print(json.dumps({"snapshot": result["snapshot"], **{k: v for k, v in variant.items() if k != "files"}}), flush=True)
        snapshots.append(result)
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("reports/storage-benchmark"))
    parser.add_argument("--max-group-mb", type=float, default=256, help="Maximum compressed source bytes per sample group, in decimal MB")
    parser.add_argument("--parquet", help="Optional small Parquet file, relative to --data-dir, to compare row groups and sorting")
    parser.add_argument("--compact-snapshots", action="store_true", help="Compare streaming snapshot repacking without changing row order")
    parser.add_argument("--skip-archives", action="store_true", help="Only run the requested Parquet experiments")
    args = parser.parse_args()
    data, out = args.data_dir.resolve(), args.out.resolve()
    if not data.is_dir():
        parser.error("--data-dir must be an existing directory")
    if out.is_relative_to(data):
        parser.error("--out must be outside --data-dir")
    if not (0 < args.max_group_mb <= 1024):
        parser.error("--max-group-mb must be between 0 and 1024")
    out.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_group_mb * 1_000_000)
    report = {"python": platform.python_version(), "pyarrow": pa.__version__, "polars": pl.__version__, "timing_note": "Single warm-cache run. Archive pack includes tar construction and disk write; deploy includes writing restored files. Hash verification is outside timings. Samples are not a corpus-wide forecast.", "inventory": inventory(data), "archives": []}
    report_path = out / "storage-results.json"
    for name, files in ({} if args.skip_archives else choose_groups(data, max_bytes)).items():
        report["archives"].append(archive_benchmark(data, out, name, files))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.parquet:
        report["parquet"] = parquet_benchmark(data, args.parquet, out, max_bytes)
    if args.compact_snapshots:
        report["compact_snapshots"] = compact_snapshots_benchmark(data, out, max_bytes)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
