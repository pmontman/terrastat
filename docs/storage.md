# Store data, then deploy it for analysis

**Received a normalized release and want to use it?** Start with the
[saved-data deployment walkthrough](deployment.md), with separate routes for online archives
and files downloaded manually. This page explains packing, storage choices and the measurements.

Keep Parquet as the working format. It already compresses the numerical values, dates and
repeated metadata, and researchers can read it without terrastat. For cold storage or transfer,
`pack` wraps an existing directory in a standard **tar.zst** archive; `deploy` verifies it and
restores the original files. No values are rounded, missing periods filled or metadata dropped.

The largest disk saving comes from choosing which **layers** need to be on your working machine.
An extra compressor around already compressed files usually saves little. Do not replace your
only copy of a research dataset until you have verified its archive and tested restoration.

## Pack and restore an existing snapshot

Stop any process writing to the selected directory, then run from the repository root:

```bash
uv run --no-sync terrastat pack data/snapshot/starter reports/archives/starter-v1.tar.zst
uv run --no-sync terrastat deploy reports/archives/starter-v1.tar.zst data/snapshot/starter-restored
```

The first command prints the archive's SHA-256, original byte count and archive byte count.
Keep that checksum with your release record. The second creates a **new** directory: it refuses
to overwrite an existing snapshot. Open the restored copy as usual:

```python
from terrastat import dataset

quarterly = dataset.load("starter-restored", frequencies=["Q"])
```

All files travel together, including `index.parquet`, the snapshot manifest, dataset card,
attributions and Croissant metadata where present. Older snapshots without per-file checksums
also work: the archive adds its own inventory without changing their original manifests.
The archive, source and restored copy all remain on disk; these commands do not free space by
deleting anything. `reports/` is ignored by Git; move archives intended for long-term retention
to your chosen storage and keep their checksums.

The same commands work on an explicit data subtree, such as `data/datasets/eurostat`, or on a
stopped `data/` tree. Use one archive per snapshot, source or manageable dataset group rather
than requiring an entire corpus download for every experiment. Archives cannot be placed inside
their input directory. Incomplete `.part`/`.tmp` files and links are rejected rather than silently
omitted. Select data directories, not a repository containing `.git` or `.env`.

## Store online and deploy later

An archive is an ordinary file: an institutional repository, object store or private file server
can hold it. Uploading is separate from packing. Use a versioned, immutable object name and retain
its checksum and citation information. No storage account or dataset has been published by this
change.

For a server providing an HTTP(S) download, pass its archive URL and the SHA-256 from `pack` to
`terrastat deploy`, using `--sha256` for the checksum. In Python, with those release values in
`archive_url` and `archive_sha256`:

```python
from terrastat import datasets

path = datasets.deploy(
    archive_url,
    "data/snapshot/downloaded-release",
    sha256=archive_sha256,
)
```

Downloads resume when the server supports byte ranges. Verified archives are cached by checksum
and reused; a server ignoring ranges causes a clean restart. The default cache is under the
normal terrastat cache directory (`TERRASTAT_CACHE` overrides it). `--cache` selects an archive
cache explicitly. An interrupted extraction restarts from that local archive; it does not need
to download again. The new destination appears only after every file passes verification.

Budget disk space for the compressed archive **plus** its expanded contents. Deployment limits
expanded payloads to 1 TiB by default; set `--max-gb` to an appropriate GiB limit for your release.
HTTP downloads require an expected archive checksum; obtain it from a trusted release record.
Archive checksums establish integrity, not permission to redistribute data. Retain and review
the provider terms described in [licensing and citation](licensing.md), including FRED restrictions.

For public datasets that people should preview or query remotely, publish the **Parquet shards
and companions directly**. The existing `datasets.fetch()` supports that arrangement, including
individual shard selection. A tar.zst archive requires whole-archive download and extraction and
cannot provide Parquet column/row-group access over HTTP. Hugging Face's
[dataset upload documentation](https://huggingface.co/docs/hub/datasets-adding) describes direct
Parquet publication. Cold archives and directly readable releases serve different uses.

## Which layers should be retained?

The local inventory measured on 2026-09-20 was approximately:

| Layer | Size, decimal GB | Role |
|---|---:|---|
| `raw/` | 31.72 | Downloaded provider payloads; useful for auditing and re-parsing |
| `datasets/` | 17.22 | Normalized observations and dataset metadata |
| `series/` | 31.86 | Derived per-series tables, expanded metadata and numerical lists |
| `snapshot/` | 0.204 | Three specific experiment selections, not the full corpus |
| `cache/` | 0.121 | Reference metadata, including fetched FRED tags |

These are file lengths, not filesystem allocation or a forecast of another crawl's size.
Keeping raw and normalized data while rebuilding `series/` on demand avoids storing about
**32 GB of derived files**, roughly 39% of this full local tree. That saving is distinct from
compression. A 204 MB snapshot covers only its selected series; it cannot replace the full corpus.

For a compact offline modeling source, retain `datasets/`, the FRED tag cache and exact experiment
snapshots. Together, the normalized tables, entire cache and current snapshots occupy about
17.55 GB before outer compression. This omits raw payloads and the full derived series layer;
it is **not** an equivalent preservation copy of the entire crawl. Keep raw data separately when
you need to audit provider responses or repeat parsing. Keep catalogues and state logs when you
need to resume ingestion.

### Rebuild the series layer offline

For the approximately 17.3 GB modeling source (normalized data plus the whole cache), pack
the two directories independently:

```bash
uv run --no-sync terrastat pack data/datasets reports/archives/observations-v1.tar.zst
uv run --no-sync terrastat pack data/cache reports/archives/reference-v1.tar.zst
```

That size is their current **unpacked** size. The sampled compression ratios above do not
establish the final size of these two complete archives. Keep both archives with the code
revision and dependency versions. To deploy them into a fresh location and build local views:

```python
import os
from pathlib import Path
from terrastat import datasets
from terrastat.series import build_series

root = Path("data-deployed").resolve()
datasets.deploy("reports/archives/observations-v1.tar.zst", root / "datasets")
datasets.deploy("reports/archives/reference-v1.tar.zst", root / "cache")
os.environ["TERRASTAT_DATA_DIR"] = str(root)
for source in ("eurostat", "oecd", "fred"):
    if (root / "datasets" / source).is_dir():
        build_series(source, force=True, min_obs=1)
```

The archive cache, normalized files and derived files occupy space independently. Keeping both
normalized data and the complete series view is roughly 49 GB for this corpus, before any raw
files, saved snapshots or compressed archive copies. You can rebuild only one source, or pass
`dataset_ids=["une_rt_q"]` to `build_series` for a specific dataset. Archive and test exact
experiment snapshots separately.

Restore normalized data to `data/datasets/` and cached metadata to `data/cache/`. In particular,
FRED's `cache/fred/series_tags.parquet` and `cache/fred/tags.parquet` preserve fetched tags and
their expanded labels. Then run only the sources present in your restored data:

```bash
uv run --no-sync terrastat series eurostat --force --min-obs 1
uv run --no-sync terrastat series oecd --force --min-obs 1
uv run --no-sync terrastat series fred --force --min-obs 1
```

These commands use local normalized data; they do not fetch provider observations. `--force`
matters if restored state logs already mark the missing series files as done. Use the original
minimum-observation threshold if it differed from 1. Preserve the code revision and dependency
versions too: rebuilding with changed parsing, descriptions or licensing code can change the
derived result. Rebuilding is not a promise of byte-identical Parquet, and regenerating a shuffled
snapshot also depends on its selection, seed, precision and exact inputs. Archive existing
experiment snapshots directly when you need exact restoration.

As a local check, rebuilding Eurostat `une_rt_q` reproduced all 7,152 series and every column
of the existing series view, with HTTP requests blocked. This validates the route on that
dataset, not every historical dataset or the run time for the full corpus.

## What the measurements say

The [benchmark script](../benchmarks/storage.py) inventories the data and compares compression
on bounded local samples. Measurements below are a single local run with warm filesystem caches,
Python 3.12.7 and PyArrow 25.0.1. Each source contributes a file near its median size and one
near 8 MB; these samples do not establish corpus-wide compression ratios. Snapshot figures cover all 19 files
in the three existing snapshots. MB/GB use decimal units.

Reproduce the codec comparison with `python benchmarks/storage.py --data-dir data --out reports/storage-benchmark`.
Add `--parquet snapshot/starter/shard-00000.parquet` for the small-file layout experiment, or
`--skip-archives --compact-snapshots` for a streaming comparison across existing snapshots.

| Input | Original MB | tar + Zstd level 3 MB | Saved | Pack / unpack seconds |
|---|---:|---:|---:|---:|
| Normalized Parquet sample | 23.94 | 21.18 | 11.55% | 0.27 / 0.11 |
| Series Parquet sample | 24.57 | 24.06 | 2.07% | 0.21 / 0.08 |
| All existing snapshots | 204.29 | 201.37 | 1.43% | 1.97 / 0.71 |
| Raw gzip sample | 19.82 | 19.78 | 0.21% | 0.15 / 0.06 |

These codec timings include file writes but exclude checksum validation; the production commands
also hash their inputs and outputs. They are not network-transfer or cold-disk timings. Gzip was
slower and generally larger than Zstd. Zstd level 9 added only fractions of a percentage point to
the outer archive savings, so `pack` uses Arrow's default Zstd stream without a new dependency
or an expensive maximum-compression mode. Arrow 25.0.1 defaults to level 1 for this stream:
the actual commands packed the three snapshots separately into **202.51 MB** in **2.13 seconds**
and restored and verified them in **1.47 seconds**. All 19 files matched their original SHA-256;
the source snapshots were unchanged. This is about **0.87%** saved, useful for packaging but
not a significant storage reduction. A custom numerical codec, lossy quantization or
calendar encoding is not justified by this experiment.

### Denser Parquet for new exports

Row-group layout matters more than wrapping Parquet in another compressor. On the `starter`
data shard, using the same Polars writer with 65,536-row groups reduced 28.19 MB to **22.31 MB**
at Zstd level 3 (20.9% saved), or **20.11 MB** at level 9 (28.7% saved). Writes took 0.48 and
0.93 seconds respectively, excluding the initial read and subsequent equality check. Both
preserved row order, column types, values and missingness. These figures describe one shard,
not a promised reduction across the normalized corpus.

For a **new** snapshot, you can try the measured settings:

```bash
uv run --no-sync terrastat export compact_v1 --public-only --per-dataset 100 --row-group-rows 65536 --compression-level 9
```

The larger row groups and codec level are recorded in its manifest. Shuffling, index positions
and value precision follow the same export settings as before. This is still ordinary Parquet;
it can be read directly or packed and deployed. Existing snapshots and default export settings
are unchanged. Use a new name: the older export command replaces an existing named snapshot.

The broader PyArrow streaming experiment (row groups capped near 128 MiB and 65,536 rows)
confirms that results depend on the data: level 9 saved 16.2% on `demo_bundle` and 24.1% on
`starter`, but made `getting_started` **22.8% larger**. That is why the project exposes explicit
settings for new exports instead of automatically rewriting existing snapshots.

Larger groups increase read buffers and can make small reads less efficient. The appropriate
size depends on series lengths, not just row count; [Parquet's configuration guide](https://parquet.apache.org/docs/file-format/configurations/)
describes the buffering and I/O tradeoff. Sorting the starter shard by dataset and series
compressed further, but would alter the shuffled layout and require index reconstruction, so it
is only a benchmark experiment. There is no custom compact binary format to maintain.

## Format and verification

The archive is standard tar inside a Zstandard stream. Its first member is `archive.json`
(`format: terrastat-archive`, `version: 1`), followed by regular files under `payload/`. The
inventory records every relative filename, byte count and SHA-256. Independent tar and Zstandard
tools can read it; terrastat supplies verified restoration and safe path handling.

`deploy` preserves file contents and relative names, not original filesystem owners, permissions
or modification times. It rejects unsafe paths, links, duplicate or unexpected files, unsupported
format versions, expansion beyond the configured bound and checksum mismatches. Packing checks
that input files have not changed during the operation, but it does not lock the crawler: pause
writers before archiving. Failed operations do not publish a partially restored destination.

This follows established practices rather than inventing a new time-series format: keep
[Parquet's native compression](https://parquet.apache.org/docs/file-format/data-pages/compression/),
use a [standard fast compressor](https://facebook.github.io/zstd/) for transport, and keep a
complete checksum inventory as in preservation formats such as
[BagIt](https://www.rfc-editor.org/rfc/rfc8493.html). This archive is not a BagIt implementation.
