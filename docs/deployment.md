# Start from a saved normalized dataset

Use this route when someone has already collected and shared the normalized data. You can
download their saved release, or use archive files you downloaded manually. In both cases,
terrastat restores the normalized tables and rebuilds the convenient per-series view locally.
It does not need to download the observations again from Eurostat, OECD or FRED.

**There is no official normalized release URL configured yet.** These instructions work with
a release you host or receive. The person sharing it must provide two archives made with
`terrastat pack`, their SHA-256 checksums, and the release's code/dependency versions:

| Archive | Contains | Restored to |
|---|---|---|
| Normalized observations | Contents of `data/datasets/`, including dataset metadata | `datasets/` |
| Reference metadata | Contents of `data/cache/`, including FRED tags when collected | `cache/` |

Keep the two archives from the **same release**. The [storage guide](storage.md#rebuild-the-series-layer-offline)
shows how to prepare them. About 17.3 GB refers to the unpacked normalized data plus cache
measured locally, not a promised download size. Rebuilding the full series view adds about
32 GB; allow roughly 49 GB plus the archive files. A smaller release needs less space.

## 1. Prepare the environment

Follow the [installation instructions](../README.md#1-install), using the release's documented
code and dependency versions when available. Open a Python notebook from the repository root.
Use a fresh kernel when switching between releases, so cached metadata from another release
is not reused.

```python
import os
from pathlib import Path
from terrastat import dataset, datasets
from terrastat.series import build_series

# Use a new directory so an existing collection stays separate.
root = Path("data/deployed-v1").resolve()
```

## 2. Choose where the archives come from

Run **one** of the following blocks, then continue to step 3.

### A. Download a saved release directly

Paste the two HTTP(S) archive download URLs supplied by the release owner. These should point
to the archive files, not a repository's landing page. No example server is assumed here.

```python
archives = {
    "datasets": input("Normalized observations archive URL: ").strip(),
    "cache": input("Reference metadata archive URL: ").strip(),
}
```

### B. Use archives downloaded manually

Download both files with your browser or copy them from a drive. Store them under
`data/downloads/`, for example, and enter their paths below. Keep them compressed: `deploy`
will unpack them. Paths can be relative to the repository root or absolute.

```python
archives = {
    "datasets": Path(input("Normalized observations archive file: ").strip()).expanduser(),
    "cache": Path(input("Reference metadata archive file: ").strip()).expanduser(),
}
```

## 3. Verify and restore the normalized data

Paste the checksums supplied with the release. The same calls handle URLs and local files:

```python
checksums = {
    "datasets": input("Normalized observations SHA-256: ").strip(),
    "cache": input("Reference metadata SHA-256: ").strip(),
}
for folder in ("datasets", "cache"):
    datasets.deploy(archives[folder], root / folder, sha256=checksums[folder])
```

For URLs, terrastat downloads to its archive cache, resumes interrupted transfers when the
server supports it, and reuses a verified cached archive. For local files, this step stays
offline. Both routes check the archive checksum and the size and checksum of every restored
file. A destination appears only after its extraction succeeds; existing destinations are
never overwritten.

If the first archive was restored but the second failed, rerun only the failed call. Do not
delete a completed deployment to retry a download. If you already have the unpacked normalized
directories, skip steps 2–3 and set `root` to the folder containing `datasets/` and `cache/`.

## 4. Build the easy-to-use series view

Unpacking restores the normalized files. This separate step produces one row per series,
with dates, values, flags, descriptions and source/licensing metadata:

```python
os.environ["TERRASTAT_DATA_DIR"] = str(root)
sources = [source for source in ("eurostat", "oecd", "fred")
           if (root / "datasets" / source).is_dir()]
if not sources:
    raise RuntimeError("No source directories found under the restored datasets/ folder.")

for source in sources:
    result = build_series(source, force=True, min_obs=1)
    print(source, result)
    if result["failed"]:
        raise RuntimeError(f"Some {source} datasets failed to build; inspect the reported errors.")
```

This step reads local files and does not call the providers. It retains missing observations
within the series; it does not fill calendar gaps. `min_obs=1` selects series with at least one
observed value. Use the release's original threshold if it differed. `force=True` ensures that
old state logs cannot make a missing derived file look finished.

To try a small subset first, replace the loop with a specific dataset present in the release:
`build_series("eurostat", dataset_ids=["une_rt_q"], force=True, min_obs=1)`.

## 5. Read the deployed data

In the same kernel, `source=None` reads the series layer under the data root set above:

```python
quarterly = dataset.load(source=None, frequencies=["Q"])
sample = quarterly.head(5).collect()
sample.select("series_uid", "title", "units", "n_obs")
```

In a new notebook or terminal, set `TERRASTAT_DATA_DIR` to this deployed root again. Set it
in `.env` if this should become the project's usual data location; see the
[configuration reference](reference.md#install).

A normalized release does not automatically create a named `starter` snapshot. To use the
[forecasting tutorial](../notebooks/quarterly_forecasting.ipynb) with an exact saved experiment,
restore that experiment's snapshot separately and set its `SNAPSHOT` path. You can also create
a new snapshot from the rebuilt series using [export](../MANUAL.md#6b-a-snapshot-for-training-export).

Archives restore file contents exactly. Rebuilding derived views with different software
versions can change descriptions, classifications or serialization, so retain release versions
and exact experiment snapshots for replication. Provider licences still apply to saved releases;
keep their attribution and review [licensing and citation](licensing.md).
