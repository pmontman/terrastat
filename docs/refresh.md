# Resume a collection or refresh its data

Use `terrastat run` or `fetch` to finish an interrupted collection. Use **`terrastat refresh`**
to check already collected datasets for new provider data. Time passing does not trigger an
update automatically, and reading a saved snapshot continues to read that fixed snapshot.

## Refresh an existing collection

Start with a dataset you have already downloaded or deployed as normalized data:

```bash
uv run --no-sync terrastat refresh eurostat --ids une_rt_q
```

To check all locally collected datasets, or only selected providers:

```bash
uv run --no-sync terrastat refresh
uv run --no-sync terrastat refresh --sources eurostat oecd
uv run --no-sync terrastat refresh eurostat --limit 10 --max-hours 2
```

The command considers local normalized datasets under `data/datasets/`, including manually
deployed ones. It does not discover and download every new dataset in a provider's catalogue.
Use `fetch` to acquire additional datasets. `--ids` requires one explicitly selected provider;
use either the positional provider or `--sources`, not both. FRED refreshes require the usual
API key; Eurostat and OECD require none.

The summary reports checked, unchanged, updated and failed datasets, followed by the path to a
JSON report under `data/refresh/reports/`. `--json` prints the full result instead. A run with
failed datasets returns exit code 1; a keyboard interruption returns 130. An exhausted time or
dataset budget is reported as an early stop rather than as a data error. Each invocation starts
checking the selected datasets again; after a limited run, use `--ids` to select further datasets
or remove the limit to check the full collection.

## What gets downloaded?

Refresh obtains fresh payloads in a separate working area and compares them with the local
normalized data. The behavior depends on the provider:

| Provider | First refresh of a local dataset | Later refreshes |
|---|---|---|
| Eurostat | Downloads a fresh copy to establish a verified refresh baseline | Reads a fresh catalogue; a valid unchanged update token can avoid the payload download. Missing or unusable change metadata requires another download |
| OECD | Downloads a fresh copy and compares it with the local data | Downloads and compares again; no observation-level delta endpoint is used |
| FRED | Downloads a fresh copy and compares it with the local data | Downloads and compares again; no observation-level delta endpoint is used |

The initial Eurostat refresh downloads even when an older catalogue appears unchanged: a cached
catalogue or completed ingestion status does not prove that the local payload matches the current
provider version. Later token checks also compare local file sizes and modification times with
the recorded baseline. This detects ordinary local edits; it is not a content-integrity checksum.
Eurostat tokens include both data and structure update dates. Since these dates have day-level
precision, dates from today or yesterday (UTC) are not used to skip downloads. Missing or invalid
dates also require a download. Tokens are publisher change hints; use `--force` when you need a
fresh payload regardless of those hints.
If a selected dataset disappears from the fresh catalogue or the catalogue cannot be obtained,
refresh reports the problem and retains the local data.

To bypass the unchanged-token shortcut and request fresh payloads for every selected local
dataset, use:

```bash
uv run --no-sync terrastat refresh eurostat --ids une_rt_q --force
```

This is **dataset-level refresh**, not a promise to download only newly appended observations.
Historical observations can be revised or removed. An unchanged dataset may still require a
full download to establish that it is unchanged.

## Changes and their audit trail

Each report compares normalized observations by `(series_key, period)`. Added and removed counts
identify records present on only one side; revised counts identify changed values or status flags
at an existing key. Date and dimension changes are reported separately. Metadata-only changes can
also cause a dataset update even when its numerical observations are unchanged.

A changed dataset's normalized files and derived series are prepared before replacing its current
local files. Raw payloads are retained when the collection already retained them; refresh does not
automatically add a permanent raw layer to a compact deployed collection. Changed series are
rebuilt automatically with `min_obs=1`; dates, flags and missingness follow the existing sparse
storage contract. Existing FRED tags are carried forward; fetching newer tags remains a separate
`terrastat tags fred` operation.

Download, build or validation failures retain the previous data. A failed refresh is recorded in
its refresh report; it does not replace a previously completed ingestion status with a failure.
Refresh rejects malformed numerical fields, malformed rows, incomplete OECD slices and failed
Eurostat metadata requests rather than treating those omissions as provider deletions. An empty
replacement or a dataset missing from the provider catalogue also requires manual review.
Additional user files in a normalized dataset directory prevent its replacement; move research
notes and other custom files outside that managed directory before refreshing it.
Publication is recoverable across interruptions, but changing several directories can briefly
make a dataset unavailable to readers. Stop other crawlers, series builders, exports and readers
using the affected data while refreshing. Rerun refresh to recover an interrupted publication;
do not delete its working or backup files manually. Allow additional disk space for the prepared
replacement and its temporary files alongside the existing dataset.

If data publication succeeds but the state log cannot be updated, the report marks the dataset
as updated with `state_pending: true`, and the command returns exit code 1. Rerun refresh to finish
recovery before doing other work. Recovery uses a journal and checksums on the same local
filesystem; it does not provide a simultaneous view of all files to active readers.

Comparisons collect scalar counts with Polars, but exact key validation and joins require memory
that grows with the number of distinct observations in a dataset. Refresh processes one dataset
at a time. The first implementation does not promise fixed-memory processing for very large
dataflows.

**Existing snapshots are never overwritten by refresh.** Their byte contents and experimental
selection stay fixed. When you want a snapshot of the updated collection, export it under a new
name and retain its manifest and attribution files. See
[snapshot replacement and recovery](storage.md#build-and-replace-a-snapshot).

Refresh reports help audit changes between checks; they are not a complete archive of every
historical vintage or observation release date. A real-time forecasting study still needs an
explicit historical information set. Retain earlier snapshots when comparing model runs.

## Command distinctions

| Operation | Existing-data behavior |
|---|---|
| Quarterly notebook, snapshot mode | Reads the same local snapshot and retrains; no network or snapshot writes |
| Quarterly notebook, source mode | Reuses existing series files; fetches only when a requested file is absent |
| `terrastat run` / normal `fetch` | Resumes eligible unfinished work and skips datasets marked `done` |
| `terrastat refresh` | Checks locally collected datasets; downloads according to provider support and rebuilds changed series |
| `terrastat refresh SOURCE --force` | Requests fresh payloads even when a usable provider token is unchanged |
| `terrastat catalog SOURCE --refresh` | Refreshes the catalogue, not the stored observations |
| `terrastat fetch SOURCE --reprocess` | Reprocesses completed datasets; source adapters may reuse cached raw payloads |
| `terrastat fetch SOURCE --force` | Deprecated spelling of `--reprocess`; retains the same cached-data behavior and prints a warning |
| `terrastat series SOURCE --force` | Rebuilds the series view from local normalized observations; no provider download |
| `terrastat export NAME` | Packages local series; does not obtain new provider data |

`catalog --refresh` rebuilds the catalogue, but source adapters can still reuse their catalogue
payload cache for up to a day. The dataset `refresh` command uses a fresh working cache for its
provider checks.

`fetch --reprocess` is useful after parser changes. Eurostat can reuse `data.tsv.gz`, OECD can
reuse a completed raw file or split manifest, and FRED can reuse its completed page list. A new
processing timestamp therefore does not establish a fresh acquisition. Use the separate
`refresh` command when freshness is the objective.
