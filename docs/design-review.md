# Design review: ingestion to a reproducible forecasting experiment

Reviewed on 2026-09-17. Scope: source adapters, storage/state, snapshots, modeling helpers,
the existing notebooks, documentation and local tests. The checks use local data and fixtures;
they are not an exhaustive replay of every provider response or a corpus-wide quality audit.
Snapshot-publication and state-log recovery follow-up: 2026-09-21.
Dataset-refresh follow-up: 2026-09-27.

## Assessment

The ingestion architecture is a sound starting point. Source-specific parsing lives behind
adapters, the pipeline owns orchestration, HTTP pacing/retries are centralized, and the files
are readable with independent Parquet tools. Atomic per-file writes, streaming paths, checksums,
provenance columns and substantial parser fixtures already exist. Keep these separations; a
rewrite into a service or database is not justified by the issues found here.

The weaker boundary was between sparse storage and model-ready arrays. Tutorial snippets also
promised stronger evaluation guarantees than their code implemented. The changes below make
that boundary explicit and add a working forecasting example.

## Corrected in this pass

| Finding | Consequence | Change |
|---|---|---|
| `dataset.load` scanned `index.parquet` with shards | Mixed schemas make snapshots fail to load | Select shard files; exclude lookup index |
| Projection preceded filters | Requesting only IDs removed columns needed by filters | Filter before projecting; regression test |
| `windows` removed NaNs and ignored dates | Nonadjacent quarters became adjacent timesteps | Bounded calendar reconstruction, masks and target dates |
| Undefined seasonal scales dropped windows | Flat or incomplete histories silently disappeared | Retain windows with NaN scale; report score coverage |
| Invalid split fractions/fold counts accepted | Degenerate assignments or arithmetic errors | Validate arguments; reserve at least one bucket per side |
| Site example unpacked a frame and a dict as array pairs | Copy-paste training example did not match the API | Use explicit fold filtering and dictionary keys |
| Schema called explicit null counts all gaps | Sparse data looked more complete than it was | Document explicit nulls versus absent calendar periods |
| Group-split claims were absolute | Users could mistake grouping for proof of independence | Separate chronological forecasting from group transfer |
| No trained model or statistical benchmark in the getting-started tour | No end-to-end example to verify | Executed quarterly CPU notebook and offline CI smoke path |

## Implemented reliability follow-up

**Verified snapshot replacement.** Previously, `export_shards` deleted existing shards before
their replacements were complete. The exporter now builds and verifies the full snapshot in a
sibling staging directory under a destination lock. Build or verification failure leaves the
published snapshot intact. Publication moves an existing directory to a backup, then moves the
verified build into place; an ordinary failure attempts rollback, and the next export to the
same name recovers an interrupted publication. If both destination and backup remain, recovery
verifies the destination before discarding the backup; failed verification preserves both.
Unknown user files in an existing destination prevent replacement. Existing snapshots require
no migration. Validation includes interrupted builds and publication failures, checksum and
row-count checks, and subprocesses terminated during publication.

The two directory renames can briefly make the destination unavailable to readers. This is not
uninterrupted replacement or a full power-loss durability guarantee. Use immutable release names
for active studies and budget for the old snapshot, new build and shuffle intermediates. The
reserved staging/backup paths support recovery, and a persistent lock file is expected; users
should not delete them manually. See [snapshot storage](storage.md#build-and-replace-a-snapshot).

**State-log tail recovery.** State operations now preserve valid final records without a newline
and recover only a syntactically incomplete final JSON object without a newline. Corruption
earlier in the file, malformed terminated records, complete malformed records and invalid events
still fail. Appends are flushed and synchronized
before the in-memory status changes. Operation locks protect log reads, recovery and appends;
they do not give one process ownership of an entire source ingestion run or its `.part` files.
Continue to run one writer per source. See [recovery in the manual](../MANUAL.md#5-stopping-resuming-failures).

**Explicit dataset refresh.** `terrastat refresh` checks existing normalized datasets, obtains
fresh provider data when needed, compares observation records and metadata, and rebuilds changed
series. The first check establishes a fresh baseline; subsequent Eurostat checks can skip a
payload only when current catalogue metadata and the recorded local baseline permit it. FRED
and OECD use fresh downloads and comparison. This is not an observation-level delta protocol.

Per-run JSON reports retain additions, removals and revisions. Prepared replacements and recovery
protect the previous data on download, parsing, validation or pre-commit publication failure;
existing snapshots are not changed. A committed publication with pending state updates is
reported separately and requires recovery. Strict refresh parsing rejects malformed observations
and incomplete downloads before comparison. Normal `run`/`fetch`
still resume ingestion. `fetch --reprocess` names the old cached-data behavior; its `--force`
spelling remains supported with a deprecation warning. Provider vintage archives, automatic
new-dataset discovery and a full historical release-time model remain separate work.

## Next reliability work, in priority order

1. **Bound the older curation path too.** `curate._on_grid` builds a date grid and a cross join
   with the selected series. The new calendar helper limits per-series reconstruction, but it
   does not retrofit that older path. Add period and total-cell limits before that cross join,
   especially for `trim="first_valid"`. Keep `fill="none"` explicit for missing-preserving work.

2. **Separate installation paths from writable data paths.** `config.PROJECT_ROOT` assumes a
   source checkout; wheel installs can derive a location inside site-packages. Define a user
   data default or require `TERRASTAT_DATA_DIR` outside editable installs. Also fix/document
   dotenv precedence: `load_dotenv()` does not override values already loaded, contrary to
   the existing comment saying the current-directory file wins. Test both installation modes.

3. **Define a stable release/schema contract.** Snapshots already record package version,
   selections and checksums. Add an explicit data-schema version and compatibility checks,
   a tested dependency constraints file for benchmark reproduction, and a release changelog.
   Keep broad library dependency ranges separate from the frozen experiment environment.

4. **Coordinate complete source-writing operations.** Destination and state-operation locks do
   not protect every crawler, metadata refresh, series build or archive read from concurrent
   changes. Define source-level ownership before supporting multiple writers to the same data
   root. For now, stop input writers before exports or packing and keep one writer per source.

## Features that would improve research use

The refresh command now accounts for additions, removals and historical revisions in collected
data. Further download savings depend on provider-specific change or delta interfaces; FRED and
OECD currently require a full comparison download. See [refreshing data](refresh.md) for the
implemented behavior and its limits.

| Feature | Why it matters | First useful deliverable |
|---|---|---|
| Published starter snapshot | Removes a long crawl from onboarding | Small checksummed, versioned bundle with provenance and an executable example |
| Saved evaluation protocol | Separates model changes from changing data/splits | Persist cohort IDs, calendar origins, masks and family assignments |
| Family and duplicate index | Supports transfer claims across indicators | Exact-value candidates plus source/unit/aggregation links, followed by numerical checks |
| Vintage and release-time data | Enables historical information-set evaluation | One provider/indicator with observation vintage and release timestamp |
| Forecast uncertainty | Point errors alone cannot assess interval calibration | Seasonal baseline intervals and empirical coverage across rolling origins |
| Broader validation matrix | Linux-only tests miss Windows and minimum-version issues | Python 3.11/3.12 plus Windows; wheel installation/import smoke |

For the next modeling iteration, vary context length and compare a linear shared model before
increasing neural capacity. Add dataset/family holdouts only after defining the transfer claim,
and retain the chronological cutoff in all training data. Report multiple seeds, per-dataset
scores and blocked uncertainty estimates rather than treating related series as independent.

Validation commands and tutorial details live in [CONTRIBUTING.md](../CONTRIBUTING.md) and
[forecasting.md](forecasting.md). The generated corpus pages remain derived from the committed
tables; they were not manually rewritten to make the review appear current.
