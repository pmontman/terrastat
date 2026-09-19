# Design review: ingestion to a reproducible forecasting experiment

Reviewed on 2026-09-17. Scope: source adapters, storage/state, snapshots, modeling helpers,
the existing notebooks, documentation and local tests. The checks use local data and fixtures;
they are not an exhaustive replay of every provider response or a corpus-wide quality audit.

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

## Next reliability work, in priority order

1. **Make snapshot publication transactional.** In `src/terrastat/export.py`, `export_shards`
   deletes existing shards before the replacement export finishes. Per-file atomic writes do
   not protect the whole release. Build in a sibling staging directory, validate checksums,
   then publish a new immutable snapshot/version. Test interruption before and after manifest
   publication. Until then, use a fresh snapshot name for every release.

2. **Recover from a truncated final state-log record.** `State.__init__` in
   `src/terrastat/storage.py` parses every JSONL line strictly. An interrupted append can leave
   an incomplete final line that prevents resume. Tolerate only a provably incomplete final
   record, report it, and continue from earlier events; corruption in the middle should fail.
   Also document/enforce single-writer ownership of a source's state and `.part` files.

3. **Bound the older curation path too.** `curate._on_grid` builds a date grid and a cross join
   with the selected series. The new calendar helper limits per-series reconstruction, but it
   does not retrofit that older path. Add period and total-cell limits before that cross join,
   especially for `trim="first_valid"`. Keep `fill="none"` explicit for missing-preserving work.

4. **Separate installation paths from writable data paths.** `config.PROJECT_ROOT` assumes a
   source checkout; wheel installs can derive a location inside site-packages. Define a user
   data default or require `TERRASTAT_DATA_DIR` outside editable installs. Also fix/document
   dotenv precedence: `load_dotenv()` does not override values already loaded, contrary to
   the existing comment saying the current-directory file wins. Test both installation modes.

5. **Define a stable release/schema contract.** Snapshots already record package version,
   selections and checksums. Add an explicit data-schema version and compatibility checks,
   a tested dependency constraints file for benchmark reproduction, and a release changelog.
   Keep broad library dependency ranges separate from the frozen experiment environment.

## Features that would improve research use

**Refresh is also missing:** resuming a crawl does not update completed datasets, and `--force`
can reuse cached raw payloads. A dataset-change-aware refresh command must account for historical
revisions as well as appended periods. See [the current behavior and intended design](refresh.md).

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
