# terrastat

Polite, resumable gathering of public economic time series (FRED, Eurostat, OECD) into a tidy,
licence-aware Parquet database, with a model-ready "one row per series" view on top.

New here? Read [MANUAL.md](MANUAL.md) first: it walks through opening a terminal, running the
commands, stopping and resuming, and reading the files. This README is the technical reference.

Design priorities, in order: clarity and ease of use, then performance. Everything is plain
Parquet and JSON on disk; there is no database server and no custom binary format.

| document | what it is for |
|---|---|
| [docs/index.html](docs/index.html) | **generated** — the project page: the facts, the tables, and how to start |
| [docs/corpus.md](docs/corpus.md) | **generated** — the same figures as a markdown annex, for reading in the repo |
| [MANUAL.md](MANUAL.md) | the walkthrough: running it, stopping it, reading the output |
| [docs/schema.md](docs/schema.md) | the four layers and all 36 series columns, with diagrams |
| [docs/leakage.md](docs/leakage.md) | twelve ways this corpus will contaminate a train/test split |
| this README | technical reference: commands, politeness, licences, scale |

> Not affiliated with, endorsed by, or connected to Eurostat, the OECD, the Federal Reserve Bank of St. Louis, or any statistical agency. The name is a description of what the tool gathers, not a claim of provenance.

**Licensing in one line:** the code here is Apache-2.0; the data it downloads is **not**, and
carries the terms of whoever published it. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the
[Licences](#licences) section. Use `terrastat export --public-only` for anything that leaves your
machine.

## Install

```bash
cd terrastat
uv venv --python 3.12
uv pip install -e ".[dev]"          # dev adds pytest and duckdb
copy .env.example .env               # then put your FRED key in .env
```

The only secret is `FRED_API_KEY` (free at https://fred.stlouisfed.org/docs/api/api_key.html).
Eurostat and OECD need no key. Data lands in `terrastat/data/` unless `TERRASTAT_DATA_DIR` is set.

### Running it

Three equivalent ways, all from the `terrastat` folder:

```powershell
# PowerShell: activate once per terminal, then use the plain command
.\.venv\Scripts\Activate.ps1
terrastat status eurostat
```

```bat
:: cmd.exe
.venv\Scripts\activate.bat
terrastat status eurostat
```

```bash
# no activation at all (any shell), or via uv
.venv\Scripts\terrastat status eurostat
uv run --no-sync terrastat status eurostat
```

If PowerShell refuses to run the activation script, start it once with
`powershell -ExecutionPolicy Bypass` or simply use the no-activation form.

### Stopping and resuming

A run can be stopped at any time (Ctrl+C, closing the terminal, `--max-hours`, a crash) and
started again with the same command; it continues with the datasets not yet finished:

1. `terrastat catalog <source>` fetches the full list of tables once and caches it
   (`data/catalog/<source>.parquet`). Add `--csv path.csv` to get the list as a spreadsheet with
   a `status` column (todo, raw, done, failed).
2. `terrastat fetch` walks that list in a fixed order (smallest first by default) and appends one
   line per finished dataset to `data/state/<source>.jsonl`: `done`, `raw` (raw-only mode) or
   `failed` with the error.
3. On the next run every dataset already `done` (or `raw`, in raw-only mode) is skipped. A dataset
   that was interrupted half-way has no state line and only a `.part` file on disk; it is simply
   redone. FRED releases are the exception: their pages are checkpointed with the API cursor, so
   an interrupted release resumes at the page where it stopped.
4. `terrastat status <source>` shows the counts, and `--retry-failed` re-attempts the failures.

Tested by killing a 45-second download after 10 seconds: the rerun redid that dataset and
finished, a third run skipped it.

## Three layers

```
data/
  raw/<source>/<dataset>/...                  exact payloads as downloaded (gzip), for reproducibility
  datasets/<source>/<dataset>/dataset.json    every piece of metadata the source gives: title, licence,
                                              attribution text, links, dimensions with code -> label maps
  datasets/<source>/<dataset>/observations.parquet   compact long table, one row per observation
  datasets/<source>/<dataset>/series.parquet         per-series metadata (FRED only: title, units, ...)
  series/<source>/freq=<F>/<dataset>.parquet  one row per series, metadata expanded to text
  catalog/<source>.parquet                    what the source offers (sizes, frequencies, links)
  cache/<source>/                             shared reference data (Eurostat codelists, catalogues)
  state/<source>.jsonl                        which datasets are done / failed, with timings and errors
```

**Layer 1, datasets.** The natural bulk unit of each source: a FRED *release*, a Eurostat
*dataset*, an OECD *dataflow*. `observations.parquet` always has `series_key, period, date,
value, flag`, plus one column per dimension for Eurostat and OECD (codes, dictionary-encoded, so
the redundancy costs almost nothing). Cells that are simply missing are not stored; cells with a
status flag but no value (confidential, not applicable) are kept so the information survives.
This is the layer to use for anything table-shaped: DuckDB, pandas, polars, R.

**Layer 2, series.** Built from layer 1 without touching the network. One row per series with
parallel list columns `dates`, `values`, `flags`, and the metadata turned into columns and into
one readable `description` sentence for text encoders. Same schema for all sources, partitioned
by canonical frequency, so the whole tree opens as one table:

```python
import polars as pl
df = pl.scan_parquet("data/series/**/*.parquet").filter(pl.col("frequency") == "M").collect()
```

```sql
-- duckdb
select source, frequency, count(*) from 'data/series/**/*.parquet' group by all;
```

Series columns: `series_uid, source, source_id, dataset_id, dataset_title, title, description,
frequency, frequency_raw, units, seasonal_adjustment, geo, geo_label, dimensions (list of
{id, name, code, label}), tags, notes, license_id, license_name, license_url, license_detail,
attribution, license_notes, origin_agencies, origin_us_federal, source_url, last_updated,
retrieved_at, vintage, start_date, end_date, n_points, n_obs, n_flagged, dates, values, flags`.

The three lists always have `n_points` elements, in date order. `values` can contain nulls where
the source published a flag without a number (confidential, not applicable); `n_obs` counts the
non-null values, so `n_obs <= n_points`. Purely missing cells are never stored.

Layer 2 can be many millions of rows for Eurostat (most Eurostat "series" are short
cross-sectional cells). That is why it is derived, filtered (`--min-obs`), and written per
dataset and per frequency: you only build the parts you need, and layer 1 stays compact.

## Commands

The one command that does everything, and that you re-run until it says nothing is left:

```bash
terrastat run                          # all sources, all frequencies, tidy + series view, one connection per source
terrastat run --max-hours 8            # same, but stop cleanly after 8 h; run again to continue
```

`run` downloads the catalogues, walks every source in its own thread (one paced connection each,
so each source sees exactly the same load as a single polite user), records every dataset as it
finishes, retries failures once at the end, and builds the series view. Ctrl+C (or the time
budget) stops it within seconds: a download in progress is abandoned and redone next time, a FRED
release resumes at the page it reached. The finer-grained commands underneath:

```bash
terrastat catalog eurostat            # download the catalogue once; shows sizes and frequencies
terrastat fetch eurostat --freq M     # all monthly datasets, smallest first, one every few seconds
terrastat fetch eurostat --freq M --raw-only     # one request per dataset, tidy later
terrastat fetch eurostat --freq A Q --max-hours 6      # stop cleanly after 6 h, re-run to resume
terrastat fetch fred                  # every FRED release, all frequencies (annual included)
terrastat fetch fred --ids 10 53      # just these releases
terrastat fetch oecd --limit 50
terrastat series fred --min-obs 24    # build the per-series view (skips datasets already built)
terrastat fetch eurostat --freq M --with-series   # build the series view right after each dataset
terrastat status fred                 # progress, counts, first failures
terrastat peek eurostat nama_10_gdp   # dataset.json summary plus a few rows
```

Run `terrastat <command> --help` for every option.

### Understanding and choosing, once the data is on disk

**Start with the project page, [docs/index.html](docs/index.html)** (or its markdown twin,
[docs/corpus.md](docs/corpus.md)) — how much data there is, how long it is, how much is still
being published and how concentrated it is. Both are generated from the same tables by
`terrastat corpus`, never written by hand, so neither can drift from the corpus or from the other.
Open the HTML file in a browser, or serve `docs/` with GitHub Pages.

The crawl is only half of it. Five commands read the series layer and answer the questions that
decide what is worth modelling:

```bash
terrastat corpus                          # regenerate docs/index.html and docs/corpus.md
terrastat corpus --check                  # no corpus needed: are the committed pages still generated?
terrastat quality --out reports/quality   # length, recency, gaps, degeneracy, per source x frequency
terrastat hierarchy --dataset demo_maeduc --dim sex --check   # is this series the sum of others?
terrastat search "chicken|poultry"        # find datasets by subject, including by code label
terrastat select --sources fred --min-rank 0.7   # rank series on recency, length and information
```

| command | answers |
|---|---|
| `corpus` | the one-page answer: how much, how long, how live, how concentrated — regenerated, not written |
| `quality` | how much data is there really, how much is current, how much is missing or constant |
| `hierarchy` | which series are exact sums of other series, and whether the units permit summing |
| `search` | which datasets are about a subject — searches every dimension code label, not just titles |
| `select` | which individual series are worth training on, ranked within their own frequency |

`corpus`, `quality` and `search` are cheap: they read only fixed-width columns or the dataset
metadata, so they answer over the whole corpus in seconds — `corpus` samples for the one thing
that needs the values themselves, and says on the page which numbers those are. `select` needs the values themselves and writes a
resumable score file. See [docs/schema.md](docs/schema.md) for the layout these all read, and
**[docs/leakage.md](docs/leakage.md) before splitting anything into train and test** — three
sources that republish each other, publish totals alongside their parts, and store latest
revisions only will defeat a naive random split.

### Cleaning, only when you ask

The corpus is deliberately uncleaned: series keep their own start and end, their own gaps and
their own missing values, because which of those are a problem depends on the exercise. One
curated alternative ships with it, and nothing calls it unless you do —
[`terrastat.curate`](src/terrastat/curate.py) reproduces the FRED curation from *Neuro-Symbolic
Models with Multimodal Data*: choose the cutoff date the data itself supports, keep the series
covering a full context window ending there, drop the too-gappy and the missing-at-an-edge, and
forward-fill what is left.

```python
from terrastat import curate
res = curate.curate(df, context=48)        # res.frame, res.report, res.curves["M"]
```

On the monthly FRED series here it picks the same December 2021 cutoff the thesis reports, and its
two filters remove the same number of series to within a handful. The module docstring documents
the three places where the thesis text and its reference implementation disagree, and how to ask
for either.

### Keeping the generated pages honest

`terrastat corpus` writes the page, the annex, `docs/corpus_tables.json`, `docs/favicon.svg`, and the logo kit in [`docs/brand/`](docs/brand/) (twelve variants plus the full still and a README saying which to use where) — the
computed figures themselves, small enough to commit. That third file is what lets a checkout with
no data re-render the pages and confirm they match, which is what `terrastat corpus --check` does
and what CI runs on every push.

It cannot tell you whether the figures are current with the latest crawl — only the machine holding
the Parquet can answer that, and the pages date-stamp themselves so a stale one says so. What it
catches is the failure that actually happens: someone edits a generated page by hand, or changes a
renderer and commits the code without regenerating what it produces.

### Politeness and timing

Every source has a pacer with two brakes: a minimum gap between requests and a cap per sliding
minute, plus exponential backoff with jitter on 429/5xx/network errors and respect for
`Retry-After`. Defaults:

| source   | gap between requests | max per minute | notes |
|----------|---------------------:|---------------:|-------|
| fred     | 2 s                  | 20             | 250k observations per page; FRED tolerates ~120/min, we use a sixth |
| eurostat | 3 s                  | 12             | Eurostat asks not to parallelise; the server needs 1-3 min to generate a large file anyway |
| oecd     | 3 s                  | 12             | one CSV per dataflow |

Override with `--min-interval` and `--max-per-minute`. A full FRED pass is a few hours. A full
Eurostat pass is dominated by the server generating each bulk file: measured in September
2026, the first byte arrives after 1.3 s and the gzipped TSV then streams at a steady 23 to
25 KB/s whatever the dataset (60 KB in 4 s, 1.1 MB in 44 s, 2.0 MB in 85 s, 3.4 MB in 150 s).
No other format is faster: SDMX-CSV of the same dataset took twice as long, the SDMX 3.0
endpoint answers with an error, and the old static bulk files are gone (HTTP 410). So the crawl
is meant to be left running with `--max-hours` and resumed, and every dataset is usable as soon
as its folder appears. Fetching is sequential by design: one open connection per source.

Two modes exist for Eurostat-sized crawls:

- `fetch --raw-only` sends exactly one request per dataset and stores the payload. Run `fetch`
  again later without the flag to tidy from the local files; only the small structure requests
  (data structure, concept scheme, and codelists not yet cached) hit the network then.
- `fetch` (default) tidies right away. Codelists are shared across datasets and cached by
  version, so after the first few hundred datasets almost none are requested any more; the
  structure overhead is then two 1-second requests per dataset.

**When a source pushes back, the whole crawl slows down, not just the request.** A 429 doubles the
gap between requests for that source and it stays doubled until twenty clean responses earn it
back. `Retry-After` is honoured in both its forms, seconds and HTTP date. A 5xx arriving during a
burst of 429s is treated as more pushback rather than as an unrelated glitch, since that is what it
usually is. If a dataset still cannot be fetched after its retries, the source is put on a five
minute cooldown and the dataset is recorded as `throttled` rather than `failed` — nothing is wrong
with it, so the next run picks it up without `--retry-failed`.

### Resumability

- Finished datasets are recorded in `state/<source>.jsonl` and skipped next time (`--force` redoes them).
- Failures are recorded with the error; `--retry-failed` tries them again.
- FRED pages are checkpointed with their cursor, so an interrupted release continues where it stopped.
- Raw payloads are kept (`--no-raw` deletes them after tidying); tidying can be redone from raw without network.
- All writes are atomic (temp file then rename), so a killed process never leaves a half-written file.

## Training snapshots: fixed-size shards

The per-dataset files are the right shape for analysis and the wrong shape for training: thousands
of small files grouped by dataset. `terrastat export` repacks a selection of the series view into
shards of a target size, following what Hugging Face `datasets`, WebDataset and Mosaic streaming
all converge on:

```bash
terrastat export public_v1 --public-only --min-obs 24 --target-mb 128
```

writes `data/snapshot/public_v1/`:

- `shard-00000.parquet ...`: same columns as the series view, `values` as float32, zstd, row groups
  of 2,048 series. Every series is assigned to a random shard and rows are shuffled inside each
  shard, so any shard is a uniform sample of the whole selection and a training loop that reads
  shards in any order sees a mix of sources, frequencies and datasets from the first batch.
- `index.parquet`: `series_uid -> shard, row` plus source, dataset, frequency, licence, length.
- `manifest.json`: the filters, seed, counts by source / frequency / licence, per-shard rows and bytes.

Why 128 MB: one shard is read at a time, so this bounds memory on a laptop with a consumer GPU;
it is small enough for many workers to each own a shard and large enough that a run is not
dominated by file opens (Hugging Face defaults to 500 MB, WebDataset to about 100 MB to 1 GB).
`terrastat.export.iter_shard_batches(name, batch_rows=256)` is the reading pattern: bounded
memory, one shard at a time, random shard order. `--public-only` keeps Eurostat, OECD, and FRED
public-domain series from US federal sources. See `notebooks/getting_started.ipynb` for a windowing
example that turns batches into (context, horizon) arrays.

## Notebooks

Install the extras with `uv pip install -e ".[notebook]"`, then start `jupyter lab` from the
`terrastat` environment. All are saved with their outputs, so they read as documentation too.

- `notebooks/parquet_basics.ipynb` — start here. One real dataset end to end (Eurostat's monthly
  poultry statistics, 1967 to 2026): searching the catalogue to find it, reading it in polars and
  pandas with the Parquet format explained along the way, decoding codes and flags, and a reusable
  `health()` check that reports length, recency and completeness so you can tell whether a dataset
  is worth modelling before you commit to it.
- `notebooks/getting_started.ipynb` — the whole tool: catalogues, fetching, all three layers,
  licences, and building a sharded snapshot for training.
- `notebooks/showcase.ipynb` — what is actually in the corpus, in plots: the monthly price of
  olive oil through the 2023 drought, inland fisheries by species, household internet adoption,
  the COVID cliff in tourist nights, house prices back to 1947. Also a cautionary example —
  221,418 series holding 1.2 observations each.
- `notebooks/quality_check.ipynb` — the interactive form of `terrastat quality`, calling the same
  functions so the notebook and the batch script cannot drift apart.

## Frequencies and dates

One vocabulary across all three sources, the union of SDMX's `CL_FREQ` (which Eurostat and OECD
build on) and the strings FRED reports, using SDMX's own letters wherever one exists:

| code | meaning | FRED | Eurostat / OECD |
|---|---|---|---|
| `H` | hourly | — | `H` |
| `D` | daily, every calendar day | `Daily...` | `D` |
| `B` | daily, business week only | — | `B` |
| `W` | weekly | `Weekly...` | `W` |
| `BW` | biweekly | `Biweekly...` | — |
| `M` | monthly | `Monthly` | `M` |
| `Q` | quarterly | `Quarterly` | `Q` |
| `S` | semiannual / half-yearly | `Semiannual` | `S` |
| `A` | annual | `Annual` | `A` |
| `A3` | the source's own "triannual" bucket, kept verbatim | — | `A3` |
| `P` | pluriannual, any fixed multi-year step | `5 Year`, `10 Year` | `P` |
| `I` | irregular / a-periodic | — | `I` |
| `OTHER` | source declares none applicable, or a code we do not know | `Not Applicable` | `NAP` |

`A3` keeps the source's code because "triannual" is ambiguous in English (three times a year vs
every three years) and neither codelist disambiguates it; rather than guess, the code is passed
through and `frequency_raw` holds the original.

The FRED column is exhaustive, verified against all 845,518 series of all 331 releases on
2026-09-04: those ten strings are the only ones FRED uses. The Eurostat column is the complete
`ESTAT:FREQ` codelist v3.9. Everything unrecognised becomes `OTHER` rather than a guess, and
`frequency_raw` always keeps the source's own label, so no mapping ever loses information.

Every period is also stored as `date`, the first calendar day of the period (`2015-Q4` ->
2015-10-01, `2015-W05` -> Monday of ISO week 5, an hourly stamp -> its day). Annual data is
included everywhere; nothing is monthly-only.

## Licences

The point of this tool is a snapshot that can be shared and used for training, so licensing is
data, not documentation: every dataset and every series row carries `license_id`,
`license_url`, the exact `attribution` text the source asks for, and `license_notes` with the
caveats. What was verified in September 2026:

**Eurostat** (`eurostat-reuse`). The copyright notice authorises re-use of statistical data for
commercial and non-commercial purposes provided the source is acknowledged and modifications are
stated. Exceptions named in the notice: some non-EU country data and some trade data are
restricted to non-commercial use; check the dataset's ESMS page (linked in `dataset.json`).

**OECD** (`oecd-terms`). Data may be extracted, adapted and distributed for any purpose,
including commercial, with the citation given in the metadata. Third-party-owned data may carry
extra restrictions; the OECD asks users to check the "source" tab of the dataset.

**FRED** (`fred-public-domain-citation-requested`, `fred-copyrighted-citation-required`,
`fred-copyrighted-preapproval-required`). The bulk endpoint reports FRED's copyright status for
every series, so filtering costs nothing. Two caveats are recorded in every FRED row:

1. FRED defines "Public Domain: Citation Requested" as series that "may be under copyright or in
   the public domain".
2. The FRED Terms of Use restrict storing/archiving FRED content and its use for training
   machine-learning systems regardless of the series status.

Works of the US federal government are public domain by statute (17 U.S.C. 105) at their origin,
so `origin_us_federal` flags releases whose sources are all US federal agencies (heuristic on
source names; the source list is stored in `dataset.json`). The defensible public snapshot is the
intersection: FRED status public domain *and* origin US federal. For anything beyond that, the
clean routes are the originating agencies' own APIs or an explicit agreement with the St. Louis
Fed. This tool records the facts; it does not decide for you.

## Vintages

terrastat stores the **latest revision** of every observation as served at `retrieved_at`.
Economic series are revised: GDP is re-estimated for years, seasonal factors change, base years
move. If a study needs the data as it was known at a point in time (real-time forecasting
evaluation), FRED's ALFRED archive serves vintages through the v1 `series/observations` endpoint
with `realtime_start`/`realtime_end`, and each FRED series row here carries `last_updated`.
Eurostat and OECD publish only the current version. A `vintage` column is present in every row
(`"latest"`) so a future vintage-aware run can coexist with this one.

## Scale, for planning

| source   | catalogue                       | rough size |
|----------|---------------------------------|------------|
| FRED     | 331 releases, 845,518 series    | 146 million observations; a few GB raw, ~1-2 GB Parquet |
| Eurostat | 6,650 datasets (+917 tables)    | 961.8 million series, ~6.2 billion observations; monthly: 226 datasets, ~1.0e9 values; quarterly: 420, ~1.1e9; annual: 6,154, ~4.4e9; weekly 14, daily 5, semiannual 19 |
| OECD     | 1,543 dataflows                 | varies widely; very large dataflows may be refused in one request (recorded as failed) |

Use `--freq`, `--max-values` (size hint from the catalogue) and `--max-hours` to shape a run.
`terrastat fetch eurostat` skips the 917 *tables* by default: they are pre-defined subsets of
datasets (`--types dataset table` includes them). At the measured 23 to 25 KB/s the monthly and
quarterly sets are each on the order of a day of crawling and the annual set several days;
plan runs with `--freq` and `--max-hours`, smallest datasets first (the default order).

## FRED tags

FRED attaches analyst tags to every series ("cpi", "usa", "county", ...). The bulk endpoint does
not return them, so `terrastat tags fred` gets them the cheap way: it lists the ~6,000 tags with
their notes (six requests), then lists the series of each tag, 1,000 per page, resumably. Three
groups (copyright status, seasonal adjustment, frequency) are derived from fields already stored
and cost nothing; the default fetches the concept, geography and geography-type groups
(about 11,000 requests, some hours at the default 30 per minute); `--groups all` adds source and
release tags. `run` performs this step after the FRED releases are complete.

Result: `tags` holds the tag names, and `description` gains the sentence the reference project
used, with the tag notes spelled out:
`Time series about Consumer Price Index ... It is associated with the tags: Consumer Price Index,
inflation, United States of America, public domain: citation requested, sa, monthly. ...`
The tag table itself is `data/cache/fred/tags.parquet` (name, group, notes, popularity, count).

## Known limitations of this version

- Eurostat ESMS reference-metadata pages (method descriptions) are linked, not downloaded.
- OECD dataflows are fetched in one request where possible, and tidied by streaming (read in
  blocks, then a streaming sort that spills to disk), so size costs time rather than memory. The
  "Local areas" drilldowns are the extreme case at about 6.6 GB of CSV and 40 minutes for 1.25M
  series, compressing to 270 MB of raw and 52 MB of Parquet. Past a 20 GB cap
  (`run --max-download-gb`) the request is abandoned and the dataflow is re-fetched a slice at a
  time, keyed on its first dimension, which for OECD is almost always the reference area. Slices
  start as batches of 8 codes and halve whenever one is still too big, codes with no data answer
  404 and are skipped, and progress is checkpointed in `raw/<id>/_parts.json`, so an interrupted
  split resumes. The parts share a schema and are concatenated on tidying, so the result is the
  same as a single-request fetch. Only a single code that alone exceeds the part cap is dropped,
  with a warning naming it.
- No parallelism, on purpose.
