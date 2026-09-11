# terrastat, the manual

A step-by-step guide for running the data gathering and using what it produces. The
[README](README.md) is the technical reference (schemas, licences, design); this document is the
walkthrough. Everything below was run on the machine this was built on (Windows 10, PowerShell).

## 1. What you are operating

`terrastat` is a command-line program. You type a command, it downloads datasets from a source
(FRED, Eurostat or OECD) one at a time, slowly and politely, and writes tidy files into
`terrastat\data\`. You can stop it at any time and start it again; it continues where it left off.

It produces three kinds of output, each usable while the next is still being built:

| folder under `data\` | what it holds | who it is for |
|---|---|---|
| `raw\<source>\<dataset>\` | the payload exactly as the source served it (gzip) | reproducibility |
| `datasets\<source>\<dataset>\` | `observations.parquet` (one row per observation) and `dataset.json` (all metadata) | tables: pandas, DuckDB, R |
| `series\<source>\freq=<F>\` | one row per time series with lists of dates and values, metadata expanded to text | models, text encoders |

Frequency codes used everywhere, finest to coarsest: `H` hourly, `D` daily, `B` daily on business
days only, `W` weekly, `BW` biweekly, `M` monthly, `Q` quarterly, `S` semiannual, `A` annual,
`A3` the source's own "triannual" bucket, `P` pluriannual (any fixed multi-year step, such as
FRED's 5 Year), `I` irregular, `OTHER` when the source declares no applicable frequency. The
source's own label is always kept alongside, in `frequency_raw`. See the README for the full
mapping table.

## 2. Opening a terminal in the right place

1. Open **PowerShell** (Start menu, type "PowerShell"). Windows Terminal is fine too.
2. Go to the project folder. Type this and press Enter:

   ```powershell
   cd path\to\terrastat        # wherever you cloned or unpacked it
   ```

   The prompt now ends with `terrastat>`. Every command in this manual assumes you are here.

### What `.venv` is, and how to "run" it

Inside the folder there is a hidden-looking folder called `.venv`. It is not a program. It is a
private copy of Python with `terrastat` and its libraries installed, kept separate from anything
else on the computer. You never run the folder itself; you run the program inside it. There are
two ways, pick one:

**Way A, no activation.** Type the path to the program every time:

```powershell
.\.venv\Scripts\terrastat --help
```

The leading `.\` means "in the current folder". PowerShell also accepts the shorter
`.venv\Scripts\terrastat` because the path contains a backslash.

**Way B, activate once per terminal window.** Activation tells this terminal window to look in
`.venv` first, so the plain name works afterwards:

```powershell
.\.venv\Scripts\Activate.ps1
terrastat --help
```

You will see `(terrastat)` at the start of the prompt while it is active. It lasts until you close
the window. On this machine PowerShell currently answers "running scripts is disabled on this
system": that is Windows' default execution policy for scripts, and it blocks Way B. Either stay
with Way A, which needs no script, or allow locally written scripts once and for all by running
this in PowerShell (it changes a setting for your user account only):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Both ways run exactly the same program. The rest of this manual writes the plain `terrastat ...`
form; with Way A, prefix each command with `.\.venv\Scripts\`.

Sanity check that everything is in place:

```powershell
terrastat --help
terrastat status fred
```

The first prints the list of commands. The second prints counts (zeros on a fresh machine) and
proves the FRED key in `.env` is readable.

## 3. The commands

Two equivalent ways to invoke any of them. If the virtualenv is activated, or you are typing the
full path to it, use the command directly:

```powershell
.\.venv\Scripts\terrastat run
```

Otherwise go through the interpreter, which needs no `Scripts` directory on PATH:

```powershell
.\.venv\Scripts\python -m terrastat run
```

The examples below use the short form; both accept exactly the same arguments.

### `run`, the only one you need for a full crawl

```powershell
terrastat run
```

Fetches every dataset of every source, at every frequency, tidies it and builds the per-series
view, keeping track of everything itself. The three sources are worked side by side, each with
its own single paced connection, so no source sees more load than one polite user. Leave the
window open; the three progress bars show where each source is. What to expect:

- FRED finishes in a few hours, OECD in a day or so, Eurostat in several days.
- Press `Ctrl+C` whenever you need the computer: the command exits within seconds. A download
  that was in progress is abandoned and redone next time (FRED releases resume at the page they
  reached). Type `terrastat run` again later and it continues where it stopped.
- `terrastat run --max-hours 8` stops by itself after 8 hours (overnight runs). A download cut
  by the budget is retried first on the next run, so give runs at least an hour: a few OECD
  dataflows are several hundred megabytes and need that long in one piece.
- The progress bars count over the whole catalogue (for example `fred: 26/331`), including
  what earlier runs finished, with an estimate of the time left. `terrastat status <source>`
  shows the same numbers from another window at any time.
- Failures are retried once at the end of each source; `terrastat status <source>` lists any
  that remain, with the reason.
- It is done when the last lines say `0 remaining` for every source. Running it again then does
  nothing (except picking up datasets the sources have added since).

Everything below is what `run` uses internally; you need it only for partial or custom crawls.

### `catalog`, the list of all tables

```powershell
terrastat catalog eurostat
terrastat catalog eurostat --csv data\eurostat_tables.csv
```

Downloads the source's full table of contents once (about a minute for Eurostat), caches it, and
prints a preview. With `--csv` you also get the whole list as a spreadsheet: id, title,
frequencies, size, last update, and a `status` column that says `todo`, `raw`, `done` or `failed`
for each table. Open it in Excel to see what exists and what has been fetched. Add `--refresh`
to re-download the catalogue (the cached copy is reused if it is less than a day old).

### `fetch`, the crawl

```powershell
terrastat fetch eurostat --freq M --raw-only --max-hours 8
```

Walks the catalogue, smallest datasets first, one at a time, and writes a folder per dataset.
The options you will actually use:

| option | meaning |
|---|---|
| `--freq M A Q` | only datasets with these frequencies (default: all) |
| `--ids id1 id2` | only these datasets (ids as in the catalogue) |
| `--raw-only` | one request per dataset, store the payload, tidy later |
| `--with-series` | also build the per-series view right after each dataset |
| `--max-hours 8` | stop cleanly after 8 hours; run again later to continue |
| `--limit 50` | stop after 50 new datasets |
| `--max-values N` | skip datasets larger than N stored values (Eurostat, size known up front) |
| `--retry-failed` | try again the datasets that failed before |
| `--force` | redo datasets already done |
| `--min-interval 5 --max-per-minute 6` | be even slower than the defaults |

The progress bar shows the dataset being processed. Each finished dataset is recorded at once,
so closing the window loses at most the dataset in progress.

Default pacing, chosen to stay well inside each source's fair-use terms:

| source | seconds between requests | requests per minute, at most |
|---|---|---|
| fred | 2 | 20 |
| eurostat | 3 | 12 |
| oecd | 3 | 12 |

Never run two fetches of the same source at the same time; one connection per source is the
whole point.

### `tags`, the FRED analyst tags

```powershell
terrastat tags fred
```

Fetches the FRED tags ("cpi", "usa", "county", ...) for every series, tag by tag, resumably,
then rebuilds the FRED series view so that `tags` is filled and `description` says
"It is associated with the tags: ...". `run` does this by itself once the FRED releases are
complete, so you only need the command if you want the tags earlier or with more groups
(`--groups all` adds source and release tags). Nothing is re-downloaded: only the small tag
listings are fetched.

### `series`, the model-ready view

```powershell
terrastat series eurostat --min-obs 24
```

Reads the dataset folders already on disk and writes `series\eurostat\freq=<F>\<dataset>.parquet`,
one row per series. `--min-obs 24` drops series with fewer than 24 non-missing values. Datasets
already converted are skipped (`--force` redoes them). No network is used.

### `status`, where things stand

```powershell
terrastat status eurostat
```

Counts of datasets done, raw-only, failed; series and observations gathered; datasets by
frequency; the first failures with their error messages.

### `peek`, look inside one dataset

```powershell
terrastat peek eurostat irt_st_m
terrastat peek fred 10
```

Prints the dataset's metadata (title, licence, attribution, dimensions with a few code labels)
and the first rows of its tables.

## 4. Recipes

**Eurostat, monthly first, in evening-sized chunks.** Each run stops after 8 hours; repeat until
`status` reports nothing left to do, then tidy and build the series view from the local files:

```powershell
terrastat catalog eurostat --csv data\eurostat_tables.csv
terrastat fetch eurostat --freq M --raw-only --max-hours 8
terrastat fetch eurostat --freq M --with-series
terrastat status eurostat
```

Then the same with `--freq Q`, then `--freq A`. Expect roughly a day for monthly, a day for
quarterly and several days for annual: Eurostat generates each file on request at about 24 KB/s,
and that is the limit, not this tool.

**FRED, everything, all frequencies (annual included):**

```powershell
terrastat fetch fred --with-series --max-hours 6
```

Repeat until `status fred` shows all 331 releases done. A full pass is a few hours.

**OECD:**

```powershell
terrastat fetch oecd --with-series --max-hours 6
```

**Just a few known tables:**

```powershell
terrastat fetch fred --ids 10 53 --with-series
terrastat fetch eurostat --ids nama_10_gdp irt_st_m --with-series
```

## 5. Stopping, resuming, failures

- Stop with `Ctrl+C`, by closing the window, or let `--max-hours` do it.
- Start again with the same command. Datasets already finished are skipped; the one that was in
  progress is redone from scratch (it left only a `.part` file). FRED releases resume at the page
  where they stopped.
- The record of what is done lives in `data\state\<source>.jsonl`, one line per dataset. Deleting
  a line makes that dataset eligible again; deleting the file restarts the source.
- A dataset that fails is recorded with its error and skipped on later runs. See the first
  failures in `status`; retry them with `fetch ... --retry-failed`. `run` already makes one
  retry pass per source at the end, so a transient failure usually clears itself.
- The logs are in `data\logs\terrastat.log`.

Each line in the state file carries one of these statuses:

| status | meaning | retried? |
|---|---|---|
| `done` | fetched and tidied | no |
| `raw` | downloaded but not tidied (`--raw-only`) | on a normal run |
| `failed` | something went wrong; the error is on the line | yes, with `--retry-failed` |
| `throttled` | the source was rate-limiting us and the retries ran out; nothing is wrong with the dataset | yes, on the next run |
| `too_large` | past `--max-download-gb` and not splittable | no, raise the cap instead |
| `not_timeseries` | a questionnaire or cross-section: no time dimension | no, retrying cannot help |
| `no_data` | listed in the catalogue but the source holds no observations (SDMX answers 404), or registered in the OECD public API while defined in another OECD space | no, same reason |

Twenty-seven OECD dataflows fall in that last group: TiVA (trade in value added), the DAC/CRS
development-finance statistics, and three archived productivity flows. The public API registers
them with `isExternalReference="true"` and a `structureURL` pointing at `sti-public`,
`dcd-public` or `archive`. Those spaces publish structure but not data (HTTP 403 on
`/rest/data`), and the public data endpoint answers HTTP 500 for them whatever the key or
format. They are skipped without a request; `data\state\oecd.jsonl` records where each one
lives, so they can be picked up from OECD's own portals if you want them.

## 6. Reading the data

Use the same environment (Way A: `.\.venv\Scripts\python`; Way B: `python` after activation),
or any Python with `polars`, `pandas` or `duckdb`.

**Layer 1, one dataset as a table:**

```python
import polars as pl, json
obs = pl.read_parquet(r"data\datasets\eurostat\irt_st_m\observations.parquet")
meta = json.load(open(r"data\datasets\eurostat\irt_st_m\dataset.json", encoding="utf-8"))
print(meta["title"], meta["license"]["attribution"])
labels = {d["id"]: d["codes"] for d in meta["dimensions"]}   # code -> label per dimension
```

Columns: `series_key, <one column per dimension>, period, date, value, flag`. For FRED the
per-series titles and units are in `series.parquet` next to it.

**Layer 2, all series of a source or frequency as one table:**

```python
import polars as pl
m = pl.scan_parquet(r"data\series\**\*.parquet").filter(pl.col("frequency") == "M").collect()
row = m.filter(pl.col("series_uid") == "fred:CPIAUCSL").row(0, named=True)
row["dates"], row["values"], row["description"]
```

```python
import duckdb
duckdb.sql("select source, frequency, count(*) from 'data/series/**/*.parquet' group by all")
```

```python
import pandas as pd
df = pd.read_parquet(r"data\series\fred\freq=M\10.parquet")   # lists become numpy arrays
```

Every series row has the same columns whatever the source. The ones you will use most:
`series_uid`, `title`, `description` (one sentence with all the metadata, for text encoders),
`frequency`, `units`, `seasonal_adjustment`, `geo`, `dimensions` (list of id, name, code, label),
`license_id`, `attribution`, `origin_us_federal`, `start_date`, `end_date`, `n_points`,
`n_obs`, `dates`, `values`, `flags`. The three lists have `n_points` elements each, in date
order; `values` can hold a null where the source published only a flag.

## 6b. A snapshot for training: `export`

> **Read this before sharing anything.** The corpus is not uniformly redistributable. Eurostat
> and OECD data may be shared with acknowledgement; FRED is a mixture, and only the series it
> marks public domain *and* whose originating body is a US federal one are free of copyright.
> The rest belong to the agencies that produced them.
>
> **Always pass `--public-only` for anything that will leave your machine.** It keeps Eurostat,
> OECD, and the FRED public-domain-from-US-federal subset, and writes `ATTRIBUTIONS.md` with the
> citations those sources require. Without it, `export` prints a warning naming every licence it
> included — but the filter is what makes the snapshot safe, not the warning.
>
> Attribution is required even for the shareable subset. The exact wording sits in the
> `attribution` column of every row, so it survives slicing, joining and re-export.

```powershell
terrastat export public_v1 --public-only --min-obs 24 --target-mb 128
```

Every snapshot folder is publishable as it stands:

| file | what it is |
|---|---|
| `shard-*.parquet` | the data, ~128 MB each, globally shuffled |
| `index.parquet` | `series_uid` to shard and row |
| `README.md` | a dataset card, with Hugging Face front-matter so the Hub previews it |
| `ATTRIBUTIONS.md` | the citation each source requires |
| `manifest.json` | filters, seed, counts |
| `croissant.json` | [MLCommons Croissant](https://mlcommons.org/croissant/) metadata, validated against `mlcroissant`; Hugging Face, Kaggle and OpenML read it directly |

Use `--per-dataset N` rather than `--limit-rows` for a representative subset: `--limit-rows`
truncates in catalogue order, and a 400,000-row limit once produced a "starter" set that was
entirely Eurostat annual series.

Repacks the series view into `data\snapshot\public_v1\`: shards of about 128 MB each, every
series placed in a random shard and shuffled inside it, `values` stored as float32, plus
`index.parquet` (which shard and row each series is in) and `manifest.json` (filters, counts).
A training loop reads one shard at a time, so memory stays flat on a laptop. Options:
`--sources fred eurostat`, `--freq M`, `--license-ids ...`, `--target-mb`, `--seed`,
`--float64`, `--columns` (keep only some columns). The notebook in `notebooks\` shows how to
iterate the shards into batches.

## 6b0. The one page to read first: `corpus`

```powershell
.\.venv\Scripts	errastat corpus
```

Writes two files from one set of tables, so they can never disagree: `docs/index.html`, the
project page to open in a browser or publish with GitHub Pages, and `docs/corpus.md`, the same
figures as a markdown annex for reading inside the repository. Both say how many series there are,
how many have at least two years of their own frequency, how many are still being published, how
repetitive the values are and how much of each frequency comes from a handful of enormous tables.
It is generated, not written — **regenerate it after every crawl and after adding a source**, and
never edit it by hand.

Counts, lengths and recency are exact over the whole corpus (they read scalar columns only, so a
billion series takes seconds). Distinctness and sign are sampled, and the page labels them so a
sampled share is never mistaken for a population count. Recency is measured against the newest
`retrieved_at` in the corpus rather than today, because otherwise a daily series looks stale the
moment the crawl ends.

## 6b2. Is the data any good? `quality`

A bird's-eye view of the series layer: how many series, how long, how fresh, how complete, and
how many are so flat they would only pad a training set.

```powershell
terrastat quality                                   # everything, tables to the screen
terrastat quality --out reports\quality             # also parquet + csv + summary.md
terrastat quality --sources eurostat --freq M Q     # one slice
terrastat quality --no-deep                         # skip the sampled value-level pass
terrastat quality --asof 2026-09-04                 # measure staleness from the crawl date
terrastat quality --datasets                        # break it down per dataset, not just per source
```

Every table is grouped by **source and frequency**, so FRED, Eurostat and OECD always appear as
separate rows. `--datasets` adds a further breakdown per dataset, which is the table to sort when
choosing what to train on; it is off by default because it reads `dataset_title`, and a string
column across a billion rows is by far the slowest part of the report.

It runs in two passes with very different costs. The **scalar pass** reads only the fixed-width
columns, never the `dates`/`values`/`flags` lists that are almost all of the 28 GB, so it covers
the whole population exactly in seconds. The **deep pass** needs `values` itself — distinct
values, constants, whether a gap sits in the middle or at the edge — and reads about 70k series a
second, so it runs on a stratified sample and says so.

What the columns mean:

- **A cycle is one year.** Length is `n_obs / periods_per_year`, so `ge_1y` at monthly means at
  least 12 observations. Actual data, not calendar span: a series covering ten years with four
  points cannot be modelled as if it had ten.
- **`fresh_le_2p`** is the share whose last point is within two of its own periods of `--asof`.
  For monthly that is the "updated within two months" figure. The default `--asof` is today, so
  it counts the age of the crawl as staleness; pass the crawl date to isolate publication lag.
- **`unique_share`** is distinct values over observations. Near zero means a constant, a step
  function, or something rounded until its variation is gone.
- **`share_interior_gap`** is what matters for imputation. Missing points at the start or end
  just make a shorter series; a hole between two observations is a problem to solve.

## 6b1. Finding series by subject: `search`

Everything else in this section asks how *good* a series is. This asks what it is *about*.

```powershell
terrastat search "chicken|poultry"
terrastat search "fisher|aquacult" --sources oecd
terrastat search "price|inflation|cpi" --freq M --min-series 1000
terrastat search "unemployment" --rebuild        # after the crawl has moved on
```

The query is a case-insensitive regular expression, so alternatives work.

It searches an index built from the `dataset.json` files — a few thousand small files rather than
the 27 GB series tree — holding each dataset's title, description, dimension names and **every
code label inside it**. Searching the labels is what makes it work: `apro_ec_poulm` is called
*Poultry farming* and never says "chicken" in its title, but "Broiler chickens" is one of the
codes of its `animals` dimension. A title-only search misses the dataset you wanted; pass
`--titles-only` if you want that behaviour anyway.

Results say *where* they matched and, for a code match, *which* codes — the difference between
"this dataset is about fish" and "this dataset has a species dimension containing tilapia".

Ranking is by relevance, not size, and that matters here. A title match is a statement about what
the dataset *is*; a code label is a statement about something it *contains*. Sorting by series
count put a 2.3-million-series structural business statistics table above Eurostat's poultry data,
because one of its NACE codes is "Processing and preserving of poultry meat".

Then pull the series out:

```python
from terrastat import search
search.search("fisher")                       # which datasets
search.series_of(["OECD.TAD.ARP__DSD_FISH_PROD_DF_FISH_INLAND__1.0"]).collect()
```

## 6b2b. Hierarchies: `hierarchy`

A property of *relations between* series rather than of any one series: is this series the exact
sum of others in the corpus? It matters twice over — hierarchical forecasting and reconciliation
need it, and a train/test split that separates an aggregate from its parts leaks.

```powershell
terrastat hierarchy                                       # summary over every dataset
terrastat hierarchy --dataset demo_maeduc --dim sex --check
```

Two structures produce it, found in different ways:

- **Explicit totals.** SDMX marks an aggregate with a reserved code, usually `_T`. A `sex`
  dimension holding `{M, F, _T}` states that `_T = M + F`. **71.6% of datasets carry an explicit
  total in some dimension** — 71.5% by the unambiguous reserved code (`_T`, `TOTAL`) and a further
  6.4% by a free-text label beginning "Total" or "All", which needs judgement. `sex` is the most
  common dimension (2,445 datasets), then `age`, then firm size class. Where a dimension carries
  several total-looking codes (8.8% of them do), `aggregation_sets` declines rather than guessing.
- **Prefix-nested classifications.** NUTS, NACE, COICOP and HS encode depth in the code itself —
  `DE` contains `DE1` contains `DE11` — so nesting comes from the codes alone with no lookup
  table. **53.6% of datasets have a prefix-nested dimension**, almost always `geo`.

**The catch is units.** An aggregation identity only holds if the measure is additive, and the
most common unit in the corpus is *Percentage* (1,927 datasets). A total row in a percentage or
index dataset is a differently-computed figure, not a sum. So `additive` is reported beside the
structure, and `--check` verifies the arithmetic instead of trusting it:

| dataset | units | identity holds | median relative error |
|---|---|---|---|
| `demo_maeduc` | Number | 40 / 40 | 0.000 |
| `hlth_ehis_pe2u` | Percentage | 1 / 40 | 0.993 |

Adding male% and female% gives roughly twice the total%. Both datasets have identical structure;
only the unit tells them apart.

`hierarchy.aggregation_sets(dataset, dim)` returns the (parent, children) groups themselves, and
`check_identity` the discrepancies. A group can fail legitimately — the children may not exhaust
the parent, the aggregate may be computed rather than summed — so it reports the error rather
than passing judgement.

## 6b3. Picking a training set: `select`

`quality` describes the corpus; `select` chooses from it, on three axes at once.

```powershell
terrastat select --sources fred --freq M Q       # score the candidates
terrastat select --report --min-rank 0.7         # then pick, without rescoring
```

**Filtering for recent series is free** — `end_date` is a fixed-width column, so it costs seconds
over the whole corpus and reads none of the 28 GB of values. **Measuring information is not**: how
many distinct values a series takes, how often it repeats, its Shannon entropy, all need the
values, at about 35,000 series a second. Over 956 million that is seven hours; over the survivors
of a cheap filter it is minutes. So `select` applies `--max-lag`, `--min-horizons` and
`--max-missing` *first*, then scores only what is left, writing one file per input file so an
interrupted run resumes.

Then the part that is a judgement call rather than a computation. Each axis becomes a percentile
and `rank_min` takes the minimum: a series at 0.8 is above the 80th percentile on recency *and*
length. Thresholding that is the conjunctive "good on all", with no weights to defend.

**Nothing is discarded.** All three axes are ranked, so a degenerate series sits at the bottom of
the information axis rather than being removed by a threshold nobody can defend. A hard cut is
available with `--floor` when a fixed dataset definition is wanted.

Four findings shaped this, all of them measured rather than assumed:

- **Length has to be relative to the frequency's own distribution.** Measuring it in forecast
  horizons looks frequency-neutral and is not: FRED annual series sit at a median of 6 horizons
  and weekly ones at 95, so ranking on horizons globally is really ranking on frequency. Length is
  therefore `n_obs` divided by the median for that frequency, which puts every frequency's typical
  series at 1.0.
- **Entropy alone cannot rank.** It saturates — for continuous data every value is distinct, so it
  sits at 1.0 and nearly everything ties — and it is mechanically anti-correlated with length,
  since `H/log2(n)` falls as repeats accumulate. The `information` axis is instead the *minimum* of
  four non-degeneracy terms (entropy, `1 − repeat_share`, `1 − flat_run_share`, `diff_variety`),
  which restores the spread: 6,218 distinct values against a largest tie of 5%.
- **Value statistics are blind to time order.** Shuffle a series and its entropy is unchanged, so
  white noise and a smooth trend with the same values score identically. Continuous (differential)
  entropy has the same flaw — the problem is not discreteness but order-invariance — and adds
  scale-dependence. The order-aware features are `acf1`, `acf1_diff`, `turning_share` and
  `flat_run_share`.
- **A tied axis will silently cap the whole selection.** Series on one publication schedule share
  an end date exactly, so after a staleness prefilter the survivors are nearly all equally fresh:
  on FRED, 98.3% of 128,699 series sat on one recency value. Under a minimum that axis caps every
  one of them, cutting a 0.5 threshold from 32,156 series to 217 for reasons having nothing to do
  with the data. `select` prints an axis report first — `n_distinct`, `max_tie_share` and the
  *attainable* `rank_ceiling` — and drops any axis putting more than 90% of series on one value,
  saying so. Enforce that property as a threshold instead; `--max-lag` already does.

`selection.recent()`, `selection.load_scores()` and `selection.add_ranks()` are the same thing
from Python.

### The aligned cohort

The `tasks` table below judges each series **on its own**: could this one series, alone, identify
a model fitted to it alone? That is right for local methods (one ARIMA per series) and wrong for a
global model that cross-learns, where a short series is not unidentifiable — it borrows its
parameters from the rest of the corpus.

The `aligned` table is the other cut. Fix a forecast time, keep every series that still has data
at that point and already had data a year earlier, whatever its frequency, and they can be lined
up on a common calendar:

```powershell
terrastat quality --origin 2025-12-31 --folds 3      # forecast 2026; CV back to 2022
terrastat quality --history-years 2                  # demand two years before the earliest origin
```

`folds_k` counts the series still alignable with the origin rolled back `k` years, so the same
cohort supports rolling-origin cross-validation: 2026 from end-2025, 2025 from end-2024, and so
on. The length bar is one year, not the horizon multiples used by `tasks`.

Recency is tested in periods rather than days, because an annual observation for 2025 is stamped
`2025-01-01`: a naive `end_date >= 2025-12-31` would demand a 2026 point and reject almost every
annual series (1,487 of FRED's 493,263 rather than 108,871).

`quality.aligned_series(...)` returns the cohort itself as a lazy frame, for feeding into a
training set or an export.

The `tasks` table counts how many series each slice contributes to each kind of exercise
(`trainable`, `real_time`, `nowcasting`, `backtest`, `seasonal`, `imputation`, `long_history`).
Those rules are deliberately simple and all live in `quality.profiles()`; change them there.

Their length requirements are multiples of the **forecast horizon** for the frequency (`HORIZON`,
following the M4 conventions: annual 6, quarterly 8, monthly 18, weekly 13, daily 14). Trainable
is two horizons, backtest and real-time three, nowcasting four. Neither obvious alternative
works. Measuring in years makes one annual observation a whole year of data, which once put 97%
of the corpus in the trainable bucket. A flat observation count fails the other way: sixty points
is five years of monthly data and sixty years of annual, so it silently demands six decades of
history from exactly the series least likely to have it. A horizon means the same thing at every
frequency.

One caution the report exists to make visible: **a series count is not a measure of how much data
there is.** SDMX dimension cross-products generate enormous numbers of very short series, so
always filter on length before quoting a total.

## 6c. The notebooks

Three notebooks live in `notebooks\`, all saved with their outputs:
`parquet_basics.ipynb` is a gentle introduction to one dataset and to the Parquet format itself,
`getting_started.ipynb` runs the whole process on a few small datasets, and
`quality_check.ipynb` is the interactive form of `terrastat quality` — same functions, so the
notebook and the batch script cannot drift apart. To open them:

```powershell
.\.venv\Scripts\jupyter lab
```

then click `notebooks/getting_started.ipynb` and run the cells top to bottom (Shift+Enter).
Re-run it after a crawl to see the full data instead of the samples.

## 6d. One curated panel, if you want it: `curate`

Nothing in terrastat cleans your series for you. Length, gaps and missing values are left exactly
as the source published them, because which of those matter depends entirely on what you are
doing. `terrastat.curate` is the one exception you have to ask for by name: it reproduces the
FRED curation used in *Neuro-Symbolic Models with Multimodal Data* — pick the cutoff date the
data itself supports, keep only series that cover a full context window ending there, drop the
ones that are too gappy or missing at an edge, and forward-fill the rest.

```python
import polars as pl
from terrastat import curate
from terrastat.quality import series_root

df = pl.scan_parquet(str(series_root() / "fred" / "freq=M" / "*.parquet")).select(
    "series_uid", "frequency", "dates", "values", "title").collect()

res = curate.curate(df, context=48)      # 48 periods of each series' own frequency
res.frame                                # one row per surviving series, all the same window
res.report                               # how many series each step left
res.windows                              # {"M": {"window_start": ..., "cutoff": ...}}
res.curves["M"]                          # C(d): the cutoff-selection curve, ready to plot
curate.check(res)                        # both columns should be zero
```

On the 224,015 monthly FRED series here it picks December 2021 — the same cutoff the thesis
reports — and keeps 197,129 of them, in about eight seconds.

Two things to know. The stored `dates`/`values` hold only the points a source actually published,
so a skipped month is *absent* rather than missing; `curate` reindexes every series onto the
regular grid of its frequency first, which is what makes the missingness filters mean anything.
And `context` counts periods of each series' own frequency, so a cutoff is chosen separately per
frequency group — the reference procedure was monthly-only.

Where the thesis text and its reference implementation disagree, `curate` follows the text.
`curate.curate(df, **curate.REFERENCE_CODE)` asks for the code's behaviour instead; the module
docstring has the table of the three differences.

## 7. Licences, in one paragraph

Every dataset and every series row says under which terms it came: `license_id`, the source's
page (`license_url`), the exact citation the source asks for (`attribution`) and the caveats
(`license_notes`). Eurostat and OECD allow re-use, including commercial, with attribution.
FRED reports a status per series; the public snapshot to aim for is the rows with
`license_id = fred-public-domain-citation-requested` **and** `origin_us_federal = true`, because
FRED's own terms restrict archiving and machine-learning use of what comes through FRED, while
US federal data is public domain at its origin. The README's "Licences" section has the details.

## 8. When something looks wrong

| symptom | cause and fix |
|---|---|
| `terrastat` is "not recognized" | you are not in the project folder, or the environment is not activated; use `.\.venv\Scripts\terrastat` |
| "running scripts is disabled" on activation | PowerShell execution policy; start `powershell -ExecutionPolicy Bypass`, or use Way A |
| `FRED_API_KEY is not set` | put `FRED_API_KEY=...` in `terrastat\.env` (see `.env.example`) |
| many `HTTP 429` or `503` lines in the log | the source is throttling; the tool already waits and retries, let it run, or pass `--min-interval 6` |
| a dataset keeps failing with `413` or a timeout | too big for one request; skip it for now |
| disk filling up | raw payloads are kept by default; `--no-raw` deletes them after tidying, or set `TERRASTAT_DATA_DIR` to a bigger drive in `.env` |
| strange characters in titles in the console | the console font; the files are UTF-8 and correct |
| you want to start a source over | delete `data\state\<source>.jsonl` (and the folders under `data\datasets\<source>` if you want them rebuilt) |
