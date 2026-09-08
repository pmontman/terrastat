"""Builds quality_check.ipynb. Editing this file and re-running it is the way to change the
notebook: it keeps the cells reviewable as plain Python instead of JSON with escaped newlines.

    python notebooks/_build_quality_check.py
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Quality check: is this corpus any good?

A bird's-eye view of the series layer, meant to be run after a crawl and re-run as it grows.
It answers the questions you need answered *before* choosing what to train on:

- how many series are there, and how much data does each really carry?
- how many are still being updated, and how many reach a given cutoff?
- how many have half a seasonal cycle, one, two, three?
- how much is missing, and are the holes in the middle or at the edges?
- how many are constant, or so rounded they barely move?

Every number comes from `terrastat.quality`, so the notebook and the batch script cannot drift
apart. To get the same tables written to disk without opening a notebook:

```powershell
.\\.venv\\Scripts\\python -m terrastat.quality --out reports\\quality
```
""")

code("""
import datetime as dt
import polars as pl
from terrastat import quality

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)

ASOF  = dt.date.today()          # reference point for "how stale is this series"
UNTIL = dt.date(2025, 12, 1)     # the cutoff you care about reaching
# Scope. FRED by default so this notebook runs anywhere: the scalar passes are cheap over the
# whole corpus, but the sampled value-level pass and the exact quantiles are not, and running
# them over all 956M series inside a notebook kernel will exhaust memory on most machines.
# Set SOURCES = None for everything, preferably from the command line:
#     terrastat quality --out reports/quality
SOURCES = ["fred"]               # None = all three
FREQ    = None                   # e.g. ["M", "Q"]; None = all

ASOF, UNTIL
""")

md("""
## 1. How much is there

The scalar pass reads only the fixed-width columns, never the `dates`/`values`/`flags` lists that
make up almost all of the 28 GB on disk. That is why it covers the whole population exactly and
still returns in seconds.

`obs_p50` is the median number of observations per series, and it is the number to look at
first: a source can contribute hundreds of millions of *series* while contributing very little
*data*, because SDMX dimension cross-products multiply thin slices. The quartiles either side of
it are exact, computed from a count of each distinct length rather than by sorting a billion-row
column (which does not fit in memory).
""")

code("""
ov = quality.overview(sources=SOURCES, frequencies=FREQ, asof=ASOF, until=UNTIL)
ov.select("source", "frequency", "n_series", "n_obs",
          "obs_p25", "obs_p50", "obs_p75",
          pl.col("years_median").round(1).alias("yrs_med"),
          "earliest", "latest")
""")

code("""
# the same thing as a share of the corpus, which is usually the more sobering view
tot_s, tot_o = ov["n_series"].sum(), ov["n_obs"].sum()
(ov.select("source", "frequency",
           (pl.col("n_series") / tot_s * 100).round(2).alias("% of series"),
           (pl.col("n_obs") / tot_o * 100).round(2).alias("% of observations"))
   .sort("% of series", descending=True)
   .head(12))
""")

md("""
## 2. Length, in cycles

A cycle is one year: 12 points at monthly, 4 at quarterly, 1 at annual. Length is measured in
years of *actual data* (`n_obs / periods_per_year`), not calendar span, because a series with a
ten-year span and four observations cannot be modelled as if it had ten years.

Half a cycle is the bare minimum for anything seasonal to be visible at all; two or three cycles
is where seasonal estimation starts to behave.
""")

code("""
ln = quality.length_table(sources=SOURCES, frequencies=FREQ)
ln
""")

code("""
# as percentages, which is easier to read across sources of very different size
cuts = [c for c in ln.columns if c.startswith("ge_")]
ln.select("source", "frequency", "n_series",
          *[(pl.col(c) / pl.col("n_series") * 100).round(1).alias(c.replace("ge_", "%>=")) for c in cuts])
""")

md("""
## 3. Recency

`fresh_le_Np` is the share whose last observation is within N periods of `ASOF`, in the series'
own frequency: for monthly, `fresh_le_2p` is the "updated within two months" figure.

Two things to keep in mind. The lag is measured from `ASOF`, so it includes however long ago the
crawl ran — set `ASOF` to the crawl date if you want publication lag alone. And a series can be
fresh and still useless if it is two points long, which is why the task table below combines
recency with length rather than reporting it on its own.
""")

code("""
fresh = [c for c in ov.columns if c.startswith("fresh_")]
reaches = [c for c in ov.columns if c.startswith("reaches_")]
ov.select("source", "frequency",
          *[(pl.col(c) * 100).round(1).alias(c) for c in fresh + reaches])
""")

md("""
## 4. Completeness

`missing_share` is `1 - n_obs / n_points`: the fraction of stored periods with no value. Sources
differ structurally here — FRED stores only the points it has, so its series are complete by
construction, while the SDMX sources lay out a full rectangle and leave holes.
""")

code("""
comp = ["missing_share_mean", "share_complete", "flagged_share_mean"]
ov.select("source", "frequency", *[(pl.col(c) * 100).round(1).alias(c) for c in comp])
""")

md("""
## 5. What is each slice actually good for (univariate-local)

These profiles are simple predicates, defined in one place in `quality.profiles()` so they can be
argued with. They are a starting point, not a standard.

They are **univariate-local** requirements: they ask whether one series, alone, carries enough
history to identify a model fitted to it alone. For a global model cross-learning across the
corpus that is the wrong test — see the aligned cohort below.

| profile | rule |
|---|---|
| `trainable` | >= 2 horizons, <= 50% missing |
| `real_time` | last point within 2 periods, >= 3 horizons, <= 20% missing |
| `nowcasting` | within 3 periods, >= 4 horizons, <= 20% missing |
| `backtest` | >= 3 horizons, <= 20% missing (no recency requirement) |
| `seasonal` | >= 3 cycles, at least quarterly, <= 20% missing |
| `imputation` | 1-50% missing, >= 2 horizons (a complete series is no use here) |
| `long_history` | spans >= 20 calendar years |

Length is measured in **forecast horizons**, following the M4 conventions (annual 6, quarterly 8,
monthly 18, weekly 13, daily 14). Neither obvious alternative works: measuring in years makes one
annual point a whole year of data, and a flat observation count scales the calendar requirement
inversely with frequency, so "60 observations" quietly demands sixty years of an annual series
and five of a monthly one. A horizon means the same thing everywhere — a backtest needs enough
history to hold one out and still fit on what is left.

In observations that comes to:

| freq | horizon | trainable | backtest | nowcasting |
|---|---|---|---|---|
| A | 6 | 12 | 18 | 24 |
| Q | 8 | 16 | 24 | 32 |
| M | 18 | 36 | 54 | 72 |
| W | 13 | 26 | 39 | 52 |
| D | 14 | 28 | 42 | 56 |
""")

code("""
tk = quality.task_table(sources=SOURCES, frequencies=FREQ, asof=ASOF, until=UNTIL)
tk
""")

code("""
# the totals, which is the line to quote in a paper
totals = tk.select(pl.exclude("source", "frequency").sum())
n = totals["n_series"][0]
pl.DataFrame({
    "profile": [c for c in totals.columns if c != "n_series"],
    "series": [totals[c][0] for c in totals.columns if c != "n_series"],
}).with_columns((pl.col("series") / n * 100).round(2).alias("% of corpus"))
""")

md("""
## 6. Value-level checks (sampled)

These need the `values` column, so they read the bulk of the data and run on a stratified sample.
Raise `sample` if you want tighter estimates and can wait.

`unique_share` is distinct values divided by observations. It is the cheapest detector of series
that inflate a corpus without teaching anything: a constant sits at ~0, a step function or a
heavily rounded index sits very low, a genuine economic series sits high.

`share_interior_gap` is the one that matters for imputation — a hole between two observations is
a problem to solve, whereas missing points at the start or end just mean a shorter series.
""")

code("""
dp = quality.deep_table(sources=SOURCES, frequencies=FREQ, sample=30_000)
dp.select("source", "frequency", "sampled",
          *[(pl.col(c) * 100).round(1).alias(c) for c in dp.columns
            if c not in ("source", "frequency", "sampled")])
""")

md("""
## 7. Zoom in: monthly

The original question, spelled out for one frequency.
""")

code("""
FOCUS = "M"
lf = quality._with_derived(quality.scan(columns=quality.SCALAR_COLUMNS, frequencies=[FOCUS]), ASOF)
m = lf.collect(engine="streaming")

print(f"{FOCUS}: {m.height:,} series, {m['n_obs'].sum():,} observations")
for label, expr in [
    ("updated within 2 months",      pl.col("lag_periods") <= 2),
    (f"reaching {UNTIL:%Y-%m}",      pl.col("end_date") >= UNTIL),
    ("at least half a cycle (6m)",   pl.col("years") >= 0.5),
    ("at least 1 cycle (12m)",       pl.col("years") >= 1),
    ("at least 2 cycles (24m)",      pl.col("years") >= 2),
    ("at least 3 cycles (36m)",      pl.col("years") >= 3),
    ("no missing values",            pl.col("missing_share") == 0),
]:
    k = m.select(expr.sum()).item()
    print(f"  {label:28} {k:>12,}  ({k / m.height:6.2%})")
""")

code("""
# where the mass actually sits: series counted by how many years of data they carry
(m.with_columns(pl.col("years").cut([0.5, 1, 2, 3, 5, 10, 20]).alias("bucket"))
   .group_by("bucket").agg(pl.len().alias("series"), pl.col("n_obs").sum().alias("obs"))
   .sort("bucket"))
""")

md("""
## 6b. The aligned cohort

Everything above judges a series **on its own**: can this one series, by itself, identify a model
fitted to it alone? That is the right question for local methods and the wrong one for a global
model that cross-learns, where a short series is not unidentifiable because its parameters come
from the corpus rather than from itself.

So this is a different cut. Fix a forecast time — say the end of 2025, to forecast 2026 — and keep
every series that still has data at that point and already had data a year earlier, whatever its
frequency. They can then be lined up on a common calendar. Roll the origin back a year at a time
and the same cohort gives you rolling-origin cross-validation: forecast 2025 from the end of 2024,
2024 from the end of 2023, and so on.

The length bar here is one year, deliberately, not the horizon multiples above.

One thing the scalar columns cannot tell us: the endpoints prove the series *brackets* the window
and `max_missing` bounds the holes overall, but neither proves the window itself is unbroken.
Checking that needs the dates list.
""")

code("""
ORIGIN = quality._default_origin()      # the most recent 31 December
al = quality.aligned_table(sources=SOURCES, frequencies=FREQ, origin=ORIGIN, folds=3)
print(f"forecast time {ORIGIN}; folds_k = usable with the origin rolled back k years")
al
""")

code("""
# how the cohort decays as the origin rolls back, as a share of each slice
al.select("source", "frequency", "n_series",
          *[(pl.col(c) / pl.col("n_series") * 100).round(1).alias(c) for c in al.columns
            if c.startswith("folds_")])
""")

code("""
# the cohort itself, ready to feed a training set or an export
cohort = quality.aligned_series(sources=SOURCES, frequencies=FREQ, origin=ORIGIN, folds=1)
print(f"{cohort.select(pl.len()).collect().item():,} series aligned with one fold of rollback")
cohort.select("series_uid", "source", "frequency", "start_date", "end_date", "n_obs").head(8).collect()
""")

md("""
## 7b. Down one level: per dataset

Everything above is per **source** (FRED / Eurostat / OECD) and frequency. The same questions one
level down are what you actually sort when choosing what to train on, because a source-level
number hides enormous variation: one Eurostat dataset can contribute a hundred million
two-observation series while the one beside it contributes ten thousand usable ones.

This reads `dataset_title`, a string column, which makes it the slowest pass here — hence
`--datasets` rather than on by default in the batch script.
""")

code("""
ds = quality.dataset_table(sources=SOURCES, frequencies=FREQ, asof=ASOF, until=UNTIL)
print(f"{ds.height:,} dataset x frequency combinations")
ds.head(15).select("source", "dataset_id", "frequency",
                   pl.col("title").str.slice(0, 40).alias("title"),
                   "n_series", "obs_mean", "backtest", "real_time", "latest")
""")

code("""
# the datasets that actually carry usable data, rather than the ones with the most rows
(ds.filter(pl.col("backtest") > 0)
   .sort("backtest", descending=True)
   .head(15)
   .select("source", "dataset_id", "frequency",
           pl.col("title").str.slice(0, 40).alias("title"),
           "n_series", "backtest",
           (pl.col("backtest") / pl.col("n_series") * 100).round(1).alias("% usable")))
""")

md("""
## 8. Dates worth a second look

None of these is necessarily an error. Dates past today are usually projections — FRED carries
CBO and OMB forecast series that legitimately run into the 2030s — and a start in the 1600s can
be a real historical reconstruction. They are listed because a mis-parsed period looks exactly
the same from here, and because a forecast series left in a training set is a leak.
""")

code("""
quality.anomalies(sources=SOURCES, frequencies=FREQ, asof=ASOF)
""")

code("""
# what the future-dated series actually are, so you can judge them
(quality.scan(columns=["series_uid", "title", "frequency", "start_date", "end_date"])
   .filter(pl.col("end_date") > ASOF)
   .head(10)
   .collect())
""")

md("""
## Running it in batch

The same tables, written to disk as Parquet and CSV plus a short written summary, with no
notebook involved:

```powershell
.\\.venv\\Scripts\\python -m terrastat.quality --out reports\\quality
.\\.venv\\Scripts\\python -m terrastat.quality --sources eurostat --freq M Q --no-deep
.\\.venv\\Scripts\\python -m terrastat.quality --asof 2026-09-04 --until 2025-12-01 --sample 200000
```

`--asof` is worth setting to the date the crawl finished: recency is measured from it, so leaving
it at today counts the age of the crawl as staleness in the data.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("quality_check.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
