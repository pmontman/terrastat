"""Regenerates parquet_basics.ipynb. Run it, then execute the notebook with nbconvert."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Finding a dataset, reading it, and deciding whether it is any good

A short tour of one real dataset gathered by `terrastat`: **Eurostat's monthly poultry statistics**,
which run from January 1967 to 2026, so nearly sixty years of monthly data on hatcheries, chicks
and eggs across 39 countries.

Three things at once:

1. how to **find** a dataset among the thousands gathered,
2. how to **read** it, which means a short introduction to **Parquet**, the format everything is
   stored in,
3. how to judge whether it is **healthy enough to model**: how long the series are, how close to
   today they run, and how much is actually missing.

Nothing here needs the network. Run the cells top to bottom.""")

code(r"""from pathlib import Path
import datetime as dt
import json

import polars as pl          # the library terrastat uses
import pandas as pd          # the one you probably already know
import pyarrow.parquet as pq # lets us inspect a Parquet file without reading it
import matplotlib.pyplot as plt

DATA = Path("../data")
if not DATA.exists():
    from terrastat.config import data_dir
    DATA = data_dir()

pl.Config.set_fmt_str_lengths(60)
TODAY = dt.date.today()
print("data directory:", DATA.resolve())""")

md(r"""## 1. Finding a dataset

Each source has a catalogue: one row per dataset, with its title, size and frequencies, saved as a
Parquet file when the crawl first runs. Searching it is a plain string filter, and this is how you
would look for anything, not just poultry.""")

code(r"""catalogue = pl.read_parquet(DATA / "catalog" / "eurostat.parquet").with_columns(
    pl.col("extra").str.json_path_match("$.type").alias("type"))

# which datasets have actually been downloaded so far?
fetched = {json.loads(l)["dataset_id"]
           for l in open(DATA / "state" / "eurostat.jsonl", encoding="utf-8")
           if l.strip() and json.loads(l)["status"] == "done"}

hits = (catalogue
        .filter((pl.col("type") == "dataset") & pl.col("title").str.contains("(?i)poultry|hen|egg"))
        .with_columns(pl.col("dataset_id").is_in(fetched).alias("fetched"))
        .sort("n_values", descending=True)
        .select("dataset_id", "title", "frequencies", "n_values", "fetched"))
print(f"{hits.height} datasets mention poultry, hens or eggs")
hits.head(8)""")

md(r"""`n_values` is the size Eurostat itself declares, so you can see how big something is before
downloading it. Note the `fetched` column: the crawl works smallest-first, so the largest datasets
are usually the ones still outstanding.

We will take `apro_ec_poulm`, the monthly one.""")

code(r"""DATASET = "apro_ec_poulm"
observations = DATA / "datasets" / "eurostat" / DATASET / "observations.parquet"
metadata     = DATA / "datasets" / "eurostat" / DATASET / "dataset.json"
series_file  = DATA / "series"   / "eurostat" / "freq=M" / f"{DATASET}.parquet"

for f in (observations, metadata, series_file):
    print(f"{f.stat().st_size/1024:9.1f} KB  {f.name}")""")

md(r"""## 2. What Parquet is, and looking before you load

If you have only used CSV, the useful idea is this: a CSV is a wall of text that you must read from
the top to learn anything about it, while a Parquet file is **columnar, typed and
self-describing**. It stores each column separately, records the types, and keeps a footer of
statistics.

That footer is why you can ask what is in a file without reading the data.""")

code(r"""pf = pq.ParquetFile(observations)
print(f"rows       : {pf.metadata.num_rows:,}")
print(f"columns    : {pf.metadata.num_columns}")
print(f"row groups : {pf.metadata.num_row_groups}   (chunks that can be read independently)")
print()
print(pf.schema_arrow)""")

md(r"""Notice the **types**: `date` is a real date and `value` a float. In a CSV both would be text and
every reader would guess, which is how a column of codes like `007` silently becomes the number 7.""")

md(r"""## 3. Read it

This is *layer 1* of terrastat: one row per observation, the shape you want for tables, SQL and
plotting.""")

code(r"""obs = pl.read_parquet(observations)
print(obs.shape)
obs.head()""")

md(r"""### The same thing in pandas

`pandas.read_parquet` reads the file directly; `.to_pandas()` converts a polars frame you already
have. The values match, but the dtypes differ slightly, which is worth seeing once: pandas reads a
Parquet date into `object`, while polars gives a proper `datetime64`.""")

code(r"""obs_pd    = pd.read_parquet(observations)
obs_pd_pl = obs.to_pandas()
print("same values:", obs_pd.astype(str).equals(obs_pd_pl.astype(str)))
pd.DataFrame({"pandas.read_parquet": obs_pd.dtypes, "polars.to_pandas": obs_pd_pl.dtypes})""")

code(r"""# ask pandas for arrow-backed types and the dates come back as dates
pd.read_parquet(observations, dtype_backend="pyarrow").dtypes["date"]""")

md(r"""## 4. Read only what you need

Because columns are stored separately, asking for two of them never touches the rest, and filters
skip whole row groups using the footer statistics. On a file this size it saves little; the same
two lines are what make a 50 GB file usable on a laptop.""")

code(r"""pl.read_parquet(observations, columns=["date", "value"]).head(3)""")

code(r"""# filter while reading; pyarrow pushes this down into the file
fr = pd.read_parquet(observations, filters=[("geo", "==", "FR")])
print(f"{len(fr):,} rows read, out of {pf.metadata.num_rows:,} in the file")
fr.head(3)""")

md(r"""## 5. Codes, and what they mean

The table stores short codes rather than repeating "France" on every row, which is why it is small.
The meanings live beside it in `dataset.json`, with the licence and the citation the source asks
for.""")

code(r"""meta = json.load(open(metadata, encoding="utf-8"))
print(meta["title"])
print("span    :", meta["time_start"], "->", meta["time_end"])
print("licence :", meta["license"]["id"])
print("cite as :", meta["license"]["attribution"][:100], "...")
print()
for d in meta["dimensions"]:
    print(f"{d['id']:9s} {d['name'][:42]:42s} {len(d['codes']):3d} codes")
    for c, label in list(d["codes"].items())[:3]:
        print(f"          {c:9s} {label}")""")

code(r"""labels = {d["id"]: d["codes"] for d in meta["dimensions"]}
(obs.with_columns(pl.col("geo").replace_strict(labels["geo"]).alias("country"),
                  pl.col("animals").replace_strict(labels["animals"]).alias("animal"))
    .select("country", "animal", "date", "value", "flag")
    .head(4))""")

md(r"""### The flags matter for modelling

A flag says *why* a number is what it is, or why it is absent. Eurostat packs two different things
into this column: the observation status (`p` provisional, `e` estimated, `b` a break in the
series) and, prefixed with `@`, the confidentiality status. `@C` means the value was suppressed as
confidential, so the row exists and the value is null on purpose.""")

code(r"""flag_labels = {c: lab for a in meta["attributes"] for c, lab in a["codes"].items()}
(obs.group_by("flag").len().sort("len", descending=True)
    .with_columns(pl.col("flag").replace_strict(flag_labels, default="(no flag: a plain observation)").alias("meaning")))""")

md(r"""## 6. One row per series, and a health check

terrastat also stores the same data with **one row per time series**, where the dates and values are
lists inside a single cell. Parquet handles nested types natively; CSV cannot express this at all.
This is the shape for modelling, and the shape to judge.""")

code(r"""series = pl.read_parquet(series_file)
print(series.shape, "-> series x metadata columns")
series.select("series_uid", "geo_label", "units", "n_obs", "start_date", "end_date").head(4)""")

md(r"""Now the question that actually decides whether you can use it. Four things matter:

- **length**: enough points to fit anything
- **recency**: does it run up to roughly today, or did it stop in 2014
- **completeness**: of the calendar between the first and last observation, how much is really there
- **flags**: how much is provisional, estimated or suppressed

There are two distinct kinds of hole, and they are worth keeping apart. A period can be *stored
with a null value*, because a flag explains it, or it can be *absent altogether*. The function
below measures both, and works on any dataset in the collection.""")

code(r"""PERIOD_DAYS = {"D": 1, "W": 7, "BW": 14, "M": 30.44, "Q": 91.3, "S": 182.6, "A": 365.25}

def health(series: pl.DataFrame, today: dt.date = TODAY, min_len: int = 60) -> pl.DataFrame:
    # one row per check; works for any terrastat series file
    step = PERIOD_DAYS.get(series["frequency"][0], 30.44)
    d = series.with_columns(
        ((pl.lit(today) - pl.col("end_date")).dt.total_days() / step).alias("stale"),
        (pl.col("n_points") - pl.col("n_obs")).alias("null_rows"),
        (((pl.col("end_date") - pl.col("start_date")).dt.total_days() / step).round(0) + 1).alias("calendar"),
    ).with_columns((pl.col("n_obs") / pl.col("calendar")).alias("completeness"))
    n = d.height
    pct = lambda k: f"{k:,}  ({100*k/n:.0f}%)"
    checks = [
        ("series",                              f"{n:,}"),
        ("observations with a value",           f"{d['n_obs'].sum():,}"),
        ("length  min / median / max",          f"{d['n_obs'].min()} / {d['n_obs'].median():.0f} / {d['n_obs'].max()}"),
        (f"long enough (>= {min_len} points)",  pct(d.filter(pl.col("n_obs") >= min_len).height)),
        ("ends within 3 periods of today",      pct(d.filter(pl.col("stale") <= 3).height)),
        ("median staleness, in periods",        f"{d['stale'].median():.1f}"),
        ("complete calendar (>= 99%)",          pct(d.filter(pl.col("completeness") >= 0.99).height)),
        ("has periods stored but null",         pct(d.filter(pl.col("null_rows") > 0).height)),
        ("has periods missing entirely",        pct(d.filter(pl.col("calendar") > pl.col("n_points")).height)),
        ("values carrying a flag",              f"{d['n_flagged'].sum():,}  ({100*d['n_flagged'].sum()/max(1, d['n_points'].sum()):.0f}%)"),
    ]
    return pl.DataFrame({"check": [c for c, _ in checks], "value": [v for _, v in checks]})

health(series)""")

md(r"""Read that as: the dataset is long and current, but only about a third of the series have a
complete calendar, and a fifth of all values carry a flag. That is normal for agricultural
statistics, where countries join late, stop reporting, or suppress figures for confidentiality.

So pick a working subset rather than using everything.""")

code(r"""step = PERIOD_DAYS["M"]
stale = (pl.lit(TODAY) - pl.col("end_date")).dt.total_days() / step
usable = (series
          .with_columns((((pl.col("end_date") - pl.col("start_date")).dt.total_days()/step).round(0)+1).alias("calendar"))
          .with_columns((pl.col("n_obs")/pl.col("calendar")).alias("completeness"))
          .filter((pl.col("n_obs") >= 60) & (pl.col("completeness") >= 0.9) & (stale <= 6)))

print(f"usable: {usable.height:,} of {series.height:,} series "
      f"({100*usable.height/series.height:.0f}%) -- at least 60 points, 90% complete, under 6 months stale")
usable.select("geo_label", "n_obs", "start_date", "end_date").head(5)""")

md(r"""## 7. Making a series human-readable

Every row already carries a `description`: the whole of the series' metadata written out as one
sentence, which is what a text encoder would be shown. It is built from the dimensions, so it is
unique to each series.

There is one catch worth knowing. For Eurostat and OECD the `title` column is the *dataset* title,
so all 2,428 series here share it; only FRED gives every series its own title. The distinctive
information lives in `dimensions`, a list of `{id, name, code, label}` per series.""")

code(r"""print("title       :", series["title"][0])
print("distinct titles across the dataset:", series["title"].n_unique(), "for", series.height, "series")
print("distinct descriptions             :", series["description"].n_unique())
print()
for d in series["dimensions"][0]:
    print(f"  {d['id']:9s} {d['name'][:36]:36s} {d['code']:9s} {d['label']}")""")

md(r"""To build a name, you have to decide **which dimensions to include**, and that depends on where the
name will be seen.

Within this dataset, `freq` and `unit` are the same for every series, so repeating "Monthly,
Thousand" on all 2,428 of them tells a reader nothing they cannot see from the dataset title. Only
`hatchitm`, `animals` and `geo` distinguish one series from another.

But a constant dimension is **not uninformative** — it is only redundant *in this context*. The
moment a series leaves its dataset, which is exactly what happens in a training set mixing millions
of series from three sources, the unit is essential: `[18270.0, 17064.0, ...]` means nothing until
you know it is thousands of chicks per month. Here is how much that matters across Eurostat:""")

code(r"""import glob, collections

const, unit_values = collections.Counter(), collections.Counter()
n_ds = 0
for p in glob.glob(str(DATA / "datasets" / "eurostat" / "*" / "dataset.json")):
    m = json.load(open(p, encoding="utf-8")); n_ds += 1
    for d in m.get("dimensions", []):
        if len(d["codes"]) == 1:                    # constant within its own dataset
            const[d["id"]] += 1
            if d["id"] == "unit":
                unit_values[next(iter(d["codes"].values()))] += 1

print(f"of {n_ds:,} Eurostat datasets, the dimension is constant in:")
for k, v in const.most_common(4):
    print(f"   {k:6s} {v:5,} datasets ({100*v/n_ds:.0f}%)")
print(f"\nyet `unit`, constant inside a dataset, still takes {len(unit_values)} different values across them:")
print("   " + " | ".join(f"{k} ({v:,})" for k, v in unit_values.most_common(6)))""")

md(r"""So the rule is: drop the constants when browsing **inside** one dataset, keep everything when the
series will be read **on its own**. The function below takes a `scope` argument for that, and
defaults to the safe one.""")

code(r"""def varying_dimensions(series: pl.DataFrame) -> list[str]:
    # dimension ids that differ within this dataset
    seen: dict[str, set] = {}
    for dims in series["dimensions"].to_list():
        for d in dims or []:
            seen.setdefault(d["id"], set()).add(d["label"])
    return [i for i, labels in seen.items() if len(labels) > 1]


def label(row, varying=None, scope: str = "standalone") -> str:
    # scope='dataset'    drops dimensions constant within the dataset: shorter, for browsing it
    # scope='standalone' keeps them all: correct once the series is read outside its dataset
    dims = row["dimensions"] or []
    if scope == "dataset" and varying is not None:
        dims = [d for d in dims if d["id"] in varying]
    parts = [d["label"] for d in dims if d["label"]]
    return f"{row['title']} - {', '.join(parts)}" if parts else row["title"]


varying = varying_dimensions(series)
constant = [d["id"] for d in series["dimensions"][0] if d["id"] not in varying]
print("varies  :", varying)
print("constant:", constant)
print()
r = series.sort("n_obs", descending=True).row(1, named=True)
print("inside the dataset :", label(r, varying, scope="dataset"))
print("on its own         :", label(r, varying, scope="standalone"))""")

md(r"""Either way `description` is the field that never drops anything: it spells out every dimension with
its concept name, so it stays correct wherever the series ends up. That is why it, and not the
label, is what a text encoder should be given.

Now the three things you actually want to look at: **title, description, values**.""")

code(r"""def readable(row, varying, n: int = 4) -> str:
    lines = [label(row, varying), "-" * 76, row["description"], ""]
    lines.append(f"{row['n_obs']} points, {row['start_date']} to {row['end_date']}, "
                 f"in {(row['units'] or 'unknown units').lower()}")
    lines.append(f"source: {'; '.join(row['origin_agencies'])}   licence: {row['license_id']}")
    pairs = [(d, v) for d, v in zip(row["dates"], row["values"]) if v is not None]
    lines.append("first : " + ",  ".join(f"{d} {v:,.0f}" for d, v in pairs[:n]))
    lines.append("last  : " + ",  ".join(f"{d} {v:,.0f}" for d, v in pairs[-n:]))
    return "\n".join(lines)


one = (usable.filter(pl.col("source_id").str.contains("A5130P") & (pl.col("geo") == "FR"))
             .sort("n_obs", descending=True).row(0, named=True))
print(readable(one, varying))""")

md(r"""The same three fields as a table, when you want to scan many series at once.""")

code(r"""readable_table = (series.sort("n_obs", descending=True).head(6)
    .with_columns(pl.struct(pl.all()).map_elements(lambda r: label(r, varying), return_dtype=pl.String).alias("label"))
    .select("label", "n_obs", "start_date", "end_date",
            pl.col("values").list.head(3).alias("first_values")))
readable_table""")

md(r"""And as plain records, which is the shape to hand to a text model or write out as JSON.""")

code(r"""def as_record(row, varying, keep: int | None = None) -> dict:
    vals = [v for v in row["values"] if v is not None]
    return {
        "title": label(row, varying),
        "description": row["description"],
        "units": row["units"],
        "frequency": row["frequency"],
        "start": str(row["start_date"]),
        "values": vals if keep is None else vals[:keep],
        "attribution": row["attribution"],
    }

rec = as_record(one, varying, keep=6)
print(json.dumps(rec, indent=2, ensure_ascii=False)[:700], "...")""")

md(r"""## 8. Look at one""")

code(r"""fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(one["dates"], one["values"], linewidth=0.9)
ax.set_title(label(one, varying)[:95]); ax.set_ylabel(one["units"])
plt.tight_layout(); plt.show()""")

code(r"""# lists survive the trip to pandas, as numpy arrays
row = series.to_pandas().query("series_uid == @one['series_uid']").iloc[0]
pd.Series(row["values"], index=pd.to_datetime(row["dates"]), name="chicks (thousands)").tail()""")

md(r"""## 9. Why not just use CSV?

Measured on this very dataset rather than asserted.""")

code(r"""import time, tempfile, os

tmp = Path(tempfile.gettempdir()) / "poultry.csv"
obs.write_csv(tmp)
t0 = time.perf_counter(); pl.read_csv(tmp);              t_csv = time.perf_counter() - t0
t0 = time.perf_counter(); pl.read_parquet(observations); t_pq  = time.perf_counter() - t0

print(f"{'':10s} {'size':>10s} {'read':>10s}")
print(f"{'CSV':10s} {os.path.getsize(tmp)/1e6:8.1f} MB {t_csv*1000:8.0f} ms")
print(f"{'Parquet':10s} {os.path.getsize(observations)/1e6:8.1f} MB {t_pq*1000:8.0f} ms")
print(f"\nParquet is {os.path.getsize(tmp)/os.path.getsize(observations):.0f}x smaller and "
      f"{t_csv/t_pq:.0f}x faster here, and it keeps the types the CSV threw away.")
tmp.unlink()""")

md(r"""## 10. Cheat sheet

```python
# find a dataset
cat = pl.read_parquet("../data/catalog/eurostat.parquet")
cat.filter(pl.col("title").str.contains("(?i)poultry"))

# inspect a file without reading it
pq.ParquetFile(path).metadata        # rows, columns, row groups
pq.ParquetFile(path).schema_arrow    # names and types

# read
pl.read_parquet(path)                            # polars
pd.read_parquet(path, dtype_backend="pyarrow")   # pandas, with real dates

# read less
pl.read_parquet(path, columns=[...])
pd.read_parquet(path, filters=[("geo", "==", "FR")])
pl.scan_parquet(path).filter(...).collect()      # lazy: only touches what it needs

# many files as one table
pl.scan_parquet("../data/series/**/*.parquet").filter(pl.col("frequency") == "M").collect()

# write
df.write_parquet(path, compression="zstd")
```

Two habits worth keeping: use `scan_parquet` rather than `read_parquet` for anything large, so
filters happen inside the file instead of in memory afterwards; and when a folder of files shares a
schema, point a glob at the folder and treat it as one table instead of looping.

And one for the data itself: run the `health` check above before trusting a dataset. Length,
recency and completeness vary enormously between sources, and a series that stopped in 2014 looks
exactly like a current one until you ask.

Next: `getting_started.ipynb` covers the whole tool, all three sources, licensing, and how the
sharded snapshots for model training are built.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (terrastat)", "language": "python"}
out = Path(__file__).with_name("parquet_basics.ipynb")
nbf.write(nb, str(out))
print(f"wrote {out}")
