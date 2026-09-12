"""The corpus at a glance: one page, regenerated from the data rather than written by hand.

`terrastat quality` answers "is this any good?" in detail, across source × frequency, with a dozen
tables. This is the other thing people need — a page you can read in fifteen seconds that says how
much there is, how long it is, how much of it is still being published, and how much of it is
repetitive or duplicated. It is aimed at somebody who does not care what the series are *about*:
they want to know what shape a training set drawn from here would have.

Every number is computed, and the page carries the date and the crawl timestamps it was computed
from, so a stale page is visibly stale. Regenerate it after a crawl, after adding a source, or
after any change to how series are built::

    terrastat corpus --out docs/corpus.md

**Exact versus sampled.** Counts, lengths and recency are exact over the whole population — they
come from scalar columns only, so a pass over a billion series costs seconds rather than reading
the 30 GB of value lists. Distinctness and sign have to look at the values themselves, so they are
estimated from a sample drawn across datasets; the page labels them and reports the sample size.
Never quote a sampled figure as a population count.
"""
from __future__ import annotations

import datetime as dt
import logging
import json
import math
import random
from pathlib import Path

import numpy as np
import polars as pl

from terrastat.quality import PERIODS_PER_YEAR, series_root

log = logging.getLogger(__name__)

FREQ_NAME = {
    "H": "hourly", "D": "daily", "B": "business-daily", "W": "weekly", "BW": "biweekly",
    "M": "monthly", "Q": "quarterly", "S": "semiannual", "A": "annual", "A3": "3-yearly",
    "P": "5-yearly", "I": "irregular", "OTHER": "unknown",
}
# Reading order, by how much of the corpus a reader is likely to care about: the three frequencies
# most economic modelling actually uses, then the rest by size, then the ones with no fixed period.
# Not fastest-first: sorting by period would open every table with 14 thousand daily series and
# bury the 620 million annual ones in the middle.
FREQ_ORDER = ["A", "M", "Q", "W", "D", "S", "A3", "P", "BW", "H", "B", "I", "OTHER"]

# The headline length bar, in years of the series' own frequency. Two years is two seasonal
# cycles: the shortest history in which a yearly pattern can be seen twice and so be told apart
# from a trend. `A3` and `P` step more than a year at a time, so their bar is two observations.
MIN_YEARS = 2.0

# "Still being published", in periods of the series' own frequency. Twelve is deliberately loose:
# it keeps a monthly series that is a year behind and an annual series that is a decade behind on
# the same footing, and it is robust to a source that publishes in irregular bursts.
LIVE_PERIODS = 12.0

# The 90% extremes of length, reported alongside the median.
LENGTH_QUANTILES = (0.05, 0.5, 0.95)


def min_obs(frequency: str, years: float = MIN_YEARS) -> int | None:
    """Observations that ``years`` years amounts to at this frequency, at least two."""
    ppy = PERIODS_PER_YEAR.get(frequency)
    return None if ppy is None else max(2, math.ceil(years * ppy))


def _order(df: pl.DataFrame) -> pl.DataFrame:
    """Put the rows in reading order.

    Applied when rendering, not only when computing: row order is presentation, so changing
    ``FREQ_ORDER`` should take effect on a `--render` in under a second rather than requiring a
    fresh pass over a billion series.
    """
    if not df.height or "frequency" not in df.columns:
        return df
    rank = {f: i for i, f in enumerate(FREQ_ORDER)}
    return (df.with_columns(pl.col("frequency").replace_strict(rank, default=99).alias("_o"))
              .sort("_o", maintain_order=True).drop("_o"))


def crawled_at(root=None, sources=None) -> pl.DataFrame:
    """When each source was last retrieved. A scalar-only pass, so it is nearly free."""
    from terrastat.quality import scan

    return (scan(root, sources, None, ["source", "retrieved_at"])
            .group_by("source")
            .agg(pl.col("retrieved_at").max().alias("last_retrieved"), pl.len().alias("n_series"))
            .sort("source").collect(engine="streaming"))


# -- exact: how many, how long, how live ---------------------------------------------------------

def _wq(value: pl.Series, weight: pl.Series, qs) -> list[int]:
    """Quantiles of a weighted, already-sorted value column.

    The corpus has a billion series and a few thousand distinct lengths, so counting each length
    and reading the quantiles off the cumulative counts is both exact and small. Asking polars for
    ``median()`` over the raw column instead materialises it — a 15 GB allocation, in the run that
    prompted this.
    """
    cum = weight.cum_sum()
    total = cum[-1]
    out = []
    for q in qs:
        idx = int((cum < q * total).sum())
        out.append(int(value[min(idx, len(value) - 1)]))
    return out


def length_table(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
                 years: float = MIN_YEARS, live_periods: float = LIVE_PERIODS) -> pl.DataFrame:
    """Per frequency: how many series, how many are long enough, how long they are, how many are
    still being published — all exact, from one pass over three scalar columns.

    ``asof`` defaults to the most recent ``retrieved_at`` in the corpus, not to today. Measured
    against today, a daily series looks stale the moment a crawl finishes, which says something
    about the crawl and nothing about the source.
    """
    from terrastat.quality import scan

    if asof is None:
        stamp = (scan(root, sources, frequencies, ["retrieved_at"])
                 .select(pl.col("retrieved_at").max()).collect(engine="streaming").item())
        asof = dt.date.fromisoformat(stamp[:10]) if stamp else dt.date.today()

    ppy = pl.col("frequency").replace_strict(PERIODS_PER_YEAR, default=None, return_dtype=pl.Float64)
    lag = ((pl.lit(asof) - pl.col("end_date")).dt.total_days() / 365.25) * ppy
    counts = (
        scan(root, sources, frequencies, ["frequency", "n_obs", "end_date"])
        .with_columns((lag <= live_periods).alias("live"))
        .group_by("frequency", "n_obs", "live")
        .agg(pl.len().alias("k"))
        .collect(engine="streaming")
    )

    rows = []
    for (freq,), g in counts.group_by(["frequency"], maintain_order=True):
        cut = min_obs(freq, years)
        n_all = int(g["k"].sum())
        row = {"frequency": freq, "name": FREQ_NAME.get(freq, freq), "n_series": n_all,
               # observations, not series: the figure a modeller actually budgets against
               "n_obs_total": int((g["n_obs"] * g["k"]).sum()),
               "min_obs": cut}
        if cut is None:                       # no period: neither a length in years nor a lag in
            rows.append({**row, "n_live": None, "live_share": None,   # periods is defined here
                         "n_long": None, "long_share": None, "n_long_live": None, "mean_obs": None,
                         **{f"p{int(q * 100)}": None for q in LENGTH_QUANTILES}})
            continue
        row["n_live"] = int(g.filter(pl.col("live"))["k"].sum())
        row["live_share"] = row["n_live"] / n_all
        long = g.filter(pl.col("n_obs") >= cut)
        row["n_long"] = int(long["k"].sum())
        row["long_share"] = row["n_long"] / n_all
        row["n_long_live"] = int(long.filter(pl.col("live"))["k"].sum())
        if row["n_long"]:
            by_len = long.group_by("n_obs").agg(pl.col("k").sum()).sort("n_obs")
            for q, v in zip(LENGTH_QUANTILES, _wq(by_len["n_obs"], by_len["k"], LENGTH_QUANTILES)):
                row[f"p{int(q * 100)}"] = v
            row["mean_obs"] = float((long["n_obs"] * long["k"]).sum() / row["n_long"])
        else:
            row.update(mean_obs=None, **{f"p{int(q * 100)}": None for q in LENGTH_QUANTILES})
        rows.append(row)

    out = _order(pl.DataFrame(rows))
    ppy_col = pl.col("frequency").replace_strict(PERIODS_PER_YEAR, default=None, return_dtype=pl.Float64)
    return out.with_columns(
        [(pl.col(f"p{int(q * 100)}") / ppy_col).alias(f"p{int(q * 100)}_years") for q in LENGTH_QUANTILES]
    ).with_columns(pl.lit(asof).alias("asof"))


# -- sampled: what the values look like ----------------------------------------------------------

def value_table(root=None, sources=None, frequencies=None, per_frequency: int = 100_000,
                max_files: int = 250, seed: int = 0) -> pl.DataFrame:
    """Distinctness and sign, estimated from a sample pooled across sources.

    Reading the value lists is the expensive part of this corpus, so this reads the head of each
    of ``max_files`` randomly chosen datasets per frequency rather than the whole thing. One file
    is one dataset, so the draw spreads over the catalogue; within a dataset the rows are its
    first series in key order, which is arbitrary with respect to how the values behave but is not
    a random sample. Treat these as indicative.

    ``unique_share`` is distinct values over observations. A low value means a series that barely
    moves — a constant, a step, or something rounded until the variation is gone — which inflates
    a corpus without teaching a model anything. ``nonpositive`` matters before anyone reaches for
    sMAPE, a log transform or a multiplicative model.

    Series with fewer than two observations are left out of every statistic here: with one point a
    series has one distinct value, which would otherwise be reported as both "100% unique" and
    "100% constant" — as it was, for the 853 million single-point annual series.
    """
    root = series_root(root)
    rng = random.Random(seed)
    pools: dict[str, list[Path]] = {}
    for src in sorted(root.iterdir()):
        if not src.is_dir() or (sources and src.name not in sources):
            continue
        for fd in sorted(src.iterdir()):
            if not fd.is_dir():
                continue
            freq = fd.name.split("=", 1)[-1]
            if frequencies and freq not in frequencies:
                continue
            pools.setdefault(freq, []).extend(fd.glob("*.parquet"))

    per_file = max(50, per_frequency // max(max_files, 1))
    rows = []
    for freq, files in pools.items():
        files = list(files)
        rng.shuffle(files)
        frames, used = [], 0
        for p in files[:max_files]:
            try:
                d = (pl.scan_parquet(p).select("n_obs", "values").head(per_file)
                     .with_columns(
                         pl.col("values").list.n_unique().alias("n_unique"),
                         pl.col("values").list.eval(pl.element().is_null()).list.any().alias("any_null"),
                         pl.col("values").list.eval((pl.element() <= 0).fill_null(False)).list.any().alias("any_nonpos"),
                     ).drop("values").collect(engine="streaming"))
            except Exception as exc:                      # one unreadable file is not the report
                log.debug("skipped %s: %s", p, exc)
                continue
            if d.height:
                frames.append(d)
                used += 1
        if not frames:
            continue
        # n_unique counts null as a value when the list holds one; take it back out
        d = pl.concat(frames).with_columns(
            (pl.col("n_unique") - pl.col("any_null").cast(pl.Int32)).alias("n_distinct"))
        m = d.filter(pl.col("n_obs") >= 2).with_columns(
            (pl.col("n_distinct") / pl.col("n_obs")).alias("unique_share"))
        rows.append({
            "frequency": freq, "name": FREQ_NAME.get(freq, freq),
            "sampled": d.height, "datasets": used,
            "share_one_point": float((d["n_obs"] < 2).mean()),
            "distinct_median": float(m["n_distinct"].median()) if m.height else None,
            "unique_share_median": float(m["unique_share"].median()) if m.height else None,
            "share_constant": float((m["n_distinct"] <= 1).mean()) if m.height else None,
            "share_near_constant": float((m["unique_share"] < 0.05).mean()) if m.height else None,
            "share_nonpositive": float(m["any_nonpos"].mean()) if m.height else None,
        })
    return _order(pl.DataFrame(rows)) if rows else pl.DataFrame()


# -- one real row, so the schema is shown rather than described ----------------------------------

# Datasets tried first when picking rows to illustrate the schema. These are chosen for variety of
# subject, not cherry-picked for flattering numbers: whichever series is picked inside them is the
# longest one that actually varies, and the page says so. Anything missing is skipped silently, so
# a partial crawl still produces a page.
SHOWCASE = ("FISH_INLAND", "ert_bil_eur_d", "prc_hicp_midx", "apro_ec_poulm", "sts_inpr_m")


# Datasets tried first for the globe's parallels: one series per frequency where the corpus has
# one, so latitude (= frequency) is actually populated. Same rule as SHOWCASE: the longest series
# in the file that actually varies, nothing cherry-picked for shape.
LOGO_SHOWCASE = ("ert_bil_eur_d", "irt_st_m", "prc_hicp_midx", "namq_10_gdp", "nama_10_gdp",
                 "FISH_INLAND", "sts_inpr_m", "demo_r_mweek3", "apro_ec_poulm", "ei_bsco_m",
                 "une_rt_m", "nrg_cb_e")


def specimens(root=None, sources=None, frequencies=None, n: int = 4, min_obs: int = 25,
              min_unique_share: float = 0.5, candidates: int = 60,
              hints: tuple[str, ...] = SHOWCASE) -> pl.DataFrame:
    """A handful of real rows, to show the schema rather than describe it.

    Three things disqualify a candidate, and the third is easy to forget: it must be long enough
    to draw, carry dimensions to show a hierarchy, and actually *vary*. The first row this picked
    was 192 monthly observations of Austrian hatchery chicks, zero in every month but one — valid
    data, and an advertisement for nothing.

    Reading the values of a billion series would be absurd, so the search never does: the
    ``SHOWCASE`` datasets are opened by name, and any remaining slots are filled from a scalar
    scan whose few dozen candidates are then read one file at a time.
    """
    rows, seen = [], set()
    for hint in hints:
        if len(rows) >= n:
            break
        for path in sorted(series_root(root).glob(f"*/freq=*/*{hint}*.parquet")):
            got = _best_in_file(path, min_obs, min_unique_share)
            if got is not None and got["dataset_id"] not in seen:
                seen.add(got["dataset_id"])
                rows.append(got)
                break
    if len(rows) < n:
        rows += _fill_from_scan(root, sources, frequencies, n - len(rows), min_obs,
                                min_unique_share, candidates, seen)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _best_in_file(path: Path, min_obs: int, min_unique_share: float) -> dict | None:
    """The longest varied series in one dataset file, with the metadata needed to show it."""
    cols = ["series_uid", "description", "units", "n_obs", "start_date", "end_date",
            "dimensions", "source", "frequency", "dataset_id", "dataset_title", "values"]
    try:
        lf = pl.scan_parquet(path)
        have = set(lf.collect_schema().names())
        if not {"series_uid", "n_obs", "values", "description"} <= have:
            return None
        df = (lf.select([c for c in cols if c in have]).filter((pl.col("n_obs") >= min_obs)
                                     & pl.col("description").is_not_null())
                .with_columns((pl.col("values").list.n_unique() / pl.col("n_obs")).alias("us"))
                .filter(pl.col("us") >= min_unique_share)
                .sort(["n_obs", "series_uid"], descending=[True, False])
                .head(1).collect(engine="streaming"))
    except Exception as exc:            # an illustration never justifies failing a build
        log.debug("specimen: %s unreadable: %s", path.name, exc)
        return None
    return _pack_specimen(df.row(0, named=True)) if df.height else None


def _fill_from_scan(root, sources, frequencies, want: int, min_obs: int, min_unique_share: float,
                    candidates: int, seen: set) -> list[dict]:
    from terrastat.quality import scan

    cols = ["series_uid", "description", "units", "n_obs", "start_date", "end_date",
            "dimensions", "source", "frequency", "dataset_id", "dataset_title"]
    lf = scan(root, sources, frequencies, None)
    have = set(lf.collect_schema().names())
    if not {"series_uid", "n_obs", "description", "source", "frequency", "dataset_id"} <= have:
        return []
    cols = [c for c in cols if c in have]
    today = dt.date.today()
    where = (pl.col("n_obs") >= min_obs) & pl.col("description").is_not_null()
    if "dimensions" in have:
        where = where & (pl.col("dimensions").list.len() >= 1)
    if "end_date" in have:
        where = where & pl.col("end_date").is_between(dt.date(today.year - 3, 1, 1), today)
    try:
        pool = (lf.select(cols).filter(where)
                  .sort(["n_obs", "series_uid"], descending=[True, False])
                  .head(candidates).collect(engine="streaming"))
    except Exception as exc:
        log.debug("specimen scan failed: %s", exc)
        return []
    out = []
    for r in pool.iter_rows(named=True):
        if len(out) >= want or r["dataset_id"] in seen:
            continue
        path = (series_root(root) / r["source"] / f'freq={r["frequency"]}'
                / f'{r["dataset_id"]}.parquet')
        got = _best_in_file(path, min_obs, min_unique_share) if path.exists() else None
        if got is not None:
            seen.add(got["dataset_id"])
            out.append(got)
    return out


def _pack_specimen(r: dict) -> dict:
    vals = [v for v in (r["values"] or []) if v is not None]
    step = max(1, len(vals) // 90)
    return {
        "series_uid": r["series_uid"],
        "dataset_title": r.get("dataset_title") or r.get("dataset_id"),
        "description": r["description"],
        "units": r.get("units"),
        "source": r["source"],
        "frequency": r["frequency"],
        "dataset_id": r["dataset_id"],
        "n_obs": r["n_obs"],
        "start_date": r.get("start_date"),
        "end_date": r.get("end_date"),
        "dims": [f'{d["name"]}: {d["label"] or d["code"]}' for d in (r.get("dimensions") or [])],
        "codes": [f'{d["id"]}={d["code"]}' for d in (r.get("dimensions") or [])],
        "spark": vals[::step][:90],
        "unique_share": len(set(vals)) / len(vals) if vals else 0.0,
    }


# -- exact: how concentrated ---------------------------------------------------------------------

def concentration_table(root=None, sources=None, frequencies=None, top: int = 10) -> pl.DataFrame:
    """How much of each frequency comes from how few datasets.

    A count of series flatters a corpus that is really a handful of enormous cross-tabulations.
    ``effective_datasets`` is the perplexity of the series-per-dataset distribution — the number of
    equally sized datasets that would be as diverse as what is actually here — and it is usually a
    small fraction of the raw dataset count. It is also the leakage warning in numeric form: the
    more concentrated a frequency, the more a random split puts near-copies on both sides of it.
    """
    from terrastat.quality import scan

    d = (scan(root, sources, frequencies, ["source", "frequency", "dataset_id"])
         .group_by("frequency", "source", "dataset_id").agg(pl.len().alias("n"))
         .collect(engine="streaming"))
    def summarise(g: pl.DataFrame, freq: str, name: str) -> dict:
        g = g.sort("n", descending=True)
        total = int(g["n"].sum())
        share = (g["n"] / total).to_numpy()
        entropy = float(-(share * np.log(np.maximum(share, 1e-300))).sum())
        return {"frequency": freq, "name": name, "datasets": g.height, "n_series": total,
                "top_dataset": f"{g['source'][0]}:{g['dataset_id'][0]}",
                "top_share": float(g["n"][0] / total), "top_n": top,
                "topn_share": float(g.head(top)["n"].sum() / total),
                "effective_datasets": math.exp(entropy)}

    rows = [summarise(g, freq, FREQ_NAME.get(freq, freq))
            for (freq,), g in d.group_by(["frequency"], maintain_order=True)]
    out = _order(pl.DataFrame(rows))
    # One dataset can appear at several frequencies, so the whole-corpus figures are recomputed
    # over the pooled table rather than summed out of the rows above.
    pooled = d.group_by("source", "dataset_id").agg(pl.col("n").sum())
    return pl.concat([out, pl.DataFrame([summarise(pooled, "*", "all frequencies")])],
                     how="diagonal_relaxed")


# -- the page ------------------------------------------------------------------------------------

def _n(x) -> str:
    """A count a person can read at a glance rather than count digits in."""
    if x is None:
        return "—"
    x = float(x)
    for scale, suffix, digits in ((1e9, "B", 2), (1e6, "M", 1), (1e3, "k", 1)):
        if x >= scale:
            return f"{x / scale:.{digits}f} {suffix}".replace(".00 ", " ").replace(".0 ", " ")
    return f"{int(x):,}"


def _pct(x, digits: int = 0) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def _num(x, digits: int = 1) -> str:
    """A number with trailing zeros trimmed, so a column of lengths reads 4 rather than 4.0."""
    return "—" if x is None else f"{x:,.{digits}f}".rstrip("0").rstrip(".")


def _fixed(x, digits: int = 2) -> str:
    """A number keeping its trailing zeros, for a column where every entry has the same scale."""
    return "—" if x is None else f"{x:.{digits}f}"


def render(length: pl.DataFrame, values: pl.DataFrame, conc: pl.DataFrame,
           crawl: pl.DataFrame, years: float = MIN_YEARS,
           live_periods: float = LIVE_PERIODS, generated: str | None = None) -> str:
    """The whole page, as markdown. ``generated`` is the day the figures were computed."""
    generated = generated or dt.date.today().isoformat()
    asof = length["asof"][0] if length.height else dt.date.today()
    total = int(length["n_series"].sum())
    long_total = int(length["n_long"].fill_null(0).sum())
    live_total = int(length["n_long_live"].fill_null(0).sum())
    nonannual = length.filter(pl.col("frequency") != "A")
    length, values = _order(length), _order(values)
    overall = conc.filter(pl.col("frequency") == "*") if conc.height else pl.DataFrame()
    conc = _order(conc.filter(pl.col("frequency") != "*")) if conc.height else conc
    top_n = int(conc["top_n"][0]) if conc.height else 10

    L = [
        "# The corpus at a glance",
        "",
        f"*Generated by `terrastat corpus` on {generated}. Recency is measured against "
        f"{asof} — the most recent retrieval in the corpus, rather than the date you are reading "
        f"this, because otherwise a daily series looks stale the moment a crawl ends.*",
        "",
        "**Do not edit this file.** Regenerate it after a crawl or after adding a source:",
        "",
        "```bash",
        "terrastat corpus --out docs/corpus.md",
        "```",
        "",
        "## Headline",
        "",
        "| | |",
        "|---|---:|",
        f"| series in total | **{_n(total)}** |",
        f"| with at least {years:g} years of their own frequency | **{_n(long_total)}** |",
        f"| …and still published (within {live_periods:g} periods) | **{_n(live_total)}** |",
        f"| …excluding annual, which dominates the count | **{_n(int(nonannual['n_long_live'].fill_null(0).sum()))}** |",
        f"| distinct datasets | {_n(overall['datasets'][0]) if overall.height else '—'} |",
        f"| …effective, once weighted by size | {_n(round(overall['effective_datasets'][0])) if overall.height else '—'} |",
        "",
        "## Length, by frequency",
        "",
        f"Exact, over every series. The bar is {years:g} years *of the series' own frequency*, so "
        f"{years:g} observations for annual and {min_obs('M', years)} for monthly; `A3` and `P` "
        "step more than a year at a time, so theirs is two observations. `p5 / median / p95` are "
        "the 90% extremes of length **within the cohort that clears the bar**.",
        "",
        f"| frequency | ≥ {years:g} years | of all | still published | length (obs) p5 / med / p95 | in years | mean obs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in length.iter_rows(named=True):
        f = r["frequency"]
        if r["n_long"] is None:
            L.append(f"| {r['name']} `{f}` | n/a | {_n(r['n_series'])} total | n/a | — | — | — |")
            continue
        L.append(
            f"| {r['name']} `{f}` | **{_n(r['n_long'])}** | {_pct(r['long_share'])} of {_n(r['n_series'])} "
            f"| {_n(r['n_long_live'])} | {r['p5']} / **{r['p50']}** / {r['p95']} "
            f"| {_num(r['p5_years'])} / **{_num(r['p50_years'])}** / {_num(r['p95_years'])} "
            f"| {_num(r['mean_obs'], 0)} |"
        )
    L += [
        "",
        "`n/a` where a frequency has no fixed period: a length in years and a lag in periods are "
        "both undefined for `I` and `OTHER`, so they are counted but not judged.",
        "",
        "## What the values look like",
        "",
    ]
    if values.height:
        s = int(values["sampled"].sum())
        L += [
            f"**Sampled, not exact** — {_n(s)} series drawn across datasets, because this is the "
            "only part that has to read the value lists. Series with a single observation are "
            "excluded from every column but the first: with one point a series is simultaneously "
            "100% unique and 100% constant.",
            "",
            "| frequency | sampled | one point only | distinct values (med) | unique share (med) | constant | near-constant | has a value ≤ 0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in values.iter_rows(named=True):
            L.append(
                f"| {r['name']} `{r['frequency']}` | {_n(r['sampled'])} ({r['datasets']} datasets) "
                f"| {_pct(r['share_one_point'])} | {_num(r['distinct_median'], 0)} "
                f"| {_fixed(r['unique_share_median'], 2)} | {_pct(r['share_constant'], 1)} "
                f"| {_pct(r['share_near_constant'], 1)} | {_pct(r['share_nonpositive'], 1)} |"
            )
        L += [
            "",
            "`unique share` is distinct values over observations. A long series with a low unique "
            "share is a step function or something rounded flat — abundant, and close to "
            "worthless as a training example. `has a value ≤ 0` is what decides whether sMAPE, a "
            "log transform or a multiplicative model is safe.",
        ]
    L += ["", "## How concentrated it is", ""]
    if conc.height:
        L += [
            "Exact. A series count flatters a corpus that is really a few enormous "
            "cross-tabulations. `effective datasets` is the perplexity of the series-per-dataset "
            "distribution: the number of *equally sized* datasets that would be as diverse as what "
            "is actually here.",
            "",
            f"| frequency | datasets | effective | largest dataset | its share | top {top_n} share |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for r in conc.iter_rows(named=True):
            L.append(
                f"| {r['name']} `{r['frequency']}` | {_n(r['datasets'])} "
                f"| **{_num(r['effective_datasets'], 0)}** | `{r['top_dataset']}` "
                f"| {_pct(r['top_share'])} | {_pct(r['topn_share'])} |"
            )
        L += [
            "",
            "Read this next to [leakage.md](leakage.md): the more concentrated a frequency, the "
            "more a random train/test split puts near-copies of the same table on both sides. "
            "`terrastat.dataset.split()` groups by dataset for exactly this reason.",
        ]
    L += ["", "## Where it came from", "", "| source | series | last retrieved |", "|---|---:|---|"]
    for r in crawl.iter_rows(named=True):
        L.append(f"| {r['source']} | {_n(r['n_series'])} | {r['last_retrieved'][:19].replace('T', ' ')} |")
    L += [
        "",
        "A crawl is resumable and rarely finishes in one sitting, so these timestamps differ. "
        "Licences and attribution are per series, not per source — see the "
        "[README](../README.md#licences) before redistributing anything.",
        "",
    ]
    return "\n".join(L)


def compute(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
            years: float = MIN_YEARS, live_periods: float = LIVE_PERIODS,
            sample: int = 100_000, max_files: int = 250, seed: int = 0,
            values: bool = True, concentration: bool = True) -> dict[str, pl.DataFrame]:
    """Every table the page is made of, so a second renderer never recomputes them."""
    log.info("length and recency (exact, whole population)")
    length = length_table(root, sources, frequencies, asof, years, live_periods)
    log.info("value statistics (sampled)")
    vals = value_table(root, sources, frequencies, sample, max_files, seed) if values else pl.DataFrame()
    log.info("concentration (exact)")
    conc = concentration_table(root, sources, frequencies) if concentration else pl.DataFrame()
    return {"length": length, "values": vals, "concentration": conc,
            "crawl": crawled_at(root, sources), "specimen": specimens(root, sources, frequencies),
            # the parallels of the globe on the page: more rings, spanning frequencies
            "rings": specimens(root, sources, frequencies, n=9, hints=LOGO_SHOWCASE),
            "generated": dt.date.today().isoformat()}


def build(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
          years: float = MIN_YEARS, live_periods: float = LIVE_PERIODS,
          sample: int = 100_000, max_files: int = 250, seed: int = 0,
          values: bool = True, concentration: bool = True) -> str:
    """Compute every table and render the markdown page."""
    t = compute(root, sources, frequencies, asof, years, live_periods, sample, max_files, seed,
                values, concentration)
    return render(t["length"], t["values"], t["concentration"], t["crawl"], years, live_periods,
                  generated=t.get("generated"))


# The computed tables, small enough to commit beside the pages they produced. Keeping them is what
# lets a checkout with no data — continuous integration, a reviewer's laptop — re-render the pages
# and confirm they match. Without it, "are these files current?" is unanswerable away from the crawl.
TABLES = "docs/corpus_tables.json"


def save_tables(tables: dict[str, pl.DataFrame], path: Path | str = TABLES) -> Path:
    """Write the tables as one readable JSON file, so a diff shows which figure moved."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [{k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in row.items()}
                      for row in df.iter_rows(named=True)]
               for name, df in tables.items() if isinstance(df, pl.DataFrame)}
    # The day the figures were computed travels with them. Rendering "generated on" from the clock
    # instead made `--check` fail on any day after the one the pages were built.
    payload["generated"] = tables.get("generated") or dt.date.today().isoformat()
    p.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
    return p


def load_tables(path: Path | str = TABLES) -> dict[str, pl.DataFrame]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    generated = payload.pop("generated", None)
    out = {name: pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
           for name, rows in payload.items()}
    out["generated"] = generated
    if out.get("length") is not None and "asof" in out["length"].columns:
        out["length"] = out["length"].with_columns(pl.col("asof").str.to_date())
    return out


def rerender(out: Path | str = "docs/corpus.md", html: Path | str = "docs/index.html",
             tables_path: Path | str = TABLES, years: float = MIN_YEARS) -> list[Path]:
    """Rewrite both pages from the committed figures, without touching the corpus.

    Recomputing takes minutes over a billion series; re-rendering takes under a second. Wording,
    layout and colour are changed far more often than the data is, so they should not share a
    cost. The figures are whatever the last real run measured, and the pages keep saying so.
    """
    from terrastat.site import write as write_page

    tables = load_tables(tables_path)
    text = render(tables["length"], tables["values"], tables["concentration"],
                  tables["crawl"], years, generated=tables.get("generated"))
    md = Path(out)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    log.info("re-rendered %s", md)
    return [md, write_page(tables, html, years)]


def check(out: Path | str = "docs/corpus.md", html: Path | str = "docs/index.html",
          tables_path: Path | str = TABLES, years: float = MIN_YEARS) -> list[str]:
    """Re-render from the committed tables and report which pages no longer match.

    This does **not** need the corpus, and cannot tell you whether the figures are up to date with
    a crawl — only the machine holding 30 GB of Parquet can answer that. What it does catch is the
    failure that actually happens: someone edits a generated page by hand, or changes a renderer
    and commits the code without regenerating what it produces.
    """
    from terrastat.site import render as render_page

    tables = load_tables(tables_path)
    problems = []
    want_md = render(tables["length"], tables["values"], tables["concentration"],
                     tables["crawl"], years, generated=tables.get("generated"))
    for path, want in ((Path(out), want_md),
                       (Path(html), render_page(tables, years))):
        if not path.exists():
            problems.append(f"{path} is missing")
        elif path.read_text(encoding="utf-8") != want:
            problems.append(f"{path} does not match {tables_path}; run `terrastat corpus`")
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="terrastat corpus", description=__doc__.split("\n")[0])
    p.add_argument("--out", default="docs/corpus.md", help="where to write the markdown annex")
    p.add_argument("--html", default="docs/index.html", help="where to write the project page")
    p.add_argument("--no-html", action="store_true", help="write only the markdown")
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--frequencies", nargs="*", default=None)
    p.add_argument("--years", type=float, default=MIN_YEARS, help="the length bar, in years")
    p.add_argument("--sample", type=int, default=100_000, help="series per frequency for the value statistics")
    p.add_argument("--no-values", action="store_true", help="skip the sampled value statistics")
    p.add_argument("--no-concentration", action="store_true", help="skip the per-dataset pass")
    p.add_argument("--tables", default=TABLES, help="where to keep the computed figures")
    p.add_argument("--check", action="store_true",
                   help="do not read the corpus: re-render from --tables and fail if the committed pages differ")
    p.add_argument("--render", action="store_true",
                   help="do not read the corpus: rewrite both pages from --tables (seconds, not minutes)")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.render:
        rerender(a.out, a.html, a.tables, a.years)
        return 0
    if a.check:
        problems = check(a.out, a.html, a.tables, a.years)
        for msg in problems:
            log.error("%s", msg)
        if not problems:
            log.info("%s and %s match %s", a.out, a.html, a.tables)
        return 1 if problems else 0

    tables = compute(sources=a.sources, frequencies=a.frequencies, years=a.years,
                     sample=a.sample, values=not a.no_values,
                     concentration=not a.no_concentration)
    text = render(tables["length"], tables["values"], tables["concentration"], tables["crawl"],
                  a.years, generated=tables.get("generated"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log.info("wrote %s (%d lines)", out, text.count(chr(10)) + 1)
    # The page and the annex come from the same tables, so the two cannot disagree.
    if not a.no_html:
        from terrastat.site import write as write_page

        write_page(tables, a.html, a.years)
    save_tables(tables, a.tables)
    return 0
