"""Corpus statistics for research use, regenerated from the stored series layer.

The report describes series counts, observation lengths, endpoint dates, observed-value
characteristics and dataset concentration. It complements the more detailed source-by-frequency
tables produced by ``terrastat quality``. These summaries describe the collected data; they do
not establish suitability for a particular model or evaluation design.

The report records its computation date and source retrieval timestamps. Regenerate it after a
crawl, after adding a source, or after any change to how series are built::

    terrastat corpus --out docs/corpus.md

**Full-corpus versus sample summaries.** Counts, lengths and endpoint statistics use every
stored series in the requested scope and require only scalar columns. Value diagnostics read
a bounded sample of value lists. Files are selected randomly, but rows within each selected file
are taken from its beginning. These unweighted sample summaries are descriptive, not estimates
with known sampling uncertainty for the full corpus.
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
# Presentation order: annual, monthly and quarterly first, followed by the remaining frequencies.
FREQ_ORDER = ["A", "M", "Q", "W", "D", "S", "A3", "P", "BW", "H", "B", "I", "OTHER"]

# The observation-count threshold is max(2, ceil(years * periods_per_year)). It does not
# establish elapsed coverage, contiguous observations, seasonality or modelling suitability.
MIN_YEARS = 2.0

# Endpoint lag cutoff, in approximate periods of the series' own frequency. This is not a
# publication-status measure; future endpoints also satisfy the one-sided lag comparison.
LIVE_PERIODS = 12.0

# Observation-count quantiles among series meeting the length threshold.
LENGTH_QUANTILES = (0.05, 0.5, 0.95)


def min_obs(frequency: str, years: float = MIN_YEARS) -> int | None:
    """Observations that ``years`` years amounts to at this frequency, at least two."""
    ppy = PERIODS_PER_YEAR.get(frequency)
    return None if ppy is None else max(2, math.ceil(years * ppy))


def _order(df: pl.DataFrame) -> pl.DataFrame:
    """Put the rows in reading order.

    Applied when rendering as well as computing, so changing ``FREQ_ORDER`` does not require
    rescanning the stored series.
    """
    if not df.height or "frequency" not in df.columns:
        return df
    rank = {f: i for i, f in enumerate(FREQ_ORDER)}
    return (df.with_columns(pl.col("frequency").replace_strict(rank, default=99).alias("_o"))
              .sort("_o", maintain_order=True).drop("_o"))


def crawled_at(root=None, sources=None) -> pl.DataFrame:
    """Latest recorded retrieval timestamp and stored series count per source."""
    from terrastat.quality import scan

    return (scan(root, sources, None, ["source", "retrieved_at"])
            .group_by("source")
            .agg(pl.col("retrieved_at").max().alias("last_retrieved"), pl.len().alias("n_series"))
            .sort("source").collect(engine="streaming"))


# -- exact: how many, how long, how live ---------------------------------------------------------

def _wq(value: pl.Series, weight: pl.Series, qs) -> list[int]:
    """Quantiles of a weighted, already-sorted value column.

    Count each distinct length, then select the first value whose cumulative count reaches
    the requested fraction. This avoids materialising the full observation-count column.
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
    """Per-frequency counts, observation lengths and endpoint lag, using all scoped series.

    ``asof`` defaults to the most recent ``retrieved_at`` in the corpus. Endpoint lag uses
    ``end_date``, which need not be the date of the last non-missing observation or a release
    date. A future endpoint passes the one-sided lag cutoff. The internal ``live`` names are
    retained for compatibility; they do not indicate whether a provider still publishes a series.
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


# -- sampled observed-value diagnostics ----------------------------------------------------------

def value_table(root=None, sources=None, frequencies=None, per_frequency: int = 100_000,
                max_files: int = 250, seed: int = 0) -> pl.DataFrame:
    """Descriptive distinctness and sign summaries for a sample pooled across sources.

    Read the first ``max(50, per_frequency // max(max_files, 1))`` rows from each of up to
    ``max_files`` randomly selected dataset files per frequency. File selection is random;
    within-file row selection is not. Rows are pooled without sampling weights, so these
    summaries should not be interpreted as representative estimates of corpus-wide prevalence.

    ``unique_share`` is the count of distinct non-null values divided by ``n_obs``. It measures
    repetition, not the magnitude or temporal pattern of variation. The legacy
    ``share_near_constant`` field is the proportion with this ratio below 0.05; it is not a
    variance threshold and includes constant series when they satisfy the ratio threshold.

    Sample counts and ``share_one_point`` use all sampled rows; despite its legacy name, the
    latter includes every row with ``n_obs < 2``. Other summaries use only ``n_obs >= 2`` and
    exclude nulls when counting distinct values or checking for nonpositive values.
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

# Datasets tried first for schema illustrations. Examples are deliberately selected by subject,
# length and distinctness; they are not a representative sample. Missing candidates are skipped so
# a partial crawl still produces a page.
SHOWCASE = ("FISH_INLAND", "ert_bil_eur_d", "prc_hicp_midx", "apro_ec_poulm", "sts_inpr_m")


def specimens(root=None, sources=None, frequencies=None, n: int = 4, min_obs: int = 25,
              min_unique_share: float = 0.5, candidates: int = 60,
              hints: tuple[str, ...] = SHOWCASE) -> pl.DataFrame:
    """Select illustrative rows with sufficient length, descriptions and distinct values.

    The named ``SHOWCASE`` datasets are tried first. Any remaining slots are filled from a
    scalar scan followed by bounded reads of candidate files. These selected examples
    illustrate the schema and should not be used to infer the distribution of the corpus.
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
    """Describe the distribution of series counts across source/dataset identifiers.

    ``effective_datasets`` is exp(-sum(p * log(p))), where p is each dataset's share of series.
    It is the number of equally sized datasets with the same entropy of series allocation,
    not a measure of independent information, subject coverage or detected duplication.
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
    if x is None:
        return "—"
    formatted = f"{x:,.{digits}f}"
    return formatted.rstrip("0").rstrip(".") if digits > 0 else formatted


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
        "# Corpus statistics",
        "",
        f"*Statistics computed by `terrastat corpus` on {generated}. Endpoint reference date: "
        f"{asof}. By default, this is the most recent retrieval date in the selected corpus.*",
        "",
        "The counts describe the stored series layer at that computation date. They do not "
        "measure all data available from the providers, statistical independence or the number "
        "of series suitable for a particular forecasting task. Values displayed below are rounded.",
        "",
        "**Do not edit this file.** Regenerate it after a crawl or after adding a source:",
        "",
        "```bash",
        "terrastat corpus --out docs/corpus.md",
        "```",
        "",
        "## Report scope",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Stored series | **{_n(total)}** |",
        f"| Meeting the observation-count threshold below | **{_n(long_total)}** |",
        f"| Of these, endpoint within the {live_periods:g}-period cutoff | **{_n(live_total)}** |",
        f"| Of these, non-annual frequencies | **{_n(int(nonannual['n_long_live'].fill_null(0).sum()))}** |",
        f"| Distinct source/dataset identifiers | {_n(overall['datasets'][0]) if overall.height else '—'} |",
        f"| Entropy-based effective dataset count | {_n(round(overall['effective_datasets'][0])) if overall.height else '—'} |",
        "",
        "## Series counts and observation lengths",
        "",
        "Computed over all stored series in each frequency. The observation-count threshold is "
        f"`n_obs >= max(2, ceil({years:g} × periods_per_year))`: {min_obs('A', years)} non-missing "
        f"observations for annual series, {min_obs('Q', years)} for quarterly and "
        f"{min_obs('M', years)} for monthly. This is a descriptive selection rule, not a minimum "
        "requirement for fitting a model. It does not establish elapsed coverage, contiguous "
        "observations or complete seasonal cycles.",
        "",
        "The 5th percentile, median, 95th percentile and mean are calculated **only among series "
        "meeting the observation-count threshold**. The year-equivalent columns divide observation "
        "counts by nominal periods per year; they are not elapsed calendar spans.",
        "",
        "Endpoint lag is `(reference_date - end_date).days / 365.25 × periods_per_year`. "
        f"The endpoint count includes threshold-eligible series with lag ≤ {live_periods:g}. "
        "This one-sided cutoff also includes future-dated endpoints, such as projections. "
        "`end_date` is the last stored date, not necessarily the last non-missing value or a "
        "publication date; this measure does not establish whether a series is still published.",
        "",
        "| Frequency | Meeting threshold | Share of frequency | Endpoint within cutoff | Observations: P5 / median / P95 | Year-equivalent: P5 / median / P95 | Mean observations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in length.iter_rows(named=True):
        f = r["frequency"]
        if r["n_long"] is None:
            L.append(f"| {r['name']} `{f}` | n/a | {_n(r['n_series'])} total | n/a | — | — | — |")
            continue
        L.append(
            f"| {r['name']} `{f}` | **{_n(r['n_long'])}** | {_pct(r['long_share'])} of {_n(r['n_series'])} "
            f"| {_n(r['n_long_live'])} | {_num(r['p5'], 0)} / **{_num(r['p50'], 0)}** / {_num(r['p95'], 0)} "
            f"| {_num(r['p5_years'])} / **{_num(r['p50_years'])}** / {_num(r['p95_years'])} "
            f"| {_num(r['mean_obs'], 0)} |"
        )
    L += [
        "",
        "`n/a` indicates that no nominal period is defined for `I` or `OTHER`. These series "
        "contribute to the total count but not to the observation-threshold or endpoint-lag cohorts.",
        "",
        "## Observed-value diagnostics",
        "",
    ]
    if values.height:
        s = int(values["sampled"].sum())
        L += [
            f"**Descriptive sample summaries** for {s:,} series. Dataset files are randomly "
            "selected within each frequency, then a bounded number of rows is read from the "
            "beginning of each file. Rows are pooled without sampling weights. Because row "
            "selection within files is not random, these results are not representative estimates "
            "of corpus-wide prevalence; no sampling uncertainty is reported.",
            "",
            "Sample counts and the share with fewer than two observations use **all sampled "
            "rows**. Every other diagnostic uses only sampled series with `n_obs >= 2`. "
            "Null values are excluded from the distinct-value count and the nonpositive-value check.",
            "",
            "| Frequency | Sampled series (files) | Fewer than 2 observations | Distinct values (median) | Distinct / observed (median) | Constant | Distinct / observed < 0.05 | Any value ≤ 0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in values.iter_rows(named=True):
            L.append(
                f"| {r['name']} `{r['frequency']}` | {_n(r['sampled'])} ({r['datasets']} files) "
                f"| {_pct(r['share_one_point'])} | {_num(r['distinct_median'], 0)} "
                f"| {_fixed(r['unique_share_median'], 2)} | {_pct(r['share_constant'], 1)} "
                f"| {_pct(r['share_near_constant'], 1)} | {_pct(r['share_nonpositive'], 1)} |"
            )
        L += [
            "",
            "**Distinct / observed** is the number of distinct non-null values divided by "
            "`n_obs`, calculated separately for each series. **Constant** means at most one "
            "distinct non-null value. **Distinct / observed < 0.05** is a repetition diagnostic, "
            "not a low-variance criterion; it includes constant series when their ratio also "
            "meets that threshold. A low ratio can occur in discrete, intermittent, rounded or "
            "repeated-value series and does not, by itself, determine their research value.",
            "",
            "**Any value ≤ 0** is the share of eligible series containing at least one "
            "nonpositive observed value. It can inform checks before using logarithms or "
            "multiplicative specifications. Metric suitability requires the metric's definition "
            "and the actual values and forecasts; this flag alone does not determine whether "
            "sMAPE is defined or appropriate.",
        ]
    L += ["", "## Dataset concentration", ""]
    if conc.height:
        L += [
            "Computed over all stored series. Dataset identity combines `source` and "
            "`dataset_id`. The effective dataset count is `exp(-Σ pᵢ log(pᵢ))`, where `pᵢ` is "
            "a dataset's share of series within the reported frequency. It equals the number "
            "of equally sized datasets with the same entropy of series allocation. It does not "
            "measure subject diversity, statistical independence or the prevalence of duplicates.",
            "",
            f"| Frequency | Datasets | Effective dataset count | Largest dataset | Its series share | Top {top_n} series share |",
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
            "Large datasets can dominate a uniformly sampled set of series. Grouping folds by "
            "dataset keeps rows from the same table together, but does not establish independence "
            "across datasets or sources. `terrastat.dataset.split()` provides dataset grouping; "
            "forecasting studies also need chronological cutoffs and checks for related series, "
            "revisions and release timing. See [evaluation considerations](leakage.md) and the "
            "[quarterly forecasting tutorial](forecasting.md).",
        ]
    L += ["", "## Sources and retrieval dates", "", "| Source | Stored series | Latest retrieval |", "|---|---:|---|"]
    for r in crawl.iter_rows(named=True):
        L.append(f"| {r['source']} | {_n(r['n_series'])} | {r['last_retrieved'][:19].replace('T', ' ')} |")
    L += [
        "",
        "Each timestamp is the most recent `retrieved_at` value for that source; it does not "
        "mean every series was fetched at that time. Resumed crawls can contain different "
        "retrieval dates. Series metadata records licence classifications and attribution; "
        "provider terms and dataset-specific exceptions still apply. See "
        "[licences and citation](licensing.md) before using or redistributing data.",
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

    Re-rendering reads only the saved summary tables. It preserves their computation date and
    measured values; wording and layout changes do not require rescanning the series layer.
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

    This does not need the corpus and cannot establish whether the saved figures reflect the
    current series layer. It detects manually edited generated pages and renderer changes whose
    outputs have not been regenerated.
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
