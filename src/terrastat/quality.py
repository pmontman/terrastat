"""A bird's-eye view of the series layer: how much there is, and how usable it is.

The point is not to validate the pipeline (the tests do that) but to answer, before any modelling
starts, questions like *how many monthly series are still being updated?*, *how many carry enough
history to fit anything seasonal?*, *how many are constant?* — and therefore which slices of the
corpus are worth training or evaluating on.

Two passes, because they cost very different amounts:

**The scalar pass** reads only fixed-width columns (``frequency``, ``start_date``, ``end_date``,
``n_points``, ``n_obs``, ``n_flagged``). Parquet stores columns separately, so the ``dates``,
``values`` and ``flags`` lists — which are essentially all of the 28 GB on disk — are never
touched. The whole population of ~1e9 series is covered exactly, in seconds.

**The deep pass** needs ``values`` itself: how many distinct values a series takes, whether it is
constant, where its gaps sit. That means reading the list columns, at roughly 70k series a
second, so it runs on a stratified sample by default. Everything it reports is an estimate and is
labelled as one.

Both are available as functions returning frames (for a notebook) and through
``terrastat quality`` (for a batch run that writes the tables and a written summary to disk).

A note on what "a cycle" means here: one year, so 12 points at monthly, 4 at quarterly, 1 at
annual. Length is then measured in years of actual data, ``n_obs / periods_per_year``, which is
the quantity that decides whether a seasonal model can be fitted at all.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import polars as pl

from terrastat.storage import data_dir

log = logging.getLogger(__name__)

# Points per year for each canonical frequency. None where the notion does not apply: I is
# irregular, OTHER is the catch-all for periods we could not place on a calendar.
PERIODS_PER_YEAR: dict[str, float | None] = {
    "H": 8760.0, "D": 365.25, "B": 260.0, "W": 52.0, "BW": 26.0, "M": 12.0,
    "Q": 4.0, "S": 2.0, "A": 1.0, "A3": 1 / 3, "P": 1 / 5, "I": None, "OTHER": None,
}

# Columns the scalar pass needs. Keeping this list short is what makes it fast.
SCALAR_COLUMNS = [
    "source", "frequency", "start_date", "end_date", "n_points", "n_obs", "n_flagged",
]

# Length thresholds, in years of data. The first three are the user-facing "half a cycle, one
# cycle, two cycles" for any frequency.
YEAR_CUTS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)

# Recency thresholds, in periods of the series' own frequency.
LAG_CUTS = (1, 2, 3, 6, 12)

# Forecast horizon by frequency, following the M4 competition's conventions. Length requirements
# below are multiples of this rather than a flat observation count, because neither of the two
# obvious rules works on its own: measuring in years makes one annual point a whole "year" of
# data, and measuring in points makes the calendar requirement scale inversely with frequency —
# 60 observations is five years of monthly data but sixty years of annual. A horizon is the unit
# that means the same thing at every frequency: a backtest needs enough history to hold one out
# and still have something left to fit on.
HORIZON: dict[str, int | None] = {
    "H": 48, "D": 14, "B": 14, "W": 13, "BW": 8, "M": 18,
    "Q": 8, "S": 4, "A": 6, "A3": 4, "P": 4, "I": None, "OTHER": None,
}


def series_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else data_dir() / "series"


def scan(
    root: Path | str | None = None,
    sources: list[str] | None = None,
    frequencies: list[str] | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """The series layer as one lazy frame, with only the columns asked for.

    Restricting ``columns`` is what keeps a full-population pass cheap: the list columns are
    never read unless they are named.
    """
    lf = pl.scan_parquet(str(series_root(root) / "**" / "*.parquet"), hive_partitioning=True)
    if columns is not None:
        lf = lf.select(columns)
    if sources:
        lf = lf.filter(pl.col("source").is_in(sources))
    if frequencies:
        lf = lf.filter(pl.col("frequency").is_in(frequencies))
    return lf


def _with_derived(lf: pl.LazyFrame, asof: dt.date) -> pl.LazyFrame:
    """Add the quantities every report below is built from."""
    ppy = pl.col("frequency").replace_strict(PERIODS_PER_YEAR, default=None, return_dtype=pl.Float64)
    return lf.with_columns(
        ppy.alias("per_year"),
        pl.col("frequency").replace_strict(HORIZON, default=None, return_dtype=pl.Int32).alias("horizon"),
        # years of actual data, the length that decides whether a seasonal model can be fitted
        (pl.col("n_obs") / ppy).alias("years"),
        # calendar span, which differs from `years` exactly when the series has gaps
        ((pl.col("end_date") - pl.col("start_date")).dt.total_days() / 365.25).alias("span_years"),
        # how stale, in the series' own periods, so "2" means two months for M and two quarters for Q
        (((pl.lit(asof) - pl.col("end_date")).dt.total_days() / 365.25) * ppy).alias("lag_periods"),
        pl.when(pl.col("n_points") > 0)
        .then(1.0 - pl.col("n_obs") / pl.col("n_points"))
        .otherwise(None)
        .alias("missing_share"),
        pl.when(pl.col("n_obs") > 0)
        .then(pl.col("n_flagged") / pl.col("n_obs"))
        .otherwise(None)
        .alias("flagged_share"),
    )


def overview(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
             until: dt.date | None = None) -> pl.DataFrame:
    """One row per source and frequency: size, length, recency, completeness.

    Exact over the whole population; reads no list column.
    """
    asof = asof or dt.date.today()
    until = until or dt.date(2025, 12, 1)
    lf = _with_derived(scan(root, sources, frequencies, SCALAR_COLUMNS), asof)
    df = (
        lf.group_by("source", "frequency").agg(_overview_aggs(until))
        .sort("source", "frequency")
        .collect(engine="streaming")
    )
    return df.join(obs_quantiles(root, sources, frequencies), on=["source", "frequency"], how="left")


def _overview_aggs(until: dt.date) -> list[pl.Expr]:
    """Defined once, so `overview()` and the combined pass in `full_report` cannot disagree."""
    return [
        pl.len().alias("n_series"),
        pl.col("n_obs").sum().alias("n_obs"),
        pl.col("n_obs").mean().alias("obs_mean"),
        pl.col("start_date").min().alias("earliest"),
        pl.col("end_date").max().alias("latest"),
        (pl.col("end_date") >= until).mean().alias(f"reaches_{until:%Y_%m}"),
        pl.col("missing_share").mean().alias("missing_share_mean"),
        (pl.col("missing_share") == 0).mean().alias("share_complete"),
        pl.col("flagged_share").mean().alias("flagged_share_mean"),
        *[(pl.col("years") >= c).mean().alias(f"ge_{c:g}y") for c in YEAR_CUTS],
        *[(pl.col("lag_periods") <= c).mean().alias(f"fresh_le_{c}p") for c in LAG_CUTS],
    ]


def _overview_names(until: dt.date) -> list[str]:
    return [e.meta.output_name() for e in _overview_aggs(until)]


def _anomaly_aggs(asof: dt.date) -> list[pl.Expr]:
    return [
        (pl.col("end_date") > asof).sum().alias("ends_in_future"),
        (pl.col("start_date") < dt.date(1800, 1, 1)).sum().alias("starts_before_1800"),
        (pl.col("n_obs") <= 1).sum().alias("single_point"),
        (pl.col("n_obs") == 0).sum().alias("empty"),
        (pl.col("end_date") < pl.col("start_date")).sum().alias("end_before_start"),
    ]


def _anomaly_names() -> list[str]:
    return [e.meta.output_name() for e in _anomaly_aggs(dt.date.today())]


def obs_quantiles(root=None, sources=None, frequencies=None,
                  qs: tuple[float, ...] = (0.25, 0.5, 0.75)) -> pl.DataFrame:
    """Exact quantiles of series length, without materialising the column.

    ``median()`` over a group of 850 million rows asks polars for the whole column at once and
    fails on any ordinary machine (a 15 GB allocation, in the run that prompted this). Counting
    each distinct ``n_obs`` instead is a streaming group-by whose result has a few thousand rows,
    and the quantiles fall straight out of the cumulative counts — exact, not approximate.
    """
    counts = (
        scan(root, sources, frequencies, ["source", "frequency", "n_obs"])
        .group_by("source", "frequency", "n_obs")
        .agg(pl.len().alias("k"))
        .collect(engine="streaming")
        .sort("source", "frequency", "n_obs")
    )
    rows = []
    for (source, freq), g in counts.group_by(["source", "frequency"], maintain_order=True):
        cum = g["k"].cum_sum()
        total = cum[-1]
        row = {"source": source, "frequency": freq}
        for q in qs:
            target = q * total
            idx = int((cum < target).sum())  # first position whose cumulative count reaches it
            row[f"obs_p{int(q * 100)}"] = int(g["n_obs"][min(idx, g.height - 1)])
        rows.append(row)
    out = pl.DataFrame(rows)
    ppy = pl.col("frequency").replace_strict(PERIODS_PER_YEAR, default=None, return_dtype=pl.Float64)
    return out.with_columns((pl.col("obs_p50") / ppy).alias("years_median"))


def length_table(root=None, sources=None, frequencies=None) -> pl.DataFrame:
    """Counts, not shares, in the cycle buckets — the "how many have one, two, three cycles" view."""
    lf = _with_derived(scan(root, sources, frequencies, SCALAR_COLUMNS), dt.date.today())
    aggs = [pl.len().alias("n_series")]
    aggs += [(pl.col("years") >= c).sum().alias(f"ge_{c:g}_cycles") for c in YEAR_CUTS]
    return lf.group_by("source", "frequency").agg(aggs).sort("source", "frequency").collect(engine="streaming")


# -- task suitability ---------------------------------------------------------------------------
#
# Each profile is a predicate over the derived columns. They are deliberately simple and stated
# in one place so they can be argued with and changed; nothing downstream depends on them.

def profiles(asof: dt.date, until: dt.date) -> dict[str, pl.Expr]:
    """What each slice is usable for, judged **series by series**.

    These are *univariate-local* requirements: they ask whether one series, on its own, carries
    enough history to identify a model fitted to it alone. That is the right question for local
    methods (ARIMA, ETS, one model per series) and the wrong one for a global model cross-learning
    across the corpus, where a short series is not unidentifiable — it borrows its parameters from
    every other series. For that case see ``aligned_table``, whose length bar is a single year.

    Length requirements are multiples of the forecast horizon for that frequency (see
    ``HORIZON``), which is the only unit that means the same thing everywhere. Measuring in years
    fails at annual frequency, where one year is one point, and a flat observation count fails in
    the other direction: sixty points is five years of monthly data and sixty years of annual, so
    it quietly demands six decades of history from exactly the series least likely to have it.

    Seasonality is the exception and keeps its cycle rule, because that is what the word means.

    In observations, the resulting thresholds are:

    ========  =======  =========  ========  ==========
    freq      horizon  trainable  backtest  nowcasting
    ========  =======  =========  ========  ==========
    A               6         12        18          24
    Q               8         16        24          32
    M              18         36        54          72
    W              13         26        39          52
    D              14         28        42          56
    ========  =======  =========  ========  ==========
    """
    mostly_there = pl.col("missing_share") <= 0.2
    h = pl.col("horizon")
    return {
        # two horizons: enough to fit something and still be asked for one forecast
        "trainable": (pl.col("n_obs") >= 2 * h) & (pl.col("missing_share") <= 0.5),
        # still being published: a genuine real-time exercise needs the series to be live
        "real_time": (pl.col("lag_periods") <= 2) & (pl.col("n_obs") >= 3 * h) & mostly_there,
        "nowcasting": (pl.col("lag_periods") <= 3) & (pl.col("n_obs") >= 4 * h) & mostly_there,
        # hold out one horizon and keep two to fit on
        "backtest": (pl.col("n_obs") >= 3 * h) & mostly_there,
        # three full cycles, which is a statement about periods per year, not about points
        "seasonal": (pl.col("years") >= 3) & (pl.col("per_year") >= 4) & mostly_there,
        # gaps are the point here, so a complete series is no use
        "imputation": (pl.col("missing_share") > 0.01) & (pl.col("missing_share") <= 0.5) & (pl.col("n_obs") >= 2 * h),
        # genuinely about calendar span rather than point count
        "long_history": pl.col("span_years") >= 20,
        "reaches_cutoff": pl.col("end_date") >= until,
    }


def task_table(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
               until: dt.date | None = None) -> pl.DataFrame:
    """How many series each source and frequency contributes to each kind of exercise."""
    asof = asof or dt.date.today()
    until = until or dt.date(2025, 12, 1)
    lf = _with_derived(scan(root, sources, frequencies, SCALAR_COLUMNS), asof)
    aggs = [pl.len().alias("n_series")] + [e.sum().alias(k) for k, e in profiles(asof, until).items()]
    return lf.group_by("source", "frequency").agg(aggs).sort("source", "frequency").collect(engine="streaming")


# -- the deep pass ------------------------------------------------------------------------------

def _sample_files(root: Path, sources, frequencies, per_group: int, seed: int) -> list[tuple[str, str, Path]]:
    """Randomly chosen parquet files per (source, frequency), enough to reach the sample size.

    Sampling files rather than rows is what keeps this quick: a row-level sample would still have
    to decode every list column it skips over. One file is one dataset, so the sample is drawn
    over datasets and then over the series inside them.
    """
    import random

    rng = random.Random(seed)
    out = []
    for src_dir in sorted(root.iterdir()):
        if not src_dir.is_dir() or (sources and src_dir.name not in sources):
            continue
        for freq_dir in sorted(src_dir.iterdir()):
            if not freq_dir.is_dir():
                continue
            freq = freq_dir.name.split("=", 1)[-1]
            if frequencies and freq not in frequencies:
                continue
            files = sorted(freq_dir.glob("*.parquet"))
            rng.shuffle(files)
            out.append((src_dir.name, freq, files))
    return out


def deep_table(root=None, sources=None, frequencies=None, sample: int = 200_000, seed: int = 0,
               max_files: int = 200) -> pl.DataFrame:
    """Value-level statistics on a sample: distinctness, constants, and where the gaps sit.

    ``unique_share`` is distinct values over observations. Low values mean a series that barely
    moves — a constant, a step function, or something rounded until its variation is gone — which
    is exactly what inflates a corpus without teaching a model anything.

    Gap position matters for imputation: a hole between two observations is a problem to solve,
    whereas missing points at the start or end just make the series shorter.

    **How the sample is drawn, and what that costs.** One parquet file is one dataset. Taking
    whole files until the quota filled meant covering only a handful of datasets — in one run,
    429,204 Eurostat monthly series drawn from perhaps two of them, whose peculiarities then stood
    in for the entire frequency. Instead this reads the first ``sample // max_files`` rows of each
    of up to ``max_files`` randomly chosen datasets, which spreads the draw across the catalogue
    at the same I/O cost. Within a dataset the rows are its first series in key order, which is
    arbitrary with respect to how the values behave but is *not* a random sample; treat every
    number here as indicative, not exact.

    Series with fewer than two observations are excluded from the distinctness statistics. With
    one point a series has one distinct value, which would otherwise be reported as both "100%
    unique" and "100% constant" — as it was, for the 853 million single-point Eurostat annual
    series.
    """
    root = series_root(root)
    per_file = max(50, sample // max(max_files, 1))
    rows = []
    for source, freq, files in _sample_files(root, sources, frequencies, sample, seed):
        got, frames, used = 0, [], 0
        for f in files[:max_files]:
            if got >= sample:
                break
            # A leading or trailing null just means a shorter series; a null between two
            # observations is a hole someone has to fill, and only the second kind makes a series
            # interesting for imputation. `cum_min` over the null mask stays 1 through the opening
            # run of nulls and 0 forever after, so its sum is the length of that run. (`arg_max`
            # on the non-null mask looks equivalent and is not: on an all-true list it returns the
            # last index, not the first.)
            nulls = pl.col("values").list.eval(pl.element().is_null().cast(pl.Int8))
            lead = pl.col("values").list.eval(pl.element().is_null().cast(pl.Int8).cum_min()).list.sum()
            trail = pl.col("values").list.eval(pl.element().is_null().cast(pl.Int8).reverse().cum_min()).list.sum()
            df = (
                pl.scan_parquet(f)
                .select("n_obs", "n_points", "values")
                .head(per_file)
                .with_columns(
                    pl.col("values").list.n_unique().alias("n_unique"),
                    nulls.list.sum().alias("n_null"),
                    lead.alias("n_lead"),
                    trail.alias("n_trail"),
                )
                .with_columns(
                    # an all-null series would count its single run at both ends; clamp it
                    pl.max_horizontal(pl.lit(0), pl.col("n_null") - pl.col("n_lead") - pl.col("n_trail")).alias("n_interior"),
                )
                .drop("values")
                .collect(engine="streaming")
            )
            if df.height:
                frames.append(df)
                got += df.height
                used += 1
        if not frames:
            continue
        d = pl.concat(frames).with_columns(
            # n_unique counts null as a value when the list holds one, so take it back out
            (pl.col("n_unique") - (pl.col("n_null") > 0).cast(pl.Int32)).alias("n_distinct"),
        )
        multi = d.filter(pl.col("n_obs") >= 2)
        multi = multi.with_columns((pl.col("n_distinct") / pl.col("n_obs")).alias("unique_share"))
        rows.append(
            {
                "source": source,
                "frequency": freq,
                "sampled": d.height,
                "datasets": used,
                "share_one_point": float((d["n_obs"] < 2).mean()),
                # everything below is over the series with at least two observations
                "n_multi": multi.height,
                "unique_share_mean": multi["unique_share"].mean(),
                "unique_share_median": multi["unique_share"].median(),
                "share_constant": float((multi["n_distinct"] <= 1).mean()) if multi.height else None,
                "share_near_constant": float((multi["unique_share"] < 0.05).mean()) if multi.height else None,
                "share_any_gap": float((d["n_null"] > 0).mean()),
                "share_interior_gap": float((d["n_interior"] > 0).mean()),
            }
        )
    return pl.DataFrame(rows).sort("source", "frequency") if rows else pl.DataFrame()


# -- putting it together ------------------------------------------------------------------------

def full_report(root=None, sources=None, frequencies=None, asof=None, until=None,
                deep: bool = True, sample: int = 200_000, seed: int = 0,
                datasets: bool = False, origin: dt.date | None = None,
                folds: int = 3, history_years: float = 1.0) -> dict[str, pl.DataFrame]:
    asof = asof or dt.date.today()
    until = until or dt.date(2025, 12, 1)
    # The four scalar tables group by the same keys over the same rows, so ask for all of their
    # aggregations in one pass and slice the result. Run separately they were four full scans of
    # a billion rows for one scan's worth of work.
    lf = _with_derived(scan(root, sources, frequencies, SCALAR_COLUMNS), asof)
    aggs = (
        _overview_aggs(until)
        + [(pl.col("years") >= c).sum().alias(f"ge_{c:g}_cycles") for c in YEAR_CUTS]
        + [e.sum().alias(f"task_{k}") for k, e in profiles(asof, until).items()]
        + _anomaly_aggs(asof)
    )
    g = (
        lf.group_by("source", "frequency").agg(aggs)
        .sort("source", "frequency")
        .collect(engine="streaming")
    )
    keys = ["source", "frequency"]
    ov_names = set(_overview_names(until))
    an_names = set(_anomaly_names())
    out = {
        "overview": g.select([*keys, *[c for c in g.columns if c in ov_names]])
                     .join(obs_quantiles(root, sources, frequencies), on=keys, how="left"),
        "length": g.select([*keys, "n_series", *[c for c in g.columns if c.endswith("_cycles")]]),
        "tasks": g.select([*keys, "n_series",
                           *[pl.col(c).alias(c[len("task_"):]) for c in g.columns if c.startswith("task_")]]),
        "anomalies": g.select([*keys, "n_series", *[c for c in g.columns if c in an_names]]),
        "aligned": aligned_table(root, sources, frequencies, origin or _default_origin(),
                                 history_years, folds),
    }
    if datasets:
        out["datasets"] = dataset_table(root, sources, frequencies, asof, until)
    if deep:
        out["deep"] = deep_table(root, sources, frequencies, sample, seed)
    return out


def summarise(tables: dict[str, pl.DataFrame], asof: dt.date, until: dt.date) -> str:
    """A short written reading of the tables, for the top of a report or a log."""
    ov, tk = tables["overview"], tables["tasks"]
    n = int(ov["n_series"].sum())
    obs = int(ov["n_obs"].sum())
    lines = [
        f"# Series-layer quality report",
        "",
        f"As of {asof:%Y-%m-%d}; recency cutoff {until:%Y-%m-%d}.",
        "",
        f"- **{n:,} series**, {obs:,} observations, {obs / max(n, 1):.1f} per series on average.",
    ]
    for name in ("trainable", "backtest", "real_time", "imputation"):
        if name in tk.columns:
            k = int(tk[name].sum())
            lines.append(f"- **{name}**: {k:,} series ({k / max(n, 1):.1%}).")
    # by observations, not by years: at annual frequency a two-point series is "two years" long
    short = ov.filter(pl.col("obs_p50") < 12)
    if short.height:
        worst = short.sort("n_series", descending=True).row(0, named=True)
        lines += [
            "",
            f"The largest short-series pool is {worst['source']} {worst['frequency']}: "
            f"{int(worst['n_series']):,} series with a median of {worst['obs_p50']:.0f} observations. "
            "Cross-products of SDMX dimensions generate very many very short series; filter on length "
            "before treating a series count as a measure of how much data there is.",
        ]
    return "\n".join(lines)


# -- the aligned subset -------------------------------------------------------------------------
#
# A different question from the profiles above. Those ask whether a series can be modelled on its
# own. This one asks whether a set of series can be modelled *together*, at one forecast time,
# which is what a cross-learned global model and a shared evaluation both need.


def _years_before(d: dt.date, years: float) -> dt.date:
    return d - dt.timedelta(days=round(years * 365.25))


def aligned_expr(origin: dt.date, history_years: float = 1.0, folds: int = 0,
                 max_missing: float = 0.2, tolerance: float = 1.0) -> pl.Expr:
    """Series that all cover the same recent window, whatever their frequency.

    ``origin`` is the forecast time: the last moment the model is allowed to see. A series
    qualifies when it still has data at the origin and already had data ``history_years`` before
    the earliest origin the folds reach back to, so the *same cohort* survives every fold of a
    rolling-origin cross-validation. Forecasting 2026 from the end of 2025, and cross-validating
    on 2025 from the end of 2024, is ``origin=2025-12-31, folds=1``.

    One year of history is deliberately the whole length bar. Under cross-learning a short series
    is not unidentifiable, because its parameters come from the corpus rather than from itself;
    a year is simply what it takes to have seen one full seasonal cycle at any frequency.

    "Still has data at the origin" is measured in the series' own periods, not in days. An annual
    observation for 2025 is stamped ``2025-01-01``, so asking for ``end_date >= 2025-12-31`` would
    demand a 2026 point and throw away almost every annual series; ``tolerance`` periods of lag is
    the frequency-independent way to say the same thing.

    What this cannot see: the endpoints tell us the series brackets the window, and
    ``max_missing`` bounds the holes overall, but neither proves the window itself is unbroken.
    Confirming that needs the dates list.
    """
    earliest = _years_before(origin, folds)
    need_from = _years_before(earliest, history_years)
    return (
        (pl.col("lag_periods") <= tolerance)
        & (pl.col("start_date") <= need_from)
        # having the span is not having the points: require the observations too
        & (pl.col("n_obs") >= (pl.col("per_year") * (history_years + folds)).ceil())
        & (pl.col("missing_share") <= max_missing)
    )


def aligned_table(root=None, sources=None, frequencies=None, origin: dt.date | None = None,
                  history_years: float = 1.0, folds: int = 3, max_missing: float = 0.2,
                  tolerance: float = 1.0) -> pl.DataFrame:
    """How many series are alignable at each origin, per source and frequency.

    One column per fold, so the cohort's decay as the origin rolls back is visible: the last
    column is the set usable for the deepest cross-validation.
    """
    origin = origin or _default_origin()
    lf = _with_derived(scan(root, sources, frequencies, SCALAR_COLUMNS), origin)
    aggs = [pl.len().alias("n_series")]
    for k in range(folds + 1):
        aggs.append(aligned_expr(origin, history_years, k, max_missing, tolerance).sum().alias(f"folds_{k}"))
    return (
        lf.group_by("source", "frequency").agg(aggs)
        .sort("source", "frequency")
        .collect(engine="streaming")
    )


def aligned_series(root=None, sources=None, frequencies=None, origin: dt.date | None = None,
                   history_years: float = 1.0, folds: int = 0, max_missing: float = 0.2,
                   tolerance: float = 1.0, columns: list[str] | None = None) -> pl.LazyFrame:
    """The cohort itself, lazily, for feeding into a training set or an export.

    ``columns`` defaults to the identifying ones; ask for ``dates``/``values`` to get the data.
    """
    origin = origin or _default_origin()
    cols = SCALAR_COLUMNS + ["series_uid"] + [c for c in (columns or []) if c not in SCALAR_COLUMNS]
    lf = _with_derived(scan(root, sources, frequencies, list(dict.fromkeys(cols))), origin)
    return lf.filter(aligned_expr(origin, history_years, folds, max_missing, tolerance))


def _default_origin(today: dt.date | None = None) -> dt.date:
    """The most recent 31 December, the natural forecast time for an annual exercise."""
    today = today or dt.date.today()
    return dt.date(today.year - 1, 12, 31)


def group_expr() -> pl.Expr:
    """The theme a dataset sits under, which is the level worth ranking.

    Each source encodes its hierarchy in the dataset id, differently:

    - Eurostat  ``aact_ali01`` -> ``aact``, the theme prefix of its code (agricultural accounts,
      national accounts ``nama``, demography ``demo``, and so on)
    - OECD      ``OECD.SDD.NAD__DSD_NAMAIN10_DF_TABLE1__1.0`` -> ``OECD.SDD.NAD``, the owning
      directorate
    - FRED      the dataset id is already the release, so the group is the release itself
    """
    return (
        pl.when(pl.col("source") == "eurostat")
        .then(pl.col("dataset_id").str.split("_").list.first())
        .when(pl.col("source") == "oecd")
        .then(pl.col("dataset_id").str.split("__").list.first())
        .otherwise(pl.col("dataset_title"))
        .alias("group")
    )


def health_table(root=None, sources=None, frequencies=None, by: str = "group",
                 min_years: float = 5.0, max_missing: float = 0.05,
                 asof: dt.date | None = None) -> pl.DataFrame:
    """Which parts of the catalogue are actually in good shape.

    "Healthy" here means a series long enough to model and close to complete: at least
    ``min_years`` of data and at most ``max_missing`` of its stored periods empty. Neither
    condition mentions recency, so this ranks the corpus by whether the data is *sound*, which is
    a separate question from whether it is *current*.

    ``by`` is ``"group"`` (the theme, see ``group_expr``) or ``"dataset"``. Rank on
    ``share_healthy`` to find pockets worth trusting, and on ``n_healthy`` to find where the
    volume is; the two orders are very different, because the largest groups are largest precisely
    because they multiply out thin dimension cross-products.
    """
    asof = asof or dt.date.today()
    cols = SCALAR_COLUMNS + ["dataset_id", "dataset_title"]
    lf = _with_derived(scan(root, sources, frequencies, cols), asof).with_columns(group_expr())
    keys = ["source", "group"] if by == "group" else ["source", "group", "dataset_id"]
    # `years` is null for frequencies with no periods-per-year (I, OTHER), which would make the
    # whole aggregate null rather than zero. A series we cannot place on a calendar is not healthy.
    healthy = ((pl.col("years") >= min_years) & (pl.col("missing_share") <= max_missing)).fill_null(False)
    aggs = [
        pl.col("dataset_id").n_unique().alias("n_datasets"),
        pl.len().alias("n_series"),
        healthy.sum().alias("n_healthy"),
        healthy.mean().alias("share_healthy"),
        pl.col("years").mean().alias("years_mean"),
        pl.col("missing_share").mean().alias("missing_mean"),
        pl.col("end_date").max().alias("latest"),
    ]
    if by == "dataset":
        aggs.insert(0, pl.col("dataset_title").first().alias("title"))
    return (
        lf.group_by(keys).agg(aggs)
        .sort("n_healthy", descending=True)
        .collect(engine="streaming")
    )


def dataset_table(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
                  until: dt.date | None = None, with_titles: bool = True) -> pl.DataFrame:
    """The same questions one level down: per dataset rather than per source.

    This is the table to sort when deciding what to actually train on. A source-level number
    hides enormous variation — one Eurostat dataset can contribute a hundred million
    two-observation series while the one next to it contributes ten thousand usable ones.

    ``with_titles`` reads ``dataset_title`` as well, which is a string column and therefore the
    most expensive part of this pass; turn it off if you only need the ids.
    """
    asof = asof or dt.date.today()
    until = until or dt.date(2025, 12, 1)
    cols = SCALAR_COLUMNS + ["dataset_id"] + (["dataset_title"] if with_titles else [])
    lf = _with_derived(scan(root, sources, frequencies, cols), asof)
    prof = profiles(asof, until)
    aggs = [
        pl.len().alias("n_series"),
        pl.col("n_obs").sum().alias("n_obs"),
        pl.col("n_obs").mean().round(1).alias("obs_mean"),
        pl.col("end_date").max().alias("latest"),
        (pl.col("years") >= 1).sum().alias("ge_1y"),
        (pl.col("years") >= 3).sum().alias("ge_3y"),
        prof["trainable"].sum().alias("trainable"),
        prof["backtest"].sum().alias("backtest"),
        prof["real_time"].sum().alias("real_time"),
    ]
    if with_titles:
        aggs.insert(0, pl.col("dataset_title").first().alias("title"))
    return (
        lf.group_by("source", "dataset_id", "frequency").agg(aggs)
        .sort("n_series", descending=True)
        .collect(engine="streaming")
    )


def anomalies(root=None, sources=None, frequencies=None, asof: dt.date | None = None) -> pl.DataFrame:
    """Series whose dates deserve a second look before anyone trains on them.

    None of these is necessarily wrong. Dates past today are usually projections (FRED carries
    CBO and OMB forecast series that legitimately run to 2036), and a start in the 1600s can be a
    genuine historical reconstruction. They are listed because a silently mis-parsed period looks
    exactly the same, and because a forecast series mixed into a training set is a leak.
    """
    asof = asof or dt.date.today()
    lf = scan(root, sources, frequencies, SCALAR_COLUMNS)
    return (
        lf.group_by("source", "frequency")
        .agg(pl.len().alias("n_series"), *_anomaly_aggs(asof))
        .sort("source", "frequency")
        .collect(engine="streaming")
    )


def _pct(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Shares as readable percentages, so a terminal table stays narrow."""
    have = [c for c in cols if c in df.columns]
    return df.with_columns([(pl.col(c) * 100).round(1).alias(c) for c in have])


def render(tables: dict[str, pl.DataFrame], until: dt.date) -> str:
    """The tables as a few narrow views. The full ones go to disk; these are for reading."""
    reaches = f"reaches_{until:%Y_%m}"
    out = []
    ov = tables["overview"]
    size = ov.select("source", "frequency", "n_series", "n_obs",
                     pl.col("obs_p25").alias("p25"), pl.col("obs_p50").alias("med"), pl.col("obs_p75").alias("p75"),
                     pl.col("years_median").round(1).alias("yrs_med"), "earliest", "latest")
    out.append(("size and span", size))

    cuts = [c for c in ov.columns if c.startswith("ge_")]
    out.append(("length, % of series with at least N years of data", _pct(ov.select("source", "frequency", *cuts), cuts)))

    fresh = [c for c in ov.columns if c.startswith("fresh_")]
    out.append((f"recency, % within N periods of today (and % reaching {until:%Y-%m})",
                _pct(ov.select("source", "frequency", *fresh, reaches), fresh + [reaches])))

    comp = ["missing_share_mean", "share_complete", "flagged_share_mean"]
    out.append(("completeness, %", _pct(ov.select("source", "frequency", *comp), comp)))

    tk = tables["tasks"]
    out.append(("series usable for each exercise", tk))
    if "anomalies" in tables:
        out.append(("dates worth a second look (counts)", tables["anomalies"]))
    if "aligned" in tables:
        al = tables["aligned"]
        out.append(("the aligned cohort: series covering the same window at each rolling origin", al))
    if "datasets" in tables:
        ds = tables["datasets"]
        top = (ds.head(15).with_columns(pl.col("title").str.slice(0, 34).alias("title"))
                 .select("source", "dataset_id", "frequency", "title", "n_series", "obs_mean",
                         "ge_1y", "backtest", "real_time", "latest"))
        out.append((f"the 15 largest datasets by series count (all {ds.height:,} are in datasets.parquet)", top))

    if "deep" in tables and tables["deep"].height:
        dp = tables["deep"]
        share = [c for c in dp.columns if c.startswith("share_") or c.startswith("unique_")]
        out.append(("value-level, sampled from `datasets` datasets each (%); distinctness is over "
                    "the `n_multi` series with 2+ observations", _pct(dp, share)))

    parts = []
    with pl.Config(tbl_rows=60, tbl_cols=30, ascii_tables=True, tbl_hide_dataframe_shape=True,
                   tbl_formatting="ASCII_MARKDOWN", set_fmt_str_lengths=30):
        for title, df in out:
            parts.append(f"\n== {title} ==\n{df}")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="terrastat quality", description=__doc__.splitlines()[0])
    p.add_argument("--root", default=None, help="series directory (default: the configured data dir)")
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--freq", nargs="*", default=None, help="canonical frequencies, e.g. M Q A")
    p.add_argument("--asof", default=None, help="reference date for recency (default: today)")
    p.add_argument("--until", default="2025-12-01", help="cutoff for 'reaches' (default: 2025-12-01)")
    p.add_argument("--no-deep", action="store_true", help="skip the sampled value-level pass")
    p.add_argument("--sample", type=int, default=200_000, help="series per source/frequency in the deep pass")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="directory to write the tables and summary into")
    p.add_argument("--datasets", action="store_true", help="also break the numbers down per dataset, not just per source (reads dataset titles, so it is the slowest pass)")
    p.add_argument("--origin", default=None, help="forecast time for the aligned cohort (default: the most recent 31 December)")
    p.add_argument("--folds", type=int, default=3, help="rolling-origin folds to report the cohort for")
    p.add_argument("--history-years", type=float, default=1.0, help="history required before the earliest origin")
    args = p.parse_args(argv)

    asof = dt.date.fromisoformat(args.asof) if args.asof else dt.date.today()
    until = dt.date.fromisoformat(args.until)
    tables = full_report(args.root, args.sources, args.freq, asof, until,
                         deep=not args.no_deep, sample=args.sample, seed=args.seed,
                         datasets=args.datasets,
                         origin=dt.date.fromisoformat(args.origin) if args.origin else None,
                         folds=args.folds, history_years=args.history_years)
    text = summarise(tables, asof, until)
    print(text)
    print(render(tables, until))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.write_parquet(out / f"{name}.parquet")
            df.write_csv(out / f"{name}.csv")
        (out / "summary.md").write_text(text, encoding="utf-8")
        log.info("wrote %d tables to %s", len(tables), out)
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
