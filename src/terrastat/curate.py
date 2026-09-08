"""Data-driven cutoff selection and window cleaning — the FRED curation procedure, opt-in.

This reproduces the cleaning described in *Neuro-Symbolic Models with Multimodal Data*
(part 3, "Step 1 — Cutoff date selection and context filtering", and Steps 2–3), whose
reference implementation is ``fred_forecast/data/clean_releases.py``. **Nothing here runs by
default.** The corpus terrastat builds is deliberately raw: series keep their own start and end,
their own gaps, and their own missing values, because which of those matter depends on the
exercise. This module is for when you want that particular curated panel.

What the procedure does, and why:

1. **Bounds.** Per series, the first and last date carrying a non-missing value. Leading and
   trailing gaps are ignored rather than imputed — filling at the extremes would fabricate data
   exactly where a model has no evidence.

2. **Cutoff.** Instead of picking a cutoff date by hand, search every candidate date ``d`` on the
   frequency's own grid and keep the one maximising

       ``C(d) = #{i : first_valid_i <= d - context  and  last_valid_i >= d}``

   the number of series that cover a full context window ending at ``d``. The data chooses the
   most inclusive time frame. ``cutoff_curve()`` returns the whole of ``C(d)``, which is the
   figure in the thesis.

3. **Window.** Every retained series is placed on the regular grid of ``[window_start, cutoff]``
   and everything after this point looks only inside that window.

4. **Missingness.** Drop a series if more than ``max_nan_ratio`` of the window is missing (0.25,
   following the imputation literature's finding that accuracy degrades past roughly 20%), and
   drop it if the *edge* of the window is missing: the first position has no past to copy from
   and the last would need extrapolation, at the very position the model is scored on.

5. **Imputation.** Fill what remains — strictly interior gaps — by forward fill (LOCF). It
   introduces no look-ahead, which is the whole reason to prefer it over anything cleverer.

**Two adaptations to terrastat's layout, both material.**

*Missingness has to be reconstructed.* The reference reads long-format release files where every
month is a row and an unpublished month is a row with ``NaN``. terrastat stores one row per
series with a ``dates``/``values`` pair holding only the points the source actually published, so
an unpublished month is simply absent. Reindexing onto the regular grid of the frequency is what
restores the reference's notion of missingness; without it every series would score 0% missing
and the NaN filters would be no-ops.

*Everything is in periods, not months.* The reference is monthly-only. Here ``context`` counts
periods of each series' own frequency, and the candidate grid follows that frequency, so a
cutoff is chosen separately per frequency group. Frequencies with no uniform calendar step
(``B``, ``H``, ``I``, ``OTHER``) are refused rather than approximated.

**Where the thesis and the reference code disagree**, this module follows the thesis and lets you
ask for the code's behaviour explicitly::

    curate(lf, **REFERENCE_CODE)

============  ==========================  =====================================
option        thesis (default)            reference code
============  ==========================  =====================================
``nan_scope`` ``"window"``                ``"valid_range"`` — measured over
                                          ``[first_valid_i, last_valid_i]``
``edges``     ``"both"``                  ``"cutoff"`` — only the last position
``trim``      ``"window"``                ``"first_valid"`` — keeps the history
                                          before ``window_start``
============  ==========================  =====================================

**Reproduction.** Run over the 224,015 monthly FRED series in this corpus with the reference
settings and ``context=48``, the procedure picks a cutoff of **December 2021** — the same date the
thesis reports — and the two filters remove almost exactly what the thesis says they removed:

=====================  ==============  ==========
step                   here            thesis
=====================  ==============  ==========
covering the window    201,037         198,116
after the NaN ratio    197,675 (-3,362) 194,786 (-3,330)
after the edge rule    197,331 (-344)  194,442 (-344)
=====================  ==============  ==========

The totals differ because this crawl is later and holds more series; the *deltas* agreeing to the
series is what says the procedure is the same one. One off-by-one is worth knowing: the criterion
is ``first_valid_i <= d - context``, so the closed window ``[d - context, d]`` holds ``context + 1``
periods (49 here). The thesis prose reports its window as the 48 months January 2018 – December
2021, one period shorter, while stating the same criterion. Pass ``context=47`` for a window of
exactly 48 periods.

One known flaw of the reference does not survive the port: it forward-fills within each Parquet
row-group, so a gap landing on a row-group boundary silently kept its ``NaN``. Here a whole
series is one row, so there is no boundary to fall across.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# The calendar step of one period, per canonical frequency. `B` (business daily) is absent on
# purpose: five-out-of-seven days is not a uniform interval, so a regular grid would either invent
# weekend observations or drop real ones. `H` is absent because the schema stores dates, not
# timestamps, so two hours of the same day are indistinguishable. `I` and `OTHER` have no step.
GRID: dict[str, str] = {
    "D": "1d", "W": "1w", "BW": "2w", "M": "1mo",
    "Q": "3mo", "S": "6mo", "A": "1y", "A3": "3y", "P": "5y",
}

# Multiples of a single unit, so `context` periods becomes one offset string.
_STEP: dict[str, tuple[int, str]] = {
    "D": (1, "d"), "W": (7, "d"), "BW": (14, "d"), "M": (1, "mo"),
    "Q": (3, "mo"), "S": (6, "mo"), "A": (12, "mo"), "A3": (36, "mo"), "P": (60, "mo"),
}

# The defaults of the reference implementation, from `configs/data.yaml`.
CONTEXT = 48
MAX_NAN_RATIO = 0.25

# Splat this to get the reference code's behaviour rather than the thesis text's.
REFERENCE_CODE = {"nan_scope": "valid_range", "edges": "cutoff", "trim": "first_valid"}


def _offset(frequency: str, periods: int) -> str:
    """`periods` steps of `frequency` as a polars offset string, e.g. (Q, 48) -> '144mo'."""
    if frequency not in _STEP:
        raise ValueError(
            f"frequency {frequency!r} has no uniform calendar step, so it has no regular grid to "
            f"reindex onto. Curatable frequencies: {', '.join(GRID)}."
        )
    n, unit = _STEP[frequency]
    return f"{n * periods}{unit}"


def _frame(data) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


# -- step 1: bounds and cutoff -------------------------------------------------------------------

def valid_bounds(data) -> pl.DataFrame:
    """First and last date carrying a non-missing value, per series.

    Not ``start_date``/``end_date``: those are the ends of the stored lists, which can be null
    values (a point present only to carry a status flag). Leading and trailing missing points are
    outside every window this module builds, which is the point — they are never imputed.
    """
    df = _frame(data).lazy().select("series_uid", "frequency", "dates", "values")
    return (
        df.explode("dates", "values", empty_as_null=True)
        .filter(pl.col("values").is_not_null())
        .group_by("series_uid", "frequency")
        .agg(
            pl.col("dates").min().alias("first_valid"),
            pl.col("dates").max().alias("last_valid"),
            pl.len().alias("n_valid"),
        )
        .collect()
    )


def cutoff_curve(bounds: pl.DataFrame, frequency: str, context: int = CONTEXT) -> pl.DataFrame:
    """``C(d)``: how many series cover a full context window ending at each candidate date.

    This is the curve the thesis plots to justify its cutoff. Candidates run from the earliest
    date at which any series could already have a full window, to the latest observation anywhere.

    Computed by counting starts and ends rather than testing every series against every date —
    the same numbers as the reference's double loop, but it finishes on a corpus this size.
    """
    b = bounds.filter(pl.col("frequency") == frequency)
    if b.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "n_series": pl.Int64})

    step = GRID[frequency] if frequency in GRID else _offset(frequency, 1)
    b = b.with_columns(
        pl.col("first_valid").dt.offset_by(_offset(frequency, context)).alias("_lo")
    )
    # The candidate range spans every series, so the curve has the same domain as the reference's
    # even where it is flat at zero.
    start, end = b["_lo"].min(), b["last_valid"].max()
    if start is None or end is None or start > end:
        return pl.DataFrame(schema={"date": pl.Date, "n_series": pl.Int64})
    # A series whose valid range is shorter than the context covers no candidate at all. Excluding
    # it is not an optimisation: leaving it in would have the sweep close its interval before
    # opening it, and the count would go negative in between -- which it did, on the real corpus.
    covering = b.filter(pl.col("_lo") <= pl.col("last_valid"))

    cand = pl.date_range(start, end, interval=step, eager=True)
    lo_s = np.sort(covering["_lo"].to_numpy())
    hi_s = np.sort(covering["last_valid"].to_numpy())
    d = cand.to_numpy()
    # started by d: first_valid + context <= d;  not yet ended at d: last_valid >= d
    n = np.searchsorted(lo_s, d, side="right") - np.searchsorted(hi_s, d, side="left")
    return pl.DataFrame({"date": cand, "n_series": n.astype(np.int64)})


def best_cutoff(bounds: pl.DataFrame, frequency: str, context: int = CONTEXT
                ) -> tuple[dt.date | None, dt.date | None, pl.DataFrame]:
    """``(cutoff, window_start, curve)`` — the date maximising ``C(d)``, earliest one on a tie.

    Earliest rather than latest deliberately: pandas' ``idxmax`` returns the first maximum, and a
    plateau in ``C(d)`` means those dates are equally inclusive, so the earlier one leaves more
    future observations available for whatever is done next.
    """
    curve = cutoff_curve(bounds, frequency, context)
    if curve.is_empty():
        return None, None, curve
    i = int(curve["n_series"].arg_max())
    cutoff = curve["date"][i]
    window_start = (
        pl.Series([cutoff]).dt.offset_by("-" + _offset(frequency, context))[0]
    )
    log.info("%s: cutoff %s, window %s -> %s, %d series cover it",
             frequency, cutoff, window_start, cutoff, curve["n_series"][i])
    return cutoff, window_start, curve


def select_series_with_context(bounds: pl.DataFrame, window_start: dt.date,
                               cutoff: dt.date) -> pl.DataFrame:
    """The bounds rows whose valid range covers the whole window."""
    return bounds.filter(
        (pl.col("first_valid") <= window_start) & (pl.col("last_valid") >= cutoff)
    )


# -- steps 2 and 3: window, filter, fill ---------------------------------------------------------

def _on_grid(data, bounds: pl.DataFrame, frequency: str, window_start: dt.date,
             cutoff: dt.date, trim: str) -> pl.DataFrame:
    """Long frame of (series_uid, date, value) on the regular grid, missing periods present as
    nulls. This is what makes terrastat's stored points comparable to the reference's rows."""
    keep = bounds.select("series_uid", "first_valid")
    start = window_start if trim == "window" else bounds["first_valid"].min()
    grid = pl.DataFrame({"date": pl.date_range(start, cutoff, interval=GRID[frequency], eager=True)})

    obs = (
        _frame(data).lazy()
        .join(keep.lazy(), on="series_uid", how="semi")
        .select("series_uid", "dates", "values")
        .explode("dates", "values", empty_as_null=True)
        .rename({"dates": "date", "values": "value"})
        .filter(pl.col("date").is_between(start, cutoff))
    )
    long = (
        keep.lazy().join(grid.lazy(), how="cross")
        .join(obs, on=["series_uid", "date"], how="left")
    )
    if trim == "first_valid":
        long = long.filter(pl.col("date") >= pl.col("first_valid"))
    return long.drop("first_valid").sort("series_uid", "date").collect()


def _n_periods(first: str, last: str, frequency: str) -> pl.Expr:
    """How many grid periods the closed interval [first, last] spans, inclusive."""
    n, unit = _STEP[frequency]
    if unit == "mo":
        diff = ((pl.col(last).dt.year() - pl.col(first).dt.year()) * 12
                + (pl.col(last).dt.month() - pl.col(first).dt.month()))
    else:
        diff = (pl.col(last) - pl.col(first)).dt.total_days()
    return diff // n + 1


def _nan_share(long: pl.DataFrame, bounds: pl.DataFrame, window_start: dt.date,
               cutoff: dt.date, scope: str, frequency: str) -> pl.DataFrame:
    """Missing share per series, over the window or over each series' own valid range.

    The valid-range variant is counted rather than materialised: a series has ``n_valid`` real
    values inside a range spanning a known number of grid periods, and everything else in that
    range is missing. That is the same quantity the reference computes row by row, and it does not
    depend on how far back the window happens to have been trimmed.
    """
    if scope == "valid_range":
        return bounds.select(
            "series_uid",
            (1 - pl.col("n_valid") / _n_periods("first_valid", "last_valid", frequency))
            .alias("nan_share"),
        )
    return (
        long.lazy()
        .filter(pl.col("date").is_between(window_start, cutoff))
        .group_by("series_uid")
        .agg(pl.col("value").is_null().mean().alias("nan_share"))
        .collect()
    )


@dataclass
class Curated:
    """The curated panel and everything needed to describe how it was reached."""

    frame: pl.DataFrame                       # one row per surviving series, windowed and filled
    windows: dict = field(default_factory=dict)   # frequency -> {window_start, cutoff, n_periods}
    curves: dict = field(default_factory=dict)    # frequency -> the full C(d) curve
    report: pl.DataFrame | None = None            # per frequency, how many series each step left
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        w = ", ".join(f"{f}: {v['window_start']}..{v['cutoff']}" for f, v in self.windows.items())
        return f"<Curated {self.frame.height} series | {w or 'no window'}>"


def curate(data, context: int | dict[str, int] = CONTEXT, max_nan_ratio: float = MAX_NAN_RATIO,
           frequencies: list[str] | None = None, nan_scope: str = "window",
           edges: str = "both", trim: str = "window", fill: str = "forward") -> Curated:
    """Run the whole procedure and return the curated panel.

    Never called anywhere else in terrastat — this is opt-in by design. ``data`` is anything with
    ``series_uid``, ``frequency``, ``dates`` and ``values``: a frame from ``dataset.load()``, from
    the series layer, or from a snapshot.

    A cutoff is chosen per frequency, since a monthly and an annual grid share no candidate dates.
    ``context`` is in periods of each frequency, or a dict to set it per frequency.

    ``fill="none"`` skips step 3 and leaves interior gaps as nulls, which is what you want if you
    intend to impute them yourself or to model missingness.
    """
    for name, value, allowed in (("nan_scope", nan_scope, {"window", "valid_range"}),
                                 ("edges", edges, {"both", "cutoff", "none"}),
                                 ("trim", trim, {"window", "first_valid"}),
                                 ("fill", fill, {"forward", "none"})):
        if value not in allowed:
            raise ValueError(f"{name}={value!r}; expected one of {sorted(allowed)}")

    df = _frame(data)
    bounds = valid_bounds(df)
    freqs = frequencies or sorted(set(bounds["frequency"].to_list()))
    unusable = [f for f in freqs if f not in GRID]
    if unusable:
        raise ValueError(
            f"no regular grid for frequency {', '.join(unusable)}. Pass frequencies=[...] to "
            f"curate the rest; curatable: {', '.join(GRID)}."
        )

    parts, rows, windows, curves = [], [], {}, {}
    for freq in freqs:
        ctx = context[freq] if isinstance(context, dict) else context
        b = bounds.filter(pl.col("frequency") == freq)
        cutoff, window_start, curve = best_cutoff(b, freq, ctx)
        curves[freq] = curve
        step = {"frequency": freq, "context": ctx, "n_series": b.height,
                "window_start": window_start, "cutoff": cutoff}
        if cutoff is None:
            rows.append({**step, "with_context": 0, "after_nan": 0, "after_edge": 0,
                         "n_periods": 0, "nulls_left": 0})
            continue

        chosen = select_series_with_context(b, window_start, cutoff)
        step["with_context"] = chosen.height
        if chosen.is_empty():
            rows.append({**step, "after_nan": 0, "after_edge": 0, "n_periods": 0, "nulls_left": 0})
            continue

        sub = df.filter(pl.col("frequency") == freq)
        long = _on_grid(sub, chosen, freq, window_start, cutoff, trim)

        share = _nan_share(long, chosen, window_start, cutoff, nan_scope, freq)
        keep = share.filter(pl.col("nan_share") <= max_nan_ratio).select("series_uid")
        long = long.join(keep, on="series_uid", how="semi")
        step["after_nan"] = keep.height

        if edges != "none" and not long.is_empty():
            at = [cutoff] if edges == "cutoff" else [window_start, cutoff]
            bad = (long.filter(pl.col("date").is_in(at) & pl.col("value").is_null())
                       .select("series_uid").unique())
            long = long.join(bad, on="series_uid", how="anti")
        step["after_edge"] = long.get_column("series_uid").n_unique() if long.height else 0

        if fill == "forward":
            long = long.with_columns(pl.col("value").forward_fill().over("series_uid"))

        packed = (
            long.group_by("series_uid")
            .agg(pl.col("date").alias("dates"), pl.col("value").alias("values"))
            .with_columns(
                pl.col("dates").list.len().alias("n_points"),
                pl.col("values").list.drop_nulls().list.len().alias("n_obs"),
                pl.col("dates").list.first().alias("start_date"),
                pl.col("dates").list.last().alias("end_date"),
            )
        )
        meta = df.drop("dates", "values", "n_points", "n_obs", "start_date", "end_date",
                       strict=False)
        parts.append(packed.join(meta, on="series_uid", how="left"))

        step["n_periods"] = len(pl.date_range(window_start, cutoff,
                                              interval=GRID[freq], eager=True))
        step["nulls_left"] = int(packed["n_points"].sum() - packed["n_obs"].sum())
        windows[freq] = {"window_start": window_start, "cutoff": cutoff,
                         "n_periods": step["n_periods"], "n_series": packed.height}
        rows.append(step)

    frame = pl.concat(parts, how="diagonal_relaxed") if parts else df.clear()
    return Curated(
        frame=frame, windows=windows, curves=curves,
        report=pl.DataFrame(rows) if rows else None,
        settings={"context": context, "max_nan_ratio": max_nan_ratio, "nan_scope": nan_scope,
                  "edges": edges, "trim": trim, "fill": fill},
    )


def check(result: Curated) -> pl.DataFrame:
    """The reference's two sanity checks, per frequency: does every series cover the window, and
    are any nulls left. Both columns should be zero."""
    rows = []
    for freq, w in result.windows.items():
        f = result.frame.filter(pl.col("frequency") == freq)
        rows.append({
            "frequency": freq,
            "n_series": f.height,
            "context_violations": int(
                f.select(((pl.col("start_date") > w["window_start"])
                          | (pl.col("end_date") < w["cutoff"])).sum()).item()
            ),
            "nulls_left": int((f["n_points"].sum() or 0) - (f["n_obs"].sum() or 0)),
        })
    return pl.DataFrame(rows)
