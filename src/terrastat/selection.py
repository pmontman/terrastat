"""Choosing which series to train on: recency, length, information — and how to trade them off.

Three separate jobs, with very different costs, kept separate for that reason.

**Filtering for recent series is free.** ``end_date`` is a fixed-width column, so a recency filter
reads none of the 28 GB of ``values`` and answers over the whole corpus in seconds. Use
``recent()``.

**Measuring information is not free.** How many distinct values a series takes, how often it
repeats, its Shannon entropy — all need the values themselves, at roughly 35,000 series a second.
Over 956 million that is seven hours. Over the survivors of a cheap filter it is minutes, which is
why ``score()`` takes a predicate and applies it before reading anything expensive.

Scores are written one file per input file, so an interrupted run resumes by skipping what it
already did, and ``load_scores()`` reads the tree back as one frame.

**Combining the three is a judgement call, not a computation.** ``add_ranks`` turns each dimension
into a percentile and takes the minimum across them: a series with ``rank_min = 0.8`` sits above
the 80th percentile on recency *and* length *and* information. Thresholding that column is exactly
the conjunctive filter "good on all three", which is the honest version of what people usually
approximate with weights they cannot justify. ``how="mean"`` is available when trading one against
another is genuinely acceptable.

A caveat worth carrying: normalised entropy is near 1 for any series whose values are all
distinct, which is most genuinely continuous economic data. Its discriminating power is at the
*low* end — constants, step functions, series rounded until their variation is gone. Do not read
a high value as "informative"; read a low one as "suspect".
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from pathlib import Path

import polars as pl
from tqdm import tqdm

from terrastat.quality import (
    HORIZON,
    PERIODS_PER_YEAR,
    SCALAR_COLUMNS,
    _default_origin,
    _with_derived,
    scan,
    series_root,
)
from terrastat.storage import data_dir

log = logging.getLogger(__name__)

_UNSET = object()   # lets `within=None` mean "rank globally" and an absent argument mean "use the default"

# The three axes, and the column that measures each. Recency is negated so that larger is better
# on every axis, which is what makes the percentile ranks comparable.
def information_score() -> pl.Expr:
    """One bounded number for "how much this series actually says", as a rank, not a filter.

    The minimum of three non-degeneracy measures, so a series is only as informative as its worst
    failing — the same conjunctive logic the axes themselves use, one level down:

    - ``entropy_norm`` is 0 for a constant and 1 when every value is distinct.
    - ``1 - repeat_share`` falls as the series stalls between observations.
    - ``1 - flat_run_share`` falls when one contiguous figure is carried forward, which neither of
      the other two catches: a series whose last three years repeat one number still has high
      entropy *and* high autocorrelation.
    - ``diff_variety`` is the share of distinct first differences, and it is the only one of the
      four that sees a linear ramp. A ramp has maximal entropy, no repeats, no runs and perfect
      autocorrelation — it looks flawless on everything else — but exactly one distinct
      difference, because it is interpolation rather than observation. More usefully in practice,
      the term also falls for series that move only in fixed increments, having been rounded until
      their changes are quantised.

    Taking the minimum is what gives this spread. Entropy alone saturates at 1.0 for anything
    continuous and cannot rank; the repeat and run terms vary continuously and pull the composite
    down wherever a series is degenerate in any one way.
    """
    return pl.min_horizontal(
        pl.col("entropy_norm"),
        1.0 - pl.col("repeat_share"),
        1.0 - pl.col("flat_run_share"),
        pl.col("diff_variety"),
    )


# All three axes, all ranked, nothing discarded: a degenerate series simply ranks low rather than
# being removed by a filter whose threshold nobody can defend.
#
# The raw quantities are used directly, because ranking happens *within frequency* by default (see
# RANK_WITHIN) and a rank is unchanged by any monotone transform inside its own group. An earlier
# version divided length by the frequency's median and centred recency on it, to make the axes
# comparable across frequencies before a global rank. Ranking within the group does the same job
# exactly rather than approximately: matching the medians still leaves the spreads different, so a
# global rank on centred values continues to mix distributions of different shape — which showed up
# as annual series behaving oddly against monthly ones.
AXES = {
    "recency": -pl.col("lag_periods"),
    "length": pl.col("n_obs"),
    "information": information_score(),
}

# Percentiles are taken inside each frequency, so "top 10%" means top 10% of annual series *and*
# top 10% of monthly ones. Without this, one global ranking of a quantity like length is largely a
# ranking of frequency, since a weekly series has far more observations than an annual one covering
# the same span.
RANK_WITHIN = ["frequency"]

# Raw normalised entropy is a poor ranking axis on its own, for two measured reasons. It
# saturates: for genuinely continuous economic data every value is distinct, so it sits at 1.0
# and almost everything ties. And it is mechanically anti-correlated with length, since
# entropy_norm is H/log2(n) and repeats accumulate as n grows. `information_score` fixes both by
# taking the minimum against the repeat and run-length terms, which do vary continuously.
# Kept so the axis report can still show what raw entropy alone would have done.
AXES_WITH_INFORMATION = AXES | {"entropy_only": pl.col("entropy_norm")}


def information_floor(min_entropy_norm: float = 0.3, max_repeat_share: float = 0.5,
                      max_flat_run_share: float = 0.3) -> pl.Expr:
    """A hard cut on degeneracy. **Optional, and off by default** — prefer the information axis.

    Ranking is the better instrument: a degenerate series ranks last and can be excluded by the
    same threshold as everything else, without a second set of numbers to defend. This exists for
    when a hard boundary is genuinely wanted, such as fixing a published dataset definition.

    Two of the thresholds were set by measurement rather than taste. ``max_flat_run_share``
    defaults to 0.3, not 0.5: at 0.5 it removed 15 further series out of 128,699, because
    ``repeat_share`` already catches a run that long, whereas at 0.3 it removes 1,578 — a
    contiguous block is more suspicious than the same number of scattered repeats. And there is no
    clause for linear ramps: every series in FRED with a single distinct first difference turned
    out to be a constant, already caught by the entropy term, and neither Eurostat nor OECD showed
    a true ramp either (0.000% of the sampled monthly and quarterly series).
    """
    return (
        (pl.col("entropy_norm") >= min_entropy_norm)
        & (pl.col("repeat_share") <= max_repeat_share)
        & (pl.col("flat_run_share") <= max_flat_run_share)
    )

ID_COLUMNS = ["series_uid", "source", "dataset_id", "frequency"]


def scores_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else data_dir() / "scores"


# -- 1. the cheap filters -----------------------------------------------------------------------

def recent_expr(max_lag_periods: float = 2.0) -> pl.Expr:
    """Series whose last observation is within N of their own periods of the reference date.

    In periods rather than days, so it means the same thing at every frequency: two months for a
    monthly series, two years for an annual one. An annual observation for 2025 is stamped
    2025-01-01, so a day-based test would demand a 2026 point.
    """
    return pl.col("lag_periods") <= max_lag_periods


def candidate_expr(min_obs: int = 3, max_lag_periods: float | None = None,
                   min_horizons: float | None = None, max_missing: float | None = None) -> pl.Expr:
    """What to bother scoring. **A cost control, not a selection decision.**

    Everything here is off by default except ``min_obs``, which is a technical floor rather than a
    judgement: first differences need two points and turning points need three, so below that the
    features are undefined rather than bad.

    The optional clauses exist for when the corpus is too large to score whole, and they are
    dangerous in a specific way that is worth stating plainly. A threshold placed here removes
    series *before the ranking ever sees them*, so it is a silent selection decision that no
    percentile can reveal — and it distorts the ranking of what remains. Filtering FRED at two
    periods of staleness dropped 715,898 of 845,518 series, and left the survivors sharing 25
    distinct lag values with 82.8% of them on one, at which point recency looked like a useless
    axis. It was not: over the full population it takes 1,693 distinct values with a largest tie
    of 30%. The filter manufactured the degeneracy that then justified ignoring the axis.

    If a slice must be cut for cost, prefer a technical one (``min_obs``) over a substantive one,
    and rank on the rest.
    """
    ok = pl.col("n_obs") >= min_obs
    if max_lag_periods is not None:
        ok = ok & recent_expr(max_lag_periods)
    if min_horizons is not None:
        ok = ok & (pl.col("n_obs") >= min_horizons * pl.col("horizon"))
    if max_missing is not None:
        ok = ok & (pl.col("missing_share") <= max_missing)
    return ok


def recent(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
           max_lag_periods: float = 2.0, columns: list[str] | None = None) -> pl.LazyFrame:
    """Every series still being published, as a lazy frame. Seconds over the whole corpus."""
    asof = asof or dt.date.today()
    cols = list(dict.fromkeys(SCALAR_COLUMNS + ID_COLUMNS + list(columns or [])))
    return _with_derived(scan(root, sources, frequencies, cols), asof).filter(recent_expr(max_lag_periods))


# -- 2. the expensive pass ----------------------------------------------------------------------

def _information_exprs() -> list[pl.Expr]:
    """Per-series statistics of the values themselves.

    Two families, and the distinction matters more than any individual feature.

    **Order-invariant** — ``entropy_bits``, ``n_unique``, ``n_repeat``. These describe the *multiset*
    of values. Shuffle a series and every one of them is unchanged, which disqualifies them as
    measures of how much a time series has to say: white noise and a smooth trend with the same
    values score identically. Differential (continuous) entropy has exactly this flaw too — the
    problem is not discreteness, it is that a marginal distribution knows nothing about order — and
    it adds a second one, scale dependence, since h(aX) = h(X) + log|a|. They are kept because they
    are excellent at spotting *degenerate* series, which is a floor, not an ordering.

    **Order-aware** — ``acf1``, ``acf1_diff``, ``n_turning``, ``longest_run``. These are the ones
    that actually see the time dimension, and all of them are scale-free:

    - ``acf1`` is lag-1 autocorrelation: ~1 for anything smooth or trending, negative for noise or
      an alternating pattern. Null when the series is constant, which is itself informative.
    - ``acf1_diff`` is the same on first differences, separating "predictable because it trends"
      from "predictable because it is genuinely autocorrelated".
    - ``n_turning`` counts local extrema. White noise turns at about two thirds of its interior
      points, a smooth series almost never, an alternating one every time.
    - ``longest_run`` is the longest stretch of identical consecutive values. Nothing else here
      catches a series whose last three years were carried forward from the same figure, because
      such a series still has high entropy and high autocorrelation.
    """
    v = pl.col("values")
    return [
        v.list.n_unique().alias("n_unique"),
        v.list.eval((pl.element().diff() == 0).cast(pl.Int8)).list.sum().alias("n_repeat"),
        v.list.eval(pl.element().diff().drop_nulls()).list.n_unique().alias("n_diff_unique"),
        v.list.eval(pl.element().drop_nulls().value_counts(normalize=True).struct.field("proportion"))
         .list.eval(-pl.element() * pl.element().log(2)).list.sum().alias("entropy_bits"),
        v.list.eval(pl.corr(pl.element(), pl.element().shift(1))).list.first().alias("acf1"),
        v.list.eval(pl.corr(pl.element().diff(), pl.element().diff().shift(1))).list.first().alias("acf1_diff"),
        v.list.eval((pl.element().diff().sign().diff().abs() > 0).cast(pl.Int8)).list.sum().alias("n_turning"),
        v.list.eval(pl.element().rle().struct.field("len").max()).list.first().alias("longest_run"),
    ]


def _derived_information() -> list[pl.Expr]:
    """Scale-free forms of the raw counts, so they compare across series of different length."""
    n = pl.col("n_obs")
    return [
        # n_unique counts null as a value when the list holds one; take it back out
        (pl.col("n_unique") - (pl.col("n_points") > n).cast(pl.Int32)).alias("n_distinct"),
        pl.when(n > 1).then(pl.col("n_repeat") / (n - 1)).otherwise(None).alias("repeat_share"),
        # normalised entropy: 1 when every value is distinct, 0 for a constant
        pl.when(n > 1)
        .then(pl.col("entropy_bits") / (n.cast(pl.Float64).log(2)))
        .otherwise(None)
        .alias("entropy_norm"),
        # turning points as a share of the interior. White noise sits near 2/3, a smooth series
        # near 0, an alternating one at 1 — so this reads as "how jagged", independent of scale.
        pl.when(n > 2).then(pl.col("n_turning") / (n - 2)).otherwise(None).alias("turning_share"),
        # the longest carried-forward stretch, as a share of the series
        pl.when(n > 0).then(pl.col("longest_run") / n).otherwise(None).alias("flat_run_share"),
        # how varied the changes are: 1 when every first difference differs, near 0 for a ramp or
        # for a series whose movements have been rounded to a fixed increment
        pl.when(n > 1).then(pl.col("n_diff_unique") / (n - 1)).otherwise(None).alias("diff_variety"),
    ]


def score(root=None, sources=None, frequencies=None, asof: dt.date | None = None,
          where: pl.Expr | None = None, out: Path | str | None = None,
          force: bool = False) -> dict:
    """Compute the information metrics for every series passing ``where``, one input file at a time.

    Writes ``<out>/<source>/freq=<F>/<name>.parquet`` and skips files it has already written, so
    an interrupted run resumes where it stopped. Returns a small summary.

    ``where`` is applied *before* the ``values`` column is touched, which is the whole point: it
    is the difference between reading 28 GB and reading a few hundred megabytes.
    """
    asof = asof or dt.date.today()
    src_root, out_root = series_root(root), scores_root(out)
    if where is None:
        where = candidate_expr()
    files = sorted(src_root.rglob("*.parquet"))
    if sources:
        files = [f for f in files if f.parent.parent.name in sources]
    if frequencies:
        files = [f for f in files if f.parent.name.split("=")[-1] in frequencies]

    keep = list(dict.fromkeys(SCALAR_COLUMNS + ID_COLUMNS))
    n_in = n_out = n_skipped = 0
    bar = tqdm(files, unit="file", dynamic_ncols=True, smoothing=0)
    for f in bar:
        rel = f.relative_to(src_root)
        dest = out_root / rel
        if dest.exists() and not force:
            n_skipped += 1
            continue
        lf = _with_derived(pl.scan_parquet(f).select(keep + ["values"]), asof).filter(where)
        df = (
            lf.with_columns(_information_exprs())
            .drop("values")
            .with_columns(_derived_information())
            .collect(engine="streaming")
        )
        n_in += 1
        if df.height:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            df.write_parquet(tmp, compression="zstd")
            tmp.replace(dest)          # atomic, so a crash cannot leave a half-written score file
            n_out += df.height
        bar.set_postfix_str(f"{n_out:,} scored")
    bar.close()
    log.info("scored %s series from %d files (%d already done)", f"{n_out:,}", n_in, n_skipped)
    return {"files_read": n_in, "files_skipped": n_skipped, "series_scored": n_out, "out": str(out_root)}


def load_scores(out: Path | str | None = None) -> pl.LazyFrame:
    """Every score file as one lazy frame."""
    return pl.scan_parquet(str(scores_root(out) / "**" / "*.parquet"), hive_partitioning=True)


# -- 3. combining the axes ----------------------------------------------------------------------

def _tie_share(df: pl.DataFrame, col: str, within: list[str] | None) -> float:
    """Fraction of series sitting on the modal value *of their own ranking group*.

    Which group matters: once percentiles are taken inside each frequency, a value shared by half
    the corpus is harmless if it is spread evenly across frequencies, and fatal if one frequency
    is entirely made of it. Measuring globally would answer a question nobody is asking.
    """
    if not df.height:
        return 1.0
    if not within:
        return float(df.get_column(col).value_counts().get_column("count").max() / df.height)
    modal = (
        df.group_by([*within, col]).agg(pl.len().alias("k"))
        .group_by(within).agg(pl.col("k").max().alias("modal"))
    )
    return float(modal.get_column("modal").sum() / df.height)


def axis_report(scores: pl.DataFrame | pl.LazyFrame, axes: dict[str, pl.Expr] | None = None,
                within: list[str] | None | object = _UNSET) -> pl.DataFrame:
    """How much each axis can actually discriminate over this set of series.

    Read this before trusting a combined rank. Ties are not a corner case here: series published
    on the same schedule share an end date exactly, so within one source and frequency the recency
    axis is often a single value repeated. An axis that cannot separate anything still contributes
    its tied percentile — 0.5 under average ranking — and because the conjunctive rank takes the
    minimum, that one axis silently caps every series at 0.5 and an 80% threshold selects nothing.

    ``max_tie_share`` is the fraction sitting on the modal value of their own ranking group, and
    ``rank_ceiling`` is the highest ``rank_min`` any series can reach given that.
    """
    axes = axes or AXES
    within = RANK_WITHIN if within is _UNSET else within
    df = scores.lazy().with_columns([e.alias(f"_ax_{k}") for k, e in axes.items()]).collect()
    rows = []
    for k in axes:
        col, s = f"_ax_{k}", df.get_column(f"_ax_{k}")
        tie = _tie_share(df, col, within)
        # The attainable maximum, not the theoretical one, and computed inside the ranking group.
        # Under average ranking a block of ties sits at the middle of the block, so a large
        # tie-block at the top pulls the best achievable percentile well below 1: if 60% of a
        # group share its best value, nothing in that group can score above about 0.7, and a 0.8
        # threshold selects nothing from it however healthy the data is. The worst group is the
        # one reported, since that is the one that will look empty.
        if within:
            ceil_by_group = (
                df.with_columns((pl.col(col).rank("average") / pl.len()).over(within).alias("_r"))
                .group_by(within).agg(pl.col("_r").max().alias("c"))
            )
            ceiling = float(ceil_by_group.get_column("c").min())
        else:
            ceiling = float((s.rank("average") / s.len()).max()) if s.len() else 0.0
        rows.append({
            "axis": k,
            "n_distinct": int(s.n_unique()),
            "max_tie_share": round(tie, 4),
            "rank_ceiling": round(ceiling, 4),
            "discriminates": bool(s.n_unique() > 1),
        })
    return pl.DataFrame(rows)


def add_ranks(scores: pl.DataFrame | pl.LazyFrame, axes: dict[str, pl.Expr] | None = None,
              within: list[str] | None | object = _UNSET, how: str = "min",
              drop_constant: bool = True, max_tie_share: float = 0.9) -> pl.DataFrame:
    """Percentile-rank each axis, then combine them into one column.

    ``rank_min`` is the conjunctive reading and the one to prefer: a series at 0.8 is above the
    80th percentile on *every* axis, so thresholding it applies all three cuts at once without
    anyone having to defend a set of weights. ``how="mean"`` averages instead, which lets a series
    buy its way in on one axis by being excellent on another — sometimes what you want, but say so
    deliberately.

    ``within`` defaults to ``RANK_WITHIN`` (frequency), so a percentile means "against series of
    the same frequency". This is not a refinement but the correct comparison: length in raw
    observations, or even in forecast horizons, is largely a proxy for frequency, and normalising
    by each frequency's median only matches the medians while leaving the spreads different. Pass
    ``within=None`` for one global ranking, or add more keys (``["source", "frequency"]``) to take
    a quota from each source as well.

    ``drop_constant`` removes any axis taking a single value over the set being ranked. Such an
    axis holds no information about which series is better, but under a minimum it would cap every
    series at 0.5 and make a conjunctive threshold select nothing at all. Dropping it is not a
    convenience: keeping it would let a non-discriminating axis veto the whole selection. The names
    of the axes actually used are logged and returned in the frame's ``rank_axes`` column.
    """
    axes = dict(axes or AXES)
    within = RANK_WITHIN if within is _UNSET else within
    df = scores.lazy().with_columns([e.alias(f"_ax_{k}") for k, e in axes.items()]).collect()
    if drop_constant:
        for k in list(axes):
            s = df.get_column(f"_ax_{k}")
            tie = _tie_share(df, f"_ax_{k}", within)
            if s.n_unique() <= 1:
                log.warning("axis %r takes one value over these series and cannot rank them; dropping it", k)
                axes.pop(k)
            elif tie > max_tie_share:
                # Not a corner case: recency after a staleness prefilter behaves exactly like this
                # on FRED, where 98.3% of the survivors share one lag because they are published on
                # a common schedule. Those series all sit at the same middling percentile, and
                # under a minimum that one axis caps every one of them — 128,699 series reduced to
                # 217 at a 0.5 threshold, none of it reflecting anything about the data.
                log.warning(
                    "axis %r puts %.1f%% of these series on a single value, so it cannot rank them "
                    "and would cap every one; dropping it. Enforce it as a threshold instead.",
                    k, tie * 100,
                )
                axes.pop(k)
        if not axes:
            raise ValueError("no axis can separate this set; nothing to rank on")
    rank_cols = []
    for k in axes:
        r = pl.col(f"_ax_{k}").rank("average")
        expr = (r / pl.len()) if not within else (r.over(within) / pl.len().over(within))
        rank_cols.append(expr.alias(f"rank_{k}"))
    df = df.with_columns(rank_cols)
    names = [f"rank_{k}" for k in axes]
    combined = pl.min_horizontal(names) if how == "min" else pl.mean_horizontal(names)
    return df.with_columns(
        combined.alias(f"rank_{how}"),
        pl.lit("+".join(axes)).alias("rank_axes"),
    ).drop([f"_ax_{k}" for k in axes])


def pick(ranked: pl.DataFrame, min_rank: float | None = None, n: int | None = None,
         how: str = "min") -> pl.DataFrame:
    """The chosen subset: everything above a percentile on all axes, or the best n."""
    col = f"rank_{how}"
    out = ranked
    if min_rank is not None:
        out = out.filter(pl.col(col) >= min_rank)
    out = out.sort(col, descending=True)
    return out.head(n) if n else out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="terrastat select", description=__doc__.splitlines()[0])
    p.add_argument("--root", default=None)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--freq", nargs="*", default=None)
    p.add_argument("--asof", default=None, help="reference date for recency (default: today)")
    p.add_argument("--min-obs", type=int, default=3, help="technical floor: fewer points and the features are undefined (default 3)")
    p.add_argument("--max-lag", type=float, default=None, help="OPTIONAL cost cut: skip series staler than this. Off by default -- recency is a ranking axis, and filtering on it here hides series from the ranking entirely")
    p.add_argument("--min-horizons", type=float, default=None, help="OPTIONAL cost cut: skip series shorter than this many horizons")
    p.add_argument("--max-missing", type=float, default=None, help="OPTIONAL cost cut: skip series with more missing than this")
    p.add_argument("--out", default=None, help="where to write the scores (default: data/scores)")
    p.add_argument("--force", action="store_true", help="rescore files already done")
    p.add_argument("--report", action="store_true", help="skip scoring; summarise the scores already on disk")
    p.add_argument("--min-rank", type=float, default=None, help="with --report: keep series above this percentile on every ranked axis")
    p.add_argument("--floor", action="store_true", help="also apply a hard degeneracy cut before ranking (off by default: degenerate series already rank last)")
    p.add_argument("--min-entropy", type=float, default=0.3, help="with --floor: minimum normalised entropy")
    p.add_argument("--max-repeat", type=float, default=0.5, help="with --floor: maximum share of consecutive equal values")
    p.add_argument("--max-flat-run", type=float, default=0.3, help="with --floor: maximum share in one carried-forward run")
    args = p.parse_args(argv)

    asof = dt.date.fromisoformat(args.asof) if args.asof else dt.date.today()
    if not args.report:
        where = candidate_expr(args.min_obs, args.max_lag, args.min_horizons, args.max_missing)
        res = score(args.root, args.sources, args.freq, asof, where=where, out=args.out, force=args.force)
        print(res)

    lf = load_scores(args.out)
    total = lf.select(pl.len()).collect().item()
    if not total:
        print("no scores on disk yet")
        return 0
    scored = lf.collect()
    print(f"\n{total:,} series scored")
    cfg = dict(tbl_rows=30, ascii_tables=True, tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True)
    with pl.Config(**cfg):
        print("\n== can each axis tell these series apart? ==")
        print(axis_report(scored, AXES_WITH_INFORMATION))
        print("\n== by source and frequency ==")
        print(scored.group_by("source", "frequency").agg(
            pl.len().alias("n"),
            pl.col("entropy_norm").median().round(3).alias("entropy_med"),
            pl.col("repeat_share").median().round(3).alias("repeat_med"),
            (pl.col("entropy_norm") < 0.3).mean().round(3).alias("share_low_info"),
        ).sort("n", descending=True).head(20))

    kept = scored
    if args.floor:
        kept = scored.filter(information_floor(args.min_entropy, args.max_repeat, args.max_flat_run))
        print(f"\noptional floor keeps {kept.height:,} of {scored.height:,} "
              f"({kept.height / max(scored.height, 1):.1%})")
        if not kept.height:
            return 0
    ranked = add_ranks(kept)
    if args.min_rank is not None:
        chosen = pick(ranked, min_rank=args.min_rank)
        axes = ranked.get_column("rank_axes")[0]
        print(f"\n== above the {args.min_rank:.0%} percentile on every ranked axis "
              f"({axes}): {chosen.height:,} series ==")
        with pl.Config(**cfg):
            print(chosen.group_by("source", "frequency").agg(pl.len().alias("n")).sort("n", descending=True).head(15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
