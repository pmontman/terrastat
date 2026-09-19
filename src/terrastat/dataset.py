"""The short path: load a corpus, split it without leaking, cut it into supervised windows.

Everything else in this package builds or interrogates the corpus. This is for the person who
just wants to train something, and it exists because the three steps between "there is data" and
"there is a model" are each easy to get subtly wrong.

**Loading.** ``load()`` reads a snapshot directory or the series layer and gives back one lazy
frame either way, so a notebook written against a 200 MB starter set runs unchanged against the
full corpus.

**Splitting.** ``split()`` assigns whole groups deterministically to folds. Related datasets,
cross-source copies and aggregates can still span those groups. Forecasting additionally needs
shared chronological cutoffs; this function does not impose them. See ``docs/leakage.md``.

**Windowing.** ``windows()`` cuts each series into (context, target) pairs with the last ``horizon``
points held out, which is the forecasting exercise rather than a generic regression.

The metrics are the ones this field actually uses: MASE, scaled by the in-sample seasonal naive
error so it is comparable across series of wildly different magnitude, and sMAPE.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import polars as pl

from terrastat.quality import series_root
from terrastat.calendar import regularize
from terrastat.storage import data_dir

log = logging.getLogger(__name__)

# Seasonal period per frequency, used for the naive baseline and for MASE scaling.
SEASONALITY = {"H": 24, "D": 7, "B": 5, "W": 52, "BW": 26, "M": 12, "Q": 4, "S": 2, "A": 1}


def load(source: str | Path | None = None, frequencies: list[str] | None = None,
         min_obs: int = 1, columns: list[str] | None = None) -> pl.LazyFrame:
    """A corpus as one lazy frame, from a snapshot directory or from the series layer.

    ``source`` may be a snapshot name (``"starter"``), a path to a directory of shards, or
    ``None`` for the full series layer. The same notebook then runs against either.
    """
    if source is None:
        pattern = str(series_root() / "**" / "*.parquet")
    else:
        p = Path(source)
        if not p.exists():
            p = data_dir() / "snapshot" / str(source)
        if not p.exists():
            raise FileNotFoundError(
                f"no snapshot or directory called {source!r}. Build one with "
                f"`terrastat export {source} --public-only`, or pass source=None for the series layer."
            )
        if p.is_dir():
            # The lookup index has a different schema from the actual series shards.
            shards = sorted(p.glob("shard-*.parquet"))
            files = shards or [f for f in sorted(p.glob("*.parquet")) if f.name != "index.parquet"]
            if not files:
                raise FileNotFoundError(f"no series Parquet files in {p}")
            pattern = [str(f) for f in files]
        else:
            pattern = str(p)
    lf = pl.scan_parquet(pattern, hive_partitioning=True)
    if frequencies:
        lf = lf.filter(pl.col("frequency").is_in(frequencies))
    if min_obs > 1:
        lf = lf.filter(pl.col("n_obs") >= min_obs)
    if columns is not None:
        lf = lf.select(columns)
    return lf


# -- splitting ----------------------------------------------------------------------------------

def _bucket(values: pl.Series, folds: int, seed: int) -> pl.Series:
    """Deterministic fold per distinct key, from a hash rather than a shuffle.

    Hashing means the assignment depends only on the key and the seed, so adding series, or
    re-running after a longer crawl, leaves every existing series in the fold it had. A shuffle
    would silently reassign everything and quietly invalidate an earlier evaluation.
    """
    out = []
    for v in values.to_list():
        h = hashlib.blake2b(f"{seed}:{v}".encode(), digest_size=8).digest()
        out.append(int.from_bytes(h, "big") % folds)
    return pl.Series(out, dtype=pl.Int32)


def split(df: pl.DataFrame | pl.LazyFrame, by: str = "dataset_id", holdout: float = 0.2,
          seed: int = 0, folds: int = 20) -> pl.DataFrame:
    """Add a ``fold`` column of ``"train"`` / ``"test"``, splitting whole groups.

    ``by`` is the unit that moves together. In increasing order of safety:

    ``"dataset_id"``   the default. Keeps within-dataset variants together; cross-dataset copies
                       and identities still need an audit.
    ``"group"``        the theme prefix (``nama``, ``apro``, ``OECD.SDD.NAD``). Coarser groups can
                       reduce related-dataset overlap but do not prove independence.
    ``"geo"``          by region. Geographic aggregates can still contain held-out regions.

    ``by="series_uid"`` is accepted and warned about, because it is the split this corpus is least
    suited to: it is exactly the random split that the duplication makes meaningless.

    The proportion is approximate — whole groups move, and groups differ in size — so the realised
    share is reported by the caller rather than guaranteed here.
    """
    if not 0 < holdout < 1:
        raise ValueError("holdout must lie strictly between 0 and 1")
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    lf = df.lazy()
    if by == "group":
        from terrastat.hierarchy import group_expr
        lf = lf.with_columns(group_expr())
    out = lf.collect()
    if by == "series_uid":
        log.warning(
            "splitting by series_uid is a random split: with 78%% of datasets publishing totals "
            "alongside their parts, and FRED republishing 80,274 OECD series, the test set will "
            "contain near-copies of training series. Prefer by='dataset_id' or by='group'."
        )
    if by not in out.columns:
        raise KeyError(f"cannot split by {by!r}: not a column. Have {out.columns[:8]}...")
    n_groups = out.get_column(by).n_unique()
    if n_groups < folds:
        # Whole groups move together, so the realised share is quantised by group. With a handful
        # of groups a 20% holdout can easily land on 0% or 50%, and the caller needs to know that
        # before reading anything into an evaluation. This bites hardest on a `head()` of the
        # series layer, which draws from very few files and therefore very few datasets.
        log.warning(
            "only %d distinct %s in this frame, so a %.0f%% holdout cannot be met precisely; "
            "the realised split may be far off, or empty. Load a wider slice, or split by a "
            "finer key.", n_groups, by, holdout * 100,
        )
    n_test = min(folds - 1, max(1, round(folds * holdout)))
    return out.with_columns(
        pl.when(_bucket(out.get_column(by), folds, seed) < n_test)
        .then(pl.lit("test")).otherwise(pl.lit("train")).alias("fold")
    )


def split_report(df: pl.DataFrame, by: str = "dataset_id") -> pl.DataFrame:
    """What the split actually produced, including whether any group leaked across it."""
    g = df.group_by("fold").agg(
        pl.len().alias("series"),
        pl.col("n_obs").sum().alias("observations"),
        pl.col(by).n_unique().alias(f"{by}s"),
    ).sort("fold")
    both = (df.group_by(by).agg(pl.col("fold").n_unique().alias("k"))
              .filter(pl.col("k") > 1).height)
    return g.with_columns(pl.lit(both).alias("groups_in_both_folds"))


# -- windowing ----------------------------------------------------------------------------------

def windows(df: pl.DataFrame, context: int = 48, horizon: int = 12,
            max_per_series: int = 1, drop_missing: bool = False,
            max_periods: int = 10_000) -> dict:
    """Cut series into supervised (context, target) pairs, last observations held out.

    Returns arrays ready for a model: ``X`` of shape (n, context), ``y`` of (n, horizon), plus the
    ``series_uid`` and ``frequency`` of each row and the in-sample seasonal-naive error used to
    scale MASE. Sparse dates are expanded before slicing. ``X_mask`` and ``y_mask`` mark
    observed values; missing entries remain NaN. ``drop_missing=True`` skips whole
    incomplete windows, never individual periods. Undefined MASE scales are NaN and
    retained for callers to report coverage. No shared cutoff across series is imposed.

    ``max_per_series`` limits how many windows one long series contributes, so a handful of daily
    series with 20,000 points cannot dominate a batch made mostly of annual ones.
    """
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
           for n in (context, horizon, max_per_series)):
        raise ValueError("context, horizon and max_per_series must be positive integers")
    need = context + horizon
    X, Y, uids, freqs, scales, target_dates = [], [], [], [], [], []
    for r in df.iter_rows(named=True):
        try:
            grid = regularize(r["dates"], r["values"], r["frequency"], max_periods)
        except ValueError as exc:
            raise ValueError(f"{r['series_uid']}: {exc}") from exc
        v = grid["values"]
        if v.size < need:
            continue
        m = SEASONALITY.get(r["frequency"], 1)
        # windows are taken from the end backwards, so the most recent data is always used
        for k in range(max_per_series):
            end = v.size - k * horizon
            if end < need:
                break
            w = v[end - need:end]
            if drop_missing and not np.isfinite(w).all():
                continue
            hist = w[:context]
            # MASE denominator: mean absolute seasonal-naive error inside the context only, so
            # nothing from the target period influences the scale
            d = np.abs(hist[m:] - hist[:-m]) if hist.size > m else np.abs(np.diff(hist))
            d = d[np.isfinite(d)]
            scale = float(np.mean(d)) if d.size else np.nan
            if scale == 0:
                scale = np.nan
            X.append(hist); Y.append(w[context:])
            uids.append(r["series_uid"]); freqs.append(r["frequency"]); scales.append(scale)
            target_dates.append(grid["dates"][end - horizon:end])
    if not X:
        return {"X": np.zeros((0, context)), "y": np.zeros((0, horizon)),
                "series_uid": [], "frequency": [], "scale": np.zeros(0), "target_dates": [],
                "X_mask": np.zeros((0, context), dtype=bool), "y_mask": np.zeros((0, horizon), dtype=bool)}
    return {"X": np.stack(X), "y": np.stack(Y), "series_uid": uids,
            "frequency": freqs, "scale": np.asarray(scales), "target_dates": target_dates,
            "X_mask": np.isfinite(X), "y_mask": np.isfinite(Y)}


# -- baselines and metrics ----------------------------------------------------------------------

def seasonal_naive(X: np.ndarray, freqs: list[str], horizon: int) -> np.ndarray:
    """Repeat the last seasonal cycle. The baseline any forecast has to beat to be worth anything."""
    out = np.empty((X.shape[0], horizon))
    for i, f in enumerate(freqs):
        m = SEASONALITY.get(f, 1)
        m = m if m <= X.shape[1] else 1
        out[i] = [X[i, -m + (h % m)] for h in range(horizon)]
    return out


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray,
         agg: str | None = "median") -> float | np.ndarray:
    """Absolute error scaled by the in-sample seasonal-naive error.

    1.0 matches the in-sample naive error scale, not necessarily a naive forecast's test error.

    Scale-free, so a series in millions of euro and one in percent count equally.

    **The aggregate defaults to the median, not the mean, and that is a departure worth knowing
    about.** Forecasting competitions report mean MASE over curated series, where it is fine. On
    this corpus it is not: MASE is a ratio, and the denominator is the in-sample variation, so a
    series that sits nearly flat and then jumps produces an enormous value. Measured on a 30,000
    series monthly sample, seasonal naive scores a median of 0.72 and a mean of 6.90 — the mean
    driven by a tail reaching 4,297, all of it air-freight series between individual airport pairs
    with tiny stable volumes. The median says the baseline behaves; the mean says nothing at all.

    Pass ``agg="mean"`` to match the competition convention, or ``agg=None`` for the per-series
    array, which is what you want for a distribution plot or for finding the offenders.
    """
    per = np.abs(y_true - y_pred).mean(axis=1) / scale
    if agg is None:
        return per
    if agg == "mean":
        return float(np.mean(per))
    if agg == "median":
        return float(np.median(per))
    raise ValueError(f"agg must be 'mean', 'median' or None, not {agg!r}")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE, in percent. Undefined where both are zero; those points are skipped."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    ok = denom > 0
    return float(200.0 * np.mean(np.abs(y_true - y_pred)[ok] / (2 * denom[ok])))
