"""Calendar and masking helpers for the quarterly CPU tutorial.

No training dependencies are imported here. The notebook contains the network and optimizer.
This is a retrospective, latest-vintage benchmark, not a real-time data release simulator.
"""
from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import polars as pl

from terrastat.calendar import regularize


def snapshot_candidates(source, dataset_ids: list[str], per_dataset: int = 512,
                        seed: int = 7, geos: list[str] | None = None) -> pl.DataFrame:
    """Read bounded quarterly Eurostat candidates from an existing snapshot, without writes.

    Uses the current shard schema (including float32 lists). Selection depends on IDs and
    optional geography, never target values or future coverage. Metadata remains unchanged.
    """
    from terrastat.dataset import load

    if not dataset_ids or per_dataset < 1 or per_dataset * len(dataset_ids) > 5000:
        raise ValueError("request 1..5000 candidates across the selected datasets")
    columns = ["series_uid", "source", "dataset_id", "title", "units", "geo", "frequency",
               "dates", "values", "license_id", "attribution", "retrieved_at"]
    lf = load(source, frequencies=["Q"]).filter(pl.col("source") == "eurostat")
    if geos is not None:
        lf = lf.filter(pl.col("geo").is_in(geos))
    parts = [lf.filter(pl.col("dataset_id") == name)
             .with_columns(pl.col("series_uid").hash(seed=seed).alias("_rank"))
             .sort("_rank").head(per_dataset).select(columns).collect() for name in dataset_ids]
    frame = pl.concat(parts)
    if frame.is_empty():
        raise ValueError("snapshot contains no quarterly Eurostat series for the requested datasets/geographies")
    return frame


def quarter_dates(start_year: int = 2000, end_year: int = 2025) -> list[dt.date]:
    if not 1 <= start_year <= end_year <= 9998 or end_year - start_year >= 500:
        raise ValueError("quarterly experiment must span 1..500 years within 1..9998")
    return [dt.date(y, m, 1) for y in range(start_year, end_year + 1) for m in (1, 4, 7, 10)]


def quarterly_panel(frame: pl.DataFrame, dates: list[dt.date], train_end: dt.date,
                    per_dataset: int = 64, seed: int = 7) -> tuple[pl.DataFrame, np.ndarray, dict]:
    """Align a small candidate cohort; choose eligibility using training observations only.

    Require >=48 observed training quarters and >=75% coverage in its last 24 quarters.
    Rank by a stable UID hash, separately within each dataset. Never select on test coverage.
    Reject corrupt/ambiguous source calendars explicitly in the audit. Bound each source
    expansion to 2,000 quarters and the full candidate panel to 5,000 rows.
    """
    if not dates or len(dates) > 2000 or dates != sorted(set(dates)):
        raise ValueError("dates must be a sorted unique quarterly grid of at most 2000 periods")
    check = regularize(dates, np.zeros(len(dates)), "Q", 2000)
    if check["dates"] != dates:
        raise ValueError("dates must be a complete grid of quarter starts")
    if frame.height > 5000 or per_dataset < 1:
        raise ValueError("use at most 5000 candidates and a positive per_dataset cap")
    if frame["series_uid"].n_unique() != frame.height:
        raise ValueError("candidate series_uid values must be unique")
    end = dates.index(train_end) + 1
    if end < 48:
        raise ValueError("need at least 48 training quarters")
    positions = {d: i for i, d in enumerate(dates)}
    eligible, audit = [], {"candidates": frame.height, "invalid_calendar": [], "insufficient_train": 0}
    for row in frame.iter_rows(named=True):
        try:
            if row["frequency"] != "Q":
                raise ValueError("expected frequency Q")
            grid = regularize(row["dates"], row["values"], "Q", max_periods=2000)
        except ValueError as exc:
            audit["invalid_calendar"].append({"series_uid": row["series_uid"], "reason": str(exc)})
            continue
        v = np.full(len(dates), np.nan)
        for d, value in zip(grid["dates"], grid["values"]):
            if d in positions:
                v[positions[d]] = value
        observed = np.isfinite(v[:end])
        if observed.sum() < 48 or observed[-24:].mean() < .75:
            audit["insufficient_train"] += 1
            continue
        rank = hashlib.blake2b(f"{seed}:{row['series_uid']}".encode(), digest_size=8).hexdigest()
        eligible.append((row, v, rank))
    counts, selected = {}, []
    for row, v, rank in sorted(eligible, key=lambda x: x[2]):
        group = (row["source"], row["dataset_id"])
        if counts.get(group, 0) < per_dataset:
            counts[group] = counts.get(group, 0) + 1
            selected.append((row, v))
    audit["eligible"] = len(eligible)
    audit["selected"] = len(selected)
    if not selected:
        raise ValueError(f"no eligible quarterly series: {audit}")
    meta = pl.DataFrame([r for r, _ in selected]).drop("dates", "values")
    return meta, np.stack([v for _, v in selected]), audit


def make_windows(panel: np.ndarray, origins, context: int = 24, horizon: int = 4) -> dict:
    """Origin is the index of the first target quarter; train origins must be bounded by caller.

    Retain windows with >=75% observed context and at least one observed target.
    Original values remain NaN. Masks are used explicitly by training and scoring.
    """
    if panel.ndim != 2 or context < 4 or horizon < 1 or np.isinf(panel).any():
        raise ValueError("expected a finite-or-NaN panel, context >=4 and horizon >=1")
    X, y, rows, starts = [], [], [], []
    for origin in origins:
        if not isinstance(origin, (int, np.integer)) or not context <= origin <= panel.shape[1] - horizon:
            raise ValueError("origin must leave a full context and horizon inside the panel")
        for i, series in enumerate(panel):
            hist, target = series[origin-context:origin], series[origin:origin+horizon]
            if np.isfinite(hist).mean() >= .75 and np.isfinite(target).any():
                X.append(hist); y.append(target); rows.append(i); starts.append(origin)
    X = np.asarray(X, dtype=float).reshape(-1, context)
    y = np.asarray(y, dtype=float).reshape(-1, horizon)
    return {"X": X, "y": y, "X_mask": np.isfinite(X), "y_mask": np.isfinite(y),
            "row": np.asarray(rows, dtype=int), "origin": np.asarray(starts, dtype=int)}


def normalize_context(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed-context mean/std only; append observation masks to zero-filled z-scores."""
    mask = np.isfinite(X)
    if X.ndim != 2 or not mask.any(axis=1).all() or np.isinf(X).any():
        raise ValueError("each context must have at least one observed value and no infinities")
    mean = np.nansum(X, axis=1, keepdims=True) / mask.sum(axis=1, keepdims=True)
    delta = np.where(mask, X - mean, 0.)
    std = np.sqrt((delta ** 2).sum(axis=1, keepdims=True) / mask.sum(axis=1, keepdims=True))
    scale = np.maximum(std, np.maximum(np.abs(mean) * 1e-6, 1e-6))
    features = np.concatenate([delta / scale, mask.astype(float)], axis=1).astype("float32")
    return features, mean, scale


def score_forecasts(y: np.ndarray, pred: np.ndarray, X: np.ndarray) -> dict:
    """Per-window MAE, seasonal MASE and sMAPE, plus observed-target counts.

    Missing targets are excluded; nonfinite predictions on observed targets are errors.
    Seasonal scale uses only observed pairs four quarters apart in the original context.
    Undefined MASE (no pairs or zero variation) is NaN, never silently replaced by zero.
    """
    if y.shape != pred.shape or y.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("incompatible forecast, target or context shapes")
    mask = np.isfinite(y)
    if not np.isfinite(pred[mask]).all():
        raise ValueError("predictions must be finite at observed targets")
    count = mask.sum(axis=1)
    error = np.where(mask, np.abs(y - pred), 0.)
    mae = np.divide(error.sum(axis=1), count, out=np.full(len(y), np.nan), where=count > 0)
    diffs = np.abs(X[:, 4:] - X[:, :-4])
    pairs = np.isfinite(diffs).sum(axis=1)
    scale = np.divide(np.nansum(diffs, axis=1), pairs, out=np.full(len(y), np.nan), where=pairs > 0)
    mase = np.divide(mae, scale, out=np.full(len(y), np.nan), where=scale > 0)
    denom = np.abs(y) + np.abs(pred)
    ratio = np.divide(200 * error, denom, out=np.zeros_like(error), where=mask & (denom > 0))
    smape = np.divide(ratio.sum(axis=1), count, out=np.full(len(y), np.nan), where=count > 0)
    return {"mae": mae, "mase": mase, "smape": smape, "observed_targets": count}
