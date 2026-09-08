"""The short path from corpus to model: loading, splitting without leaking, windowing, scoring."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from terrastat import dataset as ds

DATES = [dt.date(2000 + i // 12, i % 12 + 1, 1) for i in range(120)]


def _row(uid, dataset, values, freq="M", geo="AT"):
    return {
        "series_uid": uid, "source": "eurostat", "dataset_id": dataset, "geo": geo,
        "frequency": freq, "n_obs": len(values), "n_points": len(values),
        "dates": DATES[:len(values)], "values": values,
    }


SCHEMA = {
    "series_uid": pl.String, "source": pl.String, "dataset_id": pl.String, "geo": pl.String,
    "frequency": pl.String, "n_obs": pl.Int64, "n_points": pl.Int64,
    "dates": pl.List(pl.Date), "values": pl.List(pl.Float64),
}


@pytest.fixture
def frame():
    """40 series spread over 8 datasets, so a grouped split has something to work with."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(8):
        for s in range(5):
            base = np.tile(np.arange(12, dtype=float) + d, 10)
            rows.append(_row(f"s{d}_{s}", f"ds{d}", list(base + rng.normal(0, .1, 120))))
    return pl.DataFrame(rows, schema=SCHEMA)


def test_a_group_never_lands_in_both_folds(frame):
    """The whole point: series that share a dataset move together, so a near-copy of a test
    series cannot sit in training."""
    s = ds.split(frame, by="dataset_id", holdout=0.25, seed=1)
    rep = ds.split_report(s, "dataset_id")
    assert rep.get_column("groups_in_both_folds")[0] == 0
    assert set(s.get_column("fold").unique()) == {"train", "test"}
    # every series of a dataset carries the same fold
    per = s.group_by("dataset_id").agg(pl.col("fold").n_unique().alias("k"))
    assert per.get_column("k").max() == 1


def test_the_assignment_is_stable_when_series_are_added(frame):
    """Hashing, not shuffling: a longer crawl must not silently move existing series between
    folds and invalidate an evaluation done earlier."""
    first = ds.split(frame, by="dataset_id", seed=3).select("series_uid", "fold")
    grown = pl.concat([frame, frame.head(5).with_columns(
        pl.col("series_uid") + "_new", pl.lit("ds_new").alias("dataset_id"))])
    second = ds.split(grown, by="dataset_id", seed=3).select("series_uid", "fold")
    joined = first.join(second, on="series_uid", suffix="_2")
    assert (joined.get_column("fold") == joined.get_column("fold_2")).all()


def test_splitting_by_series_warns_that_it_is_the_unsafe_one(frame, caplog):
    with caplog.at_level("WARNING"):
        ds.split(frame, by="series_uid")
    assert "random split" in caplog.text
    assert "dataset_id" in caplog.text


def test_too_few_groups_to_honour_the_holdout_is_flagged(frame, caplog):
    """A `head()` of the series layer draws from very few files and therefore very few datasets;
    whole groups move, so the realised share can be 0% or 50% and the caller must be told."""
    small = frame.filter(pl.col("dataset_id").is_in(["ds0", "ds1"]))
    with caplog.at_level("WARNING"):
        ds.split(small, by="dataset_id", holdout=0.2)
    assert "only 2 distinct dataset_id" in caplog.text


def test_windows_hold_out_the_end_and_never_leak_the_target_into_the_scale(frame):
    w = ds.windows(frame, context=48, horizon=12)
    assert w["X"].shape == (40, 48) and w["y"].shape == (40, 12)
    # the MASE denominator must come from the context alone
    r = frame.row(0, named=True)
    v = np.asarray(r["values"])
    hist = v[len(v) - 60:len(v) - 12]
    assert w["scale"][0] == pytest.approx(np.mean(np.abs(hist[12:] - hist[:-12])), rel=1e-9)


def test_a_series_too_short_for_one_window_is_dropped(frame):
    short = frame.head(2).with_columns(
        pl.col("values").list.head(20), pl.lit(20).alias("n_obs"))
    assert ds.windows(short, context=48, horizon=12)["X"].shape[0] == 0


def test_seasonal_naive_repeats_the_last_cycle(frame):
    w = ds.windows(frame, context=48, horizon=12)
    pred = ds.seasonal_naive(w["X"], w["frequency"], 12)
    assert np.allclose(pred, w["X"][:, -12:])


def test_mase_defaults_to_the_median_because_the_mean_is_not_robust_here():
    """MASE is a ratio scaled by in-sample variation, so a nearly flat series that jumps produces
    an enormous value. On a 30,000-series monthly sample the seasonal-naive median was 0.72 and
    the mean 6.90, the mean driven by a tail reaching 4,297."""
    y = np.zeros((100, 4))
    pred = np.ones((100, 4))
    scale = np.ones(100)
    scale[0] = 1e-4                       # one pathological series
    assert ds.mase(y, pred, scale) == pytest.approx(1.0)              # median shrugs it off
    assert ds.mase(y, pred, scale, agg="mean") > 100                  # the mean does not
    assert ds.mase(y, pred, scale, agg=None).shape == (100,)


def test_smape_skips_the_points_where_it_is_undefined():
    y = np.array([[0.0, 2.0]])
    pred = np.array([[0.0, 2.0]])
    assert ds.smape(y, pred) == pytest.approx(0.0)     # the 0/0 point is skipped, not NaN
