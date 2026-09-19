"""Regression checks for the tutorial's temporal isolation and missing-data contract."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from terrastat.forecasting import (
    make_windows, normalize_context, quarter_dates, quarterly_panel, score_forecasts, snapshot_candidates,
)


def test_future_values_do_not_change_cohort_training_or_normalization():
    dates = quarter_dates()
    end = dates.index(dt.date(2022, 10, 1)) + 1
    values = np.arange(len(dates), dtype=float)
    row = dict(series_uid="s", source="eurostat", dataset_id="d", frequency="Q", dates=dates,
               values=values.tolist())
    a, panel_a, _ = quarterly_panel(pl.DataFrame([row]), dates, dates[end-1])
    values[end:] = np.nan  # even missing *all* future targets cannot change cohort membership
    row["values"] = values.tolist()
    b, panel_b, _ = quarterly_panel(pl.DataFrame([row]), dates, dates[end-1])
    assert a["series_uid"].to_list() == b["series_uid"].to_list()
    origins = range(24, end - 4 + 1, 4)
    wa, wb = make_windows(panel_a, origins), make_windows(panel_b, origins)
    np.testing.assert_equal(wa["y"], wb["y"])
    np.testing.assert_equal(normalize_context(wa["X"])[0], normalize_context(wb["X"])[0])
    assert (wa["origin"] + 4 <= end).all()


def test_mask_features_and_metrics_keep_missing_targets_out_of_loss():
    X = np.array([[1., 2., np.nan, 4., 3., 4., 5., 6.]])
    features, mean, scale = normalize_context(X)
    assert features.shape == (1, 16)
    assert features[0, 2] == 0 and features[0, 10] == 0
    assert mean.item() == pytest.approx(25 / 7)
    y = np.array([[7., np.nan, 9., 10.]])
    pred = np.array([[8., 1e20, 10., 11.]])
    metrics = score_forecasts(y, pred, X)
    assert metrics["mae"].item() == 1
    assert metrics["mase"].item() == .5
    assert metrics["observed_targets"].item() == 3
    pred[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_forecasts(y, pred, X)


def test_constant_context_has_undefined_mase_but_valid_mae():
    result = score_forecasts(np.ones((1, 4)), np.ones((1, 4)), np.ones((1, 24)))
    assert np.isnan(result["mase"]).all()
    assert result["mae"].item() == 0


def test_target_dates_do_not_overlap_training_boundary():
    p = np.arange(104, dtype=float)[None, :]
    train = make_windows(p, range(24, 92 - 4 + 1, 4))
    valid = make_windows(p, [92])
    test = make_windows(p, [96, 100])
    assert np.max(train["origin"] + 4) == np.min(valid["origin"])
    assert np.max(valid["origin"] + 4) == np.min(test["origin"])
    with pytest.raises(ValueError, match="full context"):
        make_windows(p, [102])


def test_existing_snapshot_float32_sparse_schema_is_read_without_rewriting(tmp_path):
    import hashlib
    import json

    dates = quarter_dates()
    # Matches the current snapshot: float32 values, sparse dates, flags and a separate index.
    keep = [i for i in range(len(dates)) if i != 50]
    row = dict(series_uid="eurostat:demo", source="eurostat", dataset_id="demo", frequency="Q",
               title="Demo", units="Index", geo="AT", dates=[dates[i] for i in keep],
               values=[float(i) for i in keep], flags=[None] * len(keep), license_id="eurostat-reuse",
               attribution="Eurostat", retrieved_at="2026-09-08", n_obs=len(keep))
    frame = pl.DataFrame([row]).with_columns(pl.col("values").cast(pl.List(pl.Float32)))
    frame.write_parquet(tmp_path / "shard-00000.parquet")
    frame.select("series_uid", "n_obs").write_parquet(tmp_path / "index.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"rows": 1, "shards": [{"file": "shard-00000.parquet"}]}))
    def hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    before = hashes()
    candidates = snapshot_candidates(tmp_path, ["demo"])
    assert candidates["values"].dtype == pl.List(pl.Float32)
    assert candidates["dates"][0].to_list() == row["dates"]
    _, panel, _ = quarterly_panel(candidates, dates, dt.date(2022, 10, 1))
    assert panel.shape == (1, len(dates)) and np.isnan(panel[0, 50])
    assert panel[0, 51] == 51
    assert hashes() == before


def test_snapshot_without_quarterly_data_fails_explicitly(tmp_path):
    row = dict(series_uid="monthly", source="eurostat", dataset_id="demo", frequency="M",
               title="Demo", units="Index", geo="AT", dates=[dt.date(2020, 1, 1)],
               values=[1.], license_id="eurostat-reuse", attribution="Eurostat", retrieved_at=None)
    pl.DataFrame([row]).write_parquet(tmp_path / "shard-00000.parquet")
    with pytest.raises(ValueError, match="no quarterly"):
        snapshot_candidates(tmp_path, ["demo"])
