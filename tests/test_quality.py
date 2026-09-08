"""The quality report, on a tiny hand-built series layer whose answers can be worked out by hand."""
import datetime as dt

import polars as pl
import pytest

from terrastat import quality

TODAY = dt.date(2026, 9, 1)


def _months(end: dt.date, n: int) -> list[dt.date]:
    """n monthly dates ending on `end`, counting back."""
    out = []
    y, m = end.year, end.month
    for _ in range(n):
        out.append(dt.date(y, m, 1))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def _series(uid, values, end):
    dates = _months(end, len(values))
    return {
        "series_uid": uid, "source": "fred", "frequency": "M",
        "start_date": dates[0], "end_date": dates[-1],
        "n_points": len(values), "n_obs": sum(v is not None for v in values), "n_flagged": 0,
        "dates": dates, "values": values, "flags": [None] * len(values),
    }


@pytest.fixture
def layer(tmp_path):
    """Six monthly series with deliberately different shapes. Monthly horizon is 18, so the
    thresholds in play are 36 observations (trainable), 54 (backtest, real_time) and 72
    (nowcasting).

    a  60 points, ends today             long, fresh, complete
    b  60 points, ends five years ago    long, stale
    c   4 points, ends today             far too short
    d  60 slots, two holes in the middle an imputation candidate
    e  60 points, all the same value     constant
    f  60 slots, two nulls at the front  a shorter series, not a gap
    """
    seq = [float(i) for i in range(60)]
    rows = [
        _series("a", seq, TODAY),
        _series("b", seq, dt.date(2021, 9, 1)),
        _series("c", [1.0, 2.0, 3.0, 4.0], TODAY),
        _series("d", [None if i in (20, 21) else float(i) for i in range(60)], TODAY),
        _series("e", [7.0] * 60, TODAY),
        _series("f", [None, None] + seq[2:], TODAY),
    ]
    schema = {
        "series_uid": pl.String, "source": pl.String, "frequency": pl.String,
        "start_date": pl.Date, "end_date": pl.Date,
        "n_points": pl.Int64, "n_obs": pl.Int64, "n_flagged": pl.Int64,
        "dates": pl.List(pl.Date), "values": pl.List(pl.Float64), "flags": pl.List(pl.String),
    }
    d = tmp_path / "fred" / "freq=M"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=schema).write_parquet(d / "x.parquet")
    return tmp_path


def test_overview_counts_length_and_completeness(layer):
    ov = quality.overview(root=layer, asof=TODAY, until=dt.date(2025, 12, 1))
    r = ov.row(0, named=True)
    assert r["n_series"] == 6
    assert r["n_obs"] == 60 + 60 + 4 + 58 + 60 + 58
    assert r["ge_1y"] == pytest.approx(5 / 6)   # every series but c reaches 12 observations
    assert r["ge_3y"] == pytest.approx(5 / 6)   # d and f hold 58, comfortably over 36
    assert r["ge_5y"] == pytest.approx(3 / 6)   # only a, b, e hold a full 60
    assert r["share_complete"] == pytest.approx(4 / 6)  # d and f store nulls
    # b is the only stale one, and it stops well before the cutoff
    assert r["reaches_2025_12"] == pytest.approx(5 / 6)
    assert r["fresh_le_2p"] == pytest.approx(5 / 6)


def test_quantiles_are_exact_without_materialising(layer):
    q = quality.obs_quantiles(root=layer).row(0, named=True)
    # lengths are 4, 58, 58, 60, 60, 60
    assert (q["obs_p25"], q["obs_p50"], q["obs_p75"]) == (58, 58, 60)
    assert q["years_median"] == pytest.approx(58 / 12)


def test_stale_and_short_series_are_not_real_time(layer):
    tk = quality.task_table(root=layer, asof=TODAY, until=dt.date(2025, 12, 1)).row(0, named=True)
    # thresholds are multiples of the monthly horizon of 18
    assert tk["trainable"] == 5       # >= 36 observations: everything but c
    assert tk["backtest"] == 5        # >= 54: same set, since d and f hold 58
    assert tk["real_time"] == 4       # >= 54 and fresh: b is five years stale
    assert tk["nowcasting"] == 0      # >= 72, which nothing here reaches
    assert tk["imputation"] == 2      # d and f both look gappy from the scalar columns alone


def test_only_the_deep_pass_separates_interior_gaps_from_edges(layer):
    """The scalar `imputation` profile counts f, whose nulls are only at the front. Telling the
    two apart needs the values themselves, which is the reason the deep pass exists."""
    r = quality.deep_table(root=layer, sample=100).row(0, named=True)
    assert r["sampled"] == 6
    assert r["share_constant"] == pytest.approx(1 / 6)       # e
    assert r["share_any_gap"] == pytest.approx(2 / 6)        # d and f
    assert r["share_interior_gap"] == pytest.approx(1 / 6)   # but only d's hole is in the middle


def test_anomalies_finds_nothing_wrong_here(layer):
    r = quality.anomalies(root=layer, asof=TODAY).row(0, named=True)
    assert (r["ends_in_future"], r["empty"], r["end_before_start"]) == (0, 0, 0)
    assert r["single_point"] == 0


def test_the_combined_pass_matches_the_separate_ones(layer):
    """`full_report` computes the four scalar tables in one group-by instead of four scans. The
    saving is only worth having if the numbers are identical."""
    asof, until = TODAY, dt.date(2025, 12, 1)
    r = quality.full_report(root=layer, asof=asof, until=until, deep=False)
    for name, separate in [
        ("overview", quality.overview(root=layer, asof=asof, until=until)),
        ("length", quality.length_table(root=layer)),
        ("tasks", quality.task_table(root=layer, asof=asof, until=until)),
        ("anomalies", quality.anomalies(root=layer, asof=asof)),
    ]:
        combined = r[name].select(separate.columns)
        assert combined.equals(separate), f"{name} differs between the combined and separate paths"


# -- the aligned subset -------------------------------------------------------------------------

def _annual(uid, first_year, last_year):
    dates = [dt.date(y, 1, 1) for y in range(first_year, last_year + 1)]
    return {
        "series_uid": uid, "source": "fred", "frequency": "A",
        "start_date": dates[0], "end_date": dates[-1],
        "n_points": len(dates), "n_obs": len(dates), "n_flagged": 0,
        "dates": dates, "values": [float(i) for i in range(len(dates))],
        "flags": [None] * len(dates),
    }


@pytest.fixture
def annual_layer(tmp_path):
    """Annual series ending in different years. The 2025 observation is stamped 2025-01-01, which
    is the whole reason the recency test has to be in periods rather than days."""
    schema = {
        "series_uid": pl.String, "source": pl.String, "frequency": pl.String,
        "start_date": pl.Date, "end_date": pl.Date,
        "n_points": pl.Int64, "n_obs": pl.Int64, "n_flagged": pl.Int64,
        "dates": pl.List(pl.Date), "values": pl.List(pl.Float64), "flags": pl.List(pl.String),
    }
    rows = [
        _annual("long", 2000, 2025),    # 26 points, current
        _annual("short", 2024, 2025),   # 2 points: one year of history before the origin
        _annual("tiny", 2025, 2025),    # 1 point: no history at all
        _annual("stale", 2000, 2023),   # ends two years early
    ]
    d = tmp_path / "fred" / "freq=A"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=schema).write_parquet(d / "a.parquet")
    return tmp_path


def test_an_annual_series_current_to_2025_counts_as_aligned(annual_layer):
    """An annual observation for 2025 is dated 2025-01-01, so a day-based `end_date >= origin`
    test would demand a 2026 point and reject essentially every annual series."""
    origin = dt.date(2025, 12, 31)
    t = quality.aligned_table(root=annual_layer, origin=origin, folds=0).row(0, named=True)
    assert t["n_series"] == 4
    assert t["folds_0"] == 2      # long and short; tiny has no history, stale is two years behind


def test_rolling_the_origin_back_shrinks_the_cohort(annual_layer):
    origin = dt.date(2025, 12, 31)
    t = quality.aligned_table(root=annual_layer, origin=origin, folds=3).row(0, named=True)
    # `short` covers 2024-2025, so it is still alignable with the origin at end-2024 (one year of
    # history, 2024, and 2025 left as truth) but not with it at end-2023. `long` survives all four.
    assert (t["folds_0"], t["folds_1"], t["folds_2"], t["folds_3"]) == (2, 2, 1, 1)


def test_aligned_series_returns_the_cohort_itself(annual_layer):
    got = (quality.aligned_series(root=annual_layer, origin=dt.date(2025, 12, 31))
           .select("series_uid").collect()["series_uid"].to_list())
    assert sorted(got) == ["long", "short"]


def test_one_year_of_history_is_the_whole_length_bar(annual_layer):
    """Cross-learning is the premise: a two-point series is admitted, which the univariate-local
    profiles would reject outright."""
    tk = quality.task_table(root=annual_layer, asof=dt.date(2025, 12, 31), until=dt.date(2025, 12, 1))
    # locally trainable means 2 annual horizons, i.e. 12 points: `long` (26) and `stale` (24)
    assert tk.row(0, named=True)["trainable"] == 2
    assert tk.row(0, named=True)["real_time"] == 1        # `stale` ended two years ago
    t = quality.aligned_table(root=annual_layer, origin=dt.date(2025, 12, 31), folds=0)
    # the aligned cohort swaps that around: `short` has two points and is in, `stale` is out
    assert t.row(0, named=True)["folds_0"] == 2
