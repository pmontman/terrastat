"""Scoring series for information, and combining recency / length / information into one choice."""
import datetime as dt
import math

import polars as pl
import pytest

from terrastat import selection

TODAY = dt.date(2026, 9, 1)

SCHEMA = {
    "series_uid": pl.String, "source": pl.String, "dataset_id": pl.String, "frequency": pl.String,
    "start_date": pl.Date, "end_date": pl.Date,
    "n_points": pl.Int64, "n_obs": pl.Int64, "n_flagged": pl.Int64,
    "dates": pl.List(pl.Date), "values": pl.List(pl.Float64), "flags": pl.List(pl.String),
}


def _months(end, n):
    out, y, m = [], end.year, end.month
    for _ in range(n):
        out.append(dt.date(y, m, 1))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def _row(uid, values, end=TODAY, dataset="d1"):
    dates = _months(end, len(values))
    return {
        "series_uid": uid, "source": "fred", "dataset_id": dataset, "frequency": "M",
        "start_date": dates[0], "end_date": dates[-1],
        "n_points": len(values), "n_obs": sum(v is not None for v in values), "n_flagged": 0,
        "dates": dates, "values": values, "flags": [None] * len(values),
    }


@pytest.fixture
def layer(tmp_path):
    """Monthly series with information content known by construction.

    ramp      60 distinct, linear     entropy = log2(60), but one distinct difference
    curved    60 distinct, varying    the only one that should survive every floor
    constant  one value repeated      entropy = 0
    step      two values, 30 each     entropy = 1 bit exactly
    stale     varying but old         gives the recency axis something to separate
    short     24 distinct values      gives the length axis something to separate
    """
    sq = [float(i * i) for i in range(60)]
    rows = [
        _row("ramp", [float(i) for i in range(60)]),
        _row("curved", sq),
        _row("constant", [7.0] * 60),
        _row("step", [1.0] * 30 + [2.0] * 30),
        _row("stale", sq, end=dt.date(2020, 9, 1)),
        _row("short", [float(i * i) for i in range(24)]),
    ]
    d = tmp_path / "series" / "fred" / "freq=M"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(d / "x.parquet")
    return tmp_path


def _score(layer, **kw):
    selection.score(root=layer / "series", asof=TODAY, out=layer / "scores",
                    where=pl.lit(True), **kw)
    return selection.load_scores(layer / "scores").collect().sort("series_uid")


def test_entropy_matches_the_hand_computed_value(layer):
    s = _score(layer)
    by = {r["series_uid"]: r for r in s.iter_rows(named=True)}
    assert by["constant"]["entropy_bits"] == pytest.approx(0.0)
    assert by["step"]["entropy_bits"] == pytest.approx(1.0)          # two values, equally often
    assert by["ramp"]["entropy_bits"] == pytest.approx(math.log2(60))
    # normalised: 1.0 when every value is distinct, 0 for a constant
    assert by["ramp"]["entropy_norm"] == pytest.approx(1.0)
    assert by["constant"]["entropy_norm"] == pytest.approx(0.0)


def test_repeat_and_difference_counts_catch_what_entropy_misses(layer):
    by = {r["series_uid"]: r for r in _score(layer).iter_rows(named=True)}
    # a ramp has maximal entropy but exactly one distinct first difference
    assert by["ramp"]["n_diff_unique"] == 1
    assert by["ramp"]["repeat_share"] == pytest.approx(0.0)
    assert by["curved"]["n_diff_unique"] == 59      # every difference distinct
    # the step repeats within each level
    assert by["step"]["n_repeat"] == 58
    assert by["constant"]["repeat_share"] == pytest.approx(1.0)


def test_scoring_resumes_instead_of_redoing_work(layer):
    first = selection.score(root=layer / "series", asof=TODAY, out=layer / "scores", where=pl.lit(True))
    again = selection.score(root=layer / "series", asof=TODAY, out=layer / "scores", where=pl.lit(True))
    assert first["files_read"] == 1 and first["series_scored"] == 6
    assert again["files_read"] == 0 and again["files_skipped"] == 1


def test_the_filter_runs_before_the_values_are_read(layer):
    """`where` is the whole reason this is affordable: it must exclude series before the
    expensive columns are touched, not after."""
    res = selection.score(root=layer / "series", asof=TODAY, out=layer / "scores",
                          where=pl.col("series_uid") == "curved")
    assert res["series_scored"] == 1


def test_degenerate_series_rank_last_instead_of_being_discarded(layer):
    """Nothing is filtered out: a degenerate series simply sits at the bottom of the information
    axis, so one threshold governs the whole selection instead of two sets of numbers."""
    ranked = selection.add_ranks(_score(layer))
    order = ranked.sort("rank_information").get_column("series_uid").to_list()
    assert order[0] == "constant"                       # entropy 0, repeats everywhere
    assert order.index("ramp") < order.index("curved")  # interpolation below real variation
    assert order[-1] in ("curved", "stale", "short")


def test_the_optional_floor_still_works_when_asked_for(layer):
    kept = _score(layer).filter(selection.information_floor())["series_uid"].to_list()
    assert "constant" not in kept
    assert "curved" in kept and "stale" in kept


def test_a_constant_axis_is_dropped_rather_than_vetoing_everything(layer):
    """Series published on one schedule share an end date exactly. Under a minimum, an axis that
    cannot separate anything would cap every series at 0.5 and make a conjunctive threshold
    select nothing, so it is excluded instead."""
    s = _score(layer).filter(pl.col("series_uid") != "stale")   # the rest all end today
    rep = selection.axis_report(s, selection.AXES).to_dicts()
    assert {r["axis"]: r["discriminates"] for r in rep}["recency"] is False

    ranked = selection.add_ranks(s)
    assert ranked.get_column("rank_axes")[0] == "length+information"   # recency dropped
    assert "rank_recency" not in ranked.columns


def test_axis_report_gives_the_attainable_ceiling_not_the_theoretical_one(layer):
    """With ties at the top, average ranking puts the best block at the middle of its own range,
    so the highest reachable percentile is below 1 and a threshold above it selects nothing."""
    s = _score(layer)
    rep = {r["axis"]: r for r in selection.axis_report(s, selection.AXES).to_dicts()}
    # five of the six series are 60 points long, so the top block of ties sits at ranks 2-6 and
    # averages 4, i.e. a ceiling of 0.67: a 0.7 threshold on this axis would select nothing
    # the report rounds to four decimals for legibility, hence the tolerance
    assert rep["length"]["max_tie_share"] == pytest.approx(5 / 6, abs=1e-4)
    assert rep["length"]["rank_ceiling"] == pytest.approx(4 / 6, abs=1e-4)


def test_pick_applies_the_threshold_on_every_ranked_axis(layer):
    s = _score(layer)
    ranked = selection.add_ranks(s)
    chosen = selection.pick(ranked, min_rank=0.0)
    assert chosen.height == s.height                       # a zero threshold keeps everything
    assert chosen.get_column("rank_min").is_sorted(descending=True)


# -- order-aware features -----------------------------------------------------------------------

@pytest.fixture
def shapes(tmp_path):
    """Six shapes chosen so that each feature is the only one that separates a particular pair."""
    n = 60
    rows = [
        _row("smooth", [float(i * i) for i in range(n)]),
        _row("noise", [float((-1) ** i * (1 + (i % 7) * 0.1)) for i in range(n)]),
        _row("ramp", [float(i) for i in range(n)]),
        _row("carried_forward", [float(i * i) for i in range(20)] + [361.0] * 40),
        _row("alternating", [1.0, 2.0] * (n // 2)),
    ]
    d = tmp_path / "series" / "fred" / "freq=M"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(d / "x.parquet")
    return tmp_path


def test_order_aware_features_see_what_entropy_cannot(shapes):
    by = {r["series_uid"]: r for r in _score(shapes).iter_rows(named=True)}
    # smooth and alternating both take many distinct values, so entropy cannot separate them
    assert by["smooth"]["entropy_norm"] == pytest.approx(1.0)
    # but autocorrelation and turning points do, decisively
    assert by["smooth"]["acf1"] > 0.9
    assert by["alternating"]["acf1"] < -0.9
    assert by["smooth"]["turning_share"] == pytest.approx(0.0)
    assert by["alternating"]["turning_share"] == pytest.approx(1.0)
    assert by["noise"]["turning_share"] > 0.5


def test_a_carried_forward_tail_is_caught_only_by_the_run_length(shapes):
    """Two thirds of this series is one repeated figure, yet it still looks healthy on entropy and
    on autocorrelation. The longest run is what gives it away."""
    by = {r["series_uid"]: r for r in _score(shapes).iter_rows(named=True)}
    cf = by["carried_forward"]
    assert cf["entropy_norm"] > 0.3          # would pass an entropy floor
    assert cf["acf1"] > 0.9                  # and an autocorrelation one
    assert cf["flat_run_share"] == pytest.approx(41 / 60)   # the run starts at the last real point
    assert not _score(shapes).filter(
        (pl.col("series_uid") == "carried_forward") & selection.information_floor()
    ).height
    # and it ranks below the series it would otherwise be indistinguishable from
    ranked = selection.add_ranks(_score(shapes))
    r = {x["series_uid"]: x["rank_information"] for x in ranked.iter_rows(named=True)}
    assert r["carried_forward"] < r["smooth"]


def test_a_linear_ramp_ranks_low_despite_looking_perfect(shapes):
    """A ramp has maximal entropy, perfect autocorrelation, no repeats and no runs. Only the
    variety of its first differences gives it away as interpolation rather than observation."""
    scored = _score(shapes)
    by = {r["series_uid"]: r for r in scored.iter_rows(named=True)}
    assert by["ramp"]["entropy_norm"] == pytest.approx(1.0)
    assert by["ramp"]["acf1"] > 0.99
    assert by["ramp"]["repeat_share"] == pytest.approx(0.0)
    assert by["ramp"]["n_diff_unique"] == 1
    assert by["ramp"]["diff_variety"] < 0.02
    # so the composite puts it below the genuinely varying series
    ranked = selection.add_ranks(scored)
    r = {x["series_uid"]: x["rank_information"] for x in ranked.iter_rows(named=True)}
    assert r["ramp"] < r["smooth"] and r["ramp"] < r["noise"]


def test_a_near_constant_axis_is_dropped_too_not_just_an_exactly_constant_one(tmp_path):
    """The case that actually occurs: after a staleness prefilter, the survivors are published on
    a common schedule, so almost all share one lag. On FRED that was 98.3% of 128,699 series on a
    single recency value — enough to cap every one of them under a minimum, and to cut a 0.5
    threshold from 32,156 series down to 217, none of it about the data."""
    n = 40
    rows = [_row(f"s{i}", [float(j * j + i) for j in range(20 + i)]) for i in range(n)]
    rows[0] = _row("odd", [float(j * j) for j in range(30)], end=dt.date(2024, 9, 1))
    d = tmp_path / "series" / "fred" / "freq=M"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(d / "x.parquet")
    scored = _score(tmp_path)

    rep = {r["axis"]: r for r in selection.axis_report(scored, selection.AXES).to_dicts()}
    assert rep["recency"]["n_distinct"] == 2            # not constant, so the old rule would keep it
    assert rep["recency"]["max_tie_share"] > 0.9

    ranked = selection.add_ranks(scored)
    assert "recency" not in ranked.get_column("rank_axes")[0]
    # and keeping it would have capped the selection, which is the point of dropping it
    with_it = selection.add_ranks(scored, drop_constant=False)
    assert selection.pick(with_it, min_rank=0.7).height < selection.pick(ranked, min_rank=0.7).height
