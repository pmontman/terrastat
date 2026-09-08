"""The opt-in FRED curation procedure: data-driven cutoff, window filters, forward fill."""
import datetime as dt

import polars as pl
import pytest

from terrastat import curate


def _months(start, n):
    y, m = start
    return [dt.date(y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1, 1) for i in range(n)]


def _series(uid, dates, values, frequency="M", **meta):
    row = {"series_uid": uid, "frequency": frequency, "dates": dates, "values": values,
           "source": "fred", "title": f"series {uid}"}
    row.update(meta)
    return row


def _frame(rows):
    return pl.DataFrame(rows, schema_overrides={"dates": pl.List(pl.Date),
                                                "values": pl.List(pl.Float64)})


@pytest.fixture
def panel():
    """Nine monthly series laid out so the most inclusive cutoff is in the middle of the record --
    neither the earliest candidate nor the latest -- and so that each later filter has exactly one
    series to catch."""
    full = _months((2000, 1), 180)                        # 2000-01 .. 2014-12
    mid = _months((2004, 1), 108)                         # 2004-01 .. 2012-12
    rows = [
        # present throughout: covers every candidate from 2004-01 on
        _series("fred:A1", full, [float(i) for i in range(180)]),
        _series("fred:A2", full, [float(i) for i in range(180)]),
        # the bulge: three series that exist only 2004-01..2012-12
        *[_series(f"fred:M{k}", mid, [float(i) for i in range(108)]) for k in (1, 2, 3)],
        # only old history, so it supports early candidates and nothing after 2007-12
        _series("fred:OLD", full[:96], [float(i) for i in range(96)]),
        # only recent history, so it supports nothing before 2013-01
        _series("fred:NEW", full[120:], [float(i) for i in range(60)]),
        # spans the bulge but publishes one month in four
        _series("fred:GAPPY", mid[::4], [float(i) for i in range(len(mid[::4]))]),
        # present throughout, but the one month it never published is the cutoff itself
        _series("fred:EDGE", full[:96] + full[97:],
                [float(i) for i in range(179)]),
        # shorter than the context window, so it can never cover one
        _series("fred:SHORT", full[60:72], [float(i) for i in range(12)]),
    ]
    return _frame(rows)


CUTOFF = dt.date(2008, 1, 1)          # the peak of C(d) for this panel at context=48
WINDOW_START = dt.date(2004, 1, 1)


def _brute_force(bounds, context):
    """The reference's own double loop, transcribed. Any disagreement with `cutoff_curve` is a
    bug in the fast counting, not a difference of method."""
    lo = bounds["first_valid"].min()
    hi = bounds["last_valid"].max()
    cands = pl.date_range(
        pl.Series([lo]).dt.offset_by(f"{context}mo")[0], hi, interval="1mo", eager=True)
    out = []
    for d in cands:
        start = pl.Series([d]).dt.offset_by(f"-{context}mo")[0]
        n = bounds.filter((pl.col("first_valid") <= start) & (pl.col("last_valid") >= d)).height
        out.append(n)
    return pl.DataFrame({"date": cands, "n_series": out})


# -- step 1 -------------------------------------------------------------------------------------

def test_bounds_ignore_leading_and_trailing_missing_values(panel):
    """A trailing null is not the end of the series' evidence, and treating it as one would let
    the procedure impute at the boundary -- exactly what the thesis says it avoids."""
    b = curate.valid_bounds(panel).sort("series_uid")
    by = {r["series_uid"]: r for r in b.iter_rows(named=True)}
    assert by["fred:OLD"]["last_valid"] == dt.date(2007, 12, 1)
    assert by["fred:NEW"]["first_valid"] == dt.date(2010, 1, 1)
    assert by["fred:GAPPY"]["n_valid"] == 27
    # EDGE never published 2008-01, but it has values on both sides, so its bounds span it and it
    # survives the context filter -- which is precisely why the edge rule is a separate pass
    assert by["fred:EDGE"]["first_valid"] == dt.date(2000, 1, 1)
    assert by["fred:EDGE"]["last_valid"] == dt.date(2014, 12, 1)


def test_the_fast_curve_equals_the_reference_double_loop(panel):
    """SHORT is in the panel for this: it is shorter than the context, so it covers no candidate.
    Counting starts and ends without noticing would close its interval before opening it and push
    C(d) below zero between the two -- which is what happened on the real corpus."""
    bounds = curate.valid_bounds(panel)
    fast = curate.cutoff_curve(bounds, "M", context=48)
    slow = _brute_force(bounds, 48)
    assert fast.equals(slow)
    assert fast["n_series"].min() >= 0


def test_the_cutoff_is_the_most_inclusive_date_not_the_latest(panel):
    """The latest candidate (2010-12) is covered by fewer series than the peak, which is the
    whole argument for searching rather than picking a date by hand."""
    bounds = curate.valid_bounds(panel)
    cutoff, window_start, curve = curate.best_cutoff(bounds, "M", context=48)
    peak = curve["n_series"].max()
    at_last = curve.filter(pl.col("date") == curve["date"].max())["n_series"][0]
    assert curve.filter(pl.col("date") == cutoff)["n_series"][0] == peak
    assert peak > at_last
    assert window_start == pl.Series([cutoff]).dt.offset_by("-48mo")[0]


def test_a_tie_resolves_to_the_earliest_date(panel):
    """`idxmax` in the reference returns the first maximum; a plateau means those dates are
    equally inclusive, so the earliest leaves the most future data for whatever comes next."""
    bounds = curate.valid_bounds(panel)
    cutoff, _, curve = curate.best_cutoff(bounds, "M", context=48)
    peak = curve["n_series"].max()
    assert cutoff == curve.filter(pl.col("n_series") == peak)["date"].min()


def test_context_is_counted_in_the_frequency_own_periods():
    """48 for a quarterly series means 48 quarters, not 48 months -- the reference is monthly
    only, and reading its constant as months would silently shorten every other frequency."""
    q = pl.date_range(dt.date(1990, 1, 1), dt.date(2020, 10, 1), interval="3mo", eager=True)
    df = _frame([_series("fred:Q", q.to_list(), [float(i) for i in range(len(q))], frequency="Q")])
    cutoff, window_start, _ = curate.best_cutoff(curate.valid_bounds(df), "Q", context=48)
    assert (cutoff.year - window_start.year) * 12 + cutoff.month - window_start.month == 144


def test_a_frequency_with_no_regular_grid_is_refused():
    df = _frame([_series("x:1", [dt.date(2020, 1, 1)], [1.0], frequency="B")])
    with pytest.raises(ValueError, match="no regular grid"):
        curate.curate(df)


# -- steps 2 and 3 ------------------------------------------------------------------------------

def test_unpublished_periods_become_missing_before_anything_is_measured(panel):
    """The adaptation that matters. terrastat stores only the points a source published, so a
    skipped month is absent rather than null; without reindexing onto the grid every series would
    look 0% missing and both NaN filters would be no-ops. GAPPY publishes one month in four."""
    res = curate.curate(panel, context=48)
    assert "fred:GAPPY" not in res.frame["series_uid"].to_list()
    row = res.report.filter(pl.col("frequency") == "M").row(0, named=True)
    assert row["with_context"] > row["after_nan"]


def test_a_series_missing_at_the_cutoff_is_dropped(panel):
    """EDGE has a value in every month except the cutoff. It covers the window and its missingness
    is one part in forty-nine, so only the edge rule catches it."""
    res = curate.curate(panel, context=48)
    assert (res.windows["M"]["cutoff"], res.windows["M"]["window_start"]) == (CUTOFF, WINDOW_START)
    assert "fred:EDGE" not in res.frame["series_uid"].to_list()
    row = res.report.filter(pl.col("frequency") == "M").row(0, named=True)
    assert row["after_nan"] - row["after_edge"] == 1


def test_interior_gaps_are_forward_filled_and_nothing_is_left_missing(panel):
    """One month missing in the middle is filled from the month before -- no look-ahead."""
    full = _months((2000, 1), 180)
    vals = [float(i) for i in range(180)]
    holed = _frame([
        _series("fred:A1", full, vals),
        _series("fred:HOLE", full[:30] + full[31:], vals[:30] + vals[31:]),
    ])
    res = curate.curate(holed, context=48)
    got = res.frame.filter(pl.col("series_uid") == "fred:HOLE").row(0, named=True)
    i = got["dates"].index(full[30])
    assert got["values"][i] == got["values"][i - 1]      # carried forward, not interpolated
    assert curate.check(res)["nulls_left"].to_list() == [0]
    assert curate.check(res)["context_violations"].to_list() == [0]


def test_fill_none_leaves_the_gaps_for_you_to_model(panel):
    full = _months((2000, 1), 180)
    vals = [float(i) for i in range(180)]
    holed = _frame([_series("fred:HOLE", full[:30] + full[31:], vals[:30] + vals[31:])])
    res = curate.curate(holed, context=48, fill="none")
    got = res.frame.row(0, named=True)
    assert got["values"][got["dates"].index(full[30])] is None
    assert res.report["nulls_left"][0] == 1


def test_the_window_is_exactly_context_plus_one_periods(panel):
    res = curate.curate(panel, context=48)
    assert res.windows["M"]["n_periods"] == 49
    lengths = res.frame["n_points"].unique().to_list()
    assert lengths == [49]
    for row in res.frame.iter_rows(named=True):
        assert row["dates"][0] == res.windows["M"]["window_start"]
        assert row["dates"][-1] == res.windows["M"]["cutoff"]


def test_metadata_survives_the_rewrite(panel):
    res = curate.curate(panel, context=48)
    assert set(res.frame.columns) >= {"series_uid", "frequency", "source", "title",
                                      "dates", "values", "n_points", "n_obs"}
    assert res.frame.filter(pl.col("series_uid") == "fred:A1")["title"][0] == "series fred:A1"


# -- where the thesis and the reference code part ways ------------------------------------------

def test_reference_settings_keep_the_history_before_the_window(panel):
    """The code trims each series to [first_valid_i, cutoff], the thesis to the window. Under the
    reference settings a series that starts in 2000 keeps all of it."""
    thesis = curate.curate(panel, context=48)
    ref = curate.curate(panel, context=48, **curate.REFERENCE_CODE)
    long_ref = ref.frame.filter(pl.col("series_uid") == "fred:A1").row(0, named=True)
    assert long_ref["dates"][0] == dt.date(2000, 1, 1) < WINDOW_START
    assert thesis.frame["n_points"].max() == 49 < ref.frame["n_points"].max()


def test_the_edge_rule_differs_at_the_start_of_the_window():
    """`edges="cutoff"` (the code) only looks at the last position; `edges="both"` (the thesis)
    also drops a series whose window opens on a missing value."""
    full = _months((2000, 1), 132)
    early = _months((1998, 1), 156)                    # starts two years before the others
    # the cutoff lands on 2004-01, so the window opens at 2000-01 -- the one month BEGIN skipped.
    # Its history reaches back to 1998, so it passes the context filter and only the edge rule
    # has anything to say about it.
    i = early.index(dt.date(2000, 1, 1))
    df = _frame([
        _series("fred:A1", full, [float(k) for k in range(132)]),
        _series("fred:BEGIN", early[:i] + early[i + 1:], [float(k) for k in range(155)]),
    ])
    both = curate.curate(df, context=48, edges="both")
    only_cutoff = curate.curate(df, context=48, edges="cutoff")
    assert "fred:BEGIN" not in both.frame["series_uid"].to_list()
    assert "fred:BEGIN" in only_cutoff.frame["series_uid"].to_list()
    # and with no past inside the window to copy from, forward fill cannot repair it
    kept = only_cutoff.frame.filter(pl.col("series_uid") == "fred:BEGIN").row(0, named=True)
    assert kept["values"][0] is None


def test_nan_scope_changes_which_series_the_ratio_filter_sees():
    """The code measures missingness over each series' own valid range, the thesis over the
    window. A series that is dense early and sparse recently passes one and fails the other."""
    full = _months((2000, 1), 132)
    dates = full[:84] + full[84::3]                 # dense to 2006-12, then one month in three
    df = _frame([
        # only recent history, so the cutoff is pulled into the sparse stretch
        _series("fred:A1", full[60:], [float(i) for i in range(72)]),
        _series("fred:SPARSE_LATE", dates, [float(i) for i in range(len(dates))]),
    ])
    win = curate.curate(df, context=48, nan_scope="window")
    rng = curate.curate(df, context=48, nan_scope="valid_range")
    assert "fred:SPARSE_LATE" not in win.frame["series_uid"].to_list()
    assert "fred:SPARSE_LATE" in rng.frame["series_uid"].to_list()


def test_an_impossible_setting_says_so_rather_than_guessing(panel):
    with pytest.raises(ValueError, match="nan_scope"):
        curate.curate(panel, nan_scope="whatever")


# -- it really is opt-in ------------------------------------------------------------------------

def test_nothing_in_the_pipeline_reaches_for_it():
    """The user asked for this explicitly as a function, not a default. If a later change wires
    it into the build or the CLI, this fails and the choice becomes deliberate again."""
    import inspect

    from terrastat import cli, pipeline, series
    for module in (cli, pipeline, series):
        assert "curate" not in inspect.getsource(module)
