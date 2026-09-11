"""The regenerated one-page corpus summary: exact where it claims to be, sampled where it says so."""
import datetime as dt
import pathlib

import polars as pl
import pytest

from terrastat import corpus


def _write(root, source, freq, dataset, rows):
    """One series-layer file: `rows` is a list of (n_obs, end_date, values)."""
    d = root / "series" / source / f"freq={freq}"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series_uid": [f"{source}:{dataset}:{i}" for i in range(len(rows))],
            "source": [source] * len(rows),
            "dataset_id": [dataset] * len(rows),
            "frequency": [freq] * len(rows),
            "n_obs": [r[0] for r in rows],
            "end_date": [r[1] for r in rows],
            "retrieved_at": ["2026-09-05T10:00:00+00:00"] * len(rows),
            "start_date": [dt.date(2000, 1, 1)] * len(rows),
            "units": ["Index"] * len(rows),
            "dataset_title": [dataset.replace("_", " ").title()] * len(rows),
            # the specimen picker needs a written description and labelled dimensions, which is
            # most of what makes a row worth showing
            "description": [f"{dataset} series {i}. Unit of measure: Index. Frequency: {freq}."
                            for i in range(len(rows))],
            "dimensions": [[{"id": "freq", "name": "Time frequency", "code": freq, "label": freq},
                            {"id": "geo", "name": "Geopolitical entity", "code": "AT",
                             "label": "Austria"},
                            {"id": "unit", "name": "Unit of measure", "code": "IDX",
                             "label": "Index"}] for _ in rows],
            "values": [r[2] for r in rows],
        },
        schema_overrides={"end_date": pl.Date, "start_date": pl.Date,
                          "values": pl.List(pl.Float64), "n_obs": pl.Int64},
    ).write_parquet(d / f"{dataset}.parquet")


@pytest.fixture
def corpus_root(tmp_path, monkeypatch):
    """A miniature corpus with one of everything the page has to get right.

    The chdir is not cosmetic. `corpus.main` defaults `--out`, `--html` and `--tables` to paths
    *relative to the working directory*, so any test that forgets one of them writes fixture
    numbers straight into the repository's own `docs/`. That happened twice — once to
    `index.html`, once to `corpus_tables.json` — before this fixture moved the working directory
    somewhere harmless.
    """
    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    live, stale = dt.date(2026, 8, 1), dt.date(2015, 1, 1)
    # monthly: 3 long and live, 1 long but abandoned, 1 too short. One is a flat constant.
    _write(tmp_path, "fred", "M", "big", [
        (60, live, [float(i) for i in range(60)]),
        (60, live, [1.0] * 60),                              # constant
        (24, live, [float(i % 7) for i in range(24)]),       # exactly on the 2-year bar
        (48, stale, [float(i) for i in range(48)]),          # long, not live
        (23, live, [float(i) for i in range(23)]),           # one short of the bar
    ])
    # a second, much smaller monthly dataset, so concentration has something to measure
    _write(tmp_path, "eurostat", "M", "small", [(36, live, [float(-i) for i in range(36)])])
    # annual: the 2-year bar is 2 observations, and single-point series are the common case
    _write(tmp_path, "eurostat", "A", "yearly", [
        (1, live, [5.0]),
        (1, live, [6.0]),
        (4, live, [1.0, 2.0, 3.0, 4.0]),
        (10, stale, [float(i) for i in range(10)]),
    ])
    # the same dataset id also publishes quarterly, which is why the whole-corpus figures cannot
    # be summed out of the per-frequency rows
    _write(tmp_path, "fred", "Q", "big", [(20, live, [float(i) for i in range(20)]),
                                          (20, live, [float(i) for i in range(20)])])
    # a frequency with no fixed period at all
    _write(tmp_path, "oecd", "OTHER", "odd", [(3, live, [1.0, 2.0, 3.0])])
    return tmp_path


# -- exact tables --------------------------------------------------------------------------------

def test_the_length_bar_is_in_the_frequency_own_periods(corpus_root):
    """Two years is 24 observations monthly and 2 annually. Reading the constant as one number
    for every frequency is the mistake this whole table exists to avoid."""
    t = corpus.length_table(asof=dt.date(2026, 9, 5))
    by = {r["frequency"]: r for r in t.iter_rows(named=True)}
    assert by["M"]["min_obs"] == 24 and by["A"]["min_obs"] == 2
    assert by["M"]["n_series"] == 6 and by["M"]["n_long"] == 5      # the 23-point series is out
    assert by["A"]["n_series"] == 4 and by["A"]["n_long"] == 2      # the two single points are out


def test_a_frequency_with_no_period_is_counted_but_not_judged(corpus_root):
    """`OTHER` has no periods per year, so a length in years and a lag in periods are both
    undefined. Reporting a zero there would read as "none qualify" rather than "not applicable"."""
    t = corpus.length_table(asof=dt.date(2026, 9, 5))
    row = t.filter(pl.col("frequency") == "OTHER").row(0, named=True)
    assert row["n_series"] == 1
    assert row["n_long"] is None and row["n_live"] is None and row["min_obs"] is None


def test_liveness_is_measured_against_the_crawl_not_today(corpus_root):
    """Left to default to today's date, every series looks stale the moment a crawl finishes --
    which measures the crawl, not the source."""
    t = corpus.length_table()
    assert t["asof"][0] == dt.date(2026, 9, 5)                     # max retrieved_at, not today
    by = {r["frequency"]: r for r in t.iter_rows(named=True)}
    assert by["M"]["n_long"] == 5 and by["M"]["n_long_live"] == 4   # the 2015 one is not live


def test_length_quantiles_are_exact_and_cover_the_long_cohort_only(corpus_root):
    t = corpus.length_table(asof=dt.date(2026, 9, 5))
    m = t.filter(pl.col("frequency") == "M").row(0, named=True)
    assert m["p5"] == 24 and m["p95"] == 60                        # 23 never enters the spread
    assert m["p50_years"] == pytest.approx(m["p50"] / 12)


def test_the_bar_moves_when_you_move_it(corpus_root):
    assert corpus.length_table(asof=dt.date(2026, 9, 5), years=5).filter(
        pl.col("frequency") == "M")["n_long"][0] == 2               # only the two 60-point series
    assert corpus.min_obs("Q", 2) == 8 and corpus.min_obs("A3", 2) == 2


# -- concentration -------------------------------------------------------------------------------

def test_concentration_names_the_dominant_dataset_and_discounts_it(corpus_root):
    """Five of the six monthly series come from one dataset, so the effective count is far below
    the raw count of two -- which is the point being made."""
    c = corpus.concentration_table()
    m = c.filter(pl.col("frequency") == "M").row(0, named=True)
    assert m["datasets"] == 2 and m["top_dataset"] == "fred:big"
    assert m["top_share"] == pytest.approx(5 / 6)
    assert 1.0 <= m["effective_datasets"] < 2.0


def test_the_whole_corpus_row_is_recomputed_not_summed(corpus_root):
    """A dataset can appear at more than one frequency, so summing the per-frequency rows would
    double-count it. The `*` row is computed over the pooled table instead."""
    c = corpus.concentration_table()
    overall = c.filter(pl.col("frequency") == "*").row(0, named=True)
    per_freq = c.filter(pl.col("frequency") != "*")
    assert overall["n_series"] == int(per_freq["n_series"].sum()) == 13
    # `fred:big` publishes at M and Q, so the rows count it twice and the pooled row does not
    assert int(per_freq["datasets"].sum()) == 5
    assert overall["datasets"] == 4


# -- sampled table -------------------------------------------------------------------------------

def test_value_statistics_exclude_single_point_series(corpus_root):
    """With one observation a series has one distinct value, so it is both 100% unique and 100%
    constant. Leaving the 853 million of them in made every annual number meaningless."""
    v = corpus.value_table()
    a = v.filter(pl.col("frequency") == "A").row(0, named=True)
    assert a["share_one_point"] == pytest.approx(0.5)
    assert a["share_constant"] == 0.0            # neither multi-point annual series is constant


def test_constants_and_signs_are_detected(corpus_root):
    v = corpus.value_table()
    m = v.filter(pl.col("frequency") == "M").row(0, named=True)
    assert m["share_constant"] == pytest.approx(1 / 6)     # the flat series, of six monthly
    assert m["share_nonpositive"] > 0                      # the eurostat series is all negative
    assert 0 < m["unique_share_median"] <= 1


# -- the page ------------------------------------------------------------------------------------

def test_the_page_renders_with_every_section_and_no_placeholders(corpus_root):
    md = corpus.build()
    for heading in ("# The corpus at a glance", "## Headline", "## Length, by frequency",
                    "## What the values look like", "## How concentrated it is",
                    "## Where it came from"):
        assert heading in md
    assert "terrastat corpus --out docs/corpus.md" in md
    assert "Do not edit this file" in md
    assert "{" not in md and "None" not in md               # no unformatted f-string leftovers
    assert md.count("| monthly `M` |") == 3                 # one row in each of the three tables


def test_the_page_says_what_is_sampled_and_what_is_not(corpus_root):
    """The single most misquotable thing here is a sampled share read as a population count."""
    md = corpus.build()
    assert "**Sampled, not exact**" in md
    assert "Exact, over every series." in md


def test_skipping_the_expensive_passes_still_renders(corpus_root):
    md = corpus.build(values=False, concentration=False)
    assert "## Length, by frequency" in md
    assert "| monthly `M` |" in md


def test_the_cli_writes_the_file(corpus_root, tmp_path):
    """Both outputs go where they are told. `--html` defaults to `docs/index.html` *relative to the
    working directory*, so a test that leaves it out overwrites the repository's own page with
    fixture numbers — which is exactly what happened once."""
    out, page = tmp_path / "docs" / "corpus.md", tmp_path / "docs" / "index.html"
    assert corpus.main(["--out", str(out), "--html", str(page)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# The corpus at a glance")
    assert page.exists()


def test_the_defaults_write_under_the_working_directory(corpus_root):
    """With no paths given at all, everything must land in `./docs` — which is why the fixture
    moves the working directory before any test can call this."""
    assert corpus.main([]) == 0
    for name in ("corpus.md", "index.html", "corpus_tables.json"):
        written = corpus_root / "docs" / name
        assert written.exists(), name
        assert written.is_relative_to(corpus_root)
