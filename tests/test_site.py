"""The generated project page: same numbers as the annex, and readable in either theme."""
import datetime as dt
import re

import polars as pl
import pytest

from terrastat import corpus, site
from tests.test_corpus import corpus_root  # noqa: F401  (the miniature corpus fixture)


@pytest.fixture
def tables(corpus_root):  # noqa: F811
    return corpus.compute()


def test_standalone_is_a_whole_document_and_the_fragment_is_not(tables):
    """`docs/index.html` has to work opened from disk and served by Pages, so it needs the
    skeleton. A host that supplies its own must not get a second one."""
    doc = site.render(tables, standalone=True)
    frag = site.render(tables, standalone=False)
    assert doc.startswith("<!doctype html>") and "<body>" in doc and doc.rstrip().endswith("</html>")
    for tag in ("<!doctype", "<html", "<head>", "<body>"):
        assert tag not in frag.lower()
    assert "<style>" in frag and "<title>" in frag


def test_every_colour_it_uses_is_defined_in_the_unthemed_root(tables):
    """The classic unreadable-page bug: a token defined only inside a dark-mode media query, so
    the default 'system' setting renders one theme's text on the other theme's ground."""
    css = site.CSS
    first_root = css[css.index(":root {"): css.index("}", css.index(":root {"))]
    defined = set(re.findall(r"(--[a-z-]+):", first_root))
    used = set(re.findall(r"var\((--[a-z-]+)", css))
    assert used <= defined, f"never defined on bare :root: {sorted(used - defined)}"
    assert "background: var(--paper)" in css                 # body paints its own ground


def test_both_theme_overrides_are_present_and_guarded(tables):
    """An explicit light choice must beat a dark OS, and the toggle must win in both directions."""
    assert ':root:not([data-theme="light"])' in site.CSS
    assert ':root[data-theme="dark"]' in site.CSS


def test_exact_and_sampled_rows_are_marked_as_such(tables):
    """The one distinction the page is built around: a sampled share must never be readable as a
    population count. It is carried by a class, not only by prose."""
    html = site.render(tables, standalone=False)
    assert 'class="exact"' in html and 'class="sampled"' in html
    assert "exact, over every series" in html and "estimated from a sample" in html
    # the sampled table says so in its own caption too
    assert "Estimated from" in html


def test_the_figures_match_the_tables_they_came_from(tables):
    html = site.render(tables, standalone=False)
    total = int(tables["length"]["n_series"].sum())
    assert corpus._n(total) in html
    long_total = int(tables["length"]["n_long"].fill_null(0).sum())
    assert corpus._n(long_total) in html


def test_every_frequency_appears_with_its_code(tables):
    html = site.render(tables, standalone=False)
    for freq in tables["length"]["frequency"].to_list():
        assert f'<span class="chip">{freq}</span>' in html


def test_a_frequency_with_no_period_says_so_rather_than_zero(tables):
    html = site.render(tables, standalone=False)
    assert "n/a" in html          # OTHER has no periods per year, so no length bar applies


def test_bars_are_log_scaled_and_stay_inside_their_cell(tables):
    """Counts span seven orders of magnitude here; a linear bar is one full row and ten invisible
    ones. Widths must still be percentages within range."""
    html = site.render(tables, standalone=False)
    widths = [float(w) for w in re.findall(r"width:([\d.]+)%", html)]
    assert widths and all(0 < w <= 100 for w in widths)
    assert "log-scaled" in html


def test_the_licence_band_is_present_and_names_the_restriction(tables):
    """The user asked for data permissions to be impossible to miss. FRED's restriction on
    archiving and ML use is the one that actually bites, so it must be stated, not implied."""
    html = site.render(tables, standalone=False)
    assert "The code is Apache-2.0. The data is not." in html
    assert "machine-learning use" in html and "--public-only" in html
    assert "Not affiliated with" in html


def test_it_links_back_to_the_repository(tables):
    html = site.render(tables, standalone=False, repo="https://example.invalid/x")
    assert html.count("https://example.invalid/x") >= 2


def test_nothing_unformatted_leaks_into_the_output(tables):
    html = site.render(tables, standalone=True)
    assert ">None<" not in html and "nan" not in html.lower().split("<style>")[0]
    assert "{" not in html.split("<style>")[0]                # no stray f-string braces in markup


def test_writing_it_produces_a_file(tables, tmp_path):
    p = site.write(tables, tmp_path / "docs" / "index.html")
    assert p.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_the_command_writes_both_the_page_and_the_annex(corpus_root, tmp_path):  # noqa: F811
    md, page = tmp_path / "corpus.md", tmp_path / "index.html"
    assert corpus.main(["--out", str(md), "--html", str(page)]) == 0
    assert md.read_text(encoding="utf-8").startswith("# The corpus at a glance")
    assert "<!doctype html>" in page.read_text(encoding="utf-8")


def test_no_html_skips_the_page(corpus_root, tmp_path):  # noqa: F811
    md, page = tmp_path / "corpus.md", tmp_path / "index.html"
    assert corpus.main(["--out", str(md), "--html", str(page), "--no-html"]) == 0
    assert md.exists() and not page.exists()


def test_it_still_renders_when_the_expensive_passes_are_skipped(corpus_root):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    html = site.render(t, standalone=False)
    assert "How much data is in it" in html
    assert 'class="exact"' in html


# -- staying generated -----------------------------------------------------------------------------

def test_the_tables_survive_a_round_trip_through_json(tables, tmp_path):
    """The committed figures are what lets a checkout with no corpus re-render the pages, so they
    have to come back exactly -- dates included, which JSON has no type for."""
    import datetime as dt

    p = corpus.save_tables(tables, tmp_path / "corpus_tables.json")
    back = corpus.load_tables(p)
    assert set(back) == set(tables)
    assert back["length"]["asof"].dtype == pl.Date
    assert isinstance(back["length"]["asof"][0], dt.date)
    for name, df in tables.items():
        if not isinstance(df, pl.DataFrame):          # the generation stamp rides along as a string
            assert back[name] == df
            continue
        assert back[name].height == df.height
        if "frequency" in df.columns:
            assert back[name]["frequency"].to_list() == df["frequency"].to_list()
    # and re-rendering from the round-tripped tables reproduces both pages byte for byte
    assert site.render(back) == site.render(tables)
    args = ("length", "values", "concentration", "crawl")
    assert corpus.render(*(back[k] for k in args)) == corpus.render(*(tables[k] for k in args))


def test_check_passes_on_what_the_command_just_wrote(corpus_root, tmp_path):  # noqa: F811
    md, page, tb = tmp_path / "corpus.md", tmp_path / "index.html", tmp_path / "t.json"
    assert corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb)]) == 0
    assert corpus.check(md, page, tb) == []
    assert corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb), "--check"]) == 0


def test_check_catches_a_hand_edited_page(corpus_root, tmp_path):  # noqa: F811
    """The failure this exists for: someone improves a sentence in the generated HTML, and the
    next regeneration silently throws it away. Better to fail loudly at review time."""
    md, page, tb = tmp_path / "corpus.md", tmp_path / "index.html", tmp_path / "t.json"
    corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb)])
    page.write_text(page.read_text(encoding="utf-8").replace("terrastat", "terrastat!"),
                    encoding="utf-8")
    problems = corpus.check(md, page, tb)
    assert len(problems) == 1 and "index.html" in problems[0] and "terrastat corpus" in problems[0]
    assert corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb), "--check"]) == 1


def test_check_catches_a_renderer_change_that_was_not_regenerated(corpus_root, tmp_path, monkeypatch):  # noqa: F811
    """The other failure: the stylesheet or the copy changes, the code is committed, and the pages
    it generates are not. Continuous integration has no corpus, but it can still re-render from the
    committed figures and notice."""
    md, page, tb = tmp_path / "corpus.md", tmp_path / "index.html", tmp_path / "t.json"
    corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb)])
    monkeypatch.setattr(site, "REPO", "https://example.invalid/moved")
    assert any("index.html" in p for p in corpus.check(md, page, tb))


def test_check_says_so_when_a_page_is_missing(corpus_root, tmp_path):  # noqa: F811
    md, page, tb = tmp_path / "corpus.md", tmp_path / "index.html", tmp_path / "t.json"
    corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb)])
    page.unlink()
    assert any("missing" in p for p in corpus.check(md, page, tb))


def test_the_byline_is_a_parameter_not_a_literal(tables):
    """The name in the footer has to be removable -- an anonymous submission, or a fork that is
    not this one -- without editing the renderer."""
    assert site.AUTHOR not in site.render(tables, author="Someone Else")
    assert "Someone Else" in site.render(tables, author="Someone Else")
    plain = site.render(tables, author="")
    assert site.AUTHOR not in plain
    assert "code Apache-2.0" in plain          # the rest of the footer survives


def test_no_affiliation_is_claimed_anywhere_on_the_page(tables):
    """Removing the university is a request, not a preference: the page must not imply the work
    is institutionally endorsed."""
    html = site.render(tables)
    assert "University" not in html and "Sydney" not in html


def test_render_only_rewrites_the_pages_without_reading_the_corpus(corpus_root, tmp_path, monkeypatch):  # noqa: F811
    """Recomputing is minutes over a billion series; wording and layout change far more often than
    the data does, so they must not share a cost."""
    md, page, tb = tmp_path / "corpus.md", tmp_path / "index.html", tmp_path / "t.json"
    corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb)])
    before = tb.read_text(encoding="utf-8")

    def explode(*a, **k):
        raise AssertionError("--render must not touch the corpus")

    monkeypatch.setattr(corpus, "compute", explode)
    monkeypatch.setattr(site, "AUTHOR", "Someone Else")
    assert corpus.main(["--out", str(md), "--html", str(page), "--tables", str(tb), "--render"]) == 0
    assert "Someone Else" in page.read_text(encoding="utf-8")
    assert tb.read_text(encoding="utf-8") == before      # the figures are untouched


def test_the_headline_figures_need_no_glossary(tables):
    """`effective datasets` is a perplexity: worth knowing, unreadable at a glance. It belongs in
    the table whose caption explains it, not in the six figures at the top."""
    html = site.render(tables, standalone=False)
    band = html[html.index('class="figures"'): html.index('class="legend"')]
    assert "effective" not in band and "perplexity" not in band
    assert "series in total" in band and "observations in total" in band
    assert band.count('class="v"') == 6
    # and it is still explained where it does appear
    assert "perplexity" in html


def test_the_lede_says_what_the_licence_columns_buy_you(tables):
    """The thing practitioners actually value here is not that licences are recorded but that
    acting on them is a filter rather than an afternoon of reading."""
    html = site.render(tables, standalone=False)
    lede = html[html.index('class="lede"'): html.index('class="status"')]
    assert "licence" in lede and "one-line filter" in lede
    assert "train and evaluate models" in lede


def test_the_sections_run_corpus_then_python_then_the_rest(tables):
    """Size and length, then the code, then everything else. The two deep-dive tables are worth
    having and are not what a reader needs before they can try it."""
    html = site.render(tables, standalone=False)
    order = [html.index(f'id="{name}"') for name in ("corpus", "start", "what", "licence")]
    assert order == sorted(order)
    assert html.index("How much data is in it") < html.index("<section id=\"start\"")
    assert html.index("What the values look like") > html.index("<section id=\"start\"")
    assert html.index("How concentrated it is") > html.index("<section id=\"start\"")
    nav = html[html.index("<nav>"): html.index("</nav>")]
    assert nav.index("#corpus") < nav.index("#start") < nav.index("#what")


def test_the_frequencies_are_ordered_by_what_a_reader_cares_about(tables):
    """Sorting by period opens every table with 14 thousand daily series and buries the 620
    million annual ones. Annual, monthly, quarterly first."""
    html = site.render(tables, standalone=False)
    chips = re.findall(r'class="chip">([A-Z0-9]+)<', html)
    assert chips[:3] == ["A", "M", "Q"]


def test_reordering_takes_effect_without_recomputing(tables, tmp_path, monkeypatch):
    """Row order is presentation. Changing it must be a re-render, not a fresh pass over a billion
    series -- which is only true if the sort happens when rendering, not when computing."""
    # a permutation, not a prefix: `{f: i for i, f in enumerate(...)}` lets a later duplicate win
    monkeypatch.setattr(corpus, "FREQ_ORDER",
                        ["Q"] + [f for f in corpus.FREQ_ORDER if f != "Q"])
    chips = re.findall(r'class="chip">([A-Z0-9]+)<', site.render(tables, standalone=False))
    assert chips[0] == "Q"


def test_the_length_columns_say_what_they_measure(tables):
    """`p5 / median / p95` over three bare columns does not say *of what*, and the one reading that
    must not happen is "number of series"."""
    html = site.render(tables, standalone=False)
    assert "how long one series is, in observations" in html
    assert "shortest 5%" in html and "longest 5%" in html


def test_recency_is_labelled_in_periods_not_months(tables):
    """The metric is 12 periods of each series' own frequency, so calling it "12 months" would be
    wrong for every frequency but monthly -- and an annual series stamped 2025-01-01 is not a year
    stale on 2026-09-08, it is one period old. The figure stays plain; the legend is exact."""
    html = site.render(tables, standalone=False)
    band = html[html.index('class="figures"'): html.index('class="legend"')]
    assert "still being published" in band
    assert "months" not in band and "periods" not in band
    legend = html[html.index('class="legend"'): html.index("</div>", html.index('class="legend"'))]
    assert "own frequency" in legend and "periods" in legend


# -- the schema section ----------------------------------------------------------------------------

def test_every_schema_column_is_described_exactly_once():
    """The page claims to account for all 36 columns. Adding one to the schema and forgetting the
    page would silently under-describe the data, so the partition is checked rather than trusted."""
    from terrastat.series import SERIES_SCHEMA

    listed = [c for *_, cols in site.SCHEMA_FAMILIES for c in cols]
    assert len(listed) == len(set(listed)), "a column appears in two families"
    assert set(listed) == set(SERIES_SCHEMA)


def test_the_schema_section_names_every_column_and_says_what_each_group_buys(tables):
    from terrastat.series import SERIES_SCHEMA

    html = site.render(tables, standalone=False)
    section = html[html.index('id="schema"'): html.index('id="leakage"')]
    for col in SERIES_SCHEMA:
        assert f'<span class="col">{col}</span>' in section, col
    # the claims each group makes, matched case-insensitively: this is prose, not markup
    low = section.lower()
    for phrase in ("text encoder", "nested geographies", "subset you can share", "vintage"):
        assert phrase in low, phrase
    # the data itself is described before the bookkeeping around it
    assert low.index(">the data<") < low.index(">extent<") < low.index(">provenance<")


def test_the_schema_sits_after_the_code_and_before_the_deep_dives(tables):
    html = site.render(tables, standalone=False)
    assert html.index('id="start"') < html.index('id="schema"') < html.index("What the values look like")


def test_the_specimen_is_a_real_row_with_a_drawn_series(corpus_root):  # noqa: F811
    """A list of column names does not convey that a series arrives already described and already
    placed in a hierarchy. One real row does — including its shape."""
    t = corpus.compute()
    spec = t["specimen"]
    assert spec.height >= 1
    r = spec.row(0, named=True)
    html = site.render(t, standalone=False)
    assert r["series_uid"] in html or r["dataset_title"] in html
    assert '<svg class="spark"' in html
    # the overview comes first; the rows illustrate it, they do not replace it
    assert html.index('class="families"') < html.index('class="rows"')


def test_the_sparkline_stays_inside_its_box(tables):
    html = site.render(tables, standalone=False)
    if '<svg class="spark"' not in html:
        pytest.skip("this fixture has no drawable specimen")
    svg = html[html.index('<svg class="spark"'): html.index("</svg>")]
    box = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    pts = [tuple(map(float, pair.split(","))) for pair in
           re.search(r'points="([^"]+)"', svg).group(1).split()]
    assert all(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in pts)


def test_a_corpus_with_no_usable_specimen_still_renders(tables):
    """The specimen is an illustration, never a reason a page fails to build."""
    t = dict(tables, specimen=pl.DataFrame())
    html = site.render(t, standalone=False)
    assert 'id="schema"' in html and 'class="rows"' not in html


def test_the_specimen_actually_varies(corpus_root):  # noqa: F811
    """The first row this picked was 192 monthly observations of Austrian hatchery chicks, zero
    in every month but one: valid data, and an advertisement for nothing. A specimen has to move."""
    spec = corpus.specimens(min_obs=2, min_unique_share=0.3)
    assert spec.height >= 1
    for r in spec.iter_rows(named=True):
        assert len(set(r["spark"])) > 1
        assert r["unique_share"] >= 0.3


def test_a_flat_series_loses_to_a_varied_one(tmp_path, monkeypatch):
    """Given both, the picker must not take the longer-but-flat one."""
    from tests.test_corpus import _write

    monkeypatch.setenv("TERRASTAT_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    flat = [(300, dt.date(2026, 8, 1), [0.0] * 299 + [0.3])]
    varied = [(120, dt.date(2026, 8, 1), [float(i % 37) + i * 0.5 for i in range(120)])]
    _write(tmp_path, "eurostat", "M", "aaa_flat", flat)
    _write(tmp_path, "eurostat", "M", "zzz_varied", varied)
    got = corpus.specimens(min_obs=100, n=1)
    assert got.height == 1
    assert "zzz_varied" in got.row(0, named=True)["series_uid"], "the flat series won on length"


def test_the_specimens_span_more_than_one_dataset(corpus_root):  # noqa: F811
    """Four rows that are all the same table illustrate nothing about breadth."""
    spec = corpus.compute()["specimen"]
    if spec.height > 1:
        assert spec["dataset_id"].n_unique() == spec.height


def test_each_series_is_drawn_against_its_own_range(tables):
    """Tonnes of fish and an exchange rate share no axis; these are small multiples, and the page
    has to say so or the reader will compare heights across cards."""
    html = site.render(tables, standalone=False)
    if 'class="rows"' in html:
        assert "its own range" in html


def test_the_leakage_section_names_the_revision_trap(tables):
    """The single most costly one for anybody running a non-real-time backtest: only the latest
    vintage is stored, so a 2008 observation is the number as it reads today."""
    html = site.render(tables, standalone=False)
    sec = html[html.index('id="leakage"'):]
    sec = sec[: sec.index("</section>")]
    assert "latest vintage" in sec and "real-time" in sec
    for phrase in ("Publication lag", "Accounting identities", "two sources", "Ill-formed"):
        assert phrase in sec, phrase
    # the user asked for the framing to be "prone to pitfalls", not a claim about what we withheld
    assert "prone to leakage" in sec
    assert "Nothing is cleaned for you" not in sec


def test_leakage_follows_the_schema(tables):
    html = site.render(tables, standalone=False)
    assert html.index('id="schema"') < html.index('id="leakage"')
