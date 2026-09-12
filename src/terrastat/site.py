"""The project page: the same computed figures as ``docs/corpus.md``, laid out to be read.

`terrastat corpus` writes two things from one set of tables — a markdown annex for reading inside
the repository, and this page for everyone else. Both are generated, so neither can drift from the
corpus or from each other.

The design carries one idea: **every figure here is either exact over the whole population or
estimated from a sample, and you must always be able to tell which.** Exact figures are set in the
accent blue, sampled ones in ochre, and the legend says so. It is the one distinction that decides
whether a number can be quoted, so it gets the colour system rather than a footnote.

Two outputs from one template. ``standalone=True`` emits a complete document, which is what
``docs/index.html`` needs to work as a file and as a GitHub Pages site. ``standalone=False`` emits
the body only, for hosts that supply their own document skeleton.
"""
from __future__ import annotations

import datetime as dt
import html
import logging
import math
from pathlib import Path

import polars as pl

from terrastat.corpus import (FREQ_NAME, LIVE_PERIODS, MIN_YEARS, _fixed, _n, _num,
                             _order, _pct)
from terrastat.logo import (HEIGHT, WIDTH, animation_script, favicon_svg, globe_svg,
                            mark_svg, wordmark_svg)

log = logging.getLogger(__name__)

REPO = "https://github.com/pmontman/terrastat"
# Whose name appears in the footer. A parameter rather than a literal so the page can be built
# without it -- for an anonymous submission, or for a fork that is not this one.
AUTHOR = "Iker Rosales Saiz and Pablo Montero-Manso"

# The 36 columns, grouped by what each group lets you do. Every column of SERIES_SCHEMA appears in
# exactly one family — a test enforces it, so adding a column to the schema and forgetting the page
# fails loudly instead of silently under-describing the data.
# The 36 columns, in the seven groups docs/schema.md uses, ordered by how much a reader needs
# them: the data itself before the bookkeeping around it. Every column appears exactly once, and a
# test enforces that, so extending the schema and forgetting the page fails instead of quietly
# leaving a column undescribed.
SCHEMA_FAMILIES = [
    ("identity", "keys",
     "The key for a series, and the table it came from. The dataset id is what you group on when "
     "splitting.",
     ["series_uid", "source", "source_id", "dataset_id"]),
    ("human-readable", "text",
     "A description per series, built from all the metadata the publisher gave. You can feed it "
     "to a text encoder alongside the values, or ignore it.",
     ["dataset_title", "title", "description", "notes"]),
    ("classification", "structure",
     "Where the series sits in the published table. Each dimension keeps its code and its label, "
     "so totals, their parts and nested geographies (<code>DE</code>, <code>DE1</code>, "
     "<code>DE11</code>) can be found.",
     ["frequency", "frequency_raw", "units", "seasonal_adjustment", "geo", "geo_label",
      "dimensions", "tags"]),
    ("the data", "numbers",
     "Dates, values and status flags, stored as lists on the row. Missing observations stay as "
     "nulls rather than disappearing, and flags such as provisional or estimated are kept.",
     ["dates", "values", "flags"]),
    ("licence", "licence",
     "The licence, its URL, and the credit the publisher asks for. Filtering on these gives a "
     "subset you can share.",
     ["license_id", "license_name", "license_url", "license_detail", "attribution",
      "license_notes"]),
    ("extent", "numbers",
     "Start, end and counts. They are fixed-width, so a scan that reads only these never touches "
     "the values: a pass over the whole corpus takes about twelve seconds.",
     ["start_date", "end_date", "n_points", "n_obs", "n_flagged"]),
    ("provenance", "keys",
     "Who produced the number, when it was fetched, and which vintage it is. FRED republishes "
     "80,274 OECD series, so the originating agency is worth keeping.",
     ["origin_agencies", "origin_us_federal", "source_url", "last_updated", "retrieved_at",
      "vintage"]),
]


FONTS = ("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;"
         "6..72,600&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap")

# Two colour roles do the structural work: `--exact` for figures computed over every series,
# `--sampled` for figures estimated from a draw. Everything else is ink, paper and a hairline.
CSS = """
:root {
  --paper: #f7f8fa;
  --panel: #ffffff;
  --ink: #151a21;
  --muted: #5c6779;
  --rule: #dce1e9;
  --exact: #2743c4;
  --sampled: #9a6b0b;
  --exact-wash: #eaeefb;
  --sampled-wash: #f7f0e0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0d1016;
    --panel: #141922;
    --ink: #e4e8ef;
    --muted: #8d97a8;
    --rule: #212832;
    --exact: #8aa0ff;
    --sampled: #dfaa43;
    --exact-wash: #171e33;
    --sampled-wash: #241d10;
  }
}
:root[data-theme="dark"] {
  --paper: #0d1016;
  --panel: #141922;
  --ink: #e4e8ef;
  --muted: #8d97a8;
  --rule: #212832;
  --exact: #8aa0ff;
  --sampled: #dfaa43;
  --exact-wash: #171e33;
  --sampled-wash: #241d10;
}

*, *::before, *::after { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Public Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.wrap { max-width: 1080px; margin: 0 auto; padding: 0 28px; }
.prose { max-width: 68ch; }

h1, h2, h3 { font-family: Newsreader, ui-serif, Georgia, serif; font-weight: 500;
  letter-spacing: -0.01em; text-wrap: balance; margin: 0; }
h2 { font-size: 1.75rem; line-height: 1.2; }
h3 { font-size: 1.15rem; line-height: 1.3; }
p { margin: 0 0 1em; }
a { color: var(--exact); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:focus-visible { outline: 2px solid var(--exact); outline-offset: 3px; border-radius: 2px; }
code, .mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
code { font-size: 0.87em; }

.eyebrow {
  font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; margin: 0 0 0.9rem;
}

/* -- masthead: figures under hairline rules, the way a statistical annex opens -- */

header.top {
  border-bottom: 1px solid var(--rule);
  position: sticky; top: 0; z-index: 5;
  background: color-mix(in srgb, var(--paper) 88%, transparent);
  backdrop-filter: blur(8px);
}
header.top .wrap {
  display: flex; align-items: baseline; gap: 1.25rem;
  padding-top: 0.7rem; padding-bottom: 0.7rem;
}
header.top .name { font-family: Newsreader, serif; font-size: 1.1rem; font-weight: 600;
  display: inline-flex; align-items: center; gap: 0.5rem; }
header.top .name svg { display: block; flex: none; }
header.top nav { margin-left: auto; display: flex; gap: 1.1rem; font-size: 0.85rem; }
header.top nav a { color: var(--muted); text-decoration: none; }
header.top nav a:hover { color: var(--ink); text-decoration: underline; }

.hero { padding: 4.5rem 0 2.5rem; display: grid; gap: 1.5rem 3rem; align-items: center; }
@media (min-width: 900px) { .hero { grid-template-columns: minmax(0, 1fr) 400px; }
  .hero .figures, .hero .legend { grid-column: 1 / -1; } }
.hero-globe { margin: 0; color: var(--ink); justify-self: center; max-width: 400px; }
.hero-globe svg { display: block; width: 100%; height: auto; }
.hero-globe figcaption { font-size: 0.72rem; color: var(--muted); margin-top: 0.6rem;
  text-align: center; line-height: 1.4; }
@media (max-width: 899px) { .hero-globe { max-width: 300px; } }
.hero h1 { font-size: clamp(2.6rem, 6vw, 4.1rem); line-height: 1.02; margin-bottom: 1rem; }
.hero .lede { font-size: 1.18rem; color: var(--muted); max-width: 60ch; }
.hero .lede strong { color: var(--ink); font-weight: 500; }
.hero .status {
  margin: 1.4rem 0 0; font-size: 0.9rem; color: var(--muted); max-width: 58ch;
  display: flex; gap: 0.6rem; align-items: baseline;
}
.hero .status .dot {
  flex: none; width: 7px; height: 7px; border-radius: 50%; background: var(--exact);
  transform: translateY(-2px);
}

.figures {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0; margin: 3rem 0 0;
  border-top: 2px solid var(--ink); border-bottom: 1px solid var(--rule);
}
.figures div { padding: 1.1rem 1.2rem 1.2rem 0; }
.figures .v {
  font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums;
  font-size: 1.85rem; line-height: 1; color: var(--exact); font-weight: 500;
}
.figures .k { font-size: 0.78rem; color: var(--muted); margin-top: 0.5rem; line-height: 1.35; }

section { padding: 3.25rem 0; border-bottom: 1px solid var(--rule); }
section:last-of-type { border-bottom: 0; }

/* -- tables: a statistical annex, not a set of cards -- */

.scroller { overflow-x: auto; margin-top: 1.5rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; min-width: 640px; }
caption { text-align: left; color: var(--muted); font-size: 0.85rem; padding-bottom: 0.9rem; }
th, td { padding: 0.5rem 0.8rem 0.5rem 0; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--rule); }
th { font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--ink); }
th.group { text-align: center; text-transform: none; letter-spacing: 0; font-size: 0.76rem;
  color: var(--ink); font-weight: 500; border-bottom: 1px solid var(--rule);
  padding-bottom: 0.35rem; }
th.sub { font-size: 0.68rem; }
thead th { vertical-align: bottom; }
th:first-child, td:first-child { text-align: left; }
td.num { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
td.lead { font-weight: 500; }
tbody tr:hover { background: color-mix(in srgb, var(--exact) 5%, transparent); }
tfoot td { border-bottom: 0; color: var(--muted); font-size: 0.8rem; padding-top: 0.8rem; }

.chip {
  font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; padding: 0.1rem 0.36rem;
  border: 1px solid var(--rule); border-radius: 3px; color: var(--muted); margin-left: 0.45rem;
}
.bar { display: block; height: 6px; background: var(--exact); opacity: 0.75; border-radius: 1px;
  min-width: 2px; }
.bar.dim { background: var(--muted); opacity: 0.35; }
td.barcell { width: 130px; padding-right: 0; }

/* -- exact vs sampled, the one distinction the page is built around -- */

.legend { display: flex; flex-wrap: wrap; gap: 1.4rem; font-size: 0.82rem; color: var(--muted);
  margin: 1.25rem 0 0; padding-top: 1rem; border-top: 1px solid var(--rule); }
.legend span { display: inline-flex; align-items: center; gap: 0.5rem; }
.swatch { width: 10px; height: 10px; border-radius: 2px; }
.swatch.e { background: var(--exact); }
.swatch.s { background: var(--sampled); }
.tag {
  display: inline-block; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
  font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 3px; vertical-align: 0.25em;
  margin-left: 0.7rem;
}
.tag.e { color: var(--exact); background: var(--exact-wash); }
.tag.s { color: var(--sampled); background: var(--sampled-wash); }
.sampled td.num { color: var(--sampled); }
.exact td.num { color: var(--exact); }

/* -- getting started -- */

pre {
  background: var(--panel); border: 1px solid var(--rule); border-left: 2px solid var(--exact);
  padding: 1rem 1.1rem; overflow-x: auto; font-size: 0.83rem; line-height: 1.65;
  font-family: "IBM Plex Mono", monospace; margin: 0 0 1.2rem;
}
pre .c { color: var(--muted); }
.steps { display: grid; gap: 2rem; margin-top: 1.75rem; }
@media (min-width: 800px) { .steps { grid-template-columns: 1fr 1fr; } }
.steps h3 { margin-bottom: 0.6rem; }
.steps p { font-size: 0.92rem; color: var(--muted); }

.facts { display: grid; gap: 1.6rem 2.5rem; margin-top: 1.75rem; }
@media (min-width: 760px) { .facts { grid-template-columns: 1fr 1fr; } }
.facts h3 { font-size: 1rem; margin-bottom: 0.35rem; }
.facts p { font-size: 0.92rem; color: var(--muted); margin: 0; }

/* -- the schema: four layers, seven column groups, four real rows -- */

.families { display: grid; gap: 1px; margin-top: 1.5rem; background: var(--rule);
  border: 1px solid var(--rule); }
@media (min-width: 720px) { .families { grid-template-columns: 1fr 1fr; } }
@media (min-width: 1000px) { .families { grid-template-columns: repeat(3, 1fr); } }
.family { background: var(--panel); padding: 1.1rem 1.2rem 1.2rem; }
.family h3 { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.45rem;
  font-size: 1.02rem; }
.family .count {
  font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: var(--muted);
  border: 1px solid var(--rule); border-radius: 999px; padding: 0.05rem 0.45rem;
}
.family p { font-size: 0.9rem; color: var(--muted); margin: 0 0 0.8rem; }
.family .cols { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.family .col {
  font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; padding: 0.12rem 0.4rem;
  border: 1px solid var(--rule); border-radius: 3px; color: var(--muted);
}
.family.numbers .col, .family.structure .col { color: var(--exact);
  border-color: var(--exact-wash); background: var(--exact-wash); }
.family.licence .col { color: var(--sampled); border-color: var(--sampled-wash);
  background: var(--sampled-wash); }

.specimen-lead { margin: 3rem 0 0; }
.rows { display: grid; gap: 1.5rem; margin-top: 1.5rem; }
@media (min-width: 700px) { .rows { grid-template-columns: 1fr 1fr; } }
.row-card { margin: 0; padding-top: 1rem; border-top: 1px solid var(--ink); color: var(--exact); }
.row-card figcaption { color: var(--ink); font-size: 0.92rem; margin-bottom: 0.7rem;
  display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.35rem; }
.row-card figcaption .chip { margin-left: 0; }
.row-card figcaption strong { font-weight: 500; }
svg.spark { display: block; width: 100%; height: 52px; }
.row-card .extent { font-size: 0.72rem; color: var(--muted); margin: 0.4rem 0 0.7rem; }
.row-card .desc { font-size: 0.85rem; color: var(--muted); margin: 0 0 0.7rem; }
ul.dims { list-style: none; margin: 0; padding: 0; font-size: 0.78rem; color: var(--muted); }
ul.dims li { padding: 0.18rem 0; border-bottom: 1px dotted var(--rule); }
ul.dims li:last-child { border-bottom: 0; }
ul.dims .mono { color: var(--exact); font-size: 0.72rem; margin-right: 0.5rem; }

/* -- curation and leakage -- */

.traps { display: grid; gap: 1.4rem 2.5rem; margin: 1.75rem 0 1.5rem; }
@media (min-width: 780px) { .traps { grid-template-columns: 1fr 1fr; } }
.traps h4 { font-family: Newsreader, ui-serif, Georgia, serif; font-size: 1rem; font-weight: 500;
  margin: 0 0 0.3rem; color: var(--ink); }
.traps p { font-size: 0.88rem; color: var(--muted); margin: 0; }
.traps > div { border-left: 2px solid var(--sampled); padding-left: 0.9rem; }
p.closing { border-top: 1px solid var(--rule); padding-top: 1.2rem; }

/* -- the licensing band, which must not be skimmable past -- */

.licence {
  border: 1px solid var(--sampled); border-left-width: 3px; background: var(--sampled-wash);
  padding: 1.5rem 1.6rem; margin-top: 1.5rem;
}
.licence h3 { color: var(--sampled); margin-bottom: 0.5rem; }
.licence p { margin-bottom: 0.7rem; font-size: 0.93rem; }
.licence p:last-child { margin-bottom: 0; }

footer { padding: 2.5rem 0 4rem; color: var(--muted); font-size: 0.83rem;
  border-top: 1px solid var(--rule); }
footer p { margin-bottom: 0.5rem; }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _row(cells: list[str], klass: str = "") -> str:
    return f'  <tr class="{klass}">' + "".join(cells) + "</tr>"


def _td(text: str, num: bool = True, lead: bool = False) -> str:
    cls = " ".join(c for c in ("num" if num else "", "lead" if lead else "") if c)
    return f'<td class="{cls}">{text}</td>' if cls else f"<td>{text}</td>"


def _freq_cell(freq: str) -> str:
    return f'<td>{_e(FREQ_NAME.get(freq, freq))}<span class="chip">{_e(freq)}</span></td>'


def _bar(value, largest, dim: bool = False) -> str:
    """A log-scaled bar. The counts span seven orders of magnitude, so a linear bar would be one
    full-width annual row and ten invisible ones; the caption says the scale is log."""
    if not value or not largest:
        return '<td class="barcell"></td>'
    frac = math.log10(max(value, 1)) / math.log10(max(largest, 10))
    return (f'<td class="barcell"><span class="bar{" dim" if dim else ""}" '
            f'style="width:{max(frac, 0.02) * 100:.1f}%"></span></td>')


def _length_table(length: pl.DataFrame, years: float) -> str:
    live = [r["n_long"] for r in length.iter_rows(named=True) if r["n_long"]]
    largest = max(live) if live else 0
    # A spanning header, because three bare columns called p5/median/p95 do not say *of what*.
    # They are the length of a single series, which is the one thing a reader must not misread as
    # a count of series.
    head = (
        '<thead>'
        '<tr><th rowspan="2">frequency</th><th rowspan="2"></th>'
        f'<th rowspan="2">&ge; {years:g} years</th><th rowspan="2">of all</th>'
        '<th rowspan="2">still published</th>'
        '<th colspan="3" class="group">how long one series is, in observations</th>'
        '<th rowspan="2">median, in years</th></tr>'
        '<tr><th class="sub">shortest 5%</th><th class="sub">median</th>'
        '<th class="sub">longest 5%</th></tr>'
        '</thead>'
    )
    body = []
    for r in length.iter_rows(named=True):
        if r["n_long"] is None:
            body.append(_row([
                _freq_cell(r["frequency"]), '<td class="barcell"></td>',
                _td("n/a", num=False), _td(f'{_n(r["n_series"])} total'), _td("n/a", num=False),
                _td("—"), _td("—"), _td("—"), _td("—"),
            ]))
            continue
        body.append(_row([
            _freq_cell(r["frequency"]), _bar(r["n_long"], largest),
            _td(f'<strong>{_n(r["n_long"])}</strong>', lead=True),
            _td(f'{_pct(r["long_share"])} of {_n(r["n_series"])}'),
            _td(_n(r["n_long_live"])), _td(f'{r["p5"]:,}'), _td(f'<strong>{r["p50"]:,}</strong>'),
            _td(f'{r["p95"]:,}'), _td(_num(r["p50_years"])),
        ], "exact"))
    return (
        '<div class="scroller"><table>'
        f'<caption>Exact, over every series. The bar is {years:g} years <em>of the series\' own '
        f'frequency</em> — {years:g} observations for annual, 24 for monthly. Bars are log-scaled.'
        '</caption>' + head + "<tbody>" + "\n".join(body) + "</tbody></table></div>"
    )


def _value_table(values: pl.DataFrame) -> str:
    head = ('<thead><tr><th>frequency</th><th>sampled</th><th>one point only</th>'
            '<th>distinct values</th><th>unique share</th><th>constant</th>'
            '<th>near-constant</th><th>has a value &le; 0</th></tr></thead>')
    body = [
        _row([
            _freq_cell(r["frequency"]), _td(_n(r["sampled"])), _td(_pct(r["share_one_point"])),
            _td(_num(r["distinct_median"], 0)), _td(_fixed(r["unique_share_median"], 2)),
            _td(_pct(r["share_constant"], 1)), _td(_pct(r["share_near_constant"], 1)),
            _td(_pct(r["share_nonpositive"], 1)),
        ], "sampled")
        for r in values.iter_rows(named=True)
    ]
    total = int(values["sampled"].sum()) if values.height else 0
    return (
        '<div class="scroller"><table>'
        f'<caption>Estimated from {_n(total)} series drawn across datasets — this is the only part '
        'that has to read the value lists, and 30 GB of them is too many to read for a summary. '
        'Series with a single observation are excluded from every column but the first: with one '
        'point a series is at once 100% unique and 100% constant.</caption>'
        + head + "<tbody>" + "\n".join(body) + "</tbody></table></div>"
    )


def _conc_table(conc: pl.DataFrame) -> str:
    per = _order(conc.filter(pl.col("frequency") != "*"))
    top_n = int(per["top_n"][0]) if per.height else 10
    largest = max(per["datasets"].to_list()) if per.height else 0
    head = ('<thead><tr><th>frequency</th><th></th><th>datasets</th><th>effective</th>'
            '<th>largest dataset</th><th>its share</th>'
            f'<th>top {top_n}</th></tr></thead>')
    body = [
        _row([
            _freq_cell(r["frequency"]), _bar(r["datasets"], largest, dim=True),
            _td(_n(r["datasets"])), _td(f'<strong>{_num(r["effective_datasets"], 0)}</strong>', lead=True),
            f'<td><code>{_e(r["top_dataset"])}</code></td>',
            _td(_pct(r["top_share"])), _td(_pct(r["topn_share"])),
        ], "exact")
        for r in per.iter_rows(named=True)
    ]
    return (
        '<div class="scroller"><table>'
        '<caption>Exact. <em>Effective</em> is the perplexity of the series-per-dataset '
        'distribution: the number of equally sized datasets that would be as diverse as what is '
        'actually here. Bars show the raw dataset count, log-scaled.</caption>'
        + head + "<tbody>" + "\n".join(body) + "</tbody></table></div>"
    )


def _sparkline(values, width: int = 300, height: int = 52) -> str:
    """One series drawn small. No axis, so no value is implied — it is there to show that a row
    carries a shape. Series here differ by orders of magnitude, so each is scaled to its own
    range: these are small multiples, never a shared plot."""
    pts = [v for v in (values or []) if v is not None]
    if len(pts) < 2:
        return ""
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    step = width / (len(pts) - 1)
    line = " ".join(f"{i * step:.1f},{height - 3 - (v - lo) / span * (height - 6):.1f}"
                    for i, v in enumerate(pts))
    area = f"0,{height} {line} {width},{height}"
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
            f'role="img" aria-label="the shape of this series">'
            f'<polygon points="{area}" fill="currentColor" opacity="0.10"/>'
            f'<polyline points="{line}" fill="none" stroke="currentColor" stroke-width="1.5" '
            f'stroke-linejoin="round"/></svg>')


def _schema_grid() -> str:
    blocks = []
    for title, kind, blurb, cols in SCHEMA_FAMILIES:
        chips = "".join(f'<span class="col">{_e(c)}</span>' for c in cols)
        blocks.append(
            f'<div class="family {kind}"><h3>{title}<span class="count">{len(cols)}</span></h3>'
            f'<p>{blurb}</p><div class="cols">{chips}</div></div>')
    return f'<div class="families">{"".join(blocks)}</div>'


def _specimens_block(spec: pl.DataFrame) -> str:
    if not spec.height:
        return ""
    cards = []
    for r in spec.iter_rows(named=True):
        dims = "".join(f'<li><span class="mono">{_e(c)}</span> {_e(d)}</li>'
                       for c, d in zip(r["codes"], r["dims"]))
        cards.append(f"""
      <figure class="row-card">
        <figcaption>
          <span class="chip">{_e(r["source"])}</span><span class="chip">{_e(r["frequency"])}</span>
          <strong>{_e(r["dataset_title"])}</strong>
        </figcaption>
        {_sparkline(r["spark"])}
        <p class="extent mono">{r["n_obs"]:,} obs &middot; {r["start_date"]} &rarr; {r["end_date"]}</p>
        <p class="desc">{_e(r["description"])}</p>
        <ul class="dims">{dims}</ul>
      </figure>""")
    return f"""
    <p class="prose specimen-lead">Four rows from the corpus, with different sources, frequencies
    and subjects. Each is drawn against its own range, so the heights are not comparable.</p>
    <div class="rows">{"".join(cards)}</div>"""


def _schema_section(spec: pl.DataFrame) -> str:
    total = sum(len(c) for *_, c in SCHEMA_FAMILIES)
    return f"""
  <section id="schema">
    <p class="eyebrow">The schema</p>
    <h2>One row per series, {total} columns.</h2>
    <p class="prose">The sources publish in formats that have nothing in common. They are mapped
    onto the same {total} columns, so the whole corpus opens as one table whichever source a
    series came from. The columns fall into seven groups.</p>
    {_schema_grid()}
    {_specimens_block(spec)}
  </section>
"""


def _leakage_section() -> str:
    return """
  <section id="leakage">
    <p class="eyebrow">Curation and leakage</p>
    <h2>This kind of data is prone to leakage.</h2>
    <p class="prose">Series are kept as published, with their own start, end and gaps. The part
    worth knowing about is that economic statistics come with pitfalls that are easy to miss, and
    most of them make a random train/test split look better than it is. Twelve are written up so
    far, and we keep finding more.</p>
    <div class="traps">
      <div><h4>The same series in another form</h4><p>Seasonally adjusted next to raw, an index
      next to its level, another currency, a per-capita version. Different numbers, same
      information.</p></div>
      <div><h4>Accounting identities</h4><p>Geographic and product hierarchies, national accounts
      identities, mirror trade, stock and flow pairs. A total is a sum of series that can end up
      on the other side of the split.</p></div>
      <div><h4>Revisions</h4><p>Only the latest vintage is stored. A 2008 value is the number as
      it reads now, not as it was published then. A backtest that is not real-time uses data that
      was not available at the time.</p></div>
      <div><h4>Publication lag and projections</h4><p>A quarterly figure is released weeks after
      the period it covers. Some series run to 2100 because they are forecasts. Both look like
      history.</p></div>
      <div><h4>Filters and imputation</h4><p>Smoothed, interpolated and estimated values carry
      information backwards inside a single series. Splitting cannot fix that; the status flags
      are there to find it.</p></div>
      <div><h4>Ill-formed series</h4><p>Constants, series that are still updated but have not
      moved in years, series that are almost entirely one value. They pass any length or recency
      filter and teach a model nothing. The sampled table below counts them.</p></div>
      <div><h4>The same series from two sources</h4><p>FRED republishes 80,274 OECD series, and
      there are near-duplicates within single sources too. <code>origin_agencies</code> is there
      to find them.</p></div>
    </div>
    <p class="prose closing">This is why the split groups by dataset instead of shuffling rows,
    and why the cleaning routine is opt-in rather than the default.</p>
  </section>
"""


def render(tables: dict[str, pl.DataFrame], years: float = MIN_YEARS,
           live_periods: float = LIVE_PERIODS, repo: str | None = None,
           standalone: bool = True, author: str | None = None) -> str:
    """The page. ``standalone`` decides whether a document skeleton is included; ``author=""``
    leaves the byline out entirely.

    ``repo`` and ``author`` are read from the module constants when not given, rather than bound
    as defaults at definition time — so changing ``site.AUTHOR`` changes what the page says, which
    is what anyone rebranding a fork or preparing an anonymous copy will expect it to do.
    """
    repo = REPO if repo is None else repo
    author = AUTHOR if author is None else author
    length, values = _order(tables["length"]), _order(tables["values"])
    conc, crawl = tables["concentration"], tables["crawl"]

    asof = length["asof"][0] if length.height else dt.date.today()
    overall = conc.filter(pl.col("frequency") == "*") if conc.height else pl.DataFrame()

    total = int(length["n_series"].sum())
    long_total = int(length["n_long"].fill_null(0).sum())
    live_total = int(length["n_long_live"].fill_null(0).sum())
    nonannual = int(length.filter(pl.col("frequency") != "A")["n_long_live"].fill_null(0).sum())
    sources = ", ".join(crawl["source"].to_list()) if crawl.height else ""

    # Six figures that need no glossary. The concentration numbers are deliberately not here:
    # "effective datasets" is a perplexity, which is worth knowing and impossible to read at a
    # glance, so it stays in its own table where the caption can explain it.
    observations = int(length["n_obs_total"].sum()) if "n_obs_total" in length.columns else None
    figures = [
        (_n(total), "series in total"),
        (_n(observations) if observations else "—", "observations in total"),
        (_n(long_total), f"with at least {years:g} years of their own frequency"),
        (_n(live_total), "…and still being published"),
        (_n(nonannual), "…of those, not annual"),
        (str(crawl.height), "source agencies, with more planned"),
    ]

    body = f"""
<header class="top"><div class="wrap">
  <span class="name">{mark_svg(22, ink="currentColor", paper="var(--paper)", rim=False, standalone=False)}terrastat</span>
  <nav>
    <a href="#corpus">the corpus</a>
    <a href="#start">getting started</a>
    <a href="#schema">the schema</a>
    <a href="#leakage">leakage</a>
    <a href="#what">what is in it</a>
    <a href="#licence">licences</a>
    <a href="{_e(repo)}">repository &#8599;</a>
  </nav>
</div></header>

<main>
<div class="wrap">

  <div class="hero">
   <div class="hero-text">
    <p class="eyebrow">An open dataset · in progress</p>
    <h1>A dataset of the world&#8217;s economic time series.</h1>
    <p class="lede">{_n(total)} time series of official economic statistics — prices, trade,
      output, labour, migration, health, energy — collected from <strong>FRED</strong>,
      <strong>Eurostat</strong> and the <strong>OECD</strong>, and put on one uniform schema so
      you can train and evaluate models across all of them at once. Every series arrives with its
      own licence and the credit its publisher asks for, so cutting a subset you are allowed to
      share is a one-line filter.</p>
    <p class="status"><i class="dot"></i>Three sources so far, and growing. The schema is built to
      absorb more, and more are planned — energy, trade and web-activity series among them.</p>
   </div>
   <figure class="hero-globe">
    {globe_svg()}
    <figcaption>Each arc is a time series on its parallel, coloured by subject. At the right edge
    it leaves the globe and its forecast fans out beside it.</figcaption>
   </figure>

    <div class="figures">
      {"".join(f'<div><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in figures)}
    </div>
    <div class="legend">
      <span><i class="swatch e"></i> exact, over every series</span>
      <span><i class="swatch s"></i> estimated from a sample</span>
      <span>&ldquo;still being published&rdquo; = a new observation within the last
        {live_periods:g} periods of the series&#8217; own frequency, measured against {asof},
        the newest retrieval in the corpus</span>
    </div>
  </div>

  <section id="corpus">
    <p class="eyebrow">The corpus</p>
    <h2>How much data is in it<span class="tag e">exact</span></h2>
    <p class="prose">How many series there are, how much history each one carries, and how much of
    it is still being added to. Length is the first thing a modeller asks about and the easiest to
    answer badly, because two years of data means two observations at annual frequency and
    twenty-four at monthly — so everything below counts in each series&#8217; own periods, and the
    spread is measured only among the series long enough to be usable.</p>
    {_length_table(length, years)}
  </section>

  <section id="start">
    <p class="eyebrow">Getting started</p>
    <h2>Two ways in, depending on whether you want the data or the crawl.</h2>
    <div class="steps">
      <div>
        <h3>I want to train something</h3>
        <p>Load a snapshot, split it without leaking, cut supervised windows. Runs the same against
        a 28 MB starter set or the full corpus.</p>
        <pre><span class="c"># pip install terrastat</span>
from terrastat import dataset

lf = dataset.load(<span class="c">"starter"</span>)
train, test = dataset.split(lf, by=<span class="c">"dataset_id"</span>)
X, y = dataset.windows(train, horizon=<span class="c">18</span>)</pre>
        <p>The payload is plain Parquet plus a checksummed manifest — readable with polars,
        pandas, DuckDB or Hugging Face <code>datasets</code>, with nothing of ours installed. The
        public snapshot is still being prepared; until it lands,
        <code>terrastat export starter --public-only</code> builds one from your own crawl.</p>
      </div>
      <div>
        <h3>I want to gather it myself</h3>
        <p>One command, re-run until it says nothing is left. Add a FRED key to <code>.env</code>
        first; Eurostat and the OECD need none.</p>
        <pre>pip install -e .
terrastat run <span class="c"># all sources, resumable</span>
terrastat run --max-hours 8 <span class="c"># stop cleanly, resume later</span>

terrastat corpus <span class="c"># regenerate this page</span>
terrastat quality --out reports/quality
terrastat search <span class="c">"chicken|poultry"</span></pre>
        <p>The manual walks through opening a terminal, stopping and resuming, and reading what
        lands on disk.</p>
      </div>
    </div>
  </section>

{_schema_section(tables.get("specimen", pl.DataFrame()))}
{_leakage_section()}
  <section>
    <h2>What the values look like<span class="tag s">sampled</span></h2>
    <p class="prose">A long series that barely moves is common here, and close to useless as a
    training example. <em>Unique share</em> is distinct values over observations; a low one means
    a step function, or something rounded until the variation is gone. <em>Has a value &le; 0</em>
    decides whether sMAPE, a log transform or a multiplicative model is safe.</p>
    {_value_table(values) if values.height else ""}
  </section>

  <section>
    <h2>How concentrated it is<span class="tag e">exact</span></h2>
    <p class="prose">A series count makes a corpus look broader than it is when most of it comes
    from a few very large tables. The more concentrated a frequency, the more a random split puts
    near-copies of the same table on both sides.</p>
    {_conc_table(conc) if conc.height else ""}
  </section>

  <section id="what">
    <p class="eyebrow">What is in it</p>
    <h2>What is in it.</h2>
    <div class="facts">
      <div>
        <h3>Broad, and deliberately uncurated</h3>
        <p>Daily to five-yearly, national accounts to fisheries by species, with histories reaching
        back to the seventeenth century and forward to published projections. Series keep their own start, end and
        gaps: what counts as clean depends on the question, so nothing is trimmed on your behalf.</p>
      </div>
      <div>
        <h3>Licences are sorted out</h3>
        <p>Terms differ by agency and, for FRED, by series. Every row carries its own
        <code>license_id</code>, <code>license_url</code>, <code>attribution</code> and
        originating agency, so selecting a subset you can redistribute is a filter, and the export
        writes out the citations each source asks for.</p>
      </div>
      <div>
        <h3>One row per series</h3>
        <p>The same 36 columns whatever the source, so the whole thing opens as one dataset in
        polars, pandas or DuckDB. Dates, values and status flags sit as lists on the row, next to
        the metadata.</p>
      </div>
      <div>
        <h3>Made for leakage-aware splits</h3>
        <p>The sources republish one another, totals sit next to their parts, and the same
        indicator turns up at several frequencies. Twelve of these are documented, and the
        splitting that ships groups by dataset instead of shuffling rows.</p>
      </div>
    </div>
  </section>

  <section id="licence">
    <p class="eyebrow">Before you redistribute anything</p>
    <h2>The code is Apache-2.0. The data is not.</h2>
    <div class="licence">
      <h3>Licences travel with the rows, and they differ</h3>
      <p>Eurostat and the OECD permit re-use, including commercial re-use, <strong>with
      attribution</strong>. FRED reports a copyright status per series, and its own terms restrict
      archiving and machine-learning use of material that passes through it — while the underlying
      US federal data is public domain at its origin.</p>
      <p>The subset to publish is therefore a filter, not a whole corpus:
      <code>terrastat export NAME --public-only</code> keeps Eurostat, the OECD, and the FRED rows
      that are both public domain and federal in origin, and writes an
      <code>ATTRIBUTIONS.md</code> beside the data. Check <code>license_id</code> and
      <code>attribution</code> on the rows you actually use.</p>
      <p>Not affiliated with, endorsed by, or connected to Eurostat, the OECD, the Federal Reserve
      Bank of St. Louis, or any statistical agency. The name describes what the tool gathers, not
      where it comes from.</p>
    </div>
  </section>

</div>
</main>

<footer><div class="wrap">
  <p>Every figure on this page is computed by <code>terrastat corpus</code>, never written by hand.
  Generated {_e(tables.get('generated') or dt.date.today().isoformat())} from {_e(sources)}; retrievals
  {" · ".join(f'{_e(r["source"])} {_e(r["last_retrieved"][:10])}' for r in crawl.iter_rows(named=True))}.</p>
  <p>{(_e(author) + " · ") if author else ""}code Apache-2.0 ·
  <a href="{_e(repo)}">{_e(repo.replace("https://", ""))}</a></p>
</div></footer>
{animation_script()}
"""
    title = "terrastat"
    head = (f'<title>{title}</title>\n<link rel="icon" type="image/svg+xml" href="favicon.svg">\n'
            f'<link rel="stylesheet" href="{FONTS}">\n'
            f"<style>{CSS}</style>")
    if not standalone:
        return head + body
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="description" content="A dataset of the world&#39;s economic time series: FRED, '
        'Eurostat and the OECD on one uniform schema, with the licence of every series attached.">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"{head}\n</head>\n<body>{body}</body>\n</html>\n"
    )


def write(tables: dict[str, pl.DataFrame], out: Path | str, years: float = MIN_YEARS,
          live_periods: float = LIVE_PERIODS, repo: str | None = None,
          standalone: bool = True, author: str | None = None) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(tables, years, live_periods, repo, standalone, author), encoding="utf-8")
    log.info("wrote %s", p)
    (p.parent / "favicon.svg").write_text(favicon_svg(), encoding="utf-8")
    # the still on its own, at a size you can look at; colours fixed since there is no page
    big = (globe_svg(ink="#151a21", paper="#ffffff")
           .replace('class="globe" ', 'xmlns="http://www.w3.org/2000/svg" ')
           .replace(f'width="{WIDTH}" height="{HEIGHT}"', f'width="{WIDTH * 2}" height="{HEIGHT * 2}"'))
    (p.parent / "logo.svg").write_text(big, encoding="utf-8")
    # the two forms a logo is actually asked for: a square mark and a lockup with the name
    (p.parent / "mark.svg").write_text(mark_svg(256), encoding="utf-8")
    (p.parent / "wordmark.svg").write_text(wordmark_svg(144), encoding="utf-8")
    return p
