"""The project page: the same computed figures as ``docs/corpus.md``, laid out to be read.

`terrastat corpus` writes two things from one set of tables — a markdown annex for reading inside
the repository, and this page for everyone else. Both are generated, so neither can drift from the
corpus or from each other.

The page distinguishes full-collection summaries from descriptive sample diagnostics.
Captions document eligibility, denominators and limitations; colour is an additional cue.
Stored report dates are retained when wording is regenerated without recomputing the data.

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

from terrastat.corpus import (FREQ_NAME, LIVE_PERIODS, MIN_YEARS, min_obs, _fixed, _n, _num,
                             _order, _pct)
from terrastat.logo import (BRAND_README, HEIGHT, WIDTH, animation_script, brand_kit,
                            favicon_svg, globe_svg, mark_svg)

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
     "Series and dataset identifiers support joins, provenance checks and grouped evaluation. "
     "Dataset grouping alone does not establish independence between training and test data.",
     ["series_uid", "source", "source_id", "dataset_id"]),
    ("human-readable", "text",
     "Titles, notes and descriptions derived from provider metadata. These support search, "
     "interpretation and optional text encoder inputs; descriptions are not additional observations.",
     ["dataset_title", "title", "description", "notes"]),
    ("classification", "structure",
     "Where the series sits in the published table. Each dimension keeps its code and its label, "
     "so totals, their parts and nested geographies (<code>DE</code>, <code>DE1</code>, "
     "<code>DE11</code>) can be found.",
     ["frequency", "frequency_raw", "units", "seasonal_adjustment", "geo", "geo_label",
      "dimensions", "tags"]),
    ("the data", "numbers",
     "Aligned lists of dates, values and status flags. Stored nulls and provider flags are retained; "
     "absent calendar periods may be omitted and must be reconstructed as missing for regular-grid models.",
     ["dates", "values", "flags"]),
    ("licence", "licence",
     "Recorded licence classifications, provider terms and attribution text. These support "
     "review of permitted uses; a metadata filter does not certify permission to redistribute or train models.",
     ["license_id", "license_name", "license_url", "license_detail", "attribution",
      "license_notes"]),
    ("extent", "numbers",
     "Stored endpoints and counts of points, observed values and flags. These scalar columns "
     "can be scanned without reading the numerical lists. Observation counts differ from calendar coverage.",
     ["start_date", "end_date", "n_points", "n_obs", "n_flagged"]),
    ("provenance", "keys",
     "Originating agencies, source links and retrieval metadata help trace observations and "
     "identify possible overlap across providers. The vintage field does not imply a historical revision archive.",
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
.table-notes { color: var(--muted); font-size: 0.86rem; max-width: 90ch; margin-top: 1rem; }
.table-notes p { margin-bottom: 0.65rem; }
.sample-count { display: block; font-size: 0.7rem; color: var(--muted); margin-top: 0.25rem; }
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
        '<th rowspan="2">meets count<br>threshold</th><th rowspan="2">share of all</th>'
        '<th rowspan="2">also within<br>endpoint cutoff</th>'
        '<th colspan="3" class="group">observations per eligible series</th>'
        '<th rowspan="2">median<br>year-equivalent</th></tr>'
        '<tr><th class="sub">5th percentile</th><th class="sub">median</th>'
        '<th class="sub">95th percentile</th></tr>'
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
            _td(_n(r["n_long_live"])), _td(_num(r["p5"], 0)), _td(f'<strong>{_num(r["p50"], 0)}</strong>'),
            _td(_num(r["p95"], 0)), _td(_num(r["p50_years"])),
        ], "exact"))
    return (
        '<div class="scroller"><table>'
        f'<caption>Full-collection summary. The count threshold corresponds to {years:g} '
        f'year-equivalents: {min_obs("A", years)} annual or {min_obs("M", years)} monthly observations, '
        'with a minimum of two observations. Bars show eligible series counts and are log-scaled.'
        '</caption>' + head + "<tbody>" + "\n".join(body) + "</tbody></table></div>"
    )


def _value_table(values: pl.DataFrame) -> str:
    head = ('<thead><tr><th>frequency</th><th>sampled<br>series</th><th>&lt; 2<br>observations</th>'
            '<th>distinct values<br>median</th><th>distinct / observed<br>median</th><th>constant</th>'
            '<th>distinct / observed<br>&lt; 0.05</th><th>any value<br>&le; 0</th></tr></thead>')
    body = [
        _row([
            _freq_cell(r["frequency"]),
            _td(f'{_n(r["sampled"])}<span class="sample-count">{r["datasets"]:,} files</span>'),
            _td(_pct(r["share_one_point"])),
            _td(_num(r["distinct_median"], 0)), _td(_fixed(r["unique_share_median"], 2)),
            _td(_pct(r["share_constant"], 1)), _td(_pct(r["share_near_constant"], 1)),
            _td(_pct(r["share_nonpositive"], 1)),
        ], "sampled")
        for r in values.iter_rows(named=True)
    ]
    total = int(values["sampled"].sum()) if values.height else 0
    return (
        '<div class="scroller"><table>'
        f'<caption>Descriptive sample: {total:,} series. The sample count and &lt; 2 observations '
        'column include all sampled rows; the remaining columns use only rows with at least '
        'two observed values.</caption>'
        + head + "<tbody>" + "\n".join(body) + "</tbody></table></div>"
    )


def _conc_table(conc: pl.DataFrame) -> str:
    per = _order(conc.filter(pl.col("frequency") != "*"))
    top_n = int(per["top_n"][0]) if per.height else 10
    largest = max(per["datasets"].to_list()) if per.height else 0
    head = ('<thead><tr><th>frequency</th><th></th><th>datasets</th><th>effective<br>count</th>'
            '<th>largest dataset</th><th>share of<br>series</th>'
            f'<th>top {top_n}<br>share</th></tr></thead>')
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
        '<caption>Full-collection summary, weighted by series count. The effective count '
        'is exp(&minus;&sum; p log p), where p is a dataset\'s share of series within the frequency '
        '(entropy perplexity). Bars show dataset counts and are log-scaled.</caption>'
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
    <p class="prose specimen-lead">Selected examples illustrate the numerical and metadata fields.
    They were chosen for display, not sampled for representativeness. Each is drawn against its own range;
    heights are not comparable. The curves use downsampled non-null values; their horizontal axis
    shows display order rather than elapsed time and does not represent missing periods.</p>
    <div class="rows">{"".join(cards)}</div>"""


def _schema_section(spec: pl.DataFrame) -> str:
    total = sum(len(c) for *_, c in SCHEMA_FAMILIES)
    return f"""
  <section id="schema">
    <p class="eyebrow">The schema</p>
    <h2>One row per series, {total} columns</h2>
    <p class="prose">Provider-specific formats are mapped to a common Parquet schema. Each row
    combines numerical observations with descriptions, dimensions, provenance and licensing
    metadata. Field availability varies by provider.</p>
    {_schema_grid()}
    {_specimens_block(spec)}
  </section>
"""


def _leakage_section() -> str:
    return """
  <section id="leakage">
    <p class="eyebrow">Evaluation design</p>
    <h2>Dependencies, revisions and forecast information sets</h2>
    <p class="prose">The appropriate holdout depends on the research question. Forecasting future
    observations requires chronological cutoffs; transfer to unseen series or domains also
    requires a defensible grouping strategy. The following properties need attention in either setting.</p>
    <div class="traps">
      <div><h4>Related representations</h4><p>Levels, indices, seasonal adjustments and
      per-capita measures can share underlying information. Group related indicators when the
      evaluation is intended to measure transfer to unseen economic signals.</p></div>
      <div><h4>Accounting identities</h4><p>Geographic and product hierarchies, national accounts
      identities and stock-flow relationships can link observations across series. Related
      aggregates and components may cross a dataset boundary; dataset-level splits do not resolve every dependency.</p></div>
      <div><h4>Revisions</h4><p>The collection stores the latest vintage available at retrieval,
      not a full history of releases. Historical values may therefore incorporate later revisions.
      Describe such evaluations as retrospective; real-time studies require vintage and availability data.</p></div>
      <div><h4>Publication lag and projections</h4><p>An observation date identifies its reference
      period, not necessarily its release date. Future-dated records may be published projections.
      Check both availability and series definitions before assigning forecast targets.</p></div>
      <div><h4>Transformations and imputation</h4><p>Some smoothing or imputation procedures use
      later observations. Fit model preprocessing on training data only and inspect provider
      methods; status flags can help but do not document every transformation.</p></div>
      <div><h4>Repeated or constant values</h4><p>Repeated values can reflect stable quantities,
      discrete measurements, rounding or reporting conventions. Their relevance depends on the
      task. Document any exclusion rule and assess its effect on evaluation coverage.</p></div>
      <div><h4>Overlap between sources</h4><p>Providers may republish the same indicators or related
      aggregates. Originating-agency metadata helps identify candidates, but confirming duplicate
      or equivalent series requires checking definitions, units, dates and values.</p></div>
    </div>
    <p class="prose closing">The grouping helper supports dataset-level holdouts; it does not
    establish independence or impose chronological cutoffs. Retain observation masks and document
    cohort selection, transformations and evaluation dates.</p>
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
        (_n(total), "stored series entries"),
        (_n(observations) if observations is not None else "—", "recorded non-null observations"),
        (_n(long_total), f"meet the {years:g}-year-equivalent count threshold"),
        (_n(live_total), "also meet the endpoint cutoff"),
        (_n(nonannual), "of that subset, nonannual"),
        (str(crawl.height), "data providers in this report"),
    ]

    body = f"""
<header class="top"><div class="wrap">
  <span class="name">{mark_svg(22, ink="currentColor", paper="var(--paper)", rim=False, standalone=False)}terrastat</span>
  <nav>
    <a href="#corpus">coverage</a>
    <a href="#start">getting started</a>
    <a href="#schema">the schema</a>
    <a href="#leakage">leakage</a>
    <a href="#what">scope</a>
    <a href="#licence">licences</a>
    <a href="{_e(repo)}">repository &#8599;</a>
  </nav>
</div></header>

<main>
<div class="wrap">

  <div class="hero">
   <div class="hero-text">
    <p class="eyebrow">Economic data for time-series research</p>
    <h1>Economic time series, with context.</h1>
    <p class="lede">terrastat collects data from <strong>FRED</strong>, <strong>Eurostat</strong>
      and the <strong>OECD</strong> into a common Parquet schema. Numerical observations are
      stored alongside descriptions, units, status flags, provenance and licence metadata,
      supporting data exploration and reproducible time-series studies.</p>
    <p class="status">This report describes a collected corpus, not a claim of complete provider
      coverage. Figures were computed {_e(tables.get('generated') or dt.date.today().isoformat())};
      the reference date for endpoint comparisons is {asof}. Public release hosting is not yet configured.</p>
   </div>
   <figure class="hero-globe">
    {globe_svg()}
    <figcaption>Project illustration; the curves and forecast bands are schematic.</figcaption>
   </figure>

    <div class="figures">
      {"".join(f'<div><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in figures)}
    </div>
    <div class="legend">
      <span><i class="swatch e"></i> full-collection summaries</span>
      <span><i class="swatch s"></i> descriptive sample diagnostics</span>
      <span>Displayed totals are rounded. Counts refer to stored entries, without deduplication
      of equivalent indicators across datasets or providers.</span>
    </div>
  </div>

  <section id="corpus">
    <p class="eyebrow">The corpus</p>
    <h2>Series counts and observation lengths<span class="tag e">full collection</span></h2>
    <p class="prose">The table describes recorded observation counts by frequency. The threshold
    selects a reporting cohort; it does not establish that a series is suitable for a particular
    model, has continuous calendar coverage or contains sufficient seasonal cycles.</p>
    {_length_table(length, years)}
    <div class="table-notes">
      <p><strong>Length:</strong> eligibility is <code>n_obs &ge; max(2, ceil({years:g} &times; periods_per_year))</code>.
      Percentiles and year-equivalents refer only to eligible series. Year-equivalents divide the
      observation count by periods per year; they do not measure elapsed date span. Irregular or
      unknown frequencies have no defined conversion.</p>
      <p><strong>Endpoint cutoff:</strong> the stored end date is no more than {live_periods:g}
      periods of the series&#8217; own frequency before {asof}, using days / 365.25 &times; periods_per_year.
      The rule also includes future-dated endpoints. It does not verify ongoing publication,
      recent releases, or that the final stored point has a non-null value.</p>
    </div>
  </section>

  <section id="start">
    <p class="eyebrow">Getting started</p>
    <h2>Load existing data or collect a dataset</h2>
    <div class="steps">
      <div>
        <h3>Use an existing snapshot</h3>
        <p>Read a local quarterly subset lazily, then inspect its observations and metadata.
        <code>starter</code> must already exist; this command does not download a release.</p>
        <pre><span class="c"># From a checkout: pip install -e ".[notebook,forecast]"</span>
from terrastat import dataset

lf = dataset.load(<span class="c">"starter"</span>, frequencies=[<span class="c">"Q"</span>])
sample = lf.head(5).collect()
sample.select(<span class="c">"series_uid", "title", "units", "n_obs"</span>)</pre>
        <p>Parquet files can also be read with Polars, pandas or DuckDB. For a saved normalized
        release, follow the <a href="{_e(repo)}/blob/master/docs/deployment.md">online or manual deployment guide</a>
        to restore its archives and rebuild the series view offline.</p>
        <p>The <a href="{_e(repo)}/blob/master/notebooks/quarterly_forecasting.ipynb">CPU forecasting notebook</a>
        compares a shared residual network with ETS and seasonal naive forecasts using
        chronological training, validation and test periods.</p>
      </div>
      <div>
        <h3>Download from a provider</h3>
        <p>Start with one Eurostat quarterly unemployment dataset. This example requires no API key.</p>
        <pre>pip install -e .
terrastat fetch eurostat --ids une_rt_q --with-series
terrastat peek eurostat une_rt_q</pre>
        <p>The <a href="{_e(repo)}/blob/master/MANUAL.md">manual</a> covers dataset discovery,
        larger collections, FRED key configuration and interruption recovery. Rerunning a crawl
        resumes unfinished work; it does not automatically refresh completed datasets.
        See <a href="{_e(repo)}/blob/master/docs/refresh.md">refresh behavior</a>.</p>
      </div>
    </div>
  </section>

{_schema_section(tables.get("specimen", pl.DataFrame()))}
{_leakage_section()}
  <section id="values">
    <h2>Observed-value diagnostics<span class="tag s">sampled</span></h2>
    <p class="prose">These summaries describe value repetition and the presence of nonpositive
    observations. They can inform preprocessing and evaluation choices, but do not measure
    forecastability or determine whether a series is useful for training.</p>
    {_value_table(values) if values.height else ""}
    <div class="table-notes">
      <p><strong>Sample design:</strong> files are selected in a seeded random order within each
      frequency, then the first rows of each selected file are read up to a per-file cap.
      Rows are not sampled uniformly across the corpus, and summaries are not weighted by
      dataset size. Treat the percentages as descriptions of the sampled rows, not population estimates.</p>
      <p><strong>Definitions:</strong> distinct / observed is the number of distinct non-null values
      divided by <code>n_obs</code>. Constant means at most one distinct observed value. The
      &lt; 0.05 column uses this ratio, not variance or change magnitude, and can overlap with
      the constant category. Repetition may reflect discrete quantities, rounding or stable measurements.</p>
      <p><strong>Model and metric assumptions:</strong> nonpositive observations require attention
      for log transformations and methods assuming strictly positive data. They do not by themselves
      determine whether sMAPE is appropriate: its definition, forecast values, zero denominators
      and near-zero magnitudes also matter. See the
      <a href="https://otexts.com/fpp2/accuracy.html">forecast-accuracy discussion</a>.</p>
    </div>
  </section>

  <section id="concentration">
    <h2>Dataset concentration<span class="tag e">full collection</span></h2>
    <p class="prose">This table shows how stored series are distributed across source datasets.
    Large cross-tabulations can dominate a frequency's series count. Concentration is relevant
    to sampling and weighting, but does not directly measure domain diversity, duplication or statistical independence.</p>
    {_conc_table(conc) if conc.height else ""}
  </section>

  <section id="what">
    <p class="eyebrow">Scope and interpretation</p>
    <h2>What the collection provides</h2>
    <div class="facts">
      <div>
        <h3>Economic and social indicators</h3>
        <p>The collection includes multiple subjects and frequencies, with provider-specific
        definitions, date spans and missingness. It also contains published projections.
        Counts describe the retrieved collection, not a representative sample of economic activity.</p>
      </div>
      <div>
        <h3>Documented provenance and terms</h3>
        <p>The schema records source links, retrieval dates, licence classifications and
        attribution where available. These support traceability and permission checks;
        users still need to review provider terms for their intended use.</p>
      </div>
      <div>
        <h3>One row per series</h3>
        <p>The shared 36-column schema can be queried using standard Parquet tools. Dates,
        values and status flags are stored as aligned lists. No calendar completion or imputation
        is required to read the stored series; regular-grid models need explicit gap handling.</p>
      </div>
      <div>
        <h3>Support for research workflows</h3>
        <p>Local readers, grouping helpers, missing-value masks and a CPU forecasting tutorial
        provide starting points for experiments. The
        <a href="{_e(repo)}/blob/master/docs/leakage.md">evaluation guide</a> describes dependencies
        and availability issues that require study-specific decisions.</p>
      </div>
    </div>
  </section>

  <section id="licence">
    <p class="eyebrow">Licensing and citation</p>
    <h2>Code and data have separate terms</h2>
    <div class="licence">
      <h3>Retain attribution and review the intended use</h3>
      <p>The software is Apache-2.0; that licence does not grant rights to the data.
      Provider policies and dataset-specific exceptions apply. In particular,
      <a href="https://fred.stlouisfed.org/legal/terms/">FRED's terms</a> restrict archiving and
      machine-learning use. A public-domain classification of underlying data does not override service terms.</p>
      <p><code>--public-only</code> applies an initial licence-metadata filter; it does not certify
      permission. Preserve <code>ATTRIBUTIONS.md</code>, the snapshot manifest and source citations
      with any permitted release. The <a href="{_e(repo)}/blob/master/docs/licensing.md">licensing
      and citation guide</a> links provider policies and explains the recorded fields.</p>
      <p>Not affiliated with, endorsed by, or connected to Eurostat, the OECD, the Federal Reserve
      Bank of St. Louis, or any statistical agency.</p>
    </div>
  </section>

</div>
</main>

<footer><div class="wrap">
  <p>Summary tables are generated by <code>terrastat corpus</code> from a saved report.
  Statistics computed {_e(tables.get('generated') or dt.date.today().isoformat())} from {_e(sources)}; latest retrieval per provider:
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
        '<meta name="description" content="Economic time-series data from FRED, Eurostat and the OECD '
        'in a common Parquet schema, with provenance, licensing metadata and documented research workflows.">\n'
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
    # the whole kit in one folder, so there is one place to look and one place to link
    brand = p.parent / "brand"
    brand.mkdir(parents=True, exist_ok=True)
    # not "logo": this is the explanatory hero, and calling it the logo invited the
    # question of which of the two was the real one. The wordmark is the logo.
    (brand / "hero.svg").write_text(big, encoding="utf-8")
    for name, svg in brand_kit().items():
        (brand / name).write_text(svg, encoding="utf-8")
    (brand / "README.md").write_text(BRAND_README, encoding="utf-8")
    return p
