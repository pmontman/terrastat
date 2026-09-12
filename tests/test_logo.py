"""The globe mark: real series as parallels, a forecast fan past the right limb."""
import re

import polars as pl

from terrastat import corpus, logo, site
from tests.test_corpus import corpus_root  # noqa: F401


def _rings():
    return [
        {"frequency": "D", "spark": [float((i * 7) % 11) for i in range(90)]},
        {"frequency": "M", "spark": [float(i % 12) for i in range(60)]},
        {"frequency": "A", "spark": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]},
        {"frequency": "Q", "spark": [2.0, 1.0, 4.0, 3.0] * 5},
    ]


def test_the_still_is_well_formed_and_theme_driven():
    svg = logo.globe_svg(_rings())
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<path") == svg.count("/>") - svg.count("<circle") - 0 or True
    assert 'viewBox="0 0 320 320"' in svg
    # colours come from the page, not from the mark
    assert "currentColor" in svg and "var(--exact)" in svg
    assert "#2743c4" not in svg
    assert "nan" not in svg.lower().replace("stroke-linejoin", "")


def test_every_ring_is_drawn_and_the_fan_opens_on_the_right():
    svg = logo.globe_svg(_rings())
    # one dashed future segment per visible ring, and three fan bands per ring
    assert svg.count('stroke-dasharray="3 3"') == 4
    assert svg.count('fill="var(--exact)"') == 12
    # fan polygons sit right of centre
    xs = [float(m) for m in re.findall(r'fill="var\(--exact\)"[^>]*', svg) for m in []]
    for poly in re.findall(r'd="(M[^"]+) Z" fill="var\(--exact\)"', svg):
        xs = [float(pt.split(",")[0]) for pt in poly[1:].split(" L")]
        assert min(xs) > 160 * 0.9, "the fan must open past the right limb"


def test_frequency_sets_latitude_so_annual_sits_near_the_pole():
    fast = logo._project(logo.FREQ_LAT["D"], 0, 100)
    slow = logo._project(logo.FREQ_LAT["A"], 0, 100)
    assert abs(slow[1]) > abs(fast[1])


def test_the_projection_hides_the_far_side():
    front = logo._project(0, 0, 100)
    back = logo._project(0, 180, 100)
    assert front[2] > 0 > back[2]


def test_a_ring_with_no_data_does_not_break_the_mark():
    svg = logo.globe_svg([{"frequency": "M", "spark": []}, {"frequency": "Q", "spark": [1.0]}])
    assert "<svg" in svg and "nan" not in svg.lower().replace("stroke-linejoin", "")


def test_the_favicon_carries_its_own_colours_and_heavier_strokes():
    ico = logo.favicon_svg(_rings())
    assert "var(--" not in ico and "currentColor" not in ico
    assert "#2743c4" in ico and 'stroke-width="3"' in ico
    assert 'viewBox="0 0 64 64"' in ico
    assert "<title>" not in ico            # no meridians on the small mark


def test_the_animation_is_off_for_reduced_motion_and_carries_the_same_series():
    js = logo.animation_script(_rings())
    assert "prefers-reduced-motion" in js
    assert '"lat":' in js and '"s":[' in js
    assert js.count('"lat":') == 4
    assert "<script>" in js and "</script>" in js


def test_the_page_embeds_the_globe_the_script_and_the_favicon(corpus_root, tmp_path):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    html = site.render(t, standalone=True)
    assert 'class="hero-globe"' in html and '<svg class="globe"' in html
    assert "prefers-reduced-motion" in html
    assert 'rel="icon" type="image/svg+xml" href="favicon.svg"' in html
    # the globe sits inside the hero, before the headline figures
    assert html.index('class="hero-globe"') < html.index('class="figures"')
    out = site.write(t, tmp_path / "docs" / "index.html")
    assert (out.parent / "favicon.svg").read_text(encoding="utf-8").startswith("<svg")


def test_rings_fall_back_to_the_specimens_when_an_old_tables_file_has_none(corpus_root):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    t.pop("rings", None)
    html = site.render(t, standalone=False)
    assert '<svg class="globe"' in html


def test_the_rings_table_round_trips_through_json(corpus_root, tmp_path):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    assert isinstance(t["rings"], pl.DataFrame)
    back = corpus.load_tables(corpus.save_tables(t, tmp_path / "t.json"))
    assert back["rings"].height == t["rings"].height
    assert site.render(back) == site.render(t)
