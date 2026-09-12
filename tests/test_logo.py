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


def test_the_still_is_well_formed_and_flat():
    svg = logo.globe_svg(_rings())
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'viewBox="0 0 320 320"' in svg
    assert "currentColor" in svg                       # outline follows the page ink
    assert "nan" not in svg.lower().replace("stroke-linejoin", "")
    # a logo, not a render: nothing on the far side, nothing faded by depth
    assert 'stroke-opacity="0.10"' not in svg and "hue-rotate" not in svg


def test_rings_are_coloured_by_subject_domain():
    rings = [{"frequency": "M", "spark": [1.0, 2.0, 1.5] * 8, "dataset_title": "HICP - monthly"},
             {"frequency": "A", "spark": [1.0, 2.0, 3.0, 2.0], "dataset_title": "Inland fisheries"}]
    svg = logo.globe_svg(rings)
    assert logo.PALETTE["prices"] in svg and logo.PALETTE["food"] in svg
    assert logo.domain_of("Deaths by week") == "people"
    assert logo.domain_of("Something nobody categorised") == ""


def test_every_ring_is_drawn_and_the_fan_opens_on_the_right():
    svg = logo.globe_svg(_rings())
    assert svg.count('stroke-dasharray="4 4"') == 4 * 2      # one dashed future line per ring, two passes
    assert svg.count('stroke="none"/>') == 4 * 2              # two fan bands per ring
    for poly in re.findall(r'd="(M[^"]+) Z" fill="#[0-9a-f]{6}"', svg):
        xs = [float(pt.split(",")[0]) for pt in poly[1:].split(" L")]
        assert min(xs) > 160 * 0.9, "the fan must open past the right limb"


def test_the_sketch_never_shimmers():
    """The jitter is a function of the point, not the frame: rendering twice gives the same
    bytes, and a phase change moves the series but not the wobble."""
    assert logo.globe_svg(_rings()) == logo.globe_svg(_rings())
    assert logo._wobble(3, 7) == logo._wobble(3, 7)
    assert logo._wobble(3, 7) != logo._wobble(3, 8)


def test_frequency_sets_latitude_so_annual_sits_near_the_pole():
    fast = logo._project(logo.FREQ_LAT["D"], 0, 100)
    slow = logo._project(logo.FREQ_LAT["A"], 0, 100)
    assert abs(slow[1]) > abs(fast[1])


def test_the_projection_hides_the_far_side():
    assert logo._project(0, 0, 100)[2] > 0 > logo._project(0, 180, 100)[2]


def test_a_ring_with_no_data_does_not_break_the_mark():
    svg = logo.globe_svg([{"frequency": "M", "spark": []}, {"frequency": "Q", "spark": [1.0]}])
    assert "<svg" in svg and "nan" not in svg.lower().replace("stroke-linejoin", "")


def test_the_favicon_has_a_fixed_ink_and_heavier_strokes():
    ico = logo.favicon_svg(_rings())
    assert "currentColor" not in ico and "#151a21" in ico
    assert 'stroke-width="4.0"' in ico and 'viewBox="0 0 64 64"' in ico
    assert ico.count("<path") < logo.globe_svg(_rings()).count("<path")   # no meridians


def test_the_animation_is_off_for_reduced_motion_and_carries_the_same_series():
    js = logo.animation_script(_rings())
    assert "prefers-reduced-motion" in js
    assert js.count('"lat":') == 4 and js.count('"c":"#') == 4
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
