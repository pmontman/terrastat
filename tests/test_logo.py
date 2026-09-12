"""The globe mark: real series as parallels, a forecast fan past the right limb."""
import math
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
    assert 'viewBox="0 0 320 320"' in svg and "currentColor" in svg
    assert "nan" not in svg.lower().replace("stroke-linejoin", "")
    # bold vector, not a sketch and not a render
    assert "translate(0.9,-0.7)" not in svg and "hue-rotate" not in svg
    assert 'stroke-width="3.2"' in svg


def test_the_landmasses_are_drawn_and_stay_on_the_disc():
    svg = logo.globe_svg(_rings())
    assert svg.count('<g class="land"') == 1
    assert svg.count("<path d=", 0, svg.index('class="rim"')) == len(logo.LANDMASSES)
    r, c = 320 * 0.36, 160
    for d in logo.landmass_paths(r, c, c):
        for pt in d[1:-2].split(" L"):
            x, y = map(float, pt.split(","))
            assert (x - c) ** 2 + (y - c) ** 2 <= (r * 1.001) ** 2
    assert 'class="land"' not in logo.globe_svg(_rings(), land=False)


def test_rings_are_coloured_by_subject_domain():
    rings = [{"frequency": "M", "spark": [1.0, 2.0, 1.5] * 8, "dataset_title": "HICP - monthly"},
             {"frequency": "A", "spark": [1.0, 2.0, 3.0, 2.0], "dataset_title": "Inland fisheries"}]
    svg = logo.globe_svg(rings)
    assert logo.PALETTE["prices"] in svg and logo.PALETTE["food"] in svg
    assert logo.domain_of("Deaths by week") == "people"
    assert logo.domain_of("Something nobody categorised") == ""


def test_every_ring_has_an_arc_and_a_fan_that_leaves_the_disc():
    svg = logo.globe_svg(_rings())
    assert "stroke-dasharray" not in svg                     # solid wedges, no scribble
    assert svg.count('fill-opacity="0.22"') == 4 and svg.count('fill-opacity="0.45"') == 4
    r, c = 320 * 0.36, 160
    for g in logo.ring_geometry(_rings(), r):
        x, y = g["fan"][-1]
        assert math.hypot(x, y) > r * 1.2, "the fan must reach well past the limb"
        assert x > 0, "and open on the right"


def test_frequency_sets_latitude_so_annual_sits_near_the_pole():
    fast = logo._project(logo.FREQ_LAT["D"], 0, 100)
    slow = logo._project(logo.FREQ_LAT["A"], 0, 100)
    assert abs(slow[1]) > abs(fast[1])


def test_a_ring_with_no_data_does_not_break_the_mark():
    svg = logo.globe_svg([{"frequency": "M", "spark": []}, {"frequency": "Q", "spark": [1.0]}])
    assert "<svg" in svg and "nan" not in svg.lower().replace("stroke-linejoin", "")


def test_the_favicon_has_a_fixed_ink_and_no_landmasses():
    ico = logo.favicon_svg(_rings())
    assert "currentColor" not in ico and "#151a21" in ico
    assert 'stroke-width="5.0"' in ico and 'viewBox="0 0 64 64"' in ico
    assert 'class="land"' not in ico


def test_the_loop_is_staggered_and_off_for_reduced_motion():
    js = logo.animation_script(_rings())
    assert "prefers-reduced-motion" in js
    assert "k / N" in js                       # rings offset by their index: no visible seam
    assert "0.55" in js and "0.85" in js       # draw, open, fade
    assert js.count('"c":"#') == 4 and '"fan":[[' in js
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


def test_series_are_smoothed_before_they_become_arcs():
    """A daily series sampled to ninety points still jumps at every sample. The arc should
    swoosh, not scribble, so a short moving average runs first."""
    # a ramp with sample-to-sample noise: a pure alternation would be rescaled straight back to
    # full swing after averaging, which is not the case the smoothing exists for
    jumpy = [i / 40 + 0.6 * (i % 2) for i in range(40)]
    def roughness(seq):
        return sum(abs(a - b) for a, b in zip(seq, seq[1:])) / (len(seq) - 1)
    # both are rescaled to -1..1 afterwards, so compare how much they jump step to step
    assert roughness(logo._normalise(jumpy)) < roughness(logo._normalise(jumpy, smooth=1)) / 3
