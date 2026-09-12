"""The globe mark: synthetic series as parallels, forecasts fanning out beside the globe."""
import math
import re

from terrastat import corpus, logo, site
from tests.test_corpus import corpus_root  # noqa: F401


def test_the_still_is_well_formed_and_flat():
    svg = logo.globe_svg()
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert f'viewBox="0 0 {logo.WIDTH} {logo.HEIGHT}"' in svg and "currentColor" in svg
    assert "nan" not in svg.lower().replace("stroke-linejoin", "")
    assert 'stroke-width="4.6"' in svg
    assert "hue-rotate" not in svg and "translate(" not in svg      # neither render nor tremor


def test_the_series_are_deterministic_and_have_distinct_characters():
    a, b = logo.synthetic_rings(), logo.synthetic_rings()
    assert [r["series"] for r in a] == [r["series"] for r in b]
    kinds = {r["kind"] for r in a}
    assert kinds == {"random walk", "seasonal", "trend", "regime shift", "ar1"}
    for r in a:
        assert len(r["series"]) == logo.N_ARC + logo.N_FUTURE
        assert -1.0 <= min(r["series"]) and max(r["series"]) <= 1.0
    trend = next(r for r in a if r["kind"] == "trend")["series"]
    assert sum(trend[-10:]) / 10 > sum(trend[:10]) / 10          # a trend goes somewhere
    seasonal = next(r for r in a if r["kind"] == "seasonal")["series"]
    assert max(seasonal[:12]) > 0.3 > min(seasonal[:12])           # a season swings within a year


def test_every_ring_has_one_colour_per_subject():
    rings = logo.synthetic_rings()
    assert len({r["colour"] for r in rings}) == len(rings)
    assert logo.PALETTE["trade"] in logo.globe_svg()


def test_the_fan_lives_in_the_plane_beyond_the_globe():
    for ring in logo.synthetic_rings():
        g = logo.ring_geometry(ring)
        xs = [x for x, _ in g["future"]]
        assert xs[0] >= logo.CX + logo.R * 0.4 and xs[-1] > logo.CX + logo.R * 1.5   # off the disc
        assert all(b >= a for a, b in zip(xs, xs[1:]))            # time runs left to right
        upper, lower = g["outer"]
        assert upper[0] == lower[0]                                # the cone starts closed
        assert lower[-1][1] - upper[-1][1] > logo.R * 0.5          # and ends wide open


def test_a_partial_wedge_never_crosses_the_globe():
    """The bug that showed as a polygon across the disc: the first m points of a pre-joined
    polygon are the start of the upper edge and the far END of the lower one."""
    g = logo.ring_geometry(logo.synthetic_rings()[0])
    for m in (2, 5, 12):
        poly = logo.band(g["outer"], m)
        assert len(poly) == 2 * m
        assert max(x for x, _ in poly) <= max(x for x, _ in g["future"][:m]) + 0.01
        assert min(x for x, _ in poly) >= g["future"][0][0] - 0.01


def test_the_landmasses_stay_on_the_disc():
    svg = logo.globe_svg()
    assert svg.count('<g class="land"') == 1
    for d in logo.landmass_paths():
        for pt in d[1:-2].split(" L"):
            x, y = map(float, pt.split(","))
            assert (x - logo.CX) ** 2 + (y - logo.CY) ** 2 <= (logo.R * 1.001) ** 2


def test_the_favicon_is_square_fixed_ink_and_carries_no_fans():
    ico = logo.favicon_svg()
    assert 'viewBox="0 0 64 64"' in ico and "#151a21" in ico and "currentColor" not in ico
    assert "fill-opacity" not in ico and 'stroke-width="5"' in ico


def test_the_loop_is_staggered_eased_and_off_for_reduced_motion():
    js = logo.animation_script()
    assert "prefers-reduced-motion" in js and "k / N" in js and "easeIn" in js
    assert "ARC_END = 0.42" in js and "FAN_END = 0.88" in js
    assert "lower.slice(0, m).reverse()" in js                      # the corrected wedge
    assert js.count('"c":"#') == 5


def test_the_page_embeds_the_globe_the_script_and_the_favicon(corpus_root, tmp_path):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    html = site.render(t, standalone=True)
    assert 'class="hero-globe"' in html and '<svg class="globe"' in html
    assert "prefers-reduced-motion" in html
    assert 'rel="icon" type="image/svg+xml" href="favicon.svg"' in html
    assert html.index('class="hero-globe"') < html.index('class="figures"')
    out = site.write(t, tmp_path / "docs" / "index.html")
    assert (out.parent / "favicon.svg").read_text(encoding="utf-8").startswith("<svg")


def test_the_mark_does_not_depend_on_the_tables(corpus_root):  # noqa: F811
    """Synthetic series: the page renders the same globe whatever the crawl stored."""
    t = corpus.compute(values=False, concentration=False)
    assert "rings" not in t
    a = site.render(t, standalone=False)
    t2 = dict(t, specimen=t["specimen"].clear())
    b = site.render(t2, standalone=False)
    ga = a[a.index('<svg class="globe"'): a.index("</svg>")]
    gb = b[b.index('<svg class="globe"'): b.index("</svg>")]
    assert ga == gb


def test_the_series_swing_north_south_so_they_read_as_series():
    """A radial displacement is invisible at the centre of the disc. Moving the point in latitude
    keeps it on the sphere and puts the shape where the eye can see it."""
    for ring in logo.synthetic_rings():
        arc = logo.ring_geometry(ring)["arc"]
        # compare against the undisplaced parallel: the series must move the line vertically
        flat = [logo.CY + logo._project(ring["lat"], lon, logo.R)[1] for lon in range(-90, 91, logo.STEP)]
        dev = [abs(y - fy) for (_, y), fy in zip(arc, flat)]
        assert max(dev) > logo.R * 0.08 and sum(dev) / len(dev) > logo.R * 0.025


def test_an_ar1_series_is_stationary_and_noisy():
    s = next(r for r in logo.synthetic_rings() if r["kind"] == "ar1")["series"]
    first, last = s[: len(s) // 2], s[len(s) // 2:]
    assert abs(sum(first) / len(first) - sum(last) / len(last)) < 0.5     # no drift
    assert sum(abs(a - b) for a, b in zip(s, s[1:])) / (len(s) - 1) > 0.3  # and busy


def test_a_landmass_behind_the_globe_is_not_drawn_across_it():
    """The shadow: a polygon mostly on the far side had every hidden vertex pushed to the rim and
    filled a band across the disc. A hidden run now collapses to its two limb crossings."""
    behind = [logo._project(lat, lon + 180, logo.R) for lat, lon in logo.LANDMASSES["africa"]]
    assert logo.clip_to_front(behind) == []
    # Eurasia seen from the Atlantic: some of it shows, and none of what shows is on the far side
    pts = [logo._project(lat, lon - logo.VIEW_LON, logo.R) for lat, lon in logo.LANDMASSES["eurasia"]]
    front = logo.clip_to_front(pts)
    assert 3 <= len(front) < len(pts) + 2
    hidden_x = [x for x, _, z in pts if z < 0]
    assert all(x >= -logo.R * 1.001 for x, _ in front)
    # a hidden run of many vertices contributes exactly two rim points
    n_hidden = sum(1 for _, _, z in pts if z < 0)
    n_front = sum(1 for _, _, z in pts if z >= 0)
    assert len(front) <= n_front + 2 * (n_hidden and 4)


def test_a_large_still_is_written_beside_the_page(corpus_root, tmp_path):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    out = site.write(t, tmp_path / "docs" / "index.html")
    big = (out.parent / "logo.svg").read_text(encoding="utf-8")
    assert big.startswith("<svg") and "currentColor" not in big and 'width="880"' in big
