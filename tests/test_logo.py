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
    assert logo.PALETTE["prices"] == "#1f77b4"          # matplotlib tab10
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


def test_the_compact_mark_is_three_arcs_no_land_no_fans():
    """Five arcs and five fans are legible at 440 px and mud at 32. The continents go too: at
    that size they are grey noise, and they cost the mark its silhouette."""
    m = logo.mark_svg(64)
    assert 'viewBox="0 0 64 64"' in m and "currentColor" not in m
    assert "fill-opacity" not in m                       # no fan bands
    assert m.count('<path d="M') == logo.SMALL_ARCS * 2  # one halo and one ribbon per arc
    assert logo.SMALL_ARCS == 3
    for ring in logo.synthetic_rings(3):
        assert ring["colour"] in m
    assert logo.PALETTE["people"] not in m               # the fourth ring is not drawn
    assert logo.favicon_svg() == logo.mark_svg(64)


def test_the_compact_mark_scales_its_stroke_and_can_lose_its_rim():
    small, big = logo.mark_svg(32), logo.mark_svg(256)
    assert 'viewBox="0 0 32 32"' in small and 'viewBox="0 0 256 256"' in big
    assert "<circle" in big and "<circle" not in logo.mark_svg(64, rim=False)
    # arcs stay the same share of the mark at any size
    assert 'width="256"' in big and big.count('<path d="M') == small.count('<path d="M')


def test_the_wordmark_sets_the_name_on_the_equator():
    w = logo.wordmark_svg(72)
    assert ">terrastat</text>" in w and "Newsreader" in w and "serif" in w
    import re
    y = float(re.search(r'<text x="[\d.]+" y="([\d.]+)"', w).group(1))
    assert 36 < y < 36 + 72 * 0.3            # baseline just below the globe's equator
    x = float(re.search(r'<text x="([\d.]+)"', w).group(1))
    assert x > 72                            # clear of the mark
    assert 'viewBox="0 0 ' in w and w.count("<svg") == 1 and w.count("</svg>") == 1


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
    big = (out.parent / "brand" / "logo.svg").read_text(encoding="utf-8")
    assert big.startswith("<svg") and "currentColor" not in big and 'width="880"' in big


def test_arcs_are_tapered_ribbons_not_strokes():
    """A uniform hairline reads as a plot. A ribbon that is thick in the middle and narrows to
    points reads as a drawn mark, which is the whole difference."""
    pts = [(float(x), 100.0) for x in range(0, 200, 5)]
    poly = logo.ribbon(pts, 10.0)
    n = len(pts)
    thick = abs(poly[n // 2][1] - poly[n + n // 2][1])
    thin = abs(poly[0][1] - poly[-1][1])
    assert thick > thin * 1.8 and thick <= 10.2
    # and the still paints them rather than stroking them
    svg = logo.globe_svg()
    assert "stroke-linecap" not in svg and svg.count('<path d="M') > 10


def test_each_arc_sits_on_a_paper_halo_so_arcs_occlude_arcs():
    svg = logo.globe_svg(paper="#fff")
    assert svg.count('fill="#fff"') == len(logo.synthetic_rings()) * 2      # arc and plane line
    # the halo is drawn immediately before its own ribbon. Only solid fills count: the fan
    # bands are the same colour but carry a fill-opacity.
    # per ring: both halos, then both ribbons -- an arc and its own plane line must not punch
    # each other out, only the next ring's halo may cut this one
    solid = re.findall(r'<path d="M[^"]*" fill="(#[0-9a-f]{3,6})"/>', svg)
    for i in range(0, len(solid), 4):
        c = logo.synthetic_rings()[i // 4]["colour"]
        assert solid[i:i + 4] == ["#fff", "#fff", c, c]


def test_the_rim_and_the_land_recede_behind_the_arcs():
    svg = logo.globe_svg()
    assert 'stroke-opacity="0.28"' in svg and 'stroke-width="1.6"' in svg
    assert 'fill-opacity="0.085"' in svg


def test_the_standalone_still_has_no_css_variables():
    from terrastat import site
    big = logo.globe_svg(ink="#151a21", paper="#ffffff")
    assert "var(--" not in big and "currentColor" not in big


def test_the_header_carries_the_mark_and_the_page_writes_the_kit(corpus_root, tmp_path):  # noqa: F811
    t = corpus.compute(values=False, concentration=False)
    html = site.render(t, standalone=True)
    header = html[html.index("<header"): html.index("</header>")]
    assert '<svg viewBox="0 0 22 22"' in header
    assert "<circle" not in header                       # no rim at 22 px
    # the halo must follow the theme, or the gaps between arcs stay white on a dark ground
    assert "var(--paper)" in header
    out = site.write(t, tmp_path / "docs" / "index.html")
    assert (out.parent / "favicon.svg").read_text(encoding="utf-8").startswith("<svg")
    brand = out.parent / "brand"
    assert (brand / "README.md").read_text(encoding="utf-8").startswith("# The terrastat mark")
    for name in ("logo.svg", *logo.brand_kit()):
        text = (brand / name).read_text(encoding="utf-8")
        assert text.startswith("<svg") and "var(--" not in text, name


def test_the_compact_mark_spreads_its_arcs_across_the_disc():
    """Three arcs bunched in a middle band leave the rim looking like a detached ring."""
    r = 64 * 0.40
    ys = []
    for ring, lat in zip(logo.synthetic_rings(3), logo.SMALL_LATS):
        ys += [y for _, y in logo.sphere_arc(ring, r, 32, 32, step=8,
                                             amp_deg=logo.SMALL_AMP, lat=lat,
                                             window=logo.SMALL_SMOOTH)]
    assert max(ys) - min(ys) > r * 1.25            # they span most of the disc's height


def test_the_compact_arcs_are_calmer_than_the_page_arcs():
    """A wiggle tuned for 440 px is a ragged edge at 64."""
    raw = logo.synthetic_rings(1)[0]["series"]
    sm = logo.smooth(raw, logo.SMALL_SMOOTH)
    rough = lambda v: sum(abs(a - b) for a, b in zip(v, v[1:])) / (len(v) - 1)  # noqa: E731
    assert rough(sm) < rough(raw) / 2
    assert logo.smooth(raw, 1) == raw


def test_the_page_arcs_carry_observation_markers():
    """A line that swings is a wave; a line with points on it is data. Markers are the clearest
    signal that these are series, so the page mark has them."""
    svg = logo.globe_svg()
    c = logo.synthetic_rings()[0]["colour"]
    dots = re.findall(r'<circle cx="[\d.-]+" cy="[\d.-]+" r="4\.3" fill="' + c + '"/>', svg)
    assert len(dots) == len(logo.ring_geometry(logo.synthetic_rings()[0])["arc"][::logo.MARKER_EVERY])
    # the ribbon is laid down first, so the dots sit on the line rather than under it
    assert svg.index(f'<path d="M' ) < svg.index(dots[0])
    # and not on the compact mark, where they would be noise
    assert "<circle" not in logo.mark_svg(64, rim=False)   # rim off: no circles at all


def test_markers_land_on_the_observations():
    g = logo.ring_geometry(logo.synthetic_rings()[0])
    import re
    got = re.findall(r'<circle cx="([\d.-]+)" cy="([\d.-]+)"',
                     logo.markers(g["arc"], 7.0, "#000"))
    assert len(got) == len(g["arc"][::logo.MARKER_EVERY])
    for (mx, my), (ax, ay) in zip([(float(a), float(b)) for a, b in got], g["arc"][::logo.MARKER_EVERY]):
        assert abs(mx - ax) < 0.1 and abs(my - ay) < 0.1


def test_the_segments_are_long_enough_to_see_a_vertex():
    """At a four-degree step the arcs were a smooth wave. Longer segments make each observation
    a visible corner, which is what a plotted series looks like."""
    assert logo.STEP == 6 and logo.AMPLITUDE_DEG > 9
    arc = logo.ring_geometry(logo.synthetic_rings()[0])["arc"]
    mid = arc[len(arc) // 3: 2 * len(arc) // 3]
    steps = [abs(b[0] - a[0]) for a, b in zip(mid, mid[1:])]
    assert sum(steps) / len(steps) > 8.0          # pixels between observations, near the centre


def test_markers_are_wider_than_the_line_they_sit_on():
    """At 0.42 of the ribbon width a marker is narrower than its own line and disappears inside
    it, showing only where the taper narrows. A marker has to be proud of the line."""
    for width in (7.0, 3.4):
        assert 2 * width * logo.MARKER_R > width * 1.15
