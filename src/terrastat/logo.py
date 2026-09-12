"""The mark: a globe whose parallels are time series, with their forecasts leaving it as fans.

Flat vector. A disc with stylised landmasses in a soft wash, and over it five thick coloured arcs,
one per subject. Each arc is a time series wrapped along a parallel: fast series near the
equator, slow ones towards the poles. At the right limb, *now*, the series leaves the sphere and
carries on into open space as an ordinary chart line, and a fan of widening bands opens around it
— history on the globe, the future distributed beside it, on its own flat plane where a fan chart
is legible.

The series are synthetic, and deliberately so: a random walk, a trend, a seasonal cycle, a regime
shift, a stationary autoregression. Real series sampled to fifty points look like noise; these look like the
things a forecaster recognises, each with a character of its own, and the mark no longer depends
on which series a crawl happened to store. They are generated from fixed seeds, so every render
is identical.

The motion is a loop. Each arc draws on from the left limb while the landmasses turn beneath it;
at the limb the line continues into the plane and the fan opens, easing in; then the whole ring
fades and starts again. Rings are staggered, so the globe is never empty and the cycle has no
seam. Three renderings share the geometry: the still, the favicon, and the script.
"""
from __future__ import annotations

import json
import math

# matplotlib's tab10, which anyone who plots will recognise. The five subjects the mark uses
# take blue, red, orange, purple and green: tab10's most separable set.
PALETTE = {
    "prices": "#1f77b4", "output": "#ff7f0e", "labour": "#8c564b", "trade": "#d62728",
    "people": "#9467bd", "food": "#2ca02c", "energy": "#bcbd22",
}

# Stylised landmasses, (lat, lon) polygons. Caricatures, not cartography.
LANDMASSES = {
    "north america": [(72, -95), (70, -140), (60, -165), (57, -152), (55, -131), (48, -124),
                      (38, -123), (32, -117), (23, -110), (19, -104), (16, -95), (9, -80),
                      (15, -88), (21, -87), (25, -97), (29, -90), (25, -80), (32, -80), (36, -76),
                      (41, -70), (45, -66), (47, -60), (52, -56), (60, -64), (63, -78), (58, -94),
                      (62, -83), (66, -70), (72, -80)],
    "greenland": [(83, -35), (78, -18), (70, -22), (60, -43), (66, -53), (76, -70), (82, -55)],
    "south america": [(12, -72), (10, -62), (5, -52), (-2, -44), (-8, -35), (-15, -39), (-23, -42),
                      (-33, -52), (-38, -58), (-45, -65), (-52, -69), (-55, -68), (-50, -75),
                      (-40, -73), (-30, -71), (-18, -70), (-5, -81), (1, -80), (7, -77)],
    "africa": [(37, -6), (37, 10), (32, 20), (31, 32), (22, 37), (12, 43), (11, 51), (2, 42),
               (-5, 40), (-15, 40), (-26, 33), (-34, 19), (-30, 17), (-17, 12), (-6, 12), (4, 9),
               (5, -2), (8, -13), (15, -17), (21, -17), (30, -10)],
    "eurasia": [(36, -9), (43, -9), (48, -5), (51, 2), (54, 8), (57, 10), (58, 5), (62, 5),
                (71, 25), (69, 33), (72, 60), (75, 90), (72, 130), (70, 160), (66, 190),
                (60, 163), (52, 158), (47, 140), (35, 129), (30, 122), (22, 114), (10, 107),
                (1, 104), (8, 98), (16, 94), (22, 89), (8, 77), (23, 68), (25, 62), (23, 58),
                (17, 55), (13, 45), (21, 39), (28, 34), (32, 35), (36, 36), (41, 29), (40, 26),
                (37, 22), (40, 18), (44, 12), (43, 8), (41, 2), (36, -5)],
    "australia": [(-12, 131), (-12, 142), (-19, 146), (-28, 153), (-34, 151), (-38, 147),
                  (-38, 140), (-32, 134), (-34, 123), (-32, 116), (-22, 114), (-14, 127)],
}

WIDTH, HEIGHT = 440, 300   # wider than tall: the globe on the left, the fans' plane on the right
CX, CY, R = 150.0, 150.0, 108.0
TILT = 14.0                # a slight lean so the parallels curve
VIEW_LON = -25.0           # the longitude facing the viewer in the still: the Atlantic
STEP = 6                   # degrees of longitude between samples: long enough that
                           # each observation is a visible vertex, not a smooth wave
N_ARC = len(range(-90, 91, STEP))
N_FUTURE = 26              # samples of the series once it has left the sphere
AMPLITUDE_DEG = 10.5       # how far a series swings north-south on the sphere, in degrees
PLANE_AMP = 0.21           # vertical swing of the line once in the plane, as a share of R
FAN_WIDTH = 0.34           # half-width of the outer band at the far end, as a share of R
PLANE_END = WIDTH - 14     # where the future line stops
TAPER = 0.42               # half-width at a ribbon's ends, as a share of its width in the middle
HALO = 2.6                 # paper-coloured margin drawn under each arc, so arcs occlude arcs
# Markers every few observations. A line that swings is a wave; a line with points on it is
# data. They are the clearest signal that these are series and not decoration, so the page mark
# carries them -- the compact mark does not, where they would only be noise.
# A marker has to be wider than the line or it hides inside it: at 0.42 of the ribbon width the
# dots were 5.9 across on a 7-wide line and showed only where the taper narrowed. 0.62 puts the
# diameter at 1.24 times the line, which reads as a point on a series; spacing them every third
# observation keeps them from merging into a bead chain.
MARKER_EVERY = 3
MARKER_R = 0.62            # marker radius, as a share of the ribbon width
SMALL_ARCS = 3             # arcs on the compact mark: five plus fans is mud below 64 px
# The compact mark sets its own latitudes and amplitude. At the page's +30/0/-24 the three arcs
# sat in a band across the middle and the rim read as a ring detached from them; spread wide,
# they nearly touch it and the disc reads as one object.
SMALL_LATS = (40, 1, -38)
SMALL_AMP = 10.0
SMALL_SMOOTH = 5           # a wiggle tuned for 440 px is noise at 64: average it down
SMALL_STEP = 8
SMALL_STROKE = 0.135       # share of the mark's size
# Coarser sampling and a bolder swing were tried here, to make the compact arcs read more like
# plotted series. They do, and the mark is worse for it: at a tenth of the mark's width the
# ribbon needs a calm line to stay a silhouette. The page globe carries that job instead.


def _lcg(seed: int):
    """A tiny deterministic generator: the mark must be identical on every machine."""
    state = seed * 2654435761 % 2**32 or 1
    while True:
        state = (1103515245 * state + 12345) % 2**31
        yield state / 2**31 * 2 - 1


def _series(kind: str, n: int, seed: int) -> list[float]:
    """One synthetic series with a recognisable character."""
    rnd = _lcg(seed)
    out, level = [], 0.0
    for i in range(n):
        e = next(rnd)
        if kind == "random walk":
            level += 0.35 * e
            out.append(level)
        elif kind == "trend":
            out.append(0.04 * i + 0.25 * e)
        elif kind == "seasonal":
            out.append(math.sin(2 * math.pi * i / 12) + 0.012 * i + 0.12 * e)
        elif kind == "slow cycle":
            out.append(math.sin(2 * math.pi * i / 40 + 1.0) + 0.10 * e)
        elif kind == "ar1":
            level = 0.55 * level + e
            out.append(level)
        elif kind == "regime shift":
            out.append((0.0 if i < n * 0.55 else 1.2) + 0.2 * math.sin(i / 3.0) + 0.2 * e)
        else:
            out.append(e)
    return out


def _normalise(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return [(v - lo) / span * 2 - 1 for v in values]


# The five rings on the page, in reading order: subject, series character, latitude, seed.
RINGS = [
    {"subject": "prices", "kind": "random walk", "lat": 0, "seed": 11},
    {"subject": "trade", "kind": "seasonal", "lat": -24, "seed": 7},
    {"subject": "output", "kind": "trend", "lat": 30, "seed": 5},
    {"subject": "people", "kind": "regime shift", "lat": -46, "seed": 3},
    {"subject": "food", "kind": "ar1", "lat": 58, "seed": 19},
]


def synthetic_rings(n: int = len(RINGS)) -> list[dict]:
    """The rings with their series attached: what every renderer draws from."""
    out = []
    for spec in RINGS[:n]:
        s = _normalise(_series(spec["kind"], N_ARC + N_FUTURE, spec["seed"]))
        out.append({**spec, "colour": PALETTE[spec["subject"]], "series": s})
    return out


def _project(lat: float, lon: float, r: float, tilt: float = TILT) -> tuple[float, float, float]:
    """Orthographic projection: screen x, screen y (down), depth (positive faces the viewer)."""
    phi, lam, tau = map(math.radians, (lat, lon, tilt))
    x = r * math.cos(phi) * math.sin(lam)
    y = r * math.sin(phi)
    z = r * math.cos(phi) * math.cos(lam)
    return x, -(y * math.cos(tau) - z * math.sin(tau)), y * math.sin(tau) + z * math.cos(tau)


def _path(points, close: bool = False) -> str:
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return d + " Z" if close else d


def clip_to_front(points: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Keep the facing part of a polygon. Each run of far-side vertices collapses to its two limb
    crossings, so a shape that is mostly behind the globe cannot fill the disc.

    (Pushing *every* far-side vertex to the rim looked right for a shape mostly in front, but
    Eurasia turned through the back with its vertices strung around the whole edge, and the
    polygon filled a grey band straight across the globe: the "shadow".)"""
    n = len(points)
    if n < 3 or not any(z >= 0 for _, _, z in points):
        return []
    start = next(i for i in range(n) if points[i][2] >= 0)
    out = []
    i = start
    for _ in range(n):
        x, y, z = points[i]
        if z >= 0:
            out.append((x, y))
        else:
            # first vertex of a hidden run: add its limb point, then skip to the run's end
            j = i
            while points[(j + 1) % n][2] < 0 and (j + 1) % n != start:
                j = (j + 1) % n
            for k in (i, j):
                px, py, _ = points[k]
                h = math.hypot(px, py) or 1.0
                out.append((px / h * R, py / h * R))
            i = j
        i = (i + 1) % n
        if i == start:
            break
    return out if len(out) >= 3 else []


def landmass_paths(view_lon: float = VIEW_LON) -> list[str]:
    """Screen polygons of whatever part of each landmass faces the viewer."""
    out = []
    for poly in LANDMASSES.values():
        pts = clip_to_front([_project(lat, lon - view_lon, R) for lat, lon in poly])
        if pts:
            out.append(_path([(CX + x, CY + y) for x, y in pts], close=True))
    return out


def ring_geometry(ring: dict) -> dict:
    """Screen-space geometry of one ring: the arc on the sphere, the line in the plane, and the
    upper and lower edges of the two fan bands, kept as separate edges so a partial wedge can be
    built correctly while the fan is still opening."""
    s = ring["series"]
    arc = []
    # The series moves the point north or south along the sphere. A radial displacement, towards
    # the viewer, is invisible at the centre of the disc: the arcs read as plain lines.
    for i, lon in enumerate(range(-90, 91, STEP)):
        x, y, _ = _project(ring["lat"] + AMPLITUDE_DEG * s[i], lon, R)
        arc.append((CX + x, CY + y))
    x1, y1 = arc[-1]
    base = s[N_ARC - 1]
    future, outer_u, outer_l, inner_u, inner_l = [], [], [], [], []
    for j in range(N_FUTURE):
        t = j / (N_FUTURE - 1)
        x = x1 + (PLANE_END - x1) * t
        y = y1 + PLANE_AMP * R * (s[N_ARC + j] - base)
        w = FAN_WIDTH * R * t
        future.append((x, y))
        outer_u.append((x, y - w))
        outer_l.append((x, y + w))
        inner_u.append((x, y - w * 0.5))
        inner_l.append((x, y + w * 0.5))
    return {"colour": ring["colour"], "arc": arc, "future": future,
            "outer": (outer_u, outer_l), "inner": (inner_u, inner_l)}


def ribbon(points: list[tuple[float, float]], width: float, grow: float = 0.0,
           taper: float = TAPER) -> list[tuple[float, float]]:
    """A closed polygon around ``points``: a stroke with variable width.

    SVG cannot taper a stroke, and a uniform hairline is what made the arcs look plotted rather
    than drawn. Half-width follows ``taper + (1 - taper) * sin(pi t)``, so a ribbon is thickest
    in the middle and narrows to clean points, the way the swooshes on a good globe mark do.
    ``grow`` widens the whole thing, which is how the paper-coloured halo under each arc is made.
    """
    n = len(points)
    if n < 2:
        return []
    upper, lower = [], []
    for i, (x, y) in enumerate(points):
        t = i / (n - 1)
        ax, ay = points[max(0, i - 1)]
        bx, by = points[min(n - 1, i + 1)]
        dx, dy = bx - ax, by - ay
        h = math.hypot(dx, dy) or 1.0
        w = width * 0.5 * (taper + (1 - taper) * math.sin(math.pi * t) ** 0.5) + grow
        upper.append((x - dy / h * w, y + dx / h * w))
        lower.append((x + dy / h * w, y - dx / h * w))
    return upper + lower[::-1]


def markers(points: list[tuple[float, float]], width: float, colour: str,
            every: int = MARKER_EVERY, r: float | None = None) -> str:
    """Dots on the observations. Drawn over the ribbon in its own colour, so they read as points
    on a line rather than as a second element."""
    rad = width * MARKER_R if r is None else r
    return "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{colour}"/>'
                   for x, y in points[::every])


def band(edges: tuple[list, list], m: int | None = None) -> list[tuple[float, float]]:
    """A closed wedge from the first ``m`` points of both edges: upper forward, lower back.
    (Slicing a pre-joined polygon took the first ``m`` of the *reversed* lower edge — its far end —
    and drew a sliver across the whole globe while the fan was opening.)"""
    upper, lower = edges
    m = len(upper) if m is None else m
    return upper[:m] + lower[:m][::-1]


def globe_svg(rings: list[dict] | None = None, fan: bool = True, land: bool = True,
              id_prefix: str = "g", stroke: float = 7.0, plane_stroke: float = 3.4,
              ink: str = "currentColor", paper: str = "var(--paper)") -> str:
    """The still: every arc drawn, every fan open.

    The arcs carry the sphere. The outline is a hairline and the landmasses a faint wash, so
    nothing competes with them; each arc is a tapered ribbon laid over a paper-coloured halo, so
    where two cross, the later one passes visibly in front.
    """
    rings = synthetic_rings() if rings is None else rings
    parts = [f'<svg class="globe" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
             f'role="img" aria-label="A globe whose parallels are time series; at the right edge '
             f'each series leaves the sphere and opens into a forecast fan">',
             f'<defs><clipPath id="{id_prefix}-disc"><circle cx="{CX}" cy="{CY}" r="{R}"/></clipPath></defs>',
             f'<circle class="disc" cx="{CX}" cy="{CY}" r="{R}" fill="{ink}" fill-opacity="0.035"/>']
    if land:
        parts.append(f'<g class="land" clip-path="url(#{id_prefix}-disc)" fill="{ink}" '
                     f'fill-opacity="0.085">' + "".join(f'<path d="{d}"/>' for d in landmass_paths())
                     + "</g>")
    parts.append(f'<circle class="rim" cx="{CX}" cy="{CY}" r="{R}" fill="none" stroke="{ink}" '
                 f'stroke-opacity="0.28" stroke-width="1.6"/>')
    parts.append('<g class="arcs">')
    for ring in rings:
        g = ring_geometry(ring)
        c = g["colour"]
        if fan:
            parts.append(f'<path d="{_path(band(g["outer"]), True)}" fill="{c}" fill-opacity="0.15"/>')
            parts.append(f'<path d="{_path(band(g["inner"]), True)}" fill="{c}" fill-opacity="0.30"/>')
        # halo first, then the ribbon: a later arc punches a clean gap through an earlier one
        line = g["arc"] + (g["future"] if fan else [])
        parts.append(f'<path d="{_path(ribbon(g["arc"], stroke, HALO), True)}" fill="{paper}"/>')
        if fan:
            parts.append(f'<path d="{_path(ribbon(g["future"], plane_stroke, HALO, taper=1.0), True)}" '
                         f'fill="{paper}"/>')
        parts.append(f'<path d="{_path(ribbon(g["arc"], stroke), True)}" fill="{c}"/>')
        parts.append(markers(g["arc"], stroke, c))
        if fan:
            parts.append(f'<path d="{_path(ribbon(g["future"], plane_stroke, taper=1.0), True)}" '
                         f'fill="{c}"/>')
            parts.append(markers(g["future"], plane_stroke, c))
    parts.append("</g></svg>")
    return "".join(parts)


def smooth(values: list[float], window: int) -> list[float]:
    """A centred moving average. The fine detail that gives the page arcs their character is
    just a ragged edge once the mark is 64 pixels wide."""
    if window < 2:
        return values
    h = window // 2
    return [sum(values[max(0, i - h): i + h + 1]) / len(values[max(0, i - h): i + h + 1])
            for i in range(len(values))]


def sphere_arc(ring: dict, r: float, cx: float, cy: float, step: int = 6,
               amp_deg: float = AMPLITUDE_DEG, lat: float | None = None,
               window: int = 0) -> list[tuple[float, float]]:
    """One parallel, at any size and centre: the same geometry as the page mark, scaled."""
    series = smooth(ring["series"], window) if window else ring["series"]
    base = ring["lat"] if lat is None else lat
    return [(cx + x, cy + y) for x, y, _ in
            (_project(base + amp_deg * series[i], lon, r)
             for i, lon in enumerate(range(-90, 91, step)))]


# The two grounds a mark has to survive. `ink` draws the rim and the name, `paper` the halos
# between arcs -- which is why a light-ground file is unusable on a dark slide: its halos are
# opaque white, and they show as slivers through the gaps.
LIGHT = {"ink": "#151a21", "paper": "#ffffff"}
DARK = {"ink": "#f2f4f8", "paper": "#0d1016"}


def mark_svg(size: int = 64, n_arcs: int = SMALL_ARCS, ink: str = "#151a21",
             paper: str = "#ffffff", rim: bool = True, standalone: bool = True,
             mono: bool = False) -> str:
    """The compact mark: three arcs, no continents, no fans, in a square.

    Not a shrunk copy of the page globe. Five arcs and five fans are legible at 440 px and mud
    at 32; three with air between them keep a silhouette. The continents go too — at this size
    they are grey noise, and they cost the mark the thing that identifies it.

    No markers either. The ribbon here is a tenth of the mark's width, and a dot proud of a line
    that thick is a blob; the arcs have to read as series through their own shape instead, which
    is why they are sampled coarsely, barely smoothed, and swung harder than the page's.
    """
    r, c = size * 0.40, size / 2
    stroke = size * SMALL_STROKE
    head = (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}"'
            + (' xmlns="http://www.w3.org/2000/svg"' if standalone else ' class="mark"') + ">")
    parts = [head]
    if rim:
        parts.append(f'<circle cx="{c}" cy="{c}" r="{r:.1f}" fill="none" stroke="{ink}" '
                     f'stroke-opacity="0.3" stroke-width="{size * 0.028:.1f}"/>')
    for ring, lat in zip(synthetic_rings(n_arcs), SMALL_LATS):
        pts = sphere_arc(ring, r, c, c, step=SMALL_STEP, amp_deg=SMALL_AMP, lat=lat,
                         window=SMALL_SMOOTH)
        parts.append(f'<path d="{_path(ribbon(pts, stroke, stroke * 0.2, taper=0.45), True)}" '
                     f'fill="{paper}"/>')
        parts.append(f'<path d="{_path(ribbon(pts, stroke, taper=0.45), True)}" '
                     f'fill="{ink if mono else ring["colour"]}"/>')
    parts.append("</svg>")
    return "".join(parts)


def favicon_svg(size: int = 64) -> str:
    """The compact mark, with its own colours: there is no page to inherit ink from."""
    return mark_svg(size)


def wordmark_svg(size: int = 72, text: str = "terrastat", ink: str = "#151a21",
                 paper: str = "#ffffff", gap: float = 0.24, standalone: bool = True,
                 mono: bool = False, stacked: bool = False) -> str:
    """The lockup: the compact mark and the name, side by side or one above the other.

    A logo rarely travels alone, and this is the form that goes on a slide or a paper. The name
    is set in Newsreader, the page's display face, with a serif fallback for anywhere the webfont
    is not loaded — a standalone SVG cannot carry a font with it.
    """
    fs = size * 0.60
    text_w = len(text) * fs * 0.52
    mark = mark_svg(size, ink=ink, paper=paper, standalone=False, mono=mono)
    inner = mark[mark.index(">") + 1: mark.rindex("</svg>")]

    if stacked:
        width = max(size, text_w)
        height = size + fs * 1.15
        place = f'<g transform="translate({(width - size) / 2:.1f},0)">{inner}</g>'
        label = (f'<text x="{width / 2:.1f}" y="{size + fs * 0.88:.1f}" text-anchor="middle" ')
    else:
        width = size * (1 + gap) + text_w
        height = size
        place = inner
        # the baseline sits just below the equator, so the name is optically centred on it
        label = f'<text x="{size * (1 + gap):.1f}" y="{size / 2 + fs * 0.35:.1f}" '
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
            f'height="{height:.0f}"'
            + (' xmlns="http://www.w3.org/2000/svg"' if standalone else ' class="wordmark"') + ">"
            + place
            + label
            + f'fill="{ink}" font-family="Newsreader, Georgia, \'Times New Roman\', serif" '
              f'font-size="{fs:.1f}" font-weight="500" letter-spacing="-0.01em">{text}</text>'
            + "</svg>")


def brand_kit(mark_size: int = 256, lockup_size: int = 144) -> dict[str, str]:
    """Every form the mark is actually asked for, keyed by file name.

    Three axes, and each exists for a real situation: colour or mono (a paper printed in
    greyscale turns five hues to mud), light or dark ground (a white halo is a visible sliver on
    a dark slide), and horizontal or stacked (a square space needs the stacked one).
    """
    out = {}
    for suffix, theme in (("", LIGHT), ("-dark", DARK)):
        for tone, is_mono in (("", False), ("-mono", True)):
            out[f"mark{tone}{suffix}.svg"] = mark_svg(mark_size, mono=is_mono, **theme)
            out[f"wordmark{tone}{suffix}.svg"] = wordmark_svg(lockup_size, mono=is_mono, **theme)
            out[f"wordmark-stacked{tone}{suffix}.svg"] = wordmark_svg(
                lockup_size, mono=is_mono, stacked=True, **theme)
    return out


BRAND_README = """# The terrastat mark

Generated by `terrastat corpus`, never drawn by hand. To change any of it, edit
`src/terrastat/logo.py` and re-run the command; editing these files directly means losing the
change at the next build.

## Which file

**The logo is `wordmark.svg`.** `hero.svg` is an illustration, not a logo: its job is to explain
the dataset where there is room to read it, not to identify the project in a corner.

| you are putting it | use |
|---|---|
| a slide, a paper, a README header | `wordmark.svg` — the primary logo |
| a square or narrow space | `wordmark-stacked.svg` |
| somewhere the name is already written | `mark.svg` |
| a browser tab, an avatar, anything under 64 px | `../favicon.svg` |
| the top of a page, to explain the project | `hero.svg` — the full globe with forecast fans |

Add `-dark` for a dark background and `-mono` for one colour. They combine:
`wordmark-stacked-mono-dark.svg`.

## Why the variants exist

**Dark.** The gaps between arcs are not transparent — they are an opaque halo in the page's
background colour, which is what makes a later arc pass visibly in front of an earlier one. A
light-ground file on a dark slide therefore shows white slivers through those gaps. The `-dark`
files draw the halo in the dark ground instead.

**Mono.** Five hues at 20% grey are mud. A paper printed in greyscale, a stamp, an embroidered
shirt: the `-mono` files draw every arc in the single ink colour and let the halos do the
separating.

**Stacked.** A horizontal lockup in a square space either shrinks to nothing or crops. The
stacked one puts the mark above the name.

## Rules

- **Clear space**: leave at least the radius of the globe on all four sides. Nothing else in it.
- **Minimum size**: the wordmark stops being readable below about 90 px wide; use `mark.svg`
  below that, and the favicon below 64.
- **Do not** recolour the arcs, change their number, put the mark on a busy photograph, or
  stretch either file non-uniformly.
- The name is set in **Newsreader** with a serif fallback. An SVG cannot carry a font, so
  anywhere Newsreader is not installed it will render in Georgia. If that matters for a
  particular use, convert the text to outlines in that copy.
"""


def animation_script(rings: list[dict] | None = None, cycle_s: float = 14.0,
                     paper: str = "var(--paper)") -> str:
    """The loop, from the same geometry. Does nothing under reduced-motion, so the still stands."""
    rings = synthetic_rings() if rings is None else rings
    rnd = lambda pts: [[round(x, 1), round(y, 1)] for x, y in pts]  # noqa: E731
    payload = json.dumps([{"c": g["colour"], "arc": rnd(g["arc"]), "fut": rnd(g["future"]),
                           "ou": rnd(g["outer"][0]), "ol": rnd(g["outer"][1]),
                           "iu": rnd(g["inner"][0]), "il": rnd(g["inner"][1])}
                          for g in map(ring_geometry, rings)], separators=(",", ":"))
    land = json.dumps(list(LANDMASSES.values()), separators=(",", ":"))
    return f"""<script>
(function(){{
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var svg = document.querySelector('svg.globe'); if (!svg) return;
  var CX = {CX}, CY = {CY}, R = {R}, T = {TILT} * Math.PI / 180, VIEW = {VIEW_LON};
  var rings = {payload}, land = {land}, CYCLE = {cycle_s * 1000}, N = rings.length;
  var arcs = svg.querySelector('g.arcs'), landG = svg.querySelector('g.land');
  function proj(lat, lon) {{
    var p = lat * Math.PI / 180, l = lon * Math.PI / 180;
    var x = R * Math.cos(p) * Math.sin(l), y = R * Math.sin(p), z = R * Math.cos(p) * Math.cos(l);
    return [x, -(y * Math.cos(T) - z * Math.sin(T)), y * Math.sin(T) + z * Math.cos(T)];
  }}
  function path(pts, close) {{ return 'M' + pts.map(function(p){{ return p[0] + ',' + p[1]; }}).join(' L') + (close ? ' Z' : ''); }}
  var TAPER = {TAPER}, HALO = {HALO};
  function ribbon(pts, width, grow, taper) {{
    var n = pts.length; if (n < 2) return [];
    var up = [], lo = [];
    for (var i = 0; i < n; i++) {{
      var t = i / (n - 1), a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
      var dx = b[0] - a[0], dy = b[1] - a[1], h = Math.hypot(dx, dy) || 1;
      var w = width * 0.5 * (taper + (1 - taper) * Math.pow(Math.sin(Math.PI * t), 0.5)) + grow;
      up.push([(pts[i][0] - dy / h * w).toFixed(1), (pts[i][1] + dx / h * w).toFixed(1)]);
      lo.push([(pts[i][0] + dy / h * w).toFixed(1), (pts[i][1] - dx / h * w).toFixed(1)]);
    }}
    return up.concat(lo.reverse());
  }}
  var MARK_EVERY = {MARKER_EVERY}, MARK_R = {MARKER_R};
  function dots(pts, width, c) {{
    var out = '';
    for (var i = 0; i < pts.length; i += MARK_EVERY)
      out += '<circle cx="' + pts[i][0] + '" cy="' + pts[i][1] + '" r="' + (width * MARK_R).toFixed(1) + '" fill="' + c + '"/>';
    return out;
  }}
  function band(upper, lower, m) {{ return upper.slice(0, m).concat(lower.slice(0, m).reverse()); }}
  function clipFront(pts) {{
    var n = pts.length, start = -1;
    for (var i = 0; i < n; i++) if (pts[i][2] >= 0) {{ start = i; break; }}
    if (n < 3 || start < 0) return [];
    var out = [], i = start;
    for (var c = 0; c < n; c++) {{
      var q = pts[i];
      if (q[2] >= 0) out.push([q[0], q[1]]);
      else {{
        var j = i;
        while (pts[(j + 1) % n][2] < 0 && (j + 1) % n !== start) j = (j + 1) % n;
        [i, j].forEach(function(k) {{ var h = Math.hypot(pts[k][0], pts[k][1]) || 1; out.push([pts[k][0] / h * R, pts[k][1] / h * R]); }});
        i = j;
      }}
      i = (i + 1) % n;
      if (i === start) break;
    }}
    return out.length >= 3 ? out : [];
  }}
  function drawLand(rot) {{
    if (!landG) return;
    landG.innerHTML = land.map(function(poly) {{
      var front = clipFront(poly.map(function(v) {{ return proj(v[0], v[1] - VIEW + rot); }}));
      if (!front.length) return '';
      return '<path d="' + path(front.map(function(q) {{ return [(CX + q[0]).toFixed(1), (CY + q[1]).toFixed(1)]; }}), true) + '"/>';
    }}).join('');
  }}
  // one ring's cycle: 0..0.42 draw the arc, 0.42..0.88 the line leaves the sphere and the fan
  // opens (eased, wedges fading up), 0.88..1 the whole ring fades, then it restarts
  var ARC_END = 0.42, FAN_END = 0.88;
  function easeIn(x) {{ return x * x * (3 - 2 * x) * 0.6 + x * 0.4; }}
  function drawRings(t) {{
    var out = [];
    rings.forEach(function(g, k) {{
      var u = ((t / CYCLE) + k / N) % 1, c = g.c;
      var alpha = u > FAN_END ? 1 - (u - FAN_END) / (1 - FAN_END) : 1;
      var n = Math.max(2, Math.round(g.arc.length * Math.min(1, u / ARC_END)));
      var seg = g.arc.slice(0, n);
      var s = '<g opacity="' + alpha.toFixed(2) + '">'
            + '<path d="' + path(ribbon(seg, 7.0, HALO, TAPER), true) + '" fill="{paper}"/>'
            + '<path d="' + path(ribbon(seg, 7.0, 0, TAPER), true) + '" fill="' + c + '"/>'
            + dots(seg, 7.0, c);
      if (u > ARC_END) {{
        var f = easeIn(Math.min(1, (u - ARC_END) / (FAN_END - ARC_END)));
        var m = Math.max(2, Math.round(g.fut.length * f));
        var fade = Math.min(1, f / 0.25);
        s += '<g opacity="' + fade.toFixed(2) + '">'
           + '<path d="' + path(band(g.ou, g.ol, m), true) + '" fill="' + c + '" fill-opacity="0.18"/>'
           + '<path d="' + path(band(g.iu, g.il, m), true) + '" fill="' + c + '" fill-opacity="0.34"/>'
           + '<path d="' + path(ribbon(g.fut.slice(0, m), 3.4, HALO, 1.0), true) + '" fill="{paper}"/>'
           + '<path d="' + path(ribbon(g.fut.slice(0, m), 3.4, 0, 1.0), true) + '" fill="' + c + '"/>'
           + dots(g.fut.slice(0, m), 3.4, c) + '</g>';
      }}
      out.push(s + '</g>');
    }});
    arcs.innerHTML = out.join('');
  }}
  var start = null, last = 0;
  function tick(now) {{
    if (start === null) start = now;
    if (now - last > 40) {{ last = now; var t = now - start; drawLand((t / 1000) * 4); drawRings(t); }}
    requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
</script>""".replace("{paper}", paper)
