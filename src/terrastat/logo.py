"""The mark: a globe whose parallels are time series, with their forecasts leaving it as fans.

Flat vector. A disc with stylised landmasses in a soft wash, and over it five thick coloured arcs,
one per subject. Each arc is a time series wrapped along a parallel: fast series near the
equator, slow ones towards the poles. At the right limb, *now*, the series leaves the sphere and
carries on into open space as an ordinary chart line, and a fan of widening bands opens around it
— history on the globe, the future distributed beside it, on its own flat plane where a fan chart
is legible.

The series are synthetic, and deliberately so: a random walk, a trend, a seasonal cycle, a slow
cycle, a regime shift. Real series sampled to fifty points look like noise; these look like the
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

# Flat, saturated, readable on both grounds.
PALETTE = {
    "prices": "#2743c4", "output": "#1f9d8a", "labour": "#e07a1f", "trade": "#9b1c2e",
    "people": "#7a4fd1", "food": "#3f9b2f", "energy": "#b8921a",
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
STEP = 4                   # degrees of longitude between samples along an arc
N_ARC = len(range(-90, 91, STEP))
N_FUTURE = 26              # samples of the series once it has left the sphere
AMPLITUDE = 0.075          # radial displacement on the sphere, as a share of R
PLANE_AMP = 0.16           # vertical swing of the line once in the plane, as a share of R
FAN_WIDTH = 0.34           # half-width of the outer band at the far end, as a share of R
PLANE_END = WIDTH - 14     # where the future line stops


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
    {"subject": "food", "kind": "slow cycle", "lat": 58, "seed": 19},
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


def landmass_paths(view_lon: float = VIEW_LON) -> list[str]:
    """Screen polygons. A far-side vertex is pushed to the limb rather than dropped, so a shape
    crossing the edge keeps a continuous outline."""
    out = []
    for poly in LANDMASSES.values():
        pts = []
        for lat, lon in poly:
            x, y, z = _project(lat, lon - view_lon, R)
            if z < 0:
                h = math.hypot(x, y) or 1.0
                x, y = x / h * R, y / h * R
            pts.append((CX + x, CY + y))
        out.append(_path(pts, close=True))
    return out


def ring_geometry(ring: dict) -> dict:
    """Screen-space geometry of one ring: the arc on the sphere, the line in the plane, and the
    upper and lower edges of the two fan bands, kept as separate edges so a partial wedge can be
    built correctly while the fan is still opening."""
    s = ring["series"]
    arc = []
    for i, lon in enumerate(range(-90, 91, STEP)):
        x, y, _ = _project(ring["lat"], lon, R * (1 + AMPLITUDE * s[i]))
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


def band(edges: tuple[list, list], m: int | None = None) -> list[tuple[float, float]]:
    """A closed wedge from the first ``m`` points of both edges: upper forward, lower back.
    (Slicing a pre-joined polygon took the first ``m`` of the *reversed* lower edge — its far end —
    and drew a sliver across the whole globe while the fan was opening.)"""
    upper, lower = edges
    m = len(upper) if m is None else m
    return upper[:m] + lower[:m][::-1]


def globe_svg(rings: list[dict] | None = None, fan: bool = True, land: bool = True,
              id_prefix: str = "g", stroke: float = 4.6, plane_stroke: float = 3.0) -> str:
    """The still: every arc drawn, every fan open. Outline and landmasses use ``currentColor``
    so they follow the page's ink; the arcs carry their own flat colours."""
    rings = synthetic_rings() if rings is None else rings
    parts = [f'<svg class="globe" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
             f'role="img" aria-label="A globe whose parallels are time series; at the right edge '
             f'each series leaves the sphere and opens into a forecast fan">',
             f'<defs><clipPath id="{id_prefix}-disc"><circle cx="{CX}" cy="{CY}" r="{R}"/></clipPath></defs>',
             f'<circle class="disc" cx="{CX}" cy="{CY}" r="{R}" fill="currentColor" fill-opacity="0.045"/>']
    if land:
        parts.append(f'<g class="land" clip-path="url(#{id_prefix}-disc)" fill="currentColor" '
                     f'fill-opacity="0.13">' + "".join(f'<path d="{d}"/>' for d in landmass_paths())
                     + "</g>")
    parts.append(f'<circle class="rim" cx="{CX}" cy="{CY}" r="{R}" fill="none" '
                 f'stroke="currentColor" stroke-width="{stroke * 0.65:.1f}"/>')
    parts.append('<g class="arcs">')
    for ring in rings:
        g = ring_geometry(ring)
        c = g["colour"]
        parts.append(f'<path d="{_path(g["arc"])}" fill="none" stroke="{c}" stroke-width="{stroke}" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
        if fan:
            parts.append(f'<path d="{_path(band(g["outer"]), True)}" fill="{c}" fill-opacity="0.18"/>')
            parts.append(f'<path d="{_path(band(g["inner"]), True)}" fill="{c}" fill-opacity="0.34"/>')
            parts.append(f'<path d="{_path(g["future"])}" fill="none" stroke="{c}" '
                         f'stroke-width="{plane_stroke}" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append("</g></svg>")
    return "".join(parts)


def favicon_svg(size: int = 64) -> str:
    """Square, four arcs, no landmasses and no fans, heavy strokes, fixed ink: what survives
    sixteen pixels."""
    rings = synthetic_rings(4)
    r, c = size * 0.42, size / 2
    parts = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
             f'xmlns="http://www.w3.org/2000/svg">',
             f'<circle cx="{c}" cy="{c}" r="{r:.1f}" fill="none" stroke="#151a21" stroke-width="3"/>']
    for ring in rings:
        pts = []
        for i, lon in enumerate(range(-90, 91, 6)):
            x, y, _ = _project(ring["lat"], lon, r * (1 + 0.09 * ring["series"][i]))
            pts.append((c + x, c + y))
        parts.append(f'<path d="{_path(pts)}" fill="none" stroke="{ring["colour"]}" stroke-width="5" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append("</svg>")
    return "".join(parts)


def animation_script(rings: list[dict] | None = None, cycle_s: float = 14.0) -> str:
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
  function band(upper, lower, m) {{ return upper.slice(0, m).concat(lower.slice(0, m).reverse()); }}
  function drawLand(rot) {{
    if (!landG) return;
    landG.innerHTML = land.map(function(poly) {{
      return '<path d="' + path(poly.map(function(v) {{
        var q = proj(v[0], v[1] - VIEW + rot);
        if (q[2] < 0) {{ var h = Math.hypot(q[0], q[1]) || 1; q[0] = q[0] / h * R; q[1] = q[1] / h * R; }}
        return [(CX + q[0]).toFixed(1), (CY + q[1]).toFixed(1)];
      }}), true) + '"/>';
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
      var s = '<g opacity="' + alpha.toFixed(2) + '">'
            + '<path d="' + path(g.arc.slice(0, n)) + '" fill="none" stroke="' + c + '" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>';
      if (u > ARC_END) {{
        var f = easeIn(Math.min(1, (u - ARC_END) / (FAN_END - ARC_END)));
        var m = Math.max(2, Math.round(g.fut.length * f));
        var fade = Math.min(1, f / 0.25);
        s += '<g opacity="' + fade.toFixed(2) + '">'
           + '<path d="' + path(band(g.ou, g.ol, m), true) + '" fill="' + c + '" fill-opacity="0.18"/>'
           + '<path d="' + path(band(g.iu, g.il, m), true) + '" fill="' + c + '" fill-opacity="0.34"/>'
           + '<path d="' + path(g.fut.slice(0, m)) + '" fill="none" stroke="' + c + '" stroke-width="3.0" stroke-linecap="round" stroke-linejoin="round"/></g>';
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
</script>"""
