"""The mark: a globe whose parallels are real series from the corpus, drawn flat.

Bold vector rather than a render: a disc with stylised landmasses in a soft wash, and over it a
few thick, flat-coloured arcs. Each arc is one series wrapped around the sphere, its line riding
the series' shape. Latitude is frequency, fast at the equator and annual towards the poles; colour
is subject domain. Past the right limb, *now*, an arc leaves the sphere and opens into a fan of
widening bands in its own colour — history on the sphere, the future distributed beyond it.

The motion is a loop. Each arc draws on from the left limb while the globe turns; when its head
crosses the terminator the fan begins and keeps growing until it has cleared the disc, then fades
and the arc starts again. The rings are staggered, so the globe is never empty and the cycle has
no visible seam.

Three renderings from one geometry: the still (``globe_svg``, all arcs drawn, fans open), the
favicon (``favicon_svg``, fewer rings and heavier strokes), and ``animation_script``, which
replays the loop from the same numbers.
"""
from __future__ import annotations

import json
import math

# Latitude of the parallel carrying a series of each frequency.
FREQ_LAT = {"D": 0, "B": 8, "W": 17, "BW": 24, "M": 32, "Q": 45, "S": 55, "A": 63, "A3": 72, "P": 80}

# Flat, saturated, readable on both grounds. Assigned by subject.
PALETTE = {
    "prices": "#2743c4", "output": "#1f9d8a", "labour": "#e07a1f", "trade": "#9b1c2e",
    "people": "#7a4fd1", "food": "#3f9b2f", "energy": "#b8921a",
}
DOMAIN_WORDS = {
    "prices": ("price", "hicp", "rate", "exchange", "interest", "money", "cpi", "inflation"),
    "output": ("gdp", "production", "industry", "output", "turnover", "value added"),
    "labour": ("unemploy", "employ", "labour", "labor", "wage", "earning", "job"),
    "trade": ("trade", "export", "import", "tariff", "balance of payments"),
    "people": ("death", "health", "population", "migr", "birth", "student", "educat"),
    "food": ("fish", "poultry", "agri", "crop", "livestock", "milk", "meat", "food"),
    "energy": ("energy", "electric", "gas", "oil", "fuel", "coal", "renewable"),
}

# Stylised landmasses, (lat, lon) polygons. Caricatures, not cartography: enough vertices to be
# recognised at 300 px and no more, so they read as shapes rather than as a map.
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

TILT = 14.0            # a slight lean so the parallels curve
VIEW_LON = -25.0       # the longitude facing the viewer in the still: the Atlantic, both shores
TERMINATOR = 50.0      # longitude past which an arc becomes a fan
AMPLITUDE = 0.08       # radial displacement at a series' maximum, as a share of R
STEP = 4               # degrees of longitude between samples along an arc
FAN_REACH = 0.42       # how far past the limb the fan extends, as a share of R
FAN_WIDTH = 0.13       # half-width of the outer band at full reach, as a share of R


def domain_of(title: str | None) -> str:
    t = (title or "").lower()
    for name, words in DOMAIN_WORDS.items():
        if any(w in t for w in words):
            return name
    return ""


def _colour(ring: dict, index: int) -> str:
    name = domain_of(ring.get("dataset_title") or ring.get("description"))
    if name:
        return PALETTE[name]
    keys = list(PALETTE)
    return PALETTE[keys[index % len(keys)]]


def _project(lat: float, lon: float, r: float, tilt: float = TILT) -> tuple[float, float, float]:
    """Orthographic projection: screen x, screen y (down), depth (positive faces the viewer)."""
    phi, lam, tau = map(math.radians, (lat, lon, tilt))
    x = r * math.cos(phi) * math.sin(lam)
    y = r * math.sin(phi)
    z = r * math.cos(phi) * math.cos(lam)
    return x, -(y * math.cos(tau) - z * math.sin(tau)), y * math.sin(tau) + z * math.cos(tau)


SMOOTH = 5             # moving-average window applied before a series becomes an arc


def _normalise(values: list[float], smooth: int = SMOOTH) -> list[float]:
    """Centred to -1..1, after a short moving average. A daily series sampled to ninety points
    still jumps at every sample; the arc should swoosh, not scribble."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return [0.0, 0.0]
    if smooth > 1 and len(vals) > smooth:
        half = smooth // 2
        vals = [sum(vals[max(0, i - half): i + half + 1]) / len(vals[max(0, i - half): i + half + 1])
                for i in range(len(vals))]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [(v - lo) / span * 2 - 1 for v in vals]


def _path(points, close: bool = False) -> str:
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return d + " Z" if close else d


def landmass_paths(r: float, cx: float, cy: float, view_lon: float = VIEW_LON) -> list[str]:
    """The landmasses as screen polygons. A vertex on the far side is pushed to the limb rather
    than dropped, so a shape crossing the edge keeps a continuous outline."""
    out = []
    for poly in LANDMASSES.values():
        pts = []
        for lat, lon in poly:
            x, y, z = _project(lat, lon - view_lon, r)
            if z < 0:
                h = math.hypot(x, y) or 1.0
                x, y = x / h * r, y / h * r
            pts.append((cx + x, cy + y))
        out.append(_path(pts, close=True))
    return out


def arc_points(lat: float, series: list[float], r: float) -> list[tuple[float, float, float]]:
    """(x, y, lon) along the facing side of one parallel, from the left limb to the right,
    the radius displaced by the series. Screen-space geometry only; nothing hidden by depth."""
    n = len(series)
    out = []
    for i, lon in enumerate(range(-90, 91, STEP)):
        x, y, _ = _project(lat, lon, r * (1 + AMPLITUDE * series[i % n]))
        out.append((x, y, float(lon)))
    return out


def fan_geometry(arc: list[tuple[float, float, float]], series: list[float], r: float):
    """The future: the arc past the terminator, continued straight out beyond the limb.

    Returns the centreline and the two band outlines. Width grows linearly with the fraction of
    the way along the fan, so the bands open like a cone rather than a tube."""
    future = [(x, y) for x, y, lon in arc if lon >= TERMINATOR]
    if len(future) < 2:
        return [], [], []
    # The fan opens outward from the sphere's centre through the limb point. Following the
    # arc's own tangent looks right until you notice that at the right limb an ellipse's tangent
    # runs almost vertically, so the fan slid along the rim instead of leaving the disc.
    x1, y1 = future[-1]
    h = math.hypot(x1, y1) or 1.0
    dx, dy = x1 / h, y1 / h
    n = len(series)
    m = 10
    for j in range(1, m + 1):
        t = j / m
        v = series[(len(arc) + j) % n]
        future.append((x1 + dx * t * FAN_REACH * r - dy * v * AMPLITUDE * r * 0.6,
                       y1 + dy * t * FAN_REACH * r + dx * v * AMPLITUDE * r * 0.6))
    total = len(future) - 1
    outer_u, outer_l, inner_u, inner_l = [], [], [], []
    for i, (x, y) in enumerate(future):
        frac = i / total
        w = FAN_WIDTH * r * frac
        outer_u.append((x - dy * w, y + dx * w))
        outer_l.append((x + dy * w, y - dx * w))
        inner_u.append((x - dy * w * 0.5, y + dx * w * 0.5))
        inner_l.append((x + dy * w * 0.5, y - dx * w * 0.5))
    return future, outer_u + outer_l[::-1], inner_u + inner_l[::-1]


def ring_geometry(rings: list[dict], r: float):
    """Everything the still and the animation share, per ring."""
    out = []
    sign = 1
    for k, ring in enumerate(rings):
        lat = FREQ_LAT.get(ring.get("frequency", "M"), 32) * sign
        sign = -sign
        series = _normalise(ring.get("spark") or [])
        arc = arc_points(lat, series, r)
        centre, outer, inner = fan_geometry(arc, series, r)
        out.append({"colour": _colour(ring, k), "arc": arc, "fan": centre,
                    "outer": outer, "inner": inner})
    return out


def globe_svg(rings: list[dict], size: int = 320, fan: bool = True, land: bool = True,
              id_prefix: str = "g", stroke: float = 4.4) -> str:
    """The still: every arc fully drawn, every fan open. The outline and the landmasses use
    ``currentColor`` so they follow the page's ink; the arcs carry their own flat colours."""
    r = size * 0.36
    cx = cy = size / 2
    parts = [f'<svg class="globe" viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
             f'role="img" aria-label="A globe whose parallels are time series from the corpus; '
             f'past the right limb they open into forecast fans">',
             f'<defs><clipPath id="{id_prefix}-disc"><circle cx="{cx}" cy="{cy}" r="{r:.1f}"/>'
             f'</clipPath></defs>']
    parts.append(f'<circle class="disc" cx="{cx}" cy="{cy}" r="{r:.1f}" fill="currentColor" '
                 f'fill-opacity="0.045"/>')
    if land:
        parts.append(f'<g class="land" clip-path="url(#{id_prefix}-disc)" fill="currentColor" '
                     f'fill-opacity="0.13">')
        parts += [f'<path d="{d}"/>' for d in landmass_paths(r, cx, cy)]
        parts.append("</g>")
    parts.append(f'<circle class="rim" cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" '
                 f'stroke="currentColor" stroke-width="{stroke * 0.7:.1f}"/>')

    parts.append('<g class="arcs">')
    for g in ring_geometry(rings, r):
        c = g["colour"]
        past = [(cx + x, cy + y) for x, y, lon in g["arc"] if lon <= TERMINATOR]
        if len(past) > 1:
            parts.append(f'<path d="{_path(past)}" fill="none" stroke="{c}" '
                         f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"/>')
        if fan and g["fan"]:
            sh = lambda pts: [(cx + x, cy + y) for x, y in pts]  # noqa: E731
            parts.append(f'<path d="{_path(sh(g["outer"]), True)}" fill="{c}" fill-opacity="0.22"/>')
            parts.append(f'<path d="{_path(sh(g["inner"]), True)}" fill="{c}" fill-opacity="0.45"/>')
    parts.append("</g></svg>")
    return "".join(parts)


def favicon_svg(rings: list[dict], size: int = 64) -> str:
    """Four rings, no landmasses, heavier strokes, and a fixed ink: what survives sixteen pixels."""
    svg = globe_svg(rings[:4], size=size, fan=True, land=False, id_prefix="f", stroke=5.0)
    return svg.replace('class="globe" ', "").replace("currentColor", "#151a21")


def animation_script(rings: list[dict], size: int = 320, cycle_s: float = 14.0) -> str:
    """The loop: arcs draw on from the left, cross the terminator, open into fans that clear the
    disc and fade; rings are staggered; the landmasses turn underneath. Does nothing under
    reduced-motion, so the still stands."""
    r = size * 0.36
    geo = ring_geometry(rings, r)
    payload = json.dumps([{"c": g["colour"], "arc": [[round(x, 1), round(y, 1)] for x, y, _ in g["arc"]],
                           "past": sum(1 for *_, lon in g["arc"] if lon <= TERMINATOR),
                           "fan": [[round(x, 1), round(y, 1)] for x, y in g["fan"]],
                           "outer": [[round(x, 1), round(y, 1)] for x, y in g["outer"]],
                           "inner": [[round(x, 1), round(y, 1)] for x, y in g["inner"]]}
                          for g in geo], separators=(",", ":"))
    land = json.dumps(list(LANDMASSES.values()), separators=(",", ":"))
    return f"""<script>
(function(){{
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var svg = document.querySelector('svg.globe'); if (!svg) return;
  var R = {r:.1f}, C = {size / 2}, T = {TILT} * Math.PI / 180, VIEW = {VIEW_LON};
  var rings = {payload}, land = {land}, CYCLE = {cycle_s * 1000}, N = rings.length;
  var arcs = svg.querySelector('g.arcs'), landG = svg.querySelector('g.land');
  function proj(lat, lon) {{
    var p = lat * Math.PI / 180, l = lon * Math.PI / 180;
    var x = R * Math.cos(p) * Math.sin(l), y = R * Math.sin(p), z = R * Math.cos(p) * Math.cos(l);
    return [x, -(y * Math.cos(T) - z * Math.sin(T)), y * Math.sin(T) + z * Math.cos(T)];
  }}
  function path(pts, close) {{ return 'M' + pts.map(function(p){{ return p[0] + ',' + p[1]; }}).join(' L') + (close ? ' Z' : ''); }}
  function sh(pts) {{ return pts.map(function(p){{ return [(C + p[0]).toFixed(1), (C + p[1]).toFixed(1)]; }}); }}
  function drawLand(rot) {{
    if (!landG) return;
    landG.innerHTML = land.map(function(poly) {{
      return '<path d="' + path(poly.map(function(v) {{
        var q = proj(v[0], v[1] - VIEW + rot); if (q[2] < 0) {{ var h = Math.hypot(q[0], q[1]) || 1; q[0] = q[0] / h * R; q[1] = q[1] / h * R; }}
        return [(C + q[0]).toFixed(1), (C + q[1]).toFixed(1)];
      }}), true) + '"/>';
    }}).join('');
  }}
  // one ring's cycle: 0..0.40 draw the arc, 0.40..0.88 open the fan, 0.88..1 fade, then restart.
  // The fan eases in, slow to appear and then opening at a steady pace, and its wedges fade up
  // over the first stretch so it never pops.
  var ARC_END = 0.40, FAN_END = 0.88;
  function easeIn(x) {{ return x * x * (3 - 2 * x) * 0.6 + x * 0.4; }}
  function drawRings(t) {{
    var out = [];
    rings.forEach(function(g, k) {{
      var u = ((t / CYCLE) + k / N) % 1, c = g.c;
      var head = Math.min(1, u / ARC_END);
      var n = Math.max(2, Math.round(g.arc.length * head));
      var pastN = Math.min(n, g.past);
      out.push('<path d="' + path(sh(g.arc.slice(0, pastN))) + '" fill="none" stroke="' + c + '" stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round"/>');
      var fanStart = ARC_END * (g.past / g.arc.length);
      if (u > fanStart) {{
        var f = easeIn(Math.min(1, (u - fanStart) / (FAN_END - fanStart)));
        var m = Math.max(2, Math.round(g.fan.length * f)), half = g.outer.length / 2;
        var outer = g.outer.slice(0, m).concat(g.outer.slice(half, half + m).reverse());
        var inner = g.inner.slice(0, m).concat(g.inner.slice(half, half + m).reverse());
        var fade = u > FAN_END ? 1 - (u - FAN_END) / (1 - FAN_END) : Math.min(1, f / 0.25);
        out.push('<g opacity="' + fade.toFixed(2) + '">'
          + '<path d="' + path(sh(outer), true) + '" fill="' + c + '" fill-opacity="0.22"/>'
          + '<path d="' + path(sh(inner), true) + '" fill="' + c + '" fill-opacity="0.45"/></g>');
      }}
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
