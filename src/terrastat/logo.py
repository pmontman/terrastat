"""The mark: a globe whose parallels are real series from the corpus, drawn as a logo.

A circle, a few parallels, two meridians. Each parallel is one series wrapped around the sphere,
so instead of a flat arc the line rides the series' shape. Latitude is frequency, fast series at
the equator and annual ones towards the poles. Each parallel is coloured by subject domain, flat
and saturated, the way a logo is coloured rather than the way a chart is. Past the right limb,
*now*, every ring frays into a fan of widening bands in its own colour: history on the left,
the future distributed on the right.

Deliberately not a render. Only the facing hemisphere is drawn, nothing is shaded or faded by
depth, and every line carries a small deterministic hand-drawn jitter and a second, lighter
stroke slightly offset, which is what makes it read as sketched rather than plotted.

Three renderings from one geometry: the still (``globe_svg``), the favicon (``favicon_svg``,
fewer rings and heavier strokes, so it survives sixteen pixels), and a few lines of script that
recompute the same projection each frame with the series phase advancing, so the globe turns.
"""
from __future__ import annotations

import json
import math

# Latitude, in degrees, of the parallel carrying a series of each frequency.
FREQ_LAT = {"D": 0, "B": 7, "W": 16, "BW": 22, "M": 30, "Q": 44, "S": 54, "A": 63, "A3": 72, "P": 80}

# Flat, saturated, and readable on both the light and the dark ground. Assigned by subject.
PALETTE = {
    "prices": "#2743c4",     # ultramarine
    "output": "#1f9d8a",     # teal
    "labour": "#e07a1f",     # orange
    "trade": "#c8385f",      # raspberry
    "people": "#7a4fd1",     # violet
    "food": "#3f9b2f",       # green
    "energy": "#b8921a",     # ochre
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

TILT = 16.0          # a slight lean, enough that the parallels curve
TERMINATOR = 50.0    # longitude past which the ring becomes a fan
AMPLITUDE = 0.085    # radial displacement at the series' maximum, as a share of R
STEP = 5             # degrees of longitude between samples
JITTER = 0.010       # hand-drawn wobble, as a share of R


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


def _normalise(values: list[float]) -> list[float]:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return [0.0, 0.0]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [(v - lo) / span * 2 - 1 for v in vals]


def _wobble(k: int, i: int) -> tuple[float, float]:
    """A small, repeatable offset per point: the same every render, so the sketch never
    shimmers, yet different for every point, so no line is perfectly smooth."""
    a = math.sin(i * 12.9898 + k * 78.233) * 43758.5453
    b = math.sin(i * 39.3467 + k * 11.135) * 24634.6345
    return (a - math.floor(a)) * 2 - 1, (b - math.floor(b)) * 2 - 1


def _ring(lat: float, series: list[float], r: float, k: int, phase: int = 0):
    """Facing-side points of one parallel: (x, y, lon), displaced by the series and jittered."""
    n = len(series)
    out = []
    for i, lon in enumerate(range(-180, 180, STEP)):
        v = series[(i + phase) % n]
        x, y, z = _project(lat, lon, r * (1 + AMPLITUDE * v))
        if z > 0:
            jx, jy = _wobble(k, i)
            out.append((x + jx * JITTER * r, y + jy * JITTER * r, lon))
    return out


def _path(points) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _stroke(d: str, colour: str, width: float, dash: str = "") -> str:
    """Two passes, the second lighter and nudged, which is most of the hand-drawn look."""
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra}/>'
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width * 0.6:.1f}" '
            f'stroke-opacity="0.45" stroke-linecap="round" stroke-linejoin="round" '
            f'transform="translate(0.9,-0.7)"{extra}/>')


def _fan(points, r: float, cx: float, cy: float, colour: str) -> str:
    """Bands widening past the terminator, opening away from the sphere's centre."""
    out = []
    for mult, op in ((2.2, 0.16), (1.1, 0.34)):
        upper, lower = [], []
        for x, y, lon in points:
            d = (lon - TERMINATOR) / (90 - TERMINATOR)
            s = r * 0.10 * mult * d
            h = math.hypot(x, y) or 1.0
            upper.append((cx + x + x / h * s, cy + y + y / h * s))
            lower.append((cx + x - x / h * s, cy + y - y / h * s))
        out.append(f'<path d="{_path(upper + lower[::-1])} Z" fill="{colour}" '
                   f'fill-opacity="{op}" stroke="none"/>')
    out.append(_stroke(_path([(cx + x, cy + y) for x, y, _ in points]), colour, 2.0, "4 4"))
    return "".join(out)


def globe_svg(rings: list[dict], size: int = 320, fan: bool = True, meridians: bool = True,
              phase: int = 0, id_prefix: str = "g", stroke: float = 2.4) -> str:
    """The still. ``rings`` are ``{"frequency", "spark", "dataset_title"}`` rows from the tables.
    The outline uses ``currentColor`` so it follows the page's ink; the rings carry their own
    flat colours, as a logo does."""
    r = size * 0.41
    cx = cy = size / 2
    parts = [f'<svg class="globe" viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
             f'role="img" aria-label="A globe whose parallels are time series from the corpus; '
             f'past the right limb they fray into forecast fans">',
             f'<defs><clipPath id="{id_prefix}-disc"><circle cx="{cx}" cy="{cy}" r="{r * 1.3:.1f}"/>'
             f'</clipPath></defs><g clip-path="url(#{id_prefix}-disc)">']

    # the outline, itself sketched: a jittered polygon rather than a perfect circle
    # finer steps and half the wobble for the rim: at ring resolution the outline looked
    # polygonal rather than drawn
    rim = []
    for i in range(0, 360, 2):
        jx, jy = _wobble(99, i)
        a = math.radians(i)
        rim.append((cx + math.cos(a) * r + jx * JITTER * r * 0.5,
                    cy + math.sin(a) * r + jy * JITTER * r * 0.5))
    parts.append(_stroke(_path(rim) + " Z", "currentColor", stroke * 0.9))

    if meridians:
        for k, lon0 in enumerate((-48.0, 42.0)):
            pts = [_project(lat, lon0, r) for lat in range(-90, 91, 5)]
            line = []
            for i, (x, y, z) in enumerate(pts):
                if z >= 0:
                    jx, jy = _wobble(50 + k, i)
                    line.append((cx + x + jx * JITTER * r, cy + y + jy * JITTER * r))
            if len(line) > 1:
                parts.append(_stroke(_path(line), "currentColor", stroke * 0.55))

    sign = 1
    for k, ring in enumerate(rings):
        lat = FREQ_LAT.get(ring.get("frequency", "M"), 30) * sign
        sign = -sign
        colour = _colour(ring, k)
        pts = _ring(lat, _normalise(ring.get("spark") or []), r, k, phase)
        past = [(cx + x, cy + y) for x, y, lon in pts if lon <= TERMINATOR]
        future = [(x, y, lon) for x, y, lon in pts if lon >= TERMINATOR]
        if len(past) > 1:
            parts.append(_stroke(_path(past), colour, stroke))
        if fan and len(future) > 1:
            parts.append(_fan(future, r, cx, cy, colour))
    parts.append("</g></svg>")
    return "".join(parts)


def favicon_svg(rings: list[dict], size: int = 64) -> str:
    """Five rings, no meridians, heavier strokes, outline in a fixed ink: what survives small."""
    svg = globe_svg(rings[:5], size=size, fan=True, meridians=False, id_prefix="f", stroke=4.0)
    return svg.replace('class="globe" ', "").replace("currentColor", "#151a21")


def animation_script(rings: list[dict], size: int = 320) -> str:
    """Turns the still into the moving version: same projection and the same jitter, with the
    series phase advancing each frame. Honours reduced-motion by doing nothing."""
    data = [{"lat": FREQ_LAT.get(r.get("frequency", "M"), 30) * (1 if i % 2 == 0 else -1),
             "c": _colour(r, i), "s": _normalise(r.get("spark") or [])}
            for i, r in enumerate(rings)]
    payload = json.dumps(data, separators=(",", ":"))
    return f"""<script>
(function(){{
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var svg = document.querySelector('svg.globe'); if (!svg) return;
  var R = {size * 0.41:.1f}, C = {size / 2}, T = {TILT} * Math.PI / 180, TERM = {TERMINATOR}, STEP = {STEP};
  var AMP = {AMPLITUDE}, J = {JITTER} * R, W = {2.4};
  var rings = {payload};
  function fract(a) {{ return a - Math.floor(a); }}
  function wob(k, i) {{ return [fract(Math.sin(i * 12.9898 + k * 78.233) * 43758.5453) * 2 - 1,
                                fract(Math.sin(i * 39.3467 + k * 11.135) * 24634.6345) * 2 - 1]; }}
  function proj(lat, lon, r) {{
    var p = lat * Math.PI / 180, l = lon * Math.PI / 180;
    var x = r * Math.cos(p) * Math.sin(l), y = r * Math.sin(p), z = r * Math.cos(p) * Math.cos(l);
    return [x, -(y * Math.cos(T) - z * Math.sin(T)), y * Math.sin(T) + z * Math.cos(T)];
  }}
  // keep the outline and meridians; the rings and fans are redrawn each frame
  var keep = Array.prototype.slice.call(svg.querySelectorAll('path[stroke="currentColor"]'));
  var g = svg.querySelector('g');
  var layer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.innerHTML = ''; keep.forEach(function(p){{ g.appendChild(p); }}); g.appendChild(layer);
  function path(pts) {{ return 'M' + pts.map(function(p){{ return p[0].toFixed(1) + ',' + p[1].toFixed(1); }}).join(' L'); }}
  function stroke(d, c, w, dash) {{
    var e = dash ? ' stroke-dasharray="' + dash + '"' : '';
    return '<path d="' + d + '" fill="none" stroke="' + c + '" stroke-width="' + w + '" stroke-linecap="round" stroke-linejoin="round"' + e + '/>'
         + '<path d="' + d + '" fill="none" stroke="' + c + '" stroke-width="' + (w * 0.6).toFixed(1) + '" stroke-opacity="0.45" stroke-linecap="round" stroke-linejoin="round" transform="translate(0.9,-0.7)"' + e + '/>';
  }}
  function draw(phase) {{
    var out = [];
    rings.forEach(function(ring, k) {{
      var n = ring.s.length, past = [], fut = [];
      for (var i = 0, lon = -180; lon < 180; lon += STEP, i++) {{
        var p = proj(ring.lat, lon, R * (1 + AMP * ring.s[(i + phase) % n]));
        if (p[2] <= 0) continue;
        var w = wob(k, i), x = p[0] + w[0] * J, y = p[1] + w[1] * J;
        if (lon <= TERM) past.push([C + x, C + y]);
        if (lon >= TERM) fut.push([x, y, lon]);
      }}
      if (past.length > 1) out.push(stroke(path(past), ring.c, W));
      if (fut.length > 1) {{
        [[2.2, .16], [1.1, .34]].forEach(function(b) {{
          var up = [], lo = [];
          fut.forEach(function(p) {{
            var d = (p[2] - TERM) / (90 - TERM), s = R * 0.10 * b[0] * d, h = Math.hypot(p[0], p[1]) || 1;
            up.push([C + p[0] + p[0] / h * s, C + p[1] + p[1] / h * s]); lo.push([C + p[0] - p[0] / h * s, C + p[1] - p[1] / h * s]);
          }});
          out.push('<path d="' + path(up.concat(lo.reverse())) + ' Z" fill="' + ring.c + '" fill-opacity="' + b[1] + '"/>');
        }});
        out.push(stroke(path(fut.map(function(p){{ return [C + p[0], C + p[1]]; }})), ring.c, 2.0, '4 4'));
      }}
    }});
    layer.innerHTML = out.join('');
  }}
  var phase = 0, last = 0;
  function tick(t) {{ if (t - last > 110) {{ last = t; draw(phase++); }} requestAnimationFrame(tick); }}
  requestAnimationFrame(tick);
}})();
</script>"""
