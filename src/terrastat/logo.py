"""The mark: a globe whose parallels are real series from the corpus.

A wireframe globe, orthographic, tilted a little. Each parallel is one series wrapped around the
sphere: instead of a flat circle the ring rides the series' shape. Latitude is frequency, so the
fast series sit at the equator and the annual ones near the poles; each of the seven meridians is
a subject domain, and the parallels take a faint tint from the meridian they cross. The right limb
is *now*: past it, every ring frays into a fan of widening quantile bands. History observed on the
left, the future distributed on the right, which is the whole argument of the project in one shape.

Two renderings from the same geometry: ``globe_svg`` draws a still, and the page ships a few lines
of JavaScript that recompute the same projection each frame with the series phase advancing, so
the globe turns and the series scroll through the terminator. ``favicon_svg`` is the same globe
with fewer rings and heavier strokes, so it survives sixteen pixels.
"""
from __future__ import annotations

import json
import math

# Latitude, in degrees, of the parallel that carries a series of each frequency. Fast series at
# the equator, slow ones towards the poles; both hemispheres are used, alternating.
FREQ_LAT = {"D": 0, "B": 6, "W": 14, "BW": 20, "M": 30, "Q": 42, "S": 52, "A": 62, "A3": 72, "P": 80}

# Seven subject domains, one meridian each: a hue shift applied to the accent colour so the
# domains read as tints of one family rather than a rainbow.
DOMAINS = ["prices", "output", "labour", "trade", "migration", "health", "energy"]

TILT = 22.0            # degrees the axis leans towards the viewer
TERMINATOR = 55.0      # longitude past which the ring becomes a fan
AMPLITUDE = 0.075      # radial displacement of a ring at the series' maximum, as a share of R
STEP = 4               # degrees of longitude between samples along a ring


def _project(lat: float, lon: float, r: float, tilt: float = TILT) -> tuple[float, float, float]:
    """Orthographic projection of a point on the sphere, tilted about the horizontal axis.
    Returns screen x, screen y (down), and depth (positive = facing the viewer)."""
    phi, lam, tau = map(math.radians, (lat, lon, tilt))
    x = r * math.cos(phi) * math.sin(lam)
    y = r * math.sin(phi)
    z = r * math.cos(phi) * math.cos(lam)
    y2 = y * math.cos(tau) - z * math.sin(tau)
    z2 = y * math.sin(tau) + z * math.cos(tau)
    return x, -y2, z2


def _normalise(values: list[float]) -> list[float]:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return [0.0] * max(len(vals), 2)
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [(v - lo) / span * 2 - 1 for v in vals]         # -1 .. 1, centred


def _ring_points(lat: float, series: list[float], r: float, phase: int = 0,
                 amplitude: float = AMPLITUDE) -> list[tuple[float, float, float, float]]:
    """(x, y, depth, lon) along a parallel, the radius displaced by the series."""
    n = len(series)
    out = []
    for i, lon in enumerate(range(-180, 180, STEP)):
        v = series[(i + phase) % n]
        x, y, z = _project(lat, lon, r * (1 + amplitude * v))
        out.append((x, y, z, lon))
    return out


def _path(points: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _runs(points, visible: bool):
    """Consecutive stretches of a ring that face the viewer (or not)."""
    run = []
    for x, y, z, lon in points:
        if (z > 0) == visible:
            run.append((x, y, lon))
        elif run:
            yield run
            run = []
    if run:
        yield run


def globe_svg(rings: list[dict], size: int = 320, fan: bool = True, meridians: bool = True,
              phase: int = 0, id_prefix: str = "g") -> str:
    """The still. ``rings`` is a list of ``{"frequency", "spark"}`` from the corpus tables.

    Colours come from the page's own tokens (``currentColor`` for ink, ``var(--exact)`` for the
    accent), so the mark follows the theme rather than carrying colours of its own.
    """
    r = size * 0.42
    cx = cy = size / 2
    parts = [f'<svg class="globe" viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
             f'role="img" aria-label="A globe whose parallels are time series from the corpus; '
             f'past the right limb they fray into forecast fans">']
    parts.append(f'<defs><clipPath id="{id_prefix}-disc"><circle cx="{cx}" cy="{cy}" r="{r * 1.28:.1f}"/>'
                 f'</clipPath></defs>')
    parts.append(f'<g clip-path="url(#{id_prefix}-disc)">')
    # the disc and its rim, faint
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="currentColor" '
                 f'stroke-opacity="0.18" stroke-width="1"/>')

    if meridians:
        k = len(DOMAINS)
        for j, name in enumerate(DOMAINS):
            lon0 = -90 + 180 * (j + 0.5) / k          # spread over the facing hemisphere
            pts = [_project(lat, lon0, r) for lat in range(-88, 89, 4)]
            front = [(cx + x, cy + y) for x, y, z in pts if z > 0]
            hue = (j - (k - 1) / 2) * 12               # -36 .. 36 degrees around the accent
            if len(front) > 1:
                parts.append(f'<path d="{_path(front)}" fill="none" stroke="var(--exact)" '
                             f'stroke-opacity="0.22" stroke-width="0.8" '
                             f'style="filter:hue-rotate({hue:.0f}deg)"><title>{name}</title></path>')

    # parallels: alternate hemispheres so both are used, ordered by frequency
    lat_sign = 1
    for ring in rings:
        lat = FREQ_LAT.get(ring.get("frequency", "M"), 30) * lat_sign
        lat_sign = -lat_sign
        series = _normalise(ring.get("spark") or [])
        pts = _ring_points(lat, series, r, phase)
        for run in _runs(pts, visible=False):
            parts.append(f'<path d="{_path([(cx + x, cy + y) for x, y, _ in run])}" fill="none" '
                         f'stroke="currentColor" stroke-opacity="0.10" stroke-width="0.8"/>')
        for run in _runs(pts, visible=True):
            past = [(cx + x, cy + y) for x, y, lon in run if lon <= TERMINATOR]
            future = [(x, y, lon) for x, y, lon in run if lon >= TERMINATOR]
            if len(past) > 1:
                parts.append(f'<path d="{_path(past)}" fill="none" stroke="currentColor" '
                             f'stroke-opacity="0.8" stroke-width="1.2" stroke-linejoin="round"/>')
            if fan and len(future) > 1:
                # quantile bands: the spread grows with distance past the terminator, in the
                # direction away from the sphere's centre so the fan opens outward
                for mult, op in ((3.0, 0.07), (2.0, 0.12), (1.0, 0.2)):
                    upper, lower = [], []
                    for x, y, lon in future:
                        d = (lon - TERMINATOR) / (90 - TERMINATOR)
                        s = r * 0.09 * mult * d
                        nx, ny = x / (math.hypot(x, y) or 1), y / (math.hypot(x, y) or 1)
                        upper.append((cx + x + nx * s, cy + y + ny * s))
                        lower.append((cx + x - nx * s, cy + y - ny * s))
                    parts.append(f'<path d="{_path(upper + lower[::-1])} Z" fill="var(--exact)" '
                                 f'fill-opacity="{op}" stroke="none"/>')
                parts.append(f'<path d="{_path([(cx + x, cy + y) for x, y, _ in future])}" '
                             f'fill="none" stroke="var(--exact)" stroke-opacity="0.9" '
                             f'stroke-width="1.2" stroke-dasharray="3 3"/>')
    parts.append("</g></svg>")
    return "".join(parts)


def favicon_svg(rings: list[dict], size: int = 64) -> str:
    """The same globe, reduced to what survives at sixteen pixels: five rings, no meridians, a
    single fan band, heavy strokes, and colours of its own since there is no page to inherit."""
    picked = rings[:5]
    svg = globe_svg(picked, size=size, fan=True, meridians=False, id_prefix="f")
    return (svg.replace('class="globe" ', "")
               .replace("var(--exact)", "#2743c4")
               .replace("currentColor", "#151a21")
               .replace('stroke-width="1.2"', 'stroke-width="3"')
               .replace('stroke-width="0.8"', 'stroke-width="1.5"')
               .replace('stroke-width="1"', 'stroke-width="2"'))


def animation_script(rings: list[dict], size: int = 320) -> str:
    """The few lines that turn the still into the moving version: same projection, the series
    phase advancing each frame so the globe appears to rotate. Honours reduced-motion."""
    data = [{"lat": FREQ_LAT.get(r.get("frequency", "M"), 30) * (1 if i % 2 == 0 else -1),
             "s": _normalise(r.get("spark") or [])} for i, r in enumerate(rings)]
    payload = json.dumps(data, separators=(",", ":"))
    return f"""<script>
(function(){{
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var svg = document.querySelector('svg.globe'); if (!svg) return;
  var R = {size * 0.42:.1f}, C = {size / 2}, T = {TILT} * Math.PI / 180, TERM = {TERMINATOR}, STEP = {STEP};
  var rings = {payload};
  function proj(lat, lon, r) {{
    var p = lat * Math.PI / 180, l = lon * Math.PI / 180;
    var x = r * Math.cos(p) * Math.sin(l), y = r * Math.sin(p), z = r * Math.cos(p) * Math.cos(l);
    return [x, -(y * Math.cos(T) - z * Math.sin(T)), y * Math.sin(T) + z * Math.cos(T), lon];
  }}
  var layer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  svg.appendChild(layer);
  svg.querySelectorAll('path[stroke="currentColor"], path[fill="var(--exact)"], path[stroke-dasharray]')
     .forEach(function(p){{ p.remove(); }});
  function path(pts) {{ return 'M' + pts.map(function(p){{ return p[0].toFixed(1) + ',' + p[1].toFixed(1); }}).join(' L'); }}
  function draw(phase) {{
    var out = [];
    rings.forEach(function(ring) {{
      var n = ring.s.length, pts = [];
      for (var i = 0, lon = -180; lon < 180; lon += STEP, i++)
        pts.push(proj(ring.lat, lon, R * (1 + {AMPLITUDE} * ring.s[(i + phase) % n])));
      var run = [], runs = [];
      pts.forEach(function(p) {{ if (p[2] > 0) run.push(p); else if (run.length) {{ runs.push(run); run = []; }} }});
      if (run.length) runs.push(run);
      runs.forEach(function(run) {{
        var past = run.filter(function(p){{ return p[3] <= TERM; }}).map(function(p){{ return [C + p[0], C + p[1]]; }});
        var fut = run.filter(function(p){{ return p[3] >= TERM; }});
        if (past.length > 1) out.push('<path d="' + path(past) + '" fill="none" stroke="currentColor" stroke-opacity="0.8" stroke-width="1.2" stroke-linejoin="round"/>');
        if (fut.length > 1) {{
          [[3, .07], [2, .12], [1, .2]].forEach(function(b) {{
            var up = [], lo = [];
            fut.forEach(function(p) {{
              var d = (p[3] - TERM) / (90 - TERM), s = R * 0.09 * b[0] * d, h = Math.hypot(p[0], p[1]) || 1;
              up.push([C + p[0] + p[0] / h * s, C + p[1] + p[1] / h * s]); lo.push([C + p[0] - p[0] / h * s, C + p[1] - p[1] / h * s]);
            }});
            out.push('<path d="' + path(up.concat(lo.reverse())) + ' Z" fill="var(--exact)" fill-opacity="' + b[1] + '"/>');
          }});
          out.push('<path d="' + path(fut.map(function(p){{ return [C + p[0], C + p[1]]; }})) + '" fill="none" stroke="var(--exact)" stroke-opacity="0.9" stroke-width="1.2" stroke-dasharray="3 3"/>');
        }}
      }});
    }});
    layer.innerHTML = out.join('');
  }}
  var phase = 0, last = 0;
  function tick(t) {{ if (t - last > 90) {{ last = t; draw(phase++); }} requestAnimationFrame(tick); }}
  requestAnimationFrame(tick);
}})();
</script>"""
