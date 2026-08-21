"""Hand-rolled inline-SVG chart primitives.

No charting library: the output must be a single self-contained HTML file that a
recruiter can open from a link with no CDN, no build step and no network. Every
mark follows one fixed spec sheet —

* bars capped at 24px thick, 4px rounded data-end, square at the baseline
* lines 2px with round joins, area wash at 10% opacity
* markers >= 8px carrying a 2px surface-coloured ring so they stay legible on crossings
* a 2px surface gap between touching marks, never a stroke
* hairline solid gridlines, one step off the surface, recessive
* labels wear text tokens, never the series colour

Colours arrive as CSS custom properties so light and dark themes swap in one place.
"""

from __future__ import annotations

from html import escape

# --------------------------------------------------------------------------- utils


def _fmt(value: float, kind: str = "number") -> str:
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "pct0":
        return f"{value * 100:.0f}%"
    if kind == "money":
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"${value / 1_000:.1f}K"
        return f"${value:,.0f}"
    if kind == "x":
        return f"{value:.2f}×"
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _nice_ceiling(value: float) -> float:
    """Round a max up to a clean axis top (1 / 2 / 2.5 / 5 × 10^n)."""
    if value <= 0:
        return 1.0
    import math

    exponent = math.floor(math.log10(value))
    base = 10**exponent
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= step * base:
            return step * base
    return 10 * base


def _tooltip(label: str, value: str) -> str:
    return f' data-tip="{escape(label)}" data-tipval="{escape(value)}"'


# ----------------------------------------------------------------------- line/area


def line_area(points: list[tuple[str, float]], value_kind: str = "pct",
              height: int = 230, series: str = "var(--series-1)",
              label_every: int = 2, width: int = 720) -> str:
    """Single-series line with a 10% area wash. Endpoint and extreme are labelled.

    ``width`` is the viewBox width, and it matters: an SVG scales to its container,
    so a 720-wide viewBox dropped into a 340px card shrinks every label to ~5px.
    Pass a width close to the card's rendered width and the type stays at its
    intended size.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 26, 18, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    values = [v for _, v in points]
    top = _nice_ceiling(max(values) * 1.12)

    def x_of(i: int) -> float:
        return pad_l + (plot_w * i / max(1, len(points) - 1))

    def y_of(v: float) -> float:
        return pad_t + plot_h - (plot_h * v / top)

    grid, ticks = [], []
    for step in range(5):
        v = top * step / 4
        y = y_of(v)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/>')
        ticks.append(f'<text x="{pad_l - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{_fmt(v, value_kind)}</text>')

    line = " ".join(f"{'M' if i == 0 else 'L'}{x_of(i):.1f},{y_of(v):.1f}" for i, (_, v) in enumerate(points))
    area = f"{line} L{x_of(len(points) - 1):.1f},{pad_t + plot_h} L{pad_l},{pad_t + plot_h} Z"

    x_labels = [
        f'<text x="{x_of(i):.1f}" y="{height - 10}" class="tick" text-anchor="middle">{escape(lbl)}</text>'
        for i, (lbl, _) in enumerate(points) if i % label_every == 0 or i == len(points) - 1
    ]

    # Selective direct labels: the first point (the peak of an early-life hazard)
    # and the endpoint. Never a number on every point.
    marks, labels = [], []
    highlight = {0, len(points) - 1}
    for i, (lbl, v) in enumerate(points):
        cx, cy = x_of(i), y_of(v)
        marks.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{5 if i in highlight else 4}" '
            f'fill="{series}" class="dot"{_tooltip(lbl, _fmt(v, value_kind))}/>'
        )
        if i in highlight:
            anchor = "start" if i == 0 else "end"
            dx = 10 if i == 0 else -10
            labels.append(
                f'<text x="{cx + dx:.1f}" y="{cy - 12:.1f}" class="pointlabel" '
                f'text-anchor="{anchor}">{_fmt(v, value_kind)}</text>'
            )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart" role="img" preserveAspectRatio="xMidYMid meet">
  {''.join(grid)}
  <path d="{area}" fill="{series}" fill-opacity="0.10"/>
  <path d="{line}" fill="none" stroke="{series}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
  {''.join(marks)}{''.join(labels)}
  {''.join(ticks)}{''.join(x_labels)}
</svg>"""


# -------------------------------------------------------------------------- columns


def columns(items: list[tuple[str, float]], value_kind: str = "number",
            colors: list[str] | None = None, height: int = 230,
            label_values: bool = True, series: str = "var(--series-1)",
            width: int = 720) -> str:
    """Vertical columns: 24px cap, 4px rounded cap, square baseline, 2px surface gap.

    Match ``width`` to the card the chart lands in — see ``line_area``.
    """
    pad_l, pad_r, pad_t, pad_b = 52, 20, 26, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    values = [v for _, v in items]
    top = _nice_ceiling(max(values) * 1.15) if values else 1.0
    band = plot_w / max(1, len(items))
    bar_w = min(24.0, band - 14)

    grid, ticks = [], []
    for step in range(5):
        v = top * step / 4
        y = pad_t + plot_h - plot_h * v / top
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/>')
        ticks.append(f'<text x="{pad_l - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{_fmt(v, value_kind)}</text>')

    bars, labels, caps = [], [], []
    for i, (label, v) in enumerate(items):
        cx = pad_l + band * i + band / 2
        h = max(2.0, plot_h * v / top)
        x, y = cx - bar_w / 2, pad_t + plot_h - h
        fill = (colors[i] if colors else series)
        r = min(4.0, bar_w / 2, h)
        # rounded top corners only; the baseline end stays square
        bars.append(
            f'<path d="M{x:.1f},{pad_t + plot_h} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
            f'L{x + bar_w - r:.1f},{y:.1f} Q{x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},{y + r:.1f} '
            f'L{x + bar_w:.1f},{pad_t + plot_h} Z" fill="{fill}" class="bar"'
            f'{_tooltip(label, _fmt(v, value_kind))}/>'
        )
        labels.append(
            f'<text x="{cx:.1f}" y="{height - 12}" class="tick" text-anchor="middle">{escape(label)}</text>'
        )
        if label_values:
            caps.append(
                f'<text x="{cx:.1f}" y="{y - 8:.1f}" class="pointlabel" text-anchor="middle">'
                f'{_fmt(v, value_kind)}</text>'
            )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart" role="img" preserveAspectRatio="xMidYMid meet">
  {''.join(grid)}{''.join(bars)}{''.join(caps)}{''.join(ticks)}{''.join(labels)}
</svg>"""


# ------------------------------------------------------------------------ h-bars


def hbars(items: list[tuple[str, float]], value_kind: str = "pct0",
          series: str = "var(--series-1)", row_h: int = 34,
          label_w: int = 168, width: int = 720) -> str:
    """Horizontal bars for ranked magnitude — the form that survives long labels."""
    height = row_h * len(items) + 14
    plot_w = width - label_w - 78
    top = max([v for _, v in items] + [1e-9])
    bar_h = min(24, row_h - 14)

    rows = []
    for i, (label, v) in enumerate(items):
        y = 8 + row_h * i
        w = max(2.0, plot_w * v / top)
        r = min(4.0, bar_h / 2, w)
        x = label_w
        rows.append(
            f'<text x="{label_w - 12}" y="{y + bar_h / 2 + 4:.1f}" class="rowlabel" text-anchor="end">'
            f'{escape(label)}</text>'
            f'<path d="M{x:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'L{x + w:.1f},{y + bar_h - r:.1f} Q{x + w:.1f},{y + bar_h:.1f} {x + w - r:.1f},{y + bar_h:.1f} '
            f'L{x:.1f},{y + bar_h:.1f} Z" fill="{series}" class="bar"{_tooltip(label, _fmt(v, value_kind))}/>'
            f'<text x="{x + w + 10:.1f}" y="{y + bar_h / 2 + 4:.1f}" class="pointlabel">'
            f'{_fmt(v, value_kind)}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart" role="img" preserveAspectRatio="xMidYMid meet">
  {''.join(rows)}
</svg>"""


# --------------------------------------------------------------------------- donut


def donut(items: list[tuple[str, float]], colors: list[str], centre_label: str = "",
          centre_value: str = "", size: int = 240) -> str:
    """Donut with a 2px surface gap between arcs and direct % labels outside."""
    import math

    cx = cy = size / 2
    r_outer, r_inner = size / 2 - 46, size / 2 - 76
    total = sum(v for _, v in items) or 1.0
    gap_deg = 1.6  # rendered as the 2px surface separator between arcs

    arcs, labels = [], []
    angle = -90.0
    for i, (label, v) in enumerate(items):
        sweep = 360.0 * v / total
        a0, a1 = angle + gap_deg / 2, angle + sweep - gap_deg / 2
        large = 1 if (a1 - a0) > 180 else 0
        p0 = (cx + r_outer * math.cos(math.radians(a0)), cy + r_outer * math.sin(math.radians(a0)))
        p1 = (cx + r_outer * math.cos(math.radians(a1)), cy + r_outer * math.sin(math.radians(a1)))
        p2 = (cx + r_inner * math.cos(math.radians(a1)), cy + r_inner * math.sin(math.radians(a1)))
        p3 = (cx + r_inner * math.cos(math.radians(a0)), cy + r_inner * math.sin(math.radians(a0)))
        arcs.append(
            f'<path d="M{p0[0]:.1f},{p0[1]:.1f} A{r_outer},{r_outer} 0 {large} 1 {p1[0]:.1f},{p1[1]:.1f} '
            f'L{p2[0]:.1f},{p2[1]:.1f} A{r_inner},{r_inner} 0 {large} 0 {p3[0]:.1f},{p3[1]:.1f} Z" '
            f'fill="{colors[i % len(colors)]}" class="bar"'
            f'{_tooltip(label, f"{v:,.0f} · {v / total * 100:.0f}%")}/>'
        )
        mid = math.radians((a0 + a1) / 2)
        lx, ly = cx + (r_outer + 16) * math.cos(mid), cy + (r_outer + 16) * math.sin(mid)
        anchor = "start" if math.cos(mid) > 0.15 else ("end" if math.cos(mid) < -0.15 else "middle")
        labels.append(
            f'<text x="{lx:.1f}" y="{ly + 4:.1f}" class="pointlabel" text-anchor="{anchor}">'
            f'{v / total * 100:.0f}%</text>'
        )
        angle += sweep

    centre = ""
    if centre_value:
        centre = (
            f'<text x="{cx}" y="{cy - 2}" class="donut-value" text-anchor="middle">{escape(centre_value)}</text>'
            f'<text x="{cx}" y="{cy + 18}" class="donut-label" text-anchor="middle">{escape(centre_label)}</text>'
        )

    return f"""<svg viewBox="0 0 {size} {size}" class="chart donut" role="img" preserveAspectRatio="xMidYMid meet">
  {''.join(arcs)}{''.join(labels)}{centre}
</svg>"""


# --------------------------------------------------------------------------- misc


def legend(entries: list[tuple[str, str]]) -> str:
    """Legend is always present for >= 2 series; identity never rests on colour alone."""
    items = "".join(
        f'<span class="legend-item"><i style="background:{color}"></i>{escape(label)}</span>'
        for label, color in entries
    )
    return f'<div class="legend">{items}</div>'


def meter(value: float, top: float, color: str = "var(--series-1)") -> str:
    """Track is a lighter step of the same ramp so state reads across the whole bar."""
    pct = max(0.0, min(1.0, value / top if top else 0))
    return (
        f'<div class="meter"><div class="meter-fill" style="width:{pct * 100:.1f}%;background:{color}"></div></div>'
    )
