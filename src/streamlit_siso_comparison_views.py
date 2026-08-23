"""Streamlit equivalents of pid_comparison_views.py's heatmap/radar tabs.

pid_comparison_views.py's *drawing* code is Tk-widget-specific (tk.Label
grids, FigureCanvasTkAgg) and isn't reusable here, but the data it draws
already comes from plain pid_compare.py functions (METRIC_TIERS,
RADAR_METRICS, METRIC_DIRECTION, normalize_column) — this module reuses
those and re-implements only the rendering, once per view:
  - heatmap: an HTML table (st.markdown unsafe_allow_html) rather than a
    Tk Label grid, since Streamlit has no native colored-cell grid widget.
  - radar: the same matplotlib Figure code, rendered via st.pyplot
    instead of FigureCanvasTkAgg/NavigationToolbar2Tk.
"""

from __future__ import annotations

import html

import numpy as np
import streamlit as st
import matplotlib
from matplotlib.figure import Figure

from pid_compare import METRIC_TIERS, RADAR_METRICS, METRIC_DIRECTION, normalize_column

_METRIC_LABELS = {
    "OS%": "OS %", "ts": "ts", "Rise": "Rise (10-90%)",
    "IAE": "IAE (track)", "IAE_load": "IAE (load)",
    "ISU": "ISU (effort)",
    "Ms": "Ms", "Mt": "Mt", "u_tv": "TV(u)",
}

_FOOTNOTE = (
    "ts = 2% settling time. Ms robust band ~ [1.4, 2.0]. "
    "TV(u) = control-signal total variation (smoothness). "
    "ISU = integral of u² dt, control effort. "
    "IAE(load) from a unit load step at the plant input."
)


def _heat_color(v):
    """Normalized goodness v∈[0,1] (1=best) → red→yellow→green hex."""
    if v is None or not np.isfinite(v):
        return "#cccccc"
    v = max(0.0, min(1.0, float(v)))
    if v < 0.5:
        f = v / 0.5
        r = 215 + f * (255 - 215)
        g = 48 + f * (255 - 48)
        b = 39 + f * (191 - 39)
    else:
        f = (v - 0.5) / 0.5
        r = 255 + f * (26 - 255)
        g = 255 + f * (152 - 255)
        b = 191 + f * (80 - 191)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def render_heatmap(rows):
    """Methods across columns, metrics down rows grouped by tier — same
    orientation as pid_comparison_views.draw_heatmap_tab."""
    if not rows:
        st.caption("Tune methods to compare them here.")
        return

    # All cell backgrounds are light pastel colors regardless of the app's
    # theme, so text color is fixed dark here rather than left to inherit —
    # inheriting would go invisible (light-on-light) under Streamlit's dark
    # theme, which uses light default text.
    def th(text, bg="#f0f0f0"):
        return (f'<th style="background:{bg};color:#111;padding:6px 10px;'
                f'text-align:left;">{text}</th>')

    def td(text, bg="#ffffff", align="center"):
        return (f'<td style="background:{bg};color:#111;padding:5px 10px;'
                f'text-align:{align};">{text}</td>')

    header_cells = [th("Metric")]
    for r in rows:
        stable = r.get("stable")
        # r["name"] and r.get("error") are currently always drawn from the
        # fixed method registry in pid_compare.py, never free text — escaped
        # anyway since this string is spliced into raw HTML, so a future
        # source of these values (a user-entered label, say) can't break
        # the table or inject markup.
        name = html.escape(r["name"])
        if r.get("black_box"):
            name += ' <span style="background:#5b3a9e;color:#fff;font-size:0.7em;' \
                    'padding:1px 4px;border-radius:3px;">BB</span>'
        if r.get("has_time_delay"):
            name += ' <span style="background:#a6621a;color:#fff;font-size:0.7em;' \
                    'padding:1px 4px;border-radius:3px;">L</span>'
        if not stable:
            err = html.escape(r.get("error", "failed"))
            name += f' <span title="{err}" style="color:#a00;">⚠</span>'
        header_cells.append(th(name, bg="#ffffff" if stable else "#dddddd"))

    norm = {}
    for _, metrics in METRIC_TIERS:
        for m in metrics:
            col = [r.get(m, float("inf")) if r.get("stable") else float("inf")
                   for r in rows]
            norm[m] = normalize_column(col, direction=METRIC_DIRECTION[m])

    body_rows = []
    for tier_name, metrics in METRIC_TIERS:
        body_rows.append(
            f'<tr><td colspan="{len(rows) + 1}" style="background:#e5e5e5;'
            f'color:#111;padding:4px 10px;font-weight:bold;">{tier_name}</td></tr>')
        for m in metrics:
            cells = [th(_METRIC_LABELS.get(m, m), bg="#f7f7f7")]
            for c, r in enumerate(rows):
                if not r.get("stable"):
                    cells.append(td("—", bg="#dddddd"))
                    continue
                val = r.get(m, float("nan"))
                color = _heat_color(norm[m][c])
                txt = f"{val:.3g}" if np.isfinite(val) else "—"
                cells.append(td(txt, bg=color))
            body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        '<table style="border-collapse:collapse;width:100%;font-size:0.85em;">'
        f'<thead><tr>{"".join(header_cells)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(_FOOTNOTE)


def render_radar(rows):
    """Methods are the spokes, one polygon per P0/P1 metric — same as
    pid_comparison_views.draw_radar_tab."""
    if not rows:
        st.caption("Tune methods to compare them here.")
        return
    if not any(r.get("stable") for r in rows):
        st.caption("Tune at least one stable method to see the radar.")
        return

    metrics = RADAR_METRICS
    labels = [r["name"] + (" [BB]" if r.get("black_box") else "")
              + (" [L]" if r.get("has_time_delay") else "") for r in rows]

    goodness = []
    for m in metrics:
        col = [r.get(m, float("inf")) if r.get("stable") else float("inf")
               for r in rows]
        goodness.append(normalize_column(col, direction=METRIC_DIRECTION[m]))
    goodness = np.array(goodness)

    n_ax = len(rows)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig = Figure(figsize=(7.6, 6.4), dpi=100)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_rlabel_position(0)

    cmap = matplotlib.colormaps.get_cmap("tab10")
    for i, m in enumerate(metrics):
        vals = goodness[i].tolist()
        vals += vals[:1]
        color = cmap(i % 10)
        label = _METRIC_LABELS.get(m, m)
        ax.plot(angles, vals, lw=1.6, color=color, label=label)
        ax.fill(angles, vals, color=color, alpha=0.06)

    ax.set_title("Each metric normalized so outer = better across methods\n"
                 "(P0 + P1 metrics; unstable methods read 0 on every axis)",
                 fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10),
              fontsize=8, framealpha=0.9)
    fig.tight_layout()
    st.pyplot(fig)
