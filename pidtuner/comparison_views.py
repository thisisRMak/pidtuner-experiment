"""Comparison views for PIDTuner: Heatmap Table and Radar Chart."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from compare import METRIC_TIERS, RADAR_METRICS, METRIC_DIRECTION, normalize_column
from widgets import add_tooltip

# Plain ASCII only — Tkinter's default label font on this platform doesn't
# render the Unicode subscript block (e.g. "tₛ" silently drops the "s").
_METRIC_LABELS = {
    "OS%": "OS %", "ts": "ts", "Rise": "Rise\n(10-90%)",
    "IAE": "IAE\n(track)", "IAE_load": "IAE\n(load)",
    "ISU": "ISU\n(effort)",
    "Ms": "Ms", "Mt": "Mt", "u_tv": "TV(u)",
}

_BLACK_BOX_TOOLTIP = (
    "Black-box result: tuned from a model identified purely from a "
    "published signal (step/relay response) — the true transfer function "
    "was never consulted to produce these gains."
)


def _add_black_box_badge(cell, bg):
    """A small 'BB' tag with a hover tooltip, packed into a name cell."""
    badge = tk.Label(cell, text=" BB ", bg="#5b3a9e", fg="#ffffff",
                      font=("TkDefaultFont", 7, "bold"), padx=2, pady=0)
    badge.pack(side="right", padx=(4, 6))
    add_tooltip(badge, _BLACK_BOX_TOOLTIP)
    return badge


def _add_delay_badge(cell, row):
    """A small 'L' tag with a hover tooltip, packed into a name cell —
    shown only when a fitted/known FOPDT model behind this row's gains
    carries a *significant* delay (row['has_time_delay'] is True; see
    identify._check_delay_significant for the significance judgment)."""
    delay_L = row.get("delay_L")
    badge = tk.Label(cell, text=" L ", bg="#a6621a", fg="#ffffff",
                      font=("TkDefaultFont", 7, "bold"), padx=2, pady=0)
    badge.pack(side="right", padx=(4, 2))
    tooltip = (f"Time delay detected: L={delay_L:.3g}s. {row.get('delay_reason', '')}"
               if delay_L is not None else "Time delay detected.")
    add_tooltip(badge, tooltip)
    return badge


def _heat_color(v):
    """Normalized goodness v∈[0,1] (1=best) → red→yellow→green hex."""
    if v is None or not np.isfinite(v):
        return "#cccccc"
    v = max(0.0, min(1.0, float(v)))
    # red (215,48,39) → yellow (255,255,191) → green (26,152,80)
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


def draw_heatmap_tab(parent, rows):
    """Render the heatmap table view on parent frame/tab.

    Methods run across the columns (one per tuning method), metrics run
    down the rows, grouped and ordered by priority tier (see
    compare.METRIC_TIERS: P0 first, then P2, ...). Colour is still computed
    per metric across methods — only the grid orientation is transposed
    relative to the old methods-as-rows layout.
    """
    for child in parent.winfo_children():
        child.destroy()
    if not rows:
        ttk.Label(parent, padding=20, foreground="#888",
                  text="Tune methods to compare them here.").pack()
        return

    # Footnote pinned to the bottom first so it always shows.
    ttk.Label(parent, padding=(8, 4), foreground="#555",
              text="ts = 2% settling time.  Ms robust band ~ [1.4, 2.0].  "
                   "TV(u) = control-signal total variation (smoothness).  "
                   "ISU = integral of u^2 dt, control effort.  IAE(load) "
                   "from a unit load step at the plant input.").pack(
                  side="bottom", fill="x")
    # Methods as columns (9 base tuning methods fit without scrolling),
    # metrics as rows grouped by tier — grid fills the tab.
    grid = ttk.Frame(parent, padding=(6, 6))
    grid.pack(fill="both", expand=True)

    n_methods = len(rows)
    hdr_font = ("TkDefaultFont", 9, "bold")
    tk.Label(grid, text="Metric", font=hdr_font, anchor="w",
             padx=8, pady=4).grid(row=0, column=0, sticky="nsew")

    for c, r in enumerate(rows, start=1):
        stable = r.get("stable")
        name_bg = "#ffffff" if stable else "#dddddd"
        name_cell = tk.Frame(grid, bg=name_bg)
        name_cell.grid(row=0, column=c, sticky="nsew")
        tk.Label(name_cell, text=r["name"], font=hdr_font, anchor="w",
                 padx=8, pady=4, bg=name_bg, wraplength=90, justify="left"
                 ).pack(side="left", fill="x", expand=True)
        if r.get("black_box"):
            _add_black_box_badge(name_cell, name_bg)
        if r.get("has_time_delay"):
            _add_delay_badge(name_cell, r)
        if not stable:
            warn = tk.Label(name_cell, text="⚠", bg=name_bg, fg="#a00")
            warn.pack(side="right", padx=(0, 4))
            add_tooltip(warn, r.get("error", "failed"))

    # Normalize each metric across methods (same computation as before,
    # just no longer tied to a fixed column index).
    norm = {}
    for _, metrics in METRIC_TIERS:
        for m in metrics:
            col = [r.get(m, float("inf")) if r.get("stable") else float("inf")
                   for r in rows]
            norm[m] = normalize_column(col, direction=METRIC_DIRECTION[m])

    rr = 1
    for tier_name, metrics in METRIC_TIERS:
        tk.Label(grid, text=tier_name, font=hdr_font, anchor="w",
                 padx=8, pady=2, bg="#e5e5e5"
                 ).grid(row=rr, column=0, columnspan=n_methods + 1,
                        sticky="nsew")
        rr += 1
        for m in metrics:
            tk.Label(grid, text=_METRIC_LABELS.get(m, m), anchor="w",
                     padx=8, pady=3, justify="left"
                     ).grid(row=rr, column=0, sticky="nsew")
            for c, r in enumerate(rows, start=1):
                if not r.get("stable"):
                    tk.Label(grid, text="—", padx=8, pady=3, bg="#dddddd",
                             anchor="center").grid(row=rr, column=c,
                                                   sticky="nsew")
                    continue
                val = r.get(m, float("nan"))
                color = _heat_color(norm[m][c - 1])
                txt = f"{val:.3g}" if np.isfinite(val) else "—"
                tk.Label(grid, text=txt, padx=8, pady=3, bg=color,
                         anchor="center").grid(row=rr, column=c,
                                               sticky="nsew")
            rr += 1

    grid.columnconfigure(0, weight=3, minsize=140)
    for c in range(1, n_methods + 1):
        grid.columnconfigure(c, weight=2, minsize=80)
    for i in range(rr):
        grid.rowconfigure(i, weight=1)


def draw_radar_tab(parent, rows):
    """Render the radar chart view on parent frame/tab.

    Methods are the spokes — every row gets an axis, including unstable
    ones (they simply read 0 on every metric line, same convention as the
    heatmap's greyed-out columns). Each P0/P1 metric (compare.RADAR_METRICS)
    is its own polygon across those spokes — the transpose of the old
    layout, which put metrics on the spokes and one polygon per method.
    """
    for child in parent.winfo_children():
        child.destroy()
    if not rows:
        ttk.Label(parent, padding=20, foreground="#888",
                  text="Tune methods to compare them here.").pack()
        return
    if not any(r.get("stable") for r in rows):
        ttk.Label(parent, padding=20, foreground="#888",
                  text="Tune at least one stable method to see the radar.").pack()
        return

    metrics = RADAR_METRICS
    labels = [r["name"] + (" [BB]" if r.get("black_box") else "")
              + (" [L]" if r.get("has_time_delay") else "") for r in rows]

    goodness = []  # list per-metric of normalized arrays (1=best), across methods
    for m in metrics:
        col = [r.get(m, float("inf")) if r.get("stable") else float("inf")
               for r in rows]
        goodness.append(normalize_column(col, direction=METRIC_DIRECTION[m]))
    goodness = np.array(goodness)  # shape (n_metrics, n_methods)

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
        label = _METRIC_LABELS.get(m, m).replace("\n", " ")
        ax.plot(angles, vals, lw=1.6, color=color, label=label)
        ax.fill(angles, vals, color=color, alpha=0.06)

    ax.set_title("Each metric normalized so outer = better across methods\n"
                 "(P0 + P1 metrics; unstable methods read 0 on every axis)",
                 fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10),
              fontsize=8, framealpha=0.9)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(canvas, parent)
