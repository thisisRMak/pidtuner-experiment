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
from compare import TABLE_METRICS, METRIC_DIRECTION, normalize_column

_METRIC_LABELS = {
    "OS%": "OS %", "ts": "tₛ (2%)", "IAE": "IAE\n(track)",
    "IAE_load": "IAE\n(load)", "Ms": "Mₛ", "Mt": "Mₜ", "u_tv": "TV(u)",
}


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
    """Render the heatmap table view on parent frame/tab."""
    for child in parent.winfo_children():
        child.destroy()
    if not rows:
        ttk.Label(parent, padding=20, foreground="#888",
                  text="Tune methods to compare them here.").pack()
        return

    # Footnote pinned to the bottom first so it always shows.
    ttk.Label(parent, padding=(8, 4), foreground="#555",
              text="Mₛ robust band ≈ [1.4, 2.0].  TV(u) = control-signal "
                   "total variation (smoothness).  IAE(load) from a unit "
                   "load step at the plant input.").pack(side="bottom",
                                                         fill="x")
    # 12 methods × 8 columns fits without scrolling — grid fills the tab.
    grid = ttk.Frame(parent, padding=(6, 6))
    grid.pack(fill="both", expand=True)

    metrics = TABLE_METRICS
    hdr_font = ("TkDefaultFont", 9, "bold")
    tk.Label(grid, text="Method", font=hdr_font, anchor="w",
             padx=8, pady=4).grid(row=0, column=0, sticky="nsew")
    for c, m in enumerate(metrics, start=1):
        tk.Label(grid, text=_METRIC_LABELS.get(m, m), font=hdr_font,
                 padx=8, pady=4, justify="center").grid(row=0, column=c,
                                                        sticky="nsew")

    norm = {}
    for m in metrics:
        col = [r.get(m, float("inf")) if r.get("stable") else float("inf")
               for r in rows]
        norm[m] = normalize_column(col, direction=METRIC_DIRECTION[m])

    for i, r in enumerate(rows):
        rr = i + 1
        stable = r.get("stable")
        name_bg = "#ffffff" if stable else "#dddddd"
        tk.Label(grid, text=r["name"], anchor="w", padx=8, pady=3,
                 bg=name_bg).grid(row=rr, column=0, sticky="nsew")
        if not stable:
            tk.Label(grid, text=f"— {r.get('error', 'failed')} —",
                     anchor="w", padx=8, pady=3, bg=name_bg, fg="#a00"
                     ).grid(row=rr, column=1, columnspan=len(metrics),
                            sticky="nsew")
            continue
        for c, m in enumerate(metrics, start=1):
            val = r.get(m, float("nan"))
            color = _heat_color(norm[m][i])
            txt = f"{val:.3g}" if np.isfinite(val) else "—"
            tk.Label(grid, text=txt, padx=8, pady=3, bg=color,
                     anchor="center").grid(row=rr, column=c, sticky="nsew")

    grid.columnconfigure(0, weight=3, minsize=140)
    for c in range(1, len(metrics) + 1):
        grid.columnconfigure(c, weight=2, minsize=70)
    for rr in range(len(rows) + 1):
        grid.rowconfigure(rr, weight=1)


def draw_radar_tab(parent, rows):
    """Render the radar chart view on parent frame/tab."""
    for child in parent.winfo_children():
        child.destroy()
    if not any(r.get("stable") for r in rows):
        ttk.Label(parent, padding=20, foreground="#888",
                  text="Tune at least one stable method to see the radar.").pack()
        return

    stable = [r for r in rows if r.get("stable")]
    if not stable:
        ttk.Label(parent, text="No stable methods to plot.",
                  padding=20).pack()
        return

    # Six axes, each normalized so OUTER = better.
    axes_spec = [
        ("Track\n(IAE)", "IAE", -1),
        ("Load rej.\n(IAE)", "IAE_load", -1),
        ("Robust\n(Mₛ)", "Ms", -1),
        ("Low OS\n(OS%)", "OS%", -1),
        ("Speed\n(tₛ)", "ts", -1),
        ("Smooth\n(TV)", "u_tv", -1),
    ]
    labels = [a[0] for a in axes_spec]
    goodness = []  # list per-axis of normalized arrays (1=best)
    for _, key, direction in axes_spec:
        col = [r.get(key, float("inf")) for r in stable]
        goodness.append(normalize_column(col, direction=direction))
    goodness = np.array(goodness)  # shape (n_axes, n_methods)

    n_ax = len(axes_spec)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig = Figure(figsize=(7.2, 6.4), dpi=100)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_rlabel_position(0)

    cmap = matplotlib.colormaps.get_cmap("tab20")
    for j, r in enumerate(stable):
        vals = goodness[:, j].tolist()
        vals += vals[:1]
        color = cmap(j % 20)
        ax.plot(angles, vals, lw=1.6, color=color, label=r["name"])
        ax.fill(angles, vals, color=color, alpha=0.06)

    ax.set_title("Each axis normalized so outer = best across methods",
                 fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10),
              fontsize=8, framealpha=0.9)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(canvas, parent)
