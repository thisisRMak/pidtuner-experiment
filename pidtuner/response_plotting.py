"""Response plot tab: 3-axis Matplotlib figure (PV/SP, control effort, error)
overlaying the closed-loop response of the tuned controllers in a session.
"""

from __future__ import annotations

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)


def create_response_figure(parent):
    """Build the 3-row response Figure/axes/canvas + toolbar inside parent.

    Returns (fig, ax_y, ax_u, ax_e, canvas).
    """
    fig = Figure(figsize=(9, 8), dpi=100)
    ax_y = fig.add_subplot(311)
    ax_u = fig.add_subplot(312, sharex=ax_y)
    ax_e = fig.add_subplot(313, sharex=ax_y)
    for ax in (ax_y, ax_u, ax_e):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(canvas, parent)
    return fig, ax_y, ax_u, ax_e, canvas


def draw_response_tab(fig, ax_y, ax_u, ax_e, canvas, active_entries):
    """Redraw the response plot for the given active (enabled + simulated)
    TunedEntry-like objects (must expose .label, .color, .sim).
    """
    ax_y.clear()
    ax_u.clear()
    ax_e.clear()
    for ax in (ax_y, ax_u, ax_e):
        ax.grid(True, alpha=0.3)
    ax_y.set_ylabel("PV / SP")
    ax_u.set_ylabel("control u(t)")
    ax_e.set_ylabel("error e(t)")
    ax_e.set_xlabel("time (s)")

    if not active_entries:
        ax_y.set_title("No tuned controllers shown — tune a method "
                       "or tick one in the session list.")
        canvas.draw()
        return

    seen_kinds = set()
    for entry in active_entries:
        kind = entry.sim.sp_kind
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        kind_label = f"setpoint ({kind})" if len(active_entries) > 1 else "setpoint"
        same_kind = [e for e in active_entries if e.sim.sp_kind == kind]
        longest = max(same_kind, key=lambda e: len(e.sim.t))
        ax_y.plot(longest.sim.t, longest.sim.sp, "--",
                  color="#666", linewidth=1.0, alpha=0.7,
                  label=kind_label)

    saturated_any = False
    for entry in active_entries:
        label = entry.label
        if entry.sim.metrics.get("unstable"):
            label += "  [UNSTABLE]"
        ax_y.plot(entry.sim.t, entry.sim.y, color=entry.color,
                  linewidth=1.5, label=label)
        ax_u.plot(entry.sim.t, entry.sim.u, color=entry.color,
                  linewidth=1.2, label=entry.label)
        # Mark actuator-saturated samples (u pinned at u_min/u_max) distinctly
        # — this is where conditional-integration vs. back-calculation
        # anti-windup actually differ in *behavior*, even though the u(t)
        # plateau itself looks identical between the two modes.
        sat_mask = (np.isclose(entry.sim.u, entry.sim.u_min) |
                   np.isclose(entry.sim.u, entry.sim.u_max))
        if sat_mask.any():
            saturated_any = True
            ax_u.plot(entry.sim.t[sat_mask], entry.sim.u[sat_mask],
                      color=entry.color, marker="o", markersize=3,
                      linestyle="none", alpha=0.7)
        ax_e.plot(entry.sim.t, entry.sim.e, color=entry.color,
                  linewidth=1.2, label=entry.label)

    ax_y.legend(loc="lower right", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    if saturated_any:
        ax_u.plot([], [], marker="o", markersize=3, linestyle="none",
                  color="#666", label="saturated (u at u_min/u_max)")
        ax_u.legend(loc="lower right", fontsize=7)
    stable_ys = [e.sim.y for e in active_entries if not e.sim.metrics.get("unstable")]
    if stable_ys:
        max_sp = max(np.max(np.abs(e.sim.sp)) for e in active_entries)
        ax_y.set_ylim(-0.2 * max_sp, max(2.0 * max_sp, 0.1))

    fig.tight_layout()
    canvas.draw()
