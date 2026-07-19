"""Method-specific argument panels for the Tkinter tuning UI.

Each `build_*_panel` function constructs the input-widgets Frame for one
tuning method and returns (frame, namespace), where `namespace` holds the
tk Variables (and any widgets whose state the app needs to toggle later,
e.g. entries that get enabled/disabled) that the caller wires up.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace


def build_pole_cancel_panel(parent, labeled_entry, on_mode_change):
    frame = ttk.Frame(parent)
    ttk.Label(frame, text="Cancel poles at s = −p₁, s = −p₂").pack(anchor="w")
    mode = tk.StringVar(value="auto")
    ttk.Radiobutton(frame, text="Auto (two slowest stable poles)",
                    variable=mode, value="auto",
                    command=on_mode_change).pack(anchor="w")
    ttk.Radiobutton(frame, text="Manual",
                    variable=mode, value="manual",
                    command=on_mode_change).pack(anchor="w")
    p1 = tk.StringVar(value="0.1")
    p2 = tk.StringVar(value="1.0")
    p1_entry = labeled_entry(frame, "p₁ (positive)", p1, width=12)
    p2_entry = labeled_entry(frame, "p₂ (positive)", p2, width=12)
    kd = tk.StringVar(value="1.0")
    labeled_entry(frame, "Kd (free)", kd, width=12)
    pole_info = ttk.Label(frame, text="", foreground="#06a",
                          wraplength=380, justify="left",
                          font=("TkDefaultFont", 8))
    pole_info.pack(anchor="w", pady=(4, 0))
    ns = SimpleNamespace(mode=mode, p1=p1, p2=p2, kd=kd,
                        p1_entry=p1_entry, p2_entry=p2_entry,
                        pole_info=pole_info)
    return frame, ns


def build_zn1_panel(parent, labeled_entry):
    frame = ttk.Frame(parent)
    step = tk.StringVar(value="1.0")
    noise = tk.StringVar(value="0.0")
    labeled_entry(frame, "step amplitude", step, width=12)
    labeled_entry(frame, "noise σ", noise, width=12)
    ttk.Label(frame,
              text="ZN-I is aggressive for tracking — tick 'Halve gains'\n"
                   "in Tuning method for the PEi9e recommendation.",
              foreground="#666", justify="left", font=("TkDefaultFont", 8)
              ).pack(anchor="w", pady=(2, 0))
    return frame, SimpleNamespace(step=step, noise=noise)


def build_zn2_panel(parent, labeled_entry):
    frame = ttk.Frame(parent)
    ttk.Label(frame, text="How to obtain Ku, Pu:").pack(anchor="w")
    source = tk.StringVar(value="bode")
    ttk.Radiobutton(frame, text="Analytical (Bode crossover of −180°)",
                    variable=source, value="bode").pack(anchor="w")
    ttk.Radiobutton(frame, text="Simulated relay-feedback test",
                    variable=source, value="relay").pack(anchor="w")
    relay_h = tk.StringVar(value="1.0")
    relay_T = tk.StringVar(value="80.0")
    labeled_entry(frame, "relay h", relay_h, width=12)
    labeled_entry(frame, "relay duration", relay_T, width=12)
    ttk.Label(frame,
              text="Like ZN-I, tick 'Halve gains' for tracking.",
              foreground="#666", font=("TkDefaultFont", 8)
              ).pack(anchor="w", pady=(2, 0))
    return frame, SimpleNamespace(source=source, relay_h=relay_h, relay_T=relay_T)


def build_amigo_panel(parent):
    frame = ttk.Frame(parent)
    integrating = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame,
                    text="Use integrating form (G = A·exp(−Ls)/s)",
                    variable=integrating).pack(anchor="w")
    return frame, SimpleNamespace(integrating=integrating)


def build_simc_panel(parent, labeled_entry):
    frame = ttk.Frame(parent)
    tau_c = tk.StringVar(value="")
    tau2 = tk.StringVar(value="")
    ttk.Label(frame,
              text="τc — closed-loop time constant (blank = use L)").pack(anchor="w")
    labeled_entry(frame, "τc", tau_c, width=12)
    ttk.Label(frame,
              text="τ₂ — 2nd time constant for PID (blank = PI only)",
              font=("TkDefaultFont", 8), foreground="#666").pack(anchor="w")
    labeled_entry(frame, "τ₂", tau2, width=12)
    return frame, SimpleNamespace(tau_c=tau_c, tau2=tau2)


def build_boyd_panel(parent, labeled_entry):
    frame = ttk.Frame(parent)
    Ms = tk.StringVar(value="1.4")
    Mt = tk.StringVar(value="1.4")
    labeled_entry(frame, "Ms (sens. bound)", Ms, width=12)
    labeled_entry(frame, "Mt (comp.sens. bound)", Mt, width=12)
    ttk.Label(frame,
              text="Smaller Ms/Mt → more robust, less aggressive.\n"
                   "Typical range 1.2 – 2.0.",
              foreground="#666", justify="left", font=("TkDefaultFont", 8)
              ).pack(anchor="w", pady=(2, 0))
    return frame, SimpleNamespace(Ms=Ms, Mt=Mt)


def build_cohen_coon_panel(parent, labeled_entry):
    frame = ttk.Frame(parent)
    ttk.Label(frame,
              text="Reaction-curve rules with a dead-time correction.\n"
                   "Better than ZN-I on delay-dominant plants.",
              foreground="#666", justify="left", font=("TkDefaultFont", 8)
              ).pack(anchor="w")
    step = tk.StringVar(value="1.0")
    noise = tk.StringVar(value="0.0")
    labeled_entry(frame, "step amplitude", step, width=12)
    labeled_entry(frame, "noise σ", noise, width=12)
    return frame, SimpleNamespace(step=step, noise=noise)


def build_chr_panel(parent):
    frame = ttk.Frame(parent)
    ttk.Label(frame, text="Design intent:").pack(anchor="w")
    response = tk.StringVar(value="setpoint")
    ttk.Radiobutton(frame, text="Setpoint tracking (servo)",
                    variable=response, value="setpoint").pack(anchor="w")
    ttk.Radiobutton(frame, text="Load rejection (regulator)",
                    variable=response, value="load").pack(anchor="w")
    ttk.Label(frame, text="Overshoot target:").pack(anchor="w", pady=(4, 0))
    overshoot = tk.StringVar(value="0")
    ttk.Radiobutton(frame, text="0% (aperiodic, most damped)",
                    variable=overshoot, value="0").pack(anchor="w")
    ttk.Radiobutton(frame, text="20% (quicker, some overshoot)",
                    variable=overshoot, value="20").pack(anchor="w")
    ttk.Label(frame,
              text="Note: the 0%/20% target is for the nominal FOPDT model;\n"
                   "realized overshoot on the true plant may differ.",
              foreground="#666", justify="left", font=("TkDefaultFont", 8)
              ).pack(anchor="w", pady=(2, 0))
    return frame, SimpleNamespace(response=response, overshoot=overshoot)


def build_tyreus_luyben_panel(parent, labeled_entry):
    frame = ttk.Frame(parent)
    ttk.Label(frame, text="How to obtain Ku, Pu:").pack(anchor="w")
    source = tk.StringVar(value="bode")
    ttk.Radiobutton(frame, text="Analytical (Bode crossover of −180°)",
                    variable=source, value="bode").pack(anchor="w")
    ttk.Radiobutton(frame, text="Simulated relay-feedback test",
                    variable=source, value="relay").pack(anchor="w")
    relay_h = tk.StringVar(value="1.0")
    relay_T = tk.StringVar(value="80.0")
    labeled_entry(frame, "relay h", relay_h, width=12)
    labeled_entry(frame, "relay duration", relay_T, width=12)
    pi = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="PI only (no derivative)",
                    variable=pi).pack(anchor="w", pady=(2, 0))
    ttk.Label(frame,
              text="Conservative cousin of ZN-II: larger margins, less\n"
                   "overshoot. Already detuned — no 'Halve gains' needed.",
              foreground="#666", justify="left", font=("TkDefaultFont", 8)
              ).pack(anchor="w", pady=(2, 0))
    return frame, SimpleNamespace(source=source, relay_h=relay_h, relay_T=relay_T, pi=pi)
