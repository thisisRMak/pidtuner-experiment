"""Tkinter UI for the PID tuner.

Layout (left column = controls, right = plots):
  ┌──────────────┬──────────────────────────────┐
  │ Plant        │                              │
  │ (tabs:       │   Closed-loop response       │
  │  Expression, │   (overlay of all tuned)     │
  │  MATLAB)     │                              │
  ├──────────────┤   ───────────────────────    │
  │ Method       │                              │
  │ Method args  │   Control effort u(t)        │
  ├──────────────┤                              │
  │ Tune button  │   ───────────────────────    │
  ├──────────────┤                              │
  │ Sim settings │   Error e(t)                 │
  ├──────────────┤                              │
  │ Results /    │                              │
  │ tuned list   │                              │
  └──────────────┴──────────────────────────────┘
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from plant import TransferFunction, parse_coeff_list
from pid_identify import (
    run_step_test, run_relay_test, find_ultimate_gain, FOPDT,
)
from pid_widgets import ClosableNotebook
from pid_tune import select_slowest_stable_poles
from pid_tuning_methods import (
    PIDGains, TuningResult, halve_gains,
    StablePoleCancellation, ZieglerNicholsI, ZieglerNicholsII,
    Amigo, Simc, Boyd, CohenCoon, ChienHronesReswick, TyreusLuyben,
)
from pid_compare import compare_all_methods, metric_row
from pid_comparison_views import draw_heatmap_tab, draw_radar_tab
from pid_response_plotting import create_response_figure, draw_response_tab
from pid_parameter_panels import (
    build_pole_cancel_panel, build_zn1_panel, build_zn2_panel,
    build_amigo_panel, build_simc_panel, build_boyd_panel,
    build_cohen_coon_panel, build_chr_panel, build_tyreus_luyben_panel,
)
from pid_simulate import simulate_closed_loop, format_metrics, saturation_mask


METHODS = [
    "1. Stable pole cancellation",
    "2. Ziegler–Nichols I (step / FOPDT)",
    "3. Ziegler–Nichols II (ultimate gain)",
    "4. AMIGO (FOPDT)",
    "5. SIMC (FOPDT)",
    "6. Boyd (convex-concave)",
    "7. Cohen–Coon (FOPDT)",
    "8. Chien–Hrones–Reswick (FOPDT)",
    "9. Tyreus–Luyben (ultimate gain)",
]

# Color palette for overlay plotting — distinguishable, colorblind-friendly
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
           "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
           "#7f7f7f", "#bcbd22", "#393b79", "#ad494a"]


class TunedEntry:
    """One tuned controller, kept in the session for overlay plotting."""

    def __init__(self, label, gains, result=None, sim=None):
        self.label = label       # short string for legend
        self.gains = gains       # PIDGains
        self.result = result     # TuningResult (full metadata) or None
        self.sim = sim           # ClosedLoopResult or None
        self.color = None        # assigned at plot time
        self.enabled = True      # checkbox state in the tuned list
        self.mrow = None         # cached comparison metric row (compare.metric_row)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

class PIDTunerApp:
    def __init__(self, root):
        self.root = root
        root.title("PID Tuner — ENGR105")
        # Size relative to the actual screen (not a fixed default) so it opens
        # appropriately large on any monitor. Capped width for ultrawides;
        # centered horizontally and placed near the top.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = min(int(sw * 0.88), 2400)
        h = int(sh * 0.88)
        x = max((sw - w) // 2, 0)
        y = max((sh - h) // 6, 0)
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(1100, 700)

        # Session state
        self.tuned = []            # list of TunedEntry
        self.identified = None     # last fitted FOPDT (cached)
        self.plant_cache = None    # last (plant, cache_key) tuple

        self._build_ui()
        self._refresh_plant_info()

    # ── build layout ─────────────────────────────────────────────────────────
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        # === Left column: controls (scrollable) ==============================
        # Tkinter doesn't have a native scrollable frame, so we build one:
        # Frame (fixed width) → Canvas + Scrollbar → inner Frame (where the
        # real widgets live). The inner frame's reqheight drives the canvas's
        # scrollregion so the scrollbar knows how far to scroll.
        left_outer = ttk.Frame(main, width=460)
        left_outer.pack(side="left", fill="y", padx=(0, 8))
        left_outer.pack_propagate(False)

        left_canvas = tk.Canvas(left_outer, borderwidth=0, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_outer, orient="vertical",
                                    command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        # Resize the inner frame to match the canvas width (so child widgets
        # can fill horizontally), and update scrollregion when contents change.
        def _on_inner_configure(_e):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        def _on_canvas_configure(e):
            left_canvas.itemconfigure(left_window, width=e.width)
        left.bind("<Configure>", _on_inner_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)

        # Mousewheel scrolling (cross-platform: Windows/Mac use <MouseWheel>
        # with event.delta, Linux uses <Button-4>/<Button-5>).
        def _on_mousewheel(e):
            if e.num == 4:
                left_canvas.yview_scroll(-1, "units")
            elif e.num == 5:
                left_canvas.yview_scroll(1, "units")
            else:
                left_canvas.yview_scroll(int(-e.delta / 120), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        left_canvas.bind_all("<Button-4>", _on_mousewheel)
        left_canvas.bind_all("<Button-5>", _on_mousewheel)

        self._build_plant_frame(left)
        self._build_compare_frame(left)
        self._build_method_frame(left)
        self._build_sim_frame(left)
        self._build_results_frame(left)

        # === Right column: tabbed output (empty until first tune/compare) ====
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        topbar = ttk.Frame(right)
        topbar.pack(fill="x")
        ttk.Label(topbar, text="Output", font=("TkDefaultFont", 10, "bold")
                  ).pack(side="left", padx=4, pady=2)
        ttk.Button(topbar, text="Close all tabs",
                   command=self._close_all_tabs).pack(side="right", padx=2)

        self.right_nb = ClosableNotebook(right, on_closed=self._reconcile_tabs)
        self.right_nb.pack(fill="both", expand=True)

        self.right_hint = ttk.Label(
            right, foreground="#888", anchor="center",
            text="Tune a method (or Compare all methods) to populate plots, "
                 "heatmap, and radar.")
        self.right_hint.pack(fill="x", pady=6)

        # Tab handles (None when the tab is closed / not yet created)
        self.plots_tab = None
        self.heatmap_tab = None
        self.radar_tab = None
        self.fig = self.ax_y = self.ax_u = self.ax_e = self.canvas = None

        # === Status bar ======================================================
        self.status = tk.StringVar(value="Ready. Define a plant, pick a method, tune.")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  relief="sunken", padding=(6, 2)).pack(fill="x", side="bottom")

    # ── plant frame ─────────────────────────────────────────────────────────
    def _build_plant_frame(self, parent):
        pf = ttk.LabelFrame(parent, text="Plant G(s)", padding=6)
        pf.pack(fill="x", pady=(0, 6))

        self.plant_nb = ttk.Notebook(pf)
        self.plant_nb.pack(fill="x", pady=(0, 4))

        # Tab 1 — symbolic
        expr_tab = ttk.Frame(self.plant_nb, padding=4)
        self.plant_nb.add(expr_tab, text="Symbolic")
        ttk.Label(expr_tab, text="G(s) =").pack(anchor="w")
        self.tf_expr = tk.StringVar(value="1000 / ((s+1)*(10s+1))")
        self.tf_expr.trace_add("write", lambda *_: self._refresh_plant_info())
        ttk.Entry(expr_tab, textvariable=self.tf_expr).pack(fill="x", pady=2)
        ttk.Label(expr_tab,
                  text=("examples:  1000/((s+1)(10s+1))    2/(5s+1)    "
                        "(s+2)/(s^2+3s+1)\n           1/(s(s+1))"),
                  foreground="#666", justify="left", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", pady=(2, 0))

        # Tab 2 — MATLAB-style coefficients
        coef_tab = ttk.Frame(self.plant_nb, padding=4)
        self.plant_nb.add(coef_tab, text="MATLAB form")
        self.gain_var = tk.StringVar(value="1000")
        self.num_var = tk.StringVar(value="[1]")
        self.den_var = tk.StringVar(value="[10, 11, 1]")
        for v in (self.gain_var, self.num_var, self.den_var):
            v.trace_add("write", lambda *_: self._refresh_plant_info())
        self._labeled_entry(coef_tab, "gain K", self.gain_var, width=20)
        self._labeled_entry(coef_tab, "num", self.num_var, width=20)
        self._labeled_entry(coef_tab, "den", self.den_var, width=20)
        ttk.Label(coef_tab,
                  text=("MATLAB tf(num, den) form — descending powers of s.\n"
                        "  num=[1, 2]  →  s + 2\n"
                        "  den=[10, 11, 1]  →  10s² + 11s + 1\n"
                        "G(s) = gain · num/den · exp(−Ls)"),
                  foreground="#666", justify="left", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", pady=(2, 0))

        self.plant_nb.bind("<<NotebookTabChanged>>",
                           lambda *_: self._refresh_plant_info())

        # Dead time (shared)
        self.L_var = tk.StringVar(value="0.0")
        self.L_var.trace_add("write", lambda *_: self._refresh_plant_info())
        self._labeled_entry(pf, "L (dead time, s)", self.L_var, width=12)

        self.plant_info = ttk.Label(pf, text="", foreground="#0a5",
                                    justify="left", wraplength=400)
        self.plant_info.pack(anchor="w", pady=(4, 0))

    # ── method frame ────────────────────────────────────────────────────────
    def _build_compare_frame(self, parent):
        cf = ttk.LabelFrame(parent, text="Compare all methods", padding=6)
        cf.pack(fill="x", pady=6)
        ttk.Button(cf, text="⊞  Compare all methods",
                   command=self.on_compare_all).pack(fill="x")
        ttk.Label(cf,
                  text="Tunes every applicable method at once and overlays them "
                       "on the plots, plus a heatmap and radar. Each appears in "
                       "the session list below — untick any to declutter.",
                  foreground="#666", justify="left", wraplength=420,
                  font=("TkDefaultFont", 8)).pack(anchor="w", pady=(3, 0))

    def _build_method_frame(self, parent):
        mf = ttk.LabelFrame(parent, text="Tune one method at a time", padding=6)
        mf.pack(fill="x", pady=6)

        ttk.Label(mf, text="Method").pack(anchor="w")
        self.method_var = tk.StringVar(value=METHODS[0])
        opt = ttk.OptionMenu(mf, self.method_var, METHODS[0], *METHODS,
                             command=lambda _: self._on_method_change())
        opt.pack(fill="x", pady=(0, 6))

        # ── Method-specific argument frame (swaps based on selection) ──────
        self.args_frame = ttk.Frame(mf)
        self.args_frame.pack(fill="x")

        self.pc_frame, pc = build_pole_cancel_panel(
            self.args_frame, self._labeled_entry, self._update_pc_state)
        (self.pc_mode, self.pc_p1, self.pc_p2, self.pc_kd,
         self.pc_p1_entry, self.pc_p2_entry, self.pc_pole_info) = (
            pc.mode, pc.p1, pc.p2, pc.kd, pc.p1_entry, pc.p2_entry, pc.pole_info)

        self.zn1_frame, zn1 = build_zn1_panel(self.args_frame, self._labeled_entry)
        self.zn1_step, self.zn1_noise = zn1.step, zn1.noise

        self.zn2_frame, zn2 = build_zn2_panel(self.args_frame, self._labeled_entry)
        self.zn2_source, self.zn2_relay_h, self.zn2_relay_T = (
            zn2.source, zn2.relay_h, zn2.relay_T)

        self.amigo_frame, amigo = build_amigo_panel(self.args_frame)
        self.amigo_integrating = amigo.integrating

        self.simc_frame, simc = build_simc_panel(self.args_frame, self._labeled_entry)
        self.simc_tau_c, self.simc_tau2 = simc.tau_c, simc.tau2

        self.boyd_frame, boyd = build_boyd_panel(self.args_frame, self._labeled_entry)
        self.boyd_Ms, self.boyd_Mt = boyd.Ms, boyd.Mt

        self.cc_frame, cc = build_cohen_coon_panel(self.args_frame, self._labeled_entry)
        self.cc_step, self.cc_noise = cc.step, cc.noise

        self.chr_frame, chr_ = build_chr_panel(self.args_frame)
        self.chr_response, self.chr_overshoot = chr_.response, chr_.overshoot

        self.tl_frame, tl = build_tyreus_luyben_panel(self.args_frame, self._labeled_entry)
        self.tl_source, self.tl_relay_h, self.tl_relay_T, self.tl_pi = (
            tl.source, tl.relay_h, tl.relay_T, tl.pi)

        # ── Tune button ─────────────────────────────────────────────────────
        # ── Global "halve gains" toggle ─────────────────────────────────────
        # Sits between the method-specific args and the Tune button so users
        # see it for every method. Intended for ZN-I/II per PEI9e advice;
        # exposed for all methods so the user can experiment.
        halve_row = ttk.Frame(mf)
        halve_row.pack(fill="x", pady=(8, 0))
        self.halve_gains_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(halve_row,
                        text="Halve gains (divide Kp, Ki, Kd by 2)",
                        variable=self.halve_gains_var).pack(anchor="w")
        ttk.Label(halve_row,
                  text="Recommended for ZN-I/II when tracking setpoints.",
                  foreground="#666", font=("TkDefaultFont", 8)
                  ).pack(anchor="w")

        ttk.Button(mf, text="Tune & simulate",
                   command=self.on_tune).pack(fill="x", pady=(8, 0))

        # Initialize visible args frame
        self._on_method_change()

    # ── sim frame ───────────────────────────────────────────────────────────
    def _build_sim_frame(self, parent):
        sf = ttk.LabelFrame(parent, text="Closed-loop simulation", padding=6)
        sf.pack(fill="x", pady=6)

        # Setpoint kind selector
        kind_row = ttk.Frame(sf)
        kind_row.pack(fill="x", pady=2)
        ttk.Label(kind_row, text="Setpoint", width=18).pack(side="left")
        self.sp_kind_var = tk.StringVar(value="step")
        ttk.OptionMenu(kind_row, self.sp_kind_var, "step",
                       "step", "ramp", "pulse").pack(side="left", fill="x", expand=True)

        self.sp_var = tk.StringVar(value="1.0")
        self.t_end_var = tk.StringVar(value="")
        self.umin_var = tk.StringVar(value="-100")
        self.umax_var = tk.StringVar(value="100")
        self._labeled_entry(sf, "amplitude", self.sp_var, width=12)
        self._labeled_entry(sf, "duration (blank=auto)", self.t_end_var, width=12)
        self._labeled_entry(sf, "u min", self.umin_var, width=12)
        self._labeled_entry(sf, "u max", self.umax_var, width=12)
        self.d_filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Derivative filter (N=10) — recommended",
                        variable=self.d_filter_var).pack(anchor="w")
        ttk.Label(sf,
                  text="ramp: linear 0 → amp over duration.   "
                       "pulse: amp during [25%, 50%] of duration.",
                  foreground="#666", justify="left", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", pady=(2, 0))

        # Anti-windup: only visible/relevant once u min/u max above can
        # actually saturate the actuator (see pid_simulate.py module docstring).
        aw_row = ttk.Frame(sf)
        aw_row.pack(fill="x", pady=(4, 0))
        ttk.Label(aw_row, text="Anti-windup", width=18).pack(side="left")
        self.antiwindup_var = tk.StringVar(value="conditional")
        ttk.OptionMenu(aw_row, self.antiwindup_var, "conditional",
                       "conditional", "back_calc").pack(side="left", fill="x", expand=True)
        self.ka_var = tk.StringVar(value="")
        self._labeled_entry(sf, "Ka override (blank=auto)", self.ka_var, width=12)
        ttk.Label(sf,
                  text="conditional: freeze integral while saturated (default, "
                       "unchanged from before). back_calc: Astrom & Hagglund "
                       "back-calculation, Ka=1/Tt auto-derived from Ti,Td unless "
                       "overridden. Neither has any effect unless u min/u max "
                       "above actually saturate the actuator.",
                  foreground="#666", justify="left", wraplength=260,
                  font=("TkDefaultFont", 8)).pack(anchor="w", pady=(2, 0))

    # ── results frame ───────────────────────────────────────────────────────
    def _build_results_frame(self, parent):
        rf = ttk.LabelFrame(parent, text="Tuned controllers (session overlay)",
                            padding=6)
        rf.pack(fill="both", expand=True, pady=6)

        # Scrollable list of tuned controllers
        list_frame = ttk.Frame(rf)
        list_frame.pack(fill="both", expand=True)

        self.tuned_canvas = tk.Canvas(list_frame, height=360, highlightthickness=0)
        scroll = ttk.Scrollbar(list_frame, orient="vertical",
                               command=self.tuned_canvas.yview)
        self.tuned_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tuned_canvas.pack(side="left", fill="both", expand=True)

        self.tuned_inner = ttk.Frame(self.tuned_canvas)
        self.tuned_canvas.create_window((0, 0), window=self.tuned_inner,
                                        anchor="nw")
        self.tuned_inner.bind("<Configure>",
                              lambda e: self.tuned_canvas.configure(
                                  scrollregion=self.tuned_canvas.bbox("all")))

        # Buttons
        btn_frame = ttk.Frame(rf)
        btn_frame.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_frame, text="Select all",
                   command=self.select_all_tuned).pack(side="left")
        ttk.Button(btn_frame, text="Deselect all",
                   command=self.deselect_all_tuned).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Clear all",
                   command=self.clear_tuned).pack(side="left", padx=(12, 0))
        ttk.Button(btn_frame, text="Remove unchecked",
                   command=self.remove_unchecked).pack(side="left", padx=4)

        # Last-result details
        self.last_lbl = ttk.Label(rf, text="", justify="left",
                                  wraplength=400, font=("TkDefaultFont", 9))
        self.last_lbl.pack(anchor="w", pady=(4, 0))

    # ── helpers ─────────────────────────────────────────────────────────────
    def _labeled_entry(self, parent, label, var, width=12):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _on_method_change(self):
        for frame in (self.pc_frame, self.zn1_frame, self.zn2_frame,
                      self.amigo_frame, self.simc_frame, self.boyd_frame,
                      self.cc_frame, self.chr_frame, self.tl_frame):
            frame.pack_forget()
        m = self.method_var.get()
        if m.startswith("1."):
            self.pc_frame.pack(fill="x")
            self._update_pc_state()
            self._update_pole_info()
        elif m.startswith("2."):
            self.zn1_frame.pack(fill="x")
        elif m.startswith("3."):
            self.zn2_frame.pack(fill="x")
        elif m.startswith("4."):
            self.amigo_frame.pack(fill="x")
        elif m.startswith("5."):
            self.simc_frame.pack(fill="x")
        elif m.startswith("6."):
            self.boyd_frame.pack(fill="x")
        elif m.startswith("7."):
            self.cc_frame.pack(fill="x")
        elif m.startswith("8."):
            self.chr_frame.pack(fill="x")
        elif m.startswith("9."):
            self.tl_frame.pack(fill="x")

    def _update_pc_state(self):
        enabled = self.pc_mode.get() == "manual"
        state = "normal" if enabled else "disabled"
        self.pc_p1_entry.configure(state=state)
        self.pc_p2_entry.configure(state=state)

    def _update_pole_info(self):
        try:
            plant = self._build_plant()
            poles = plant.poles()
            if len(poles) == 0:
                txt = "Plant has no finite poles."
            else:
                pole_strs = []
                for p in poles:
                    if abs(np.imag(p)) < 1e-9:
                        pole_strs.append(f"{p.real:+.4g}")
                    else:
                        pole_strs.append(f"{p.real:+.4g} {p.imag:+.4g}j")
                txt = "Plant poles: " + ", ".join(pole_strs)
                if plant.has_rhp_poles():
                    txt += "\n⚠ Plant has RHP poles — cancellation is unsafe."
            self.pc_pole_info.config(text=txt)
        except Exception as exc:
            self.pc_pole_info.config(text=f"(plant invalid: {exc})")

    # ── plant build / cache ────────────────────────────────────────────────
    def _read_L_field(self):
        try:
            return float(self.L_var.get())
        except (ValueError, tk.TclError):
            return 0.0

    def _build_plant(self):
        L = self._read_L_field()
        idx = self.plant_nb.index(self.plant_nb.select())
        if idx == 0:
            return TransferFunction.parse(self.tf_expr.get(), L=L)
        gain = float(self.gain_var.get())
        num = parse_coeff_list(self.num_var.get())
        den = parse_coeff_list(self.den_var.get())
        return TransferFunction.from_coeffs(num=num, den=den, L=L, gain=gain)

    def _refresh_plant_info(self):
        if not hasattr(self, "plant_info"):
            return
        try:
            plant = self._build_plant()
            info = f"{plant.pretty()}\n{plant.latex_summary()}"
            # The L field itself contributed nothing (<=0), so a nonzero
            # plant.L must have been parsed out of the expression text —
            # confirm the detection back to the user, same as TransferFunction
            # .parse would combine/reject the two sources.
            if plant.L > 0 and self._read_L_field() <= 0:
                info += f"\n✓ Detected time delay L={plant.L:g}s from expression."
            self.plant_info.config(text=info, foreground="#0a5")
            if hasattr(self, "pc_pole_info"):
                self._update_pole_info()
        except Exception as exc:
            self.plant_info.config(text=f"⚠ {exc}", foreground="#a00")

    # ── tune dispatch ──────────────────────────────────────────────────────
    def on_tune(self):
        try:
            plant = self._build_plant()
        except Exception as exc:
            messagebox.showerror("Plant error", str(exc))
            return

        method = self.method_var.get()
        try:
            if method.startswith("1."):
                result = self._tune_pole_cancel(plant)
            elif method.startswith("2."):
                result = self._tune_zn1(plant)
            elif method.startswith("3."):
                result = self._tune_zn2(plant)
            elif method.startswith("4."):
                result = self._tune_amigo(plant)
            elif method.startswith("5."):
                result = self._tune_simc(plant)
            elif method.startswith("6."):
                result = self._tune_boyd(plant)
            elif method.startswith("7."):
                result = self._tune_cohen_coon(plant)
            elif method.startswith("8."):
                result = self._tune_chr(plant)
            elif method.startswith("9."):
                result = self._tune_tyreus_luyben(plant)
            else:
                raise RuntimeError(f"unknown method {method}")
        except Exception as exc:
            messagebox.showerror("Tuning failed", str(exc))
            self.status.set(f"Error: {exc}")
            return

        # Post-hoc halving (toggle applies to any method, intended for ZN)
        if self.halve_gains_var.get():
            result = halve_gains(result)

        # Closed-loop sim
        try:
            sim = self._run_closed_loop(plant, result.gains)
        except Exception as exc:
            messagebox.showerror("Simulation failed", str(exc))
            return

        label = self._next_label(method, halved=self.halve_gains_var.get(), sim=sim)
        entry = TunedEntry(label=label, gains=result.gains,
                           result=result, sim=sim)
        entry.mrow = metric_row(plant, label, result.gains,
                                black_box=result.black_box, fopdt=result.fopdt)
        self.tuned.append(entry)
        self._refresh_tuned_list()
        self._show_last_result(result, sim)
        self._open_all_tabs_and_draw()
        self.status.set(f"Tuned: {label}  ({len(self.tuned)} in session)")

    # ── method-specific tune handlers ──────────────────────────────────────
    def _tune_pole_cancel(self, plant):
        if plant.has_rhp_poles():
            raise ValueError(
                "Plant has RHP poles — pole cancellation is unsafe. "
                "Use a different method."
            )
        if self.pc_mode.get() == "auto":
            p1, p2 = select_slowest_stable_poles(plant)
            # auto-pick may return complex pair; coerce to real if pair, else error
            if abs(np.imag(p1)) > 1e-9 or abs(np.imag(p2)) > 1e-9:
                # complex pair — handle via Kd*(s² + 2ζω·s + ω²)/s controller
                # For simplicity in this UI, refuse and ask user to pick.
                raise ValueError(
                    "Auto-selected the slowest poles are complex-conjugate. "
                    "Switch to Manual mode and pick two real poles (or use "
                    "SIMC / Boyd which handle this natively)."
                )
            p1 = float(np.real(p1))
            p2 = float(np.real(p2))
        else:
            p1 = float(self.pc_p1.get())
            p2 = float(self.pc_p2.get())
        Kd_str = self.pc_kd.get().strip()
        # If user left Kd at its default, scale it so the resulting open-loop
        # crossover sits at roughly ω = 1 rad/s. After cancellation the open
        # loop is K·Kd/s, so we want K·Kd ≈ 1.
        if Kd_str == "" or Kd_str == "1.0":
            K_dc = abs(plant.dc_gain())
            if np.isfinite(K_dc) and K_dc > 1e-9:
                Kd = 1.0 / K_dc
            else:
                Kd = 1.0
        else:
            Kd = float(Kd_str)
        return StablePoleCancellation(plant, p1, p2, Kd=Kd).tune()

    def _tune_zn1(self, plant):
        step = float(self.zn1_step.get())
        noise = float(self.zn1_noise.get())
        _, _, _, _, fopdt = run_step_test(plant, step_amp=step,
                                          noise_sigma=noise, seed=0)
        self.identified = fopdt
        return ZieglerNicholsI(fopdt).tune()

    def _tune_zn2(self, plant):
        if self.zn2_source.get() == "bode":
            Ku, Pu, w180 = find_ultimate_gain(plant)
        else:
            h = float(self.zn2_relay_h.get())
            T = float(self.zn2_relay_T.get())
            Ku, Pu, _, _, _ = run_relay_test(plant, t_max=T, h=h)
        return ZieglerNicholsII(Ku, Pu).tune()

    def _tune_amigo(self, plant):
        # Always run a step test to get the FOPDT
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0,
                                          noise_sigma=0.0, seed=0)
        self.identified = fopdt
        return Amigo(fopdt, integrating=self.amigo_integrating.get()).tune()

    def _tune_simc(self, plant):
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0,
                                          noise_sigma=0.0, seed=0)
        self.identified = fopdt
        tau_c_str = self.simc_tau_c.get().strip()
        tau_c = float(tau_c_str) if tau_c_str else None
        tau2_str = self.simc_tau2.get().strip()
        tau2 = float(tau2_str) if tau2_str else None
        return Simc(fopdt, tau_c=tau_c, tau2=tau2).tune()

    def _tune_boyd(self, plant):
        Ms = float(self.boyd_Ms.get())
        Mt = float(self.boyd_Mt.get())
        # Seed with SIMC on a quickly-fitted FOPDT
        try:
            _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0,
                                              noise_sigma=0.0, seed=0)
            seed = Simc(fopdt).tune().gains
        except Exception:
            seed = None
        return Boyd(plant, Ms=Ms, Mt=Mt, seed_gains=seed).tune()

    def _tune_cohen_coon(self, plant):
        step = float(self.cc_step.get())
        noise = float(self.cc_noise.get())
        _, _, _, _, fopdt = run_step_test(plant, step_amp=step,
                                          noise_sigma=noise, seed=0)
        self.identified = fopdt
        return CohenCoon(fopdt).tune()

    def _tune_chr(self, plant):
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0,
                                          noise_sigma=0.0, seed=0)
        self.identified = fopdt
        return ChienHronesReswick(fopdt,
                                  response=self.chr_response.get(),
                                  overshoot=int(self.chr_overshoot.get())).tune()

    def _tune_tyreus_luyben(self, plant):
        if self.tl_source.get() == "bode":
            Ku, Pu, _ = find_ultimate_gain(plant)
        else:
            h = float(self.tl_relay_h.get())
            T = float(self.tl_relay_T.get())
            Ku, Pu, _, _, _ = run_relay_test(plant, t_max=T, h=h)
        return TyreusLuyben(Ku, Pu, use_derivative=not self.tl_pi.get()).tune()

    # ── compare all methods ─────────────────────────────────────────────────
    def on_compare_all(self):
        try:
            plant = self._build_plant()
        except Exception as exc:
            messagebox.showerror("Plant error", str(exc))
            return
        self.status.set("Comparing all methods…")
        self.root.update_idletasks()
        try:
            rows = compare_all_methods(plant)
        except Exception as exc:
            messagebox.showerror("Comparison failed", str(exc))
            self.status.set("Ready.")
            return

        # Replace the session with one entry per method, simulated with the
        # user's current setpoint/limit settings so the overlay is comparable.
        self.tuned.clear()
        n_ok = 0
        for row in rows:
            gains = row.get("gains")
            if gains is None:        # method failed to tune — skip the overlay
                continue
            try:
                sim = self._run_closed_loop(plant, gains)
            except Exception:
                continue
            entry = TunedEntry(label=row["name"] + self._antiwindup_tag(sim),
                               gains=gains, result=None, sim=sim)
            entry.mrow = row
            self.tuned.append(entry)
            n_ok += 1

        self._refresh_tuned_list()
        self._open_all_tabs_and_draw()
        self.status.set(f"Compared {n_ok} methods. Untick any in the session "
                        f"list to declutter.")

    # ── closed-loop sim ────────────────────────────────────────────────────
    def _run_closed_loop(self, plant, gains):
        try:
            t_end = float(self.t_end_var.get()) if self.t_end_var.get().strip() else None
        except ValueError:
            t_end = None
        sp = float(self.sp_var.get())
        umin = float(self.umin_var.get())
        umax = float(self.umax_var.get())
        kind = self.sp_kind_var.get()
        antiwindup = self.antiwindup_var.get()
        ka_str = self.ka_var.get().strip()
        Ka = float(ka_str) if ka_str else None
        return simulate_closed_loop(
            plant, gains, t_end=t_end, setpoint=sp,
            setpoint_kind=kind,
            u_min=umin, u_max=umax, use_d_filter=self.d_filter_var.get(),
            antiwindup=antiwindup, Ka=Ka,
        )

    # ── label naming for overlay legend ────────────────────────────────────
    def _antiwindup_tag(self, sim):
        """Legend/label suffix for a non-default anti-windup mode.

        Conditional-integration is the original default and stays untagged
        (unchanged legends for everyone not using this feature); back_calc
        is opt-in, so it's worth flagging — otherwise two overlaid entries
        for the same method with different anti-windup settings are
        indistinguishable in the legend (same gains, same base label).

        Ka is only shown if the actuator actually saturated in this sim —
        back_calc's correction term is zero otherwise, so an unconditional
        Ka would misleadingly suggest it did something.
        """
        if sim.antiwindup != "back_calc":
            return ""
        if not np.any(saturation_mask(sim)):
            return " [back_calc: never saturated]"
        return f" [back_calc, Ka={sim.Ka:.3g}]"

    def _next_label(self, method, halved=False, sim=None):
        base = method.split(". ", 1)[1] if ". " in method else method
        base = base.split(" (")[0]
        if halved:
            base = base + " ½"
        # Tag with setpoint kind so a mixed-setpoint session legend reads cleanly.
        kind = self.sp_kind_var.get()
        if kind != "step":
            base = f"{base} ({kind})"
        if sim is not None:
            base += self._antiwindup_tag(sim)
        count = sum(1 for e in self.tuned if e.label.split(" #")[0] == base)
        if count == 0:
            return base
        return f"{base} #{count + 1}"

    # ── tuned-list UI ──────────────────────────────────────────────────────
    def _refresh_tuned_list(self):
        for child in self.tuned_inner.winfo_children():
            child.destroy()
        for i, entry in enumerate(self.tuned):
            color = PALETTE[i % len(PALETTE)]
            entry.color = color
            row = ttk.Frame(self.tuned_inner)
            row.pack(fill="x", pady=1)
            var = tk.BooleanVar(value=entry.enabled)
            cb = ttk.Checkbutton(
                row, variable=var, text="",
                command=lambda e=entry, v=var: self._toggle_entry(e, v))
            cb.pack(side="left")
            tk.Label(row, text="●", fg=color, font=("TkDefaultFont", 14)
                     ).pack(side="left")
            ttk.Label(row, text=entry.label, width=24, anchor="w").pack(side="left")
            g = entry.gains
            ttk.Label(row,
                      text=f"Kp={g.Kp:.3g}  Ki={g.Ki:.3g}  Kd={g.Kd:.3g}",
                      font=("TkDefaultFont", 8), foreground="#444"
                      ).pack(side="left")
            # store ref so the toggle closure can find it
            entry._cb_var = var

    def _toggle_entry(self, entry, var):
        entry.enabled = var.get()
        self._refresh_open_tabs()

    def _set_all_enabled(self, value):
        for e in self.tuned:
            e.enabled = value
            if getattr(e, "_cb_var", None) is not None:
                e._cb_var.set(value)      # sync the visible checkbox
        self._refresh_open_tabs()

    def select_all_tuned(self):
        self._set_all_enabled(True)

    def deselect_all_tuned(self):
        self._set_all_enabled(False)

    def clear_tuned(self):
        self.tuned.clear()
        self._refresh_tuned_list()
        self._refresh_open_tabs()

    def remove_unchecked(self):
        self.tuned = [e for e in self.tuned if e.enabled]
        self._refresh_tuned_list()
        self._refresh_open_tabs()

    # ── last-result display ────────────────────────────────────────────────
    def _show_last_result(self, result, sim):
        parts = [f"▸ {result.method}", "", result.gains.pretty()]
        if result.fopdt is not None:
            parts.append(f"\nIdentified FOPDT: {result.fopdt.pretty()}")
        if result.Ku is not None:
            parts.append(f"\nUltimate: Ku = {result.Ku:.4g}, Pu = {result.Pu:.4g} s")
        if result.cancelled_poles:
            cp = ", ".join(f"{p:+.3g}" for p in result.cancelled_poles)
            parts.append(f"\nCancelled poles: s = {cp}")
        if result.notes:
            parts.append(f"\n{result.notes}")
        if sim.antiwindup == "back_calc":
            if np.any(saturation_mask(sim)):
                tt_str = f"{sim.Tt:.4g} s" if np.isfinite(sim.Tt) else "inf (no integral action)"
                parts.append(f"\nAnti-windup: back_calc, Ka = {sim.Ka:.4g}  (Tt = {tt_str})")
            else:
                parts.append("\nAnti-windup: back_calc requested, but the actuator "
                             "never saturated in this simulation — Ka had no effect.")
        parts.append("")
        parts.append("Metrics:  " + format_metrics(sim.metrics))
        self.last_lbl.config(text="\n".join(parts))

    # ── tabbed output management ────────────────────────────────────────────
    def _session_rows(self):
        """Metric rows for the enabled session entries (for heatmap/radar)."""
        rows = []
        for e in self.tuned:
            if not e.enabled:
                continue
            row = getattr(e, "mrow", None)
            if row is None:
                row = {"name": e.label, "stable": False, "error": "no metrics"}
            rows.append(row)
        return rows

    def _ensure_plots_tab(self):
        if self.plots_tab is not None and self.plots_tab.winfo_exists():
            return
        frame = ttk.Frame(self.right_nb)
        self.right_nb.add(frame, text="Response  ")
        self.fig, self.ax_y, self.ax_u, self.ax_e, self.canvas = \
            create_response_figure(frame)
        self.plots_tab = frame

    def _ensure_heatmap_tab(self):
        if self.heatmap_tab is not None and self.heatmap_tab.winfo_exists():
            return
        frame = ttk.Frame(self.right_nb)
        self.right_nb.add(frame, text="Heatmap  ")
        self.heatmap_tab = frame

    def _ensure_radar_tab(self):
        if self.radar_tab is not None and self.radar_tab.winfo_exists():
            return
        frame = ttk.Frame(self.right_nb)
        self.right_nb.add(frame, text="Radar  ")
        self.radar_tab = frame

    def _reconcile_tabs(self):
        """After a tab is closed, drop references to tabs no longer present."""
        present = set(self.right_nb.tabs())
        def gone(w):
            return w is None or str(w) not in present
        if gone(self.plots_tab):
            self.plots_tab = None
            self.fig = self.ax_y = self.ax_u = self.ax_e = self.canvas = None
        if gone(self.heatmap_tab):
            self.heatmap_tab = None
        if gone(self.radar_tab):
            self.radar_tab = None
        self._update_right_hint()

    def _close_all_tabs(self):
        for tab in list(self.right_nb.tabs()):
            self.right_nb.forget(tab)
        self.plots_tab = self.heatmap_tab = self.radar_tab = None
        self.fig = self.ax_y = self.ax_u = self.ax_e = self.canvas = None
        self._update_right_hint()

    def _update_right_hint(self):
        if self.right_nb.tabs():
            self.right_hint.pack_forget()
        else:
            self.right_hint.pack(fill="x", pady=6)

    def _open_all_tabs_and_draw(self):
        """Create the three tabs if missing, then draw all of them."""
        self._ensure_plots_tab()
        self._ensure_heatmap_tab()
        self._ensure_radar_tab()
        self._update_right_hint()
        self._refresh_open_tabs()

    def _refresh_open_tabs(self):
        """Redraw whichever tabs are currently open (don't reopen closed ones)."""
        self._draw_plots_tab()
        self._draw_heatmap_tab()
        self._draw_radar_tab()

    # ── plots tab ───────────────────────────────────────────────────────────
    def _draw_plots_tab(self):
        if self.plots_tab is None or self.canvas is None:
            return
        active = [e for e in self.tuned if e.enabled and e.sim is not None]
        draw_response_tab(self.fig, self.ax_y, self.ax_u, self.ax_e,
                          self.canvas, active)

    # ── heatmap tab ─────────────────────────────────────────────────────────
    def _draw_heatmap_tab(self):
        if self.heatmap_tab is None or not self.heatmap_tab.winfo_exists():
            return
        draw_heatmap_tab(self.heatmap_tab, self._session_rows())

    # ── radar tab ───────────────────────────────────────────────────────────
    def _draw_radar_tab(self):
        if self.radar_tab is None or not self.radar_tab.winfo_exists():
            return
        draw_radar_tab(self.radar_tab, self._session_rows())



def main():
    root = tk.Tk()
    PIDTunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()