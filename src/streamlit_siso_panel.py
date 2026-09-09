"""Streamlit SISO PID panel — Build plan Step 3 (docs/gui_plan.md).

Ports pid_app.py's controls/tune/simulate/session-list flow onto
Streamlit widgets, calling the same UI-agnostic backend functions
(pid_tuning_methods.py, pid_simulate.py, pid_compare.py). Session
state (tuned-controllers list) goes through streamlit_gui_state.py rather than
Tkinter instance attributes.

Heatmap/radar comparison views live in streamlit_siso_comparison_views.py — a
Streamlit-native reimplementation of pid_comparison_views.py's drawing
code (which is Tk-widget-specific), reusing the same plain
pid_compare.py data functions.
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st
from matplotlib.figure import Figure

from plant import TransferFunction, parse_coeff_list
from pid_identify import run_step_test, run_relay_test, find_ultimate_gain
from pid_tune import select_slowest_stable_poles
from pid_tuning_methods import (
    halve_gains,
    StablePoleCancellation, ZieglerNicholsI, ZieglerNicholsII,
    Amigo, Simc, Boyd, CohenCoon, ChienHronesReswick, TyreusLuyben,
)
from pid_compare import compare_all_methods, metric_row
from pid_simulate import simulate_closed_loop, format_metrics, saturation_mask
from lqg_simulate import auto_plot_window

import streamlit_gui_state as gs
import streamlit_siso_comparison_views as scv

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

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
           "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
           "#7f7f7f", "#bcbd22", "#393b79", "#ad494a"]


def _assign_colors(entries):
    """entry.color isn't a stored field (see gs.ControllerEntry) — it's
    assigned here, by current list position, so removing/reordering
    entries reflows the palette rather than leaving gaps. Used to only
    run as a side effect of _render_session_list(), which was fine while
    render() always called both halves together — now that
    render_controls() (Mode=Manual only) and render_plots() (every Mode)
    can run independently, render_plots() needs its own call too, or an
    LLM-only session (no manual controls ever rendered this run) would
    plot with unset colors."""
    for i, entry in enumerate(entries):
        entry.color = PALETTE[i % len(PALETTE)]


def _download_fig_button(fig, filename, key):
    """PNG download button for a matplotlib Figure already shown via
    st.pyplot — used so the stacked Response/Heatmap/Radar views (no
    longer switched one-at-a-time via the View radio) are each
    individually downloadable."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Download PNG", data=buf.getvalue(), file_name=filename,
                       mime="image/png", key=key)


# ── plant ────────────────────────────────────────────────────────────────
def _build_plant():
    L = st.session_state["siso_L"]
    form = st.session_state["siso_plant_form"]
    if form == "Symbolic":
        return TransferFunction.parse(st.session_state["siso_tf_expr"], L=L)
    gain = float(st.session_state["siso_gain"])
    num = parse_coeff_list(st.session_state["siso_num"])
    den = parse_coeff_list(st.session_state["siso_den"])
    return TransferFunction.from_coeffs(num=num, den=den, L=L, gain=gain)


def _render_plant_controls():
    st.subheader("Plant G(s)")
    st.radio("Plant form", ["Symbolic", "MATLAB coefficients"],
             key="siso_plant_form", horizontal=True)
    if st.session_state["siso_plant_form"] == "Symbolic":
        st.text_input("G(s) =", value="1000 / ((s+1)*(10s+1))", key="siso_tf_expr")
        st.caption("examples:  1000/((s+1)(10s+1))    2/(5s+1)    "
                   "(s+2)/(s^2+3s+1)    1/(s(s+1))")
    else:
        st.text_input("gain K", value="1000", key="siso_gain")
        st.text_input("num", value="[1]", key="siso_num")
        st.text_input("den", value="[10, 11, 1]", key="siso_den")
        st.caption("MATLAB tf(num, den) form, descending powers of s.  "
                   "num=[1, 2] → s + 2   den=[10, 11, 1] → 10s² + 11s + 1")
    st.number_input("L (dead time, s)", value=0.0, key="siso_L")

    try:
        plant = _build_plant()
        info = f"{plant.pretty()}\n\n{plant.latex_summary()}"
        if plant.L > 0 and st.session_state["siso_L"] <= 0:
            info += f"\n\n✓ Detected time delay L={plant.L:g}s from expression."
        st.success(info)
        poles = plant.poles()
        if len(poles):
            pole_strs = [f"{p.real:+.4g}" if abs(np.imag(p)) < 1e-9
                        else f"{p.real:+.4g} {p.imag:+.4g}j" for p in poles]
            txt = "Plant poles: " + ", ".join(pole_strs)
            if plant.has_rhp_poles():
                txt += "  ⚠ Plant has RHP poles — cancellation is unsafe."
            st.caption(txt)
        return plant
    except Exception as exc:
        st.error(str(exc))
        return None


# ── method-specific args ────────────────────────────────────────────────
def _render_method_args(method):
    if method.startswith("1."):
        mode = st.radio("Pole selection", ["auto", "manual"], key="pc_mode",
                        horizontal=True)
        st.caption("Cancel poles at s = −p₁, s = −p₂")
        st.number_input("p₁ (positive)", value=0.1, key="pc_p1", disabled=mode == "auto")
        st.number_input("p₂ (positive)", value=1.0, key="pc_p2", disabled=mode == "auto")
        st.text_input("Kd (blank/1.0 = auto-scaled)", value="1.0", key="pc_kd")
    elif method.startswith("2."):
        st.number_input("step amplitude", value=1.0, key="zn1_step")
        st.number_input("noise sigma", value=0.0, key="zn1_noise")
    elif method.startswith("3."):
        st.radio("Ultimate gain source", ["bode", "relay"], key="zn2_source",
                 horizontal=True)
        st.number_input("relay h", value=1.0, key="zn2_relay_h")
        st.number_input("relay T (s)", value=50.0, key="zn2_relay_T")
    elif method.startswith("4."):
        st.checkbox("Integrating process", value=False, key="amigo_integrating")
    elif method.startswith("5."):
        st.text_input("tau_c (blank=auto)", value="", key="simc_tau_c")
        st.text_input("tau2 (blank=auto)", value="", key="simc_tau2")
    elif method.startswith("6."):
        st.number_input("Ms", value=1.4, key="boyd_Ms")
        st.number_input("Mt", value=1.4, key="boyd_Mt")
    elif method.startswith("7."):
        st.number_input("step amplitude", value=1.0, key="cc_step")
        st.number_input("noise sigma", value=0.0, key="cc_noise")
    elif method.startswith("8."):
        st.radio("Response", ["setpoint", "load"], key="chr_response",
                 horizontal=True)
        st.radio("Overshoot", [0, 20], key="chr_overshoot", horizontal=True)
    elif method.startswith("9."):
        st.radio("Ultimate gain source", ["bode", "relay"], key="tl_source",
                 horizontal=True)
        st.number_input("relay h", value=1.0, key="tl_relay_h")
        st.number_input("relay T (s)", value=50.0, key="tl_relay_T")
        st.checkbox("PI only (no derivative)", value=False, key="tl_pi")


def _tune_dispatch(method, plant):
    if method.startswith("1."):
        if plant.has_rhp_poles():
            raise ValueError("Plant has RHP poles — pole cancellation is unsafe. "
                             "Use a different method.")
        if st.session_state["pc_mode"] == "auto":
            p1, p2 = select_slowest_stable_poles(plant)
            if abs(np.imag(p1)) > 1e-9 or abs(np.imag(p2)) > 1e-9:
                raise ValueError(
                    "Auto-selected the slowest poles are complex-conjugate. "
                    "Switch to Manual mode and pick two real poles (or use "
                    "SIMC / Boyd which handle this natively).")
            p1, p2 = float(np.real(p1)), float(np.real(p2))
        else:
            p1 = float(st.session_state["pc_p1"])
            p2 = float(st.session_state["pc_p2"])
        Kd_str = st.session_state["pc_kd"].strip()
        if Kd_str in ("", "1.0"):
            K_dc = abs(plant.dc_gain())
            Kd = 1.0 / K_dc if np.isfinite(K_dc) and K_dc > 1e-9 else 1.0
        else:
            Kd = float(Kd_str)
        return StablePoleCancellation(plant, p1, p2, Kd=Kd).tune()

    if method.startswith("2."):
        _, _, _, _, fopdt = run_step_test(
            plant, step_amp=st.session_state["zn1_step"],
            noise_sigma=st.session_state["zn1_noise"], seed=0)
        return ZieglerNicholsI(fopdt).tune()

    if method.startswith("3."):
        if st.session_state["zn2_source"] == "bode":
            Ku, Pu, _ = find_ultimate_gain(plant)
        else:
            Ku, Pu, _, _, _ = run_relay_test(
                plant, t_max=st.session_state["zn2_relay_T"],
                h=st.session_state["zn2_relay_h"])
        return ZieglerNicholsII(Ku, Pu).tune()

    if method.startswith("4."):
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0, noise_sigma=0.0, seed=0)
        return Amigo(fopdt, integrating=st.session_state["amigo_integrating"]).tune()

    if method.startswith("5."):
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0, noise_sigma=0.0, seed=0)
        tau_c_str = st.session_state["simc_tau_c"].strip()
        tau2_str = st.session_state["simc_tau2"].strip()
        tau_c = float(tau_c_str) if tau_c_str else None
        tau2 = float(tau2_str) if tau2_str else None
        return Simc(fopdt, tau_c=tau_c, tau2=tau2).tune()

    if method.startswith("6."):
        Ms, Mt = st.session_state["boyd_Ms"], st.session_state["boyd_Mt"]
        try:
            _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0, noise_sigma=0.0, seed=0)
            seed = Simc(fopdt).tune().gains
        except Exception:
            seed = None
        return Boyd(plant, Ms=Ms, Mt=Mt, seed_gains=seed).tune()

    if method.startswith("7."):
        _, _, _, _, fopdt = run_step_test(
            plant, step_amp=st.session_state["cc_step"],
            noise_sigma=st.session_state["cc_noise"], seed=0)
        return CohenCoon(fopdt).tune()

    if method.startswith("8."):
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0, noise_sigma=0.0, seed=0)
        return ChienHronesReswick(
            fopdt, response=st.session_state["chr_response"],
            overshoot=int(st.session_state["chr_overshoot"])).tune()

    if method.startswith("9."):
        if st.session_state["tl_source"] == "bode":
            Ku, Pu, _ = find_ultimate_gain(plant)
        else:
            Ku, Pu, _, _, _ = run_relay_test(
                plant, t_max=st.session_state["tl_relay_T"],
                h=st.session_state["tl_relay_h"])
        return TyreusLuyben(Ku, Pu, use_derivative=not st.session_state["tl_pi"]).tune()

    raise RuntimeError(f"unknown method {method}")


# ── simulation ───────────────────────────────────────────────────────────
def _render_sim_settings():
    st.subheader("Closed-loop simulation")
    st.radio("Setpoint", ["step", "ramp", "pulse"], key="sp_kind", horizontal=True)
    st.number_input("amplitude", value=1.0, key="sp_amp")
    st.text_input("duration (blank=auto)", value="", key="sp_t_end")
    st.number_input("u min", value=-100.0, key="u_min")
    st.number_input("u max", value=100.0, key="u_max")
    st.number_input("Derivative filter N (0=disable)", value=80.0, min_value=0.0, key="N")
    st.caption("ramp: linear 0→amp over duration. pulse: amp during [25%, 50%] of duration.")
    st.radio("Anti-windup", ["conditional", "back_calc"], key="antiwindup",
             horizontal=True)
    st.text_input("Ka override (blank=auto)", value="", key="ka_override")
    st.caption("conditional: freeze integral while saturated. back_calc: "
               "Astrom & Hagglund back-calculation. Neither has any effect "
               "unless u min/u max actually saturate the actuator.")


def _sim_settings():
    """Read the closed-loop sim settings widgets into simulate_closed_loop()
    kwargs — shared by _run_closed_loop() and _do_compare_all() so the two
    stay in lockstep (the latter reuses compare_all_methods()'s own sim
    rather than re-simulating, which only produces the same trace as long
    as both call sites pass identical settings)."""
    t_end_str = st.session_state["sp_t_end"].strip()
    t_end = float(t_end_str) if t_end_str else None
    ka_str = st.session_state["ka_override"].strip()
    Ka = float(ka_str) if ka_str else None
    return dict(
        t_end=t_end, setpoint=st.session_state["sp_amp"],
        setpoint_kind=st.session_state["sp_kind"],
        u_min=st.session_state["u_min"], u_max=st.session_state["u_max"],
        N=st.session_state["N"],
        antiwindup=st.session_state["antiwindup"], Ka=Ka,
    )


def _run_closed_loop(plant, gains):
    return simulate_closed_loop(plant, gains, **_sim_settings())


def _antiwindup_tag(sim):
    if sim.antiwindup != "back_calc":
        return ""
    if not np.any(saturation_mask(sim)):
        return " [back_calc: never saturated]"
    return f" [back_calc, Ka={sim.Ka:.3g}]"


def _next_label(method, halved, sim):
    base = method.split(". ", 1)[1] if ". " in method else method
    base = base.split(" (")[0]
    if halved:
        base += " ½"
    kind = st.session_state["sp_kind"]
    if kind != "step":
        base = f"{base} ({kind})"
    base += _antiwindup_tag(sim)
    siso_entries = gs.get_by_kind("siso")
    count = sum(1 for e in siso_entries if e.label.split(" #")[0] == base)
    return base if count == 0 else f"{base} #{count + 1}"


# ── actions ──────────────────────────────────────────────────────────────
def _do_tune(plant, method):
    try:
        result = _tune_dispatch(method, plant)
    except Exception as exc:
        st.error(f"Tuning failed: {exc}")
        return
    if st.session_state.get("halve_gains"):
        result = halve_gains(result)
    try:
        sim = _run_closed_loop(plant, result.gains)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return
    label = _next_label(method, st.session_state.get("halve_gains", False), sim)
    entry = gs.ControllerEntry(kind="siso", label=label, params=result.gains,
                               result=result, sim=sim, plant=plant.pretty())
    entry.mrow = metric_row(plant, label, result.gains,
                            black_box=result.black_box, fopdt=result.fopdt)
    gs.add_controller(entry)
    st.session_state["siso_last_result"] = (result, sim)
    st.success(f"Tuned: {label}")


def _do_compare_all(plant):
    # return_sim=True: compare_all_methods() already simulates each method's
    # setpoint-tracking response internally (for OS%/ts/IAE) — always a unit
    # step against a wide-open actuator (u_min=-1e6, u_max=1e6, N=80, no
    # t_end override), deliberately unconstrained so every method is scored
    # on equal footing regardless of this panel's own actuator widgets. That
    # trace is reusable as the plotted one too, but only when the widgets
    # can't tell the difference: same setpoint/kind, same N, no t_end
    # override, and the trace never actually needed bounds wider than the
    # widgets' own u_min/u_max (i.e. it never would have saturated there
    # either). Otherwise fall back to a fresh, widget-bounded simulation —
    # same as before this reuse existed — rather than risk mislabeling a
    # constrained-actuator trace as the panel's own settings.
    try:
        rows = compare_all_methods(plant, return_sim=True)
    except Exception as exc:
        st.error(f"Comparison failed: {exc}")
        return
    settings = _sim_settings()
    reusable = (settings["setpoint_kind"] == "step"
                and settings["setpoint"] == 1.0
                and settings["N"] == 80.0
                and settings["t_end"] is None)
    gs.clear_by_kind("siso")
    n_ok = 0
    for row in rows:
        gains = row.get("gains")
        base_sim = row.get("sim")
        if gains is None or base_sim is None:
            continue
        if (reusable and float(np.min(base_sim.u)) >= settings["u_min"]
                and float(np.max(base_sim.u)) <= settings["u_max"]):
            sim = base_sim
        else:
            try:
                sim = _run_closed_loop(plant, gains)
            except Exception:
                continue
        entry = gs.ControllerEntry(
            kind="siso", label=row["name"] + _antiwindup_tag(sim),
            params=gains, result=None, sim=sim, plant=plant.pretty())
        entry.mrow = row
        gs.add_controller(entry)
        n_ok += 1
    st.success(f"Compared {n_ok} methods. Untick any below to declutter.")


def absorb_llm_rows(plant_id, rows):
    """Turn raw run_whitebox_benchmark(return_sim=True) rows (row["sim"]
    intact) into session entries — the same gs.ControllerEntry shape
    _do_compare_all builds for a manual "Compare all methods" click, so
    an LLM-triggered run lands in the same session list/plots/heatmap/
    radar as a manual one, tagged source="llm" so the two are still
    distinguishable. Called by streamlit_llm_panel.py's _drain_plot_calls
    after each chat turn — see supervisor_session_pid.Session.plot_calls.
    A row with no gains/sim (a method that failed) is skipped, same as
    _do_compare_all's own guard."""
    # plant_id is the raw plant_tf string the LLM passed (see
    # supervisor_session_pid.Session's plant_of); re-parse it to the same
    # pretty()-formatted form _render_plant_controls() tags manual entries
    # with, so the same plant doesn't show two differently-formatted tags
    # depending on who ran it. Falls back to the raw string on a parse
    # failure -- cosmetic only, never blocks absorbing the rows themselves
    # (the tool already parsed and simulated against this same string).
    plant_label = plant_id
    try:
        plant_label = TransferFunction.parse(plant_id).pretty()
    except Exception:
        pass
    n_ok = 0
    for row in rows:
        gains = row.get("gains")
        sim = row.get("sim")
        if gains is None or sim is None:
            continue
        entry = gs.ControllerEntry(
            kind="siso", label=row["name"] + _antiwindup_tag(sim),
            params=gains, result=None, sim=sim, source="llm", plant=plant_label)
        entry.mrow = row
        gs.add_controller(entry)
        n_ok += 1
    return n_ok


# ── session list ─────────────────────────────────────────────────────────
def _render_session_list():
    st.subheader("Tuned controllers (session overlay)")
    siso_entries = gs.get_by_kind("siso")
    if not siso_entries:
        st.caption("Tune a method (or Compare all methods) to populate this list.")
        return

    # No st.rerun() needed after these — they mutate state that the rest
    # of this same render() pass (the checkbox loop right below, and the
    # plots column after it) reads fresh, so the mutation is already
    # reflected by the time this script run finishes. Clear/remove-unchecked
    # replace the list itself rather than mutating entries in place, so
    # siso_entries is re-fetched afterward instead of relying on the
    # now-stale snapshot from the top of this function.
    #
    # Select/Deselect all also have to force-write each checkbox's own
    # session_state key (siso_en_<id>), not just entry.enabled: once a
    # checkbox with a given key has rendered once, Streamlit ignores a
    # later value= on that same key and keeps the widget's own recorded
    # state — so without this, the checkbox loop below would read back
    # its own stale True/False and immediately overwrite entry.enabled
    # right back to what it was before the click (via the "enabled !=
    # entry.enabled" sync a few lines down).
    cols = st.columns(4)
    if cols[0].button("Select all", key="siso_select_all"):
        gs.set_all_enabled_by_kind("siso", True)
        for e in siso_entries:
            st.session_state[f"siso_en_{e.id}"] = True
    if cols[1].button("Deselect all", key="siso_deselect_all"):
        gs.set_all_enabled_by_kind("siso", False)
        for e in siso_entries:
            st.session_state[f"siso_en_{e.id}"] = False
    if cols[2].button("Clear all", key="siso_clear_all"):
        gs.clear_by_kind("siso")
    if cols[3].button("Remove unchecked", key="siso_remove_unchecked"):
        gs.remove_unchecked_by_kind("siso")
    siso_entries = gs.get_by_kind("siso")

    _assign_colors(siso_entries)
    for i, entry in enumerate(siso_entries):
        c1, c2, c3 = st.columns([1, 3, 4])
        checkbox_key = f"siso_en_{entry.id}"
        # Once a widget's key has a value in session_state, Streamlit warns
        # (and eventually ignores) a value= passed alongside it — so this
        # only seeds the key the first time this entry is ever rendered;
        # every rerun after that reads through the key alone, and the bulk
        # actions above write straight into this same key to change it.
        st.session_state.setdefault(checkbox_key, entry.enabled)
        enabled = c1.checkbox("enabled", key=checkbox_key,
                              label_visibility="collapsed")
        if enabled != entry.enabled:
            gs.set_enabled(entry.id, enabled)
        tag = "  🤖 LLM" if entry.source == "llm" else ""
        c2.markdown(f"{_circle_shortcode(entry.color)} {entry.label}{tag}")
        g = entry.params
        plant_tag = f"  ·  {entry.plant}" if entry.plant else ""
        c3.caption(f"Kp={g.Kp:.3g}  Ki={g.Ki:.3g}  Kd={g.Kd:.3g}{plant_tag}")


def _palette_name(hex_color):
    # Streamlit's markdown colored-circle emoji only covers a fixed set of
    # names; approximate rather than pull in a color-distance library.
    names = {"#1f77b4": "blue", "#d62728": "red", "#2ca02c": "green",
             "#9467bd": "purple", "#ff7f0e": "orange", "#17becf": "blue",
             "#8c564b": "brown", "#e377c2": "purple", "#7f7f7f": "black",
             "#bcbd22": "yellow", "#393b79": "blue", "#ad494a": "red"}
    return names.get(hex_color, "blue")


def _circle_shortcode(hex_color):
    """Markdown emoji shortcode for a colored-circle swatch. Only
    :large_blue_circle: carries a "large_" prefix (a legacy alias) --
    every other color's real shortcode (:red_circle:, :yellow_circle:,
    :green_circle:, ...) has none, so using "large_" unconditionally
    (the previous behavior here) rendered as literal text for every
    color but blue -- confirmed against the `emoji` package's own alias
    table, not assumed. Pre-existing bug (not introduced by this
    session), also present in streamlit_mimo_panel.py's own copy."""
    name = _palette_name(hex_color)
    return f":large_{name}_circle:" if name == "blue" else f":{name}_circle:"


# ── comparison views (heatmap / radar) ──────────────────────────────────
def _session_rows():
    """Metric rows for the enabled SISO session entries — mirrors
    pid_app.py's _session_rows()."""
    rows = []
    for e in gs.get_by_kind("siso"):
        if not e.enabled:
            continue
        row = e.mrow or {"name": e.label, "stable": False, "error": "no metrics"}
        rows.append(row)
    return rows


# ── response plot ────────────────────────────────────────────────────────
def _render_response_plot():
    active = [e for e in gs.get_by_kind("siso") if e.enabled and e.sim is not None]
    fig = Figure(figsize=(9, 8), dpi=100)
    ax_y = fig.add_subplot(311)
    ax_u = fig.add_subplot(312, sharex=ax_y)
    ax_e = fig.add_subplot(313, sharex=ax_y)
    for ax in (ax_y, ax_u, ax_e):
        ax.grid(True, alpha=0.3)
    ax_y.set_ylabel("PV / SP")
    ax_u.set_ylabel("control u(t)")
    ax_e.set_ylabel("error e(t)")
    ax_e.set_xlabel("time (s)")

    if not active:
        ax_y.set_title("No tuned controllers shown — tune a method or "
                       "tick one in the session list.")
        st.pyplot(fig)
        _download_fig_button(fig, "siso_response.png", key="siso_response_dl")
        return

    seen_kinds = set()
    for entry in active:
        kind = entry.sim.sp_kind
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        kind_label = f"setpoint ({kind})" if len(active) > 1 else "setpoint"
        same_kind = [e for e in active if e.sim.sp_kind == kind]
        longest = max(same_kind, key=lambda e: len(e.sim.t))
        ax_y.plot(longest.sim.t, longest.sim.sp, "--", color="#666",
                  linewidth=1.0, alpha=0.7, label=kind_label)

    saturated_any = False
    for entry in active:
        label = entry.label
        if entry.sim.metrics.get("unstable"):
            label += "  [UNSTABLE]"
        ax_y.plot(entry.sim.t, entry.sim.y, color=entry.color, linewidth=1.5, label=label)
        ax_u.plot(entry.sim.t, entry.sim.u, color=entry.color, linewidth=1.2, label=entry.label)
        sat_mask = saturation_mask(entry.sim)
        if sat_mask.any():
            saturated_any = True
            ax_u.plot(entry.sim.t[sat_mask], entry.sim.u[sat_mask], color=entry.color,
                      marker="o", markersize=3, linestyle="none", alpha=0.7)
        ax_e.plot(entry.sim.t, entry.sim.e, color=entry.color, linewidth=1.2, label=entry.label)

    ax_y.legend(loc="lower right", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    if saturated_any:
        ax_u.plot([], [], marker="o", markersize=3, linestyle="none", color="#666",
                  label="saturated (u at u_min/u_max)")
        ax_u.legend(loc="lower right", fontsize=7)
    stable_ys = [e.sim.y for e in active if not e.sim.metrics.get("unstable")]
    if stable_ys:
        max_sp = max(np.max(np.abs(e.sim.sp)) for e in active)
        ax_y.set_ylim(-0.2 * max_sp, max(2.0 * max_sp, 0.1))
    # Crop the shared time axis to where the traces actually settle — the
    # plant's auto t_end (sized for a correct settling-time measurement,
    # not for viewing) can run 10x longer than anything visually
    # interesting. Same auto_plot_window() the LQG track plots use; widest
    # window across the overlaid entries wins, and since ax_u/ax_e share
    # ax_y's x-axis, one set_xlim covers all three subplots.
    xmax = max(auto_plot_window(e.sim.t, e.sim.y, e.sim.u, e.sim.e) for e in active)
    ax_y.set_xlim(0.0, xmax)
    fig.tight_layout()
    st.pyplot(fig)
    _download_fig_button(fig, "siso_response.png", key="siso_response_dl")


def _render_last_result():
    last = st.session_state.get("siso_last_result")
    if last is None:
        return
    result, sim = last
    parts = [f"**{result.method}**", "", result.gains.pretty()]
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
    parts.append("\n\nMetrics: " + format_metrics(sim.metrics))
    st.info("\n".join(parts))


# ── entry point ──────────────────────────────────────────────────────────
# Every plant-form/sim-setting field this panel owns -- listed once so
# render_controls() can preserve them all across a Track/Mode switch (see
# streamlit_gui_state.preserve_widget_state's own note on why). Both
# plant forms' fields are listed even though only one renders at a time
# (siso_plant_form picks which) -- snapshot_widget_state skips whichever
# one didn't render this turn, so this is harmless, not a second gap.
# Method-specific arg widgets (pc_p1, zn1_step, ...) aren't included --
# only one method's args render at a time even within an active Manual+
# SISO session, so protecting those needs the same treatment applied
# inside _render_method_args itself; not done here, narrower gap.
_PROTECTED_KEYS = [
    "siso_plant_form", "siso_tf_expr", "siso_gain", "siso_num", "siso_den", "siso_L",
    "siso_method", "halve_gains",
    "sp_kind", "sp_amp", "sp_t_end", "u_min", "u_max", "N", "antiwindup", "ka_override",
]


def render_controls():
    """The left-hand controls half — called by streamlit_unified_panel.py
    when Track=SISO/PID, Mode=Manual. Split from what used to be one
    render() (see git history) so the unified layout can put this in its
    own column and streamlit_llm_panel.py's chat can occupy the same slot
    for Mode=LLM Supervisor instead, without the two ever coexisting in
    the same script run — see streamlit_unified_panel.py's docstring for
    why that matters."""
    gs.preserve_widget_state(_PROTECTED_KEYS)
    plant = _render_plant_controls()

    st.subheader("Compare all methods")
    if st.button("⊞  Compare all methods", key="siso_compare_all",
                disabled=plant is None):
        _do_compare_all(plant)

    st.subheader("Tune one method at a time")
    method = st.selectbox("Method", METHODS, key="siso_method")
    _render_method_args(method)
    st.checkbox("Halve gains (divide Kp, Ki, Kd by 2)", value=False,
               key="halve_gains")
    st.caption("Recommended for ZN-I/II when tracking setpoints.")
    if st.button("Tune & simulate", key="siso_tune", disabled=plant is None):
        _do_tune(plant, method)

    _render_sim_settings()
    gs.snapshot_widget_state(_PROTECTED_KEYS)
    _render_session_list()
    _render_last_result()


def render_plots():
    """The right-hand plots half — called by streamlit_unified_panel.py
    whenever Track=SISO/PID, regardless of Mode: entries in
    gs.get_by_kind("siso") come from either the manual controls above or
    streamlit_llm_panel.py's absorb_llm_rows(), tagged by entry.source,
    so this reads and draws the same session list either way. Stacked
    vertically rather than switched via tabs/radio — an inner st.tabs
    nested inside the outer layout render unreliably in Streamlit's
    frontend (no error at the Python level, but the inner tab bar can end
    up invisible/non-interactive); showing all three at once sidesteps
    that entirely."""
    _assign_colors(gs.get_by_kind("siso"))
    st.subheader("Response")
    _render_response_plot()
    st.subheader("Heatmap")
    scv.render_heatmap(_session_rows())
    st.subheader("Radar")
    scv.render_radar(_session_rows())
