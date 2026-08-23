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
    st.checkbox("Derivative filter (N=80) — recommended", value=True, key="d_filter")
    st.caption("ramp: linear 0→amp over duration. pulse: amp during [25%, 50%] of duration.")
    st.radio("Anti-windup", ["conditional", "back_calc"], key="antiwindup",
             horizontal=True)
    st.text_input("Ka override (blank=auto)", value="", key="ka_override")
    st.caption("conditional: freeze integral while saturated. back_calc: "
               "Astrom & Hagglund back-calculation. Neither has any effect "
               "unless u min/u max actually saturate the actuator.")


def _run_closed_loop(plant, gains):
    t_end_str = st.session_state["sp_t_end"].strip()
    t_end = float(t_end_str) if t_end_str else None
    ka_str = st.session_state["ka_override"].strip()
    Ka = float(ka_str) if ka_str else None
    return simulate_closed_loop(
        plant, gains, t_end=t_end, setpoint=st.session_state["sp_amp"],
        setpoint_kind=st.session_state["sp_kind"],
        u_min=st.session_state["u_min"], u_max=st.session_state["u_max"],
        use_d_filter=st.session_state["d_filter"],
        antiwindup=st.session_state["antiwindup"], Ka=Ka,
    )


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
                               result=result, sim=sim)
    entry.mrow = metric_row(plant, label, result.gains,
                            black_box=result.black_box, fopdt=result.fopdt)
    gs.add_controller(entry)
    st.session_state["siso_last_result"] = (result, sim)
    st.success(f"Tuned: {label}")


def _do_compare_all(plant):
    try:
        rows = compare_all_methods(plant)
    except Exception as exc:
        st.error(f"Comparison failed: {exc}")
        return
    gs.clear_by_kind("siso")
    n_ok = 0
    for row in rows:
        gains = row.get("gains")
        if gains is None:
            continue
        try:
            sim = _run_closed_loop(plant, gains)
        except Exception:
            continue
        entry = gs.ControllerEntry(
            kind="siso", label=row["name"] + _antiwindup_tag(sim),
            params=gains, result=None, sim=sim)
        entry.mrow = row
        gs.add_controller(entry)
        n_ok += 1
    st.success(f"Compared {n_ok} methods. Untick any below to declutter.")


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

    for i, entry in enumerate(siso_entries):
        entry.color = PALETTE[i % len(PALETTE)]
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
        c2.markdown(f":large_{_palette_name(entry.color)}_circle: {entry.label}")
        g = entry.params
        c3.caption(f"Kp={g.Kp:.3g}  Ki={g.Ki:.3g}  Kd={g.Kd:.3g}")


def _palette_name(hex_color):
    # Streamlit's markdown colored-circle emoji only covers a fixed set of
    # names; approximate rather than pull in a color-distance library.
    names = {"#1f77b4": "blue", "#d62728": "red", "#2ca02c": "green",
             "#9467bd": "purple", "#ff7f0e": "orange", "#17becf": "blue",
             "#8c564b": "brown", "#e377c2": "purple", "#7f7f7f": "black",
             "#bcbd22": "yellow", "#393b79": "blue", "#ad494a": "red"}
    return names.get(hex_color, "blue")


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
    fig.tight_layout()
    st.pyplot(fig)


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
def render():
    controls, plots = st.columns([2, 3])

    with controls:
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
        _render_session_list()
        _render_last_result()

    with plots:
        # A radio switch, not a nested st.tabs — the outer app already uses
        # st.tabs for SISO/MIMO/LLM Chat, and nesting a second st.tabs
        # inside one of those tabs renders unreliably in Streamlit's
        # frontend (no error at the Python level, but the inner tab bar
        # can end up invisible/non-interactive).
        view = st.radio("View", ["Response", "Heatmap", "Radar"],
                        key="siso_view", horizontal=True)
        if view == "Response":
            _render_response_plot()
        elif view == "Heatmap":
            scv.render_heatmap(_session_rows())
        else:
            scv.render_radar(_session_rows())
