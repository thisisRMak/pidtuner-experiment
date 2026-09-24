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

import datetime
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
from pid_compare import compare_all_methods, metric_row, TABLE_METRICS
from pid_simulate import simulate_closed_loop, format_metrics, saturation_mask
from lqg_simulate import auto_plot_window

import report_html
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


def _current_manual_plant_object() -> TransferFunction | None:
    """The actual TransferFunction currently sitting in this panel's own
    plant widgets, read via shadow state (gs.peek()) -- same resolution
    current_manual_plant() uses (split out so a caller needing the real
    object, not just its .pretty() string, can get it too -- see
    build_entries_report_sections()'s own "Plant" section). Returns None
    under the same conditions current_manual_plant() does."""
    form = gs.peek("siso_plant_form")
    if form is None:
        return None
    L = gs.peek("siso_L", 0.0)
    try:
        if form == "Symbolic":
            return TransferFunction.parse(gs.peek("siso_tf_expr", ""), L=L)
        gain = float(gs.peek("siso_gain"))
        num = parse_coeff_list(gs.peek("siso_num"))
        den = parse_coeff_list(gs.peek("siso_den"))
        return TransferFunction.from_coeffs(num=num, den=den, L=L, gain=gain)
    except Exception:
        return None


def current_manual_plant() -> str | None:
    """The pretty()-formatted plant currently sitting in this panel's own
    plant widgets -- see _current_manual_plant_object()'s docstring for
    the resolution and why it's shadow-state-based. Same pretty()-
    formatted string ControllerEntry.plant already carries, so a caller
    like streamlit_llm_panel._render_manual_plant_hint can compare it
    against existing LLM entries directly."""
    plant = _current_manual_plant_object()
    return plant.pretty() if plant is not None else None


def _symbolic_poly(coeffs) -> str:
    """Descending-power coefficients (plant.py's own convention, same as
    np.polyval) -> an 'as^n + bs^(n-1) + ...' HTML string, e.g.
    [10, 11, 1] -> '10s<sup>2</sup> + 11s + 1'. Not a general-purpose
    polynomial formatter -- just enough to make a report's plant
    human-readable, no attempt at e.g. suppressing a redundant '+' sign
    beyond the leading term."""
    n = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        power = n - i
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        if power == 0:
            term = f"{mag:g}"
        elif power == 1:
            term = "s" if mag == 1 else f"{mag:g}s"
        else:
            term = f"s<sup>{power}</sup>" if mag == 1 else f"{mag:g}s<sup>{power}</sup>"
        parts.append((sign, term))
    if not parts:
        return "0"
    out = parts[0][1] if parts[0][0] == "+" else f"-{parts[0][1]}"
    for sign, term in parts[1:]:
        out += f" {sign} {term}"
    return out


def _symbolic_tf(plant: TransferFunction) -> str:
    """plant -> '(num) / (den)', optionally '· e^-Ls' -- the symbolic
    counterpart to plant.pretty()'s MATLAB num=[]/den=[] form, for a
    report reader who thinks in transfer-function fractions, not
    coefficient vectors."""
    expr = f"({_symbolic_poly(plant.num)}) / ({_symbolic_poly(plant.den)})"
    if plant.L > 0:
        expr += f" &middot; e<sup>-{plant.L:g}s</sup>"
    return expr


def _render_plant_controls():
    st.subheader("Plant G(s)")
    # default= (not just a pre-set session_state key) would trip Streamlit's
    # "widget created with a default value but also had its value set via
    # the Session State API" warning on every Track/Mode switch, since
    # preserve_widget_state() above already reseeds this key from its
    # shadow before this widget renders -- setdefault() instead only seeds
    # the very first time this key has never existed.
    st.session_state.setdefault("siso_plant_form", "Symbolic")
    st.segmented_control("Plant form", ["Symbolic", "MATLAB coefficients"],
                         key="siso_plant_form", required=True)
    if st.session_state["siso_plant_form"] == "Symbolic":
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("siso_tf_expr", "1000 / ((s+1)*(10s+1))")
        st.text_input("G(s) =", key="siso_tf_expr")
        st.caption("examples:  1000/((s+1)(10s+1))    2/(5s+1)    "
                   "(s+2)/(s^2+3s+1)    1/(s(s+1))")
    else:
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("siso_gain", "1000")
        st.text_input("gain K", key="siso_gain")
        st.session_state.setdefault("siso_num", "[1]")
        st.text_input("num", key="siso_num")
        st.session_state.setdefault("siso_den", "[10, 11, 1]")
        st.text_input("den", key="siso_den")
        st.caption("MATLAB tf(num, den) form, descending powers of s.  "
                   "num=[1, 2] → s + 2   den=[10, 11, 1] → 10s² + 11s + 1")
    # setdefault(), not value= -- see siso_plant_form's comment above.
    st.session_state.setdefault("siso_L", 0.0)
    st.number_input("L (dead time, s)", key="siso_L")

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
# Every method's arg-widget keys -- only one method's args render at a time
# (_render_method_args dispatches on `method`), so without this a Boyd
# Ms/Mt (say) typed in gets wiped by Streamlit the moment a different
# method is selected, same class of bug _PROTECTED_KEYS guards against
# for a Track/Mode switch. See streamlit_gui_state.preserve_widget_state's
# module note.
_METHOD_ARG_KEYS = [
    "pc_mode", "pc_p1", "pc_p2", "pc_kd",
    "zn1_step", "zn1_noise",
    "zn2_source", "zn2_relay_h", "zn2_relay_T",
    "amigo_integrating",
    "simc_tau_c", "simc_tau2",
    "boyd_Ms", "boyd_Mt",
    "cc_step", "cc_noise",
    "chr_response", "chr_overshoot",
    "tl_source", "tl_relay_h", "tl_relay_T", "tl_pi",
]


def _render_method_args(method):
    gs.preserve_widget_state(_METHOD_ARG_KEYS)
    if method.startswith("1."):
        # setdefault(), not default= -- see siso_plant_form's comment above.
        st.session_state.setdefault("pc_mode", "auto")
        mode = st.segmented_control("Pole selection", ["auto", "manual"], key="pc_mode",
                                    required=True)
        st.caption("Cancel poles at s = −p₁, s = −p₂")
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("pc_p1", 0.1)
        st.number_input("p₁ (positive)", key="pc_p1", disabled=mode == "auto")
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("pc_p2", 1.0)
        st.number_input("p₂ (positive)", key="pc_p2", disabled=mode == "auto")
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("pc_kd", "1.0")
        st.text_input("Kd (blank/1.0 = auto-scaled)", key="pc_kd")
    elif method.startswith("2."):
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("zn1_step", 1.0)
        st.number_input("step amplitude", key="zn1_step")
        st.session_state.setdefault("zn1_noise", 0.0)
        st.number_input("noise sigma", key="zn1_noise")
    elif method.startswith("3."):
        # setdefault(), not default= -- see siso_plant_form's comment above.
        st.session_state.setdefault("zn2_source", "bode")
        st.segmented_control("Ultimate gain source", ["bode", "relay"], key="zn2_source",
                             required=True)
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("zn2_relay_h", 1.0)
        st.number_input("relay h", key="zn2_relay_h")
        st.session_state.setdefault("zn2_relay_T", 50.0)
        st.number_input("relay T (s)", key="zn2_relay_T")
    elif method.startswith("4."):
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("amigo_integrating", False)
        st.checkbox("Integrating process", key="amigo_integrating")
    elif method.startswith("5."):
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("simc_tau_c", "")
        st.text_input("tau_c (blank=auto)", key="simc_tau_c")
        st.session_state.setdefault("simc_tau2", "")
        st.text_input("tau2 (blank=auto)", key="simc_tau2")
    elif method.startswith("6."):
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("boyd_Ms", 1.4)
        st.number_input("Ms", key="boyd_Ms")
        st.session_state.setdefault("boyd_Mt", 1.4)
        st.number_input("Mt", key="boyd_Mt")
    elif method.startswith("7."):
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("cc_step", 1.0)
        st.number_input("step amplitude", key="cc_step")
        st.session_state.setdefault("cc_noise", 0.0)
        st.number_input("noise sigma", key="cc_noise")
    elif method.startswith("8."):
        # setdefault(), not default= -- see siso_plant_form's comment above.
        st.session_state.setdefault("chr_response", "setpoint")
        st.segmented_control("Response", ["setpoint", "load"], key="chr_response",
                             required=True)
        st.session_state.setdefault("chr_overshoot", 0)
        st.segmented_control("Overshoot", [0, 20], key="chr_overshoot", required=True)
    elif method.startswith("9."):
        # setdefault(), not default= -- see siso_plant_form's comment above.
        st.session_state.setdefault("tl_source", "bode")
        st.segmented_control("Ultimate gain source", ["bode", "relay"], key="tl_source",
                             required=True)
        # setdefault(), not value= -- see siso_plant_form's comment above.
        st.session_state.setdefault("tl_relay_h", 1.0)
        st.number_input("relay h", key="tl_relay_h")
        st.session_state.setdefault("tl_relay_T", 50.0)
        st.number_input("relay T (s)", key="tl_relay_T")
        st.session_state.setdefault("tl_pi", False)
        st.checkbox("PI only (no derivative)", key="tl_pi")
    gs.snapshot_widget_state(_METHOD_ARG_KEYS)


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
    # setdefault(), not default= -- see siso_plant_form's comment above.
    st.session_state.setdefault("sp_kind", "step")
    st.segmented_control("Setpoint", ["pulse", "step", "ramp"], key="sp_kind",
                         required=True)
    # setdefault(), not value= -- see siso_plant_form's comment above.
    st.session_state.setdefault("sp_amp", 1.0)
    st.number_input("amplitude", key="sp_amp")
    st.session_state.setdefault("sp_t_end", "")
    st.text_input("duration (blank=auto)", key="sp_t_end")
    st.session_state.setdefault("u_min", -100.0)
    st.number_input("u min", key="u_min")
    st.session_state.setdefault("u_max", 100.0)
    st.number_input("u max", key="u_max")
    st.session_state.setdefault("N", 80.0)
    st.number_input("Derivative filter N (0=disable)", min_value=0.0, key="N")
    st.caption("ramp: linear 0→amp over duration. pulse: amp during [25%, 50%] of duration.")
    # setdefault(), not default= -- see siso_plant_form's comment above.
    st.session_state.setdefault("antiwindup", "conditional")
    st.segmented_control("Anti-windup", ["conditional", "back_calc"], key="antiwindup",
                         required=True)
    # setdefault(), not value= -- see siso_plant_form's comment above.
    st.session_state.setdefault("ka_override", "")
    st.text_input("Ka override (blank=auto)", key="ka_override")
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
    # Only "you"-sourced entries -- clearing the whole kind (the previous
    # behavior here) also silently wiped any source="llm" entries the
    # user had no reason to expect this button to touch. "Clear all"
    # (below) stays the explicit, unscoped way to drop everything.
    gs.clear_by_kind_and_source("siso", "you")
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


def absorb_llm_rows(plant_id, rows, delay=0.0):
    """Turn raw run_whitebox_benchmark(return_sim=True) rows (row["sim"]
    intact) into session entries — the same gs.ControllerEntry shape
    _do_compare_all builds for a manual "Compare all methods" click, so
    an LLM-triggered run lands in the same session list/plots/heatmap/
    radar as a manual one, tagged source="llm" so the two are still
    distinguishable. Called by streamlit_llm_panel.py's _drain_plot_calls
    after each chat turn — see supervisor_session_pid.Session.plot_calls.
    A row with no gains/sim (a method that failed) is skipped, same as
    _do_compare_all's own guard.

    row["sim"] here comes from the same run_whitebox_benchmark ->
    compare_all_methods(..., return_sim=True) path _do_compare_all uses
    for a manual "Compare all methods" click -- always a wide-open-
    actuator unit-step trace, reusable as this panel's own plotted trace
    only under the same conditions _do_compare_all checks for (see its
    own comment): same setpoint/kind, same N, no t_end override, and the
    trace never actually needed bounds wider than this panel's u_min/
    u_max. Otherwise it's re-simulated against this panel's actual sim
    settings, same as _do_compare_all's fallback -- without this, an
    LLM-triggered entry would always show the unconstrained trace even
    when the panel's own actuator limits say otherwise."""
    # plant_id is the raw plant_tf string the LLM passed, `delay` its raw
    # "delay" kwarg (see supervisor_session_pid.Session's plot_calls);
    # re-parse into the same pretty()-formatted form _render_plant_
    # controls() tags manual entries with, so the same plant doesn't show
    # two differently-formatted tags depending on who ran it. Falls back
    # to the raw string on a parse failure -- cosmetic only, never blocks
    # absorbing the rows themselves (the tool already parsed and
    # simulated against this same plant_id/delay pair).
    plant_label = plant_id
    plant_obj = None
    try:
        plant_obj = TransferFunction.parse(plant_id, L=delay)
        plant_label = plant_obj.pretty()
    except Exception:
        pass

    # Called from streamlit_llm_panel.py's _drain_plot_calls, which only
    # ever runs while Mode=LLM Supervisor -- this panel's own sim-setting
    # widgets (u_min, sp_t_end, ...) aren't instantiated on that script
    # run, so Streamlit has already dropped them from session_state by
    # the time this executes (confirmed live: _sim_settings() below raises
    # KeyError on "sp_t_end" without this). render_controls() guards its
    # own read of these same keys with this exact call, at its top, before
    # they're re-instantiated -- see streamlit_gui_state.preserve_widget_
    # state's module note. A no-op once Manual mode has rendered them
    # again this run.
    gs.preserve_widget_state(_PROTECTED_KEYS)
    settings = _sim_settings()
    reusable = (settings["setpoint_kind"] == "step"
                and settings["setpoint"] == 1.0
                and settings["N"] == 80.0
                and settings["t_end"] is None)

    n_ok = 0
    for row in rows:
        gains = row.get("gains")
        base_sim = row.get("sim")
        if gains is None or base_sim is None:
            continue
        if (reusable and float(np.min(base_sim.u)) >= settings["u_min"]
                and float(np.max(base_sim.u)) <= settings["u_max"]):
            sim = base_sim
        elif plant_obj is not None:
            try:
                sim = _run_closed_loop(plant_obj, gains)
            except Exception:
                continue
        else:
            # plant_id failed to re-parse above (see try/except) -- can't
            # rebuild the plant to re-simulate against, so fall back to
            # the unconstrained trace rather than dropping the row.
            sim = base_sim
        entry = gs.add_llm_entry(
            kind="siso", label=row["name"] + _antiwindup_tag(sim),
            params=gains, result=None, sim=sim, plant=plant_label,
            plant_tf=plant_id, plant_L=delay)
        entry.mrow = row
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
    # of this same render_plots() pass (the checkbox loop right below,
    # and the Response/Heatmap/Radar plots after it) reads fresh, so the
    # mutation is already reflected by the time this script run finishes.
    # Clear/remove-unchecked
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
    cols = st.columns(5)
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
    if cols[4].button("Clear LLM plots", key="siso_clear_llm_plots"):
        # Only source="llm" entries -- the one bulk action here that's
        # scoped by provenance, not just by kind. Lives here (not in
        # streamlit_llm_panel.py/streamlit_judge_panel.py's own controls)
        # because it only ever acts on this list, and works the same
        # regardless of which Mode is currently active -- a Manual-mode
        # user with LLM clutter left over from an earlier chat can use it
        # too, not just while actually in a chat Mode.
        gs.clear_by_kind_and_source("siso", "llm")
    siso_entries = gs.get_by_kind("siso")

    gs.assign_colors(siso_entries)
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
        tag = "  :violet-badge[🤖 LLM]" if entry.source == "llm" else ""
        c2.markdown(f"{_circle_emoji(entry.color)} {entry.label}{tag}")
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


_CIRCLE_EMOJI = {
    "blue": "🔵", "red": "🔴", "green": "🟢", "purple": "🟣",
    "orange": "🟠", "brown": "🟤", "black": "⚫", "yellow": "🟡",
}


def _circle_emoji(hex_color):
    """Colored-circle swatch for a session-list row, as a literal
    Unicode character rather than a markdown :shortcode: (e.g.
    :green_circle:). An earlier version of this function returned
    shortcode text instead, on the theory that Streamlit's markdown
    renderer converts :shortcode: to the matching emoji the way GitHub's
    does -- checked, at the time, against the `emoji` Python package's
    own alias table. Confirmed live in an actual browser (not just
    AppTest, which only inspects the server-side element tree/raw
    markdown source and so cannot see this) that Streamlit's frontend
    only recognizes a narrow subset: :large_blue_circle:, :red_circle:,
    and :black_circle: converted correctly, but :green_circle:,
    :purple_circle:, :orange_circle:, :brown_circle:, and
    :yellow_circle: all rendered as literal, unconverted text. Embedding
    the actual emoji character sidesteps shortcode conversion (and
    whatever narrower set Streamlit happens to support) entirely -- the
    same way the 🤖 tag elsewhere in this file has always worked, since
    it was never a shortcode to begin with. Also present, identically,
    in streamlit_mimo_panel.py's own copy."""
    return _CIRCLE_EMOJI[_palette_name(hex_color)]


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
def _build_response_fig(active):
    """The Figure-building half of _render_response_plot(), split out so
    a caller that isn't rendering to the screen (report_html-based report
    generation) can build the exact same figure without going through
    st.pyplot(). `active` is already the enabled/sim-present filter the
    caller wants plotted -- this function has no opinion on session state."""
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
        return fig

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

    ax_y.legend(loc="lower right", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=3)
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
    return fig


def _render_response_plot():
    active = [e for e in gs.get_by_kind("siso") if e.enabled and e.sim is not None]
    fig = _build_response_fig(active)
    st.pyplot(fig)
    _download_fig_button(fig, "siso_response.png", key="siso_response_dl")


def _fmt_metric(val):
    """Same convention streamlit_siso_comparison_views.py's heatmap cells
    already use (f"{val:.3g}" if finite else an em dash) -- kept
    consistent so a number in the report matches the number on screen."""
    if val is None or not np.isfinite(val):
        return "—"
    return f"{val:.3g}"


def build_entries_report_sections():
    """(subtitle, [section_html, ...]) for every currently-enabled
    session-list entry in this Track (any source -- the graph already
    mixes Manual and LLM-triggered entries by design, see streamlit_
    unified_panel.py's own docstring, and this reflects exactly what's on
    screen, nothing more): a metrics table, the response plot, the
    heatmap, and the radar -- the same four things render_plots() always
    shows stacked together, via the same builder functions
    (_build_response_fig(), scv.build_heatmap_html(), scv.build_radar_fig()),
    not a re-implementation of any of them. Also a "Plant" section stating
    the currently-selected plant in both MATLAB (num/den) and symbolic
    (fraction) form -- see the comment where it's built for why this is
    the *currently selected* plant, not necessarily every active entry's.

    Public (no leading underscore) and parameter-free -- called both by
    this module's own _build_report_html() below and directly by
    streamlit_llm_panel.py/streamlit_judge_panel.py, which embed this same
    "what's currently plotted" section inside their own, bigger
    conversation reports rather than duplicating this table/plot logic."""
    # entry.color isn't a stored field -- render_plots() assigns it by
    # list position every run (gs.assign_colors()) right before drawing.
    # render_controls() (where this is called from) runs *before*
    # render_plots() in streamlit_unified_panel.render()'s own column
    # order, so nothing has assigned it yet this run without this call --
    # same full-kind-list call render_plots() itself makes, so the report
    # gets the identical colors the on-screen plot has.
    gs.assign_colors(gs.get_by_kind("siso"))
    active = [e for e in gs.get_by_kind("siso") if e.enabled]
    plants = sorted({e.plant for e in active if e.plant})
    subtitle = "SISO / PID — " + ("; ".join(plants) if plants else "no plant selected")

    # The currently-selected plant, not each entry's own -- entries don't
    # retain their originating TransferFunction object (ControllerEntry.
    # plant is only ever a pre-formatted .pretty() string, see gs.py), so
    # there's nothing reliable to derive a symbolic form FROM per entry.
    # This matches the common case (one plant per report) and needs no
    # change to any entry-creation path; it just won't match every entry
    # if the session compared more than one plant. current_plant is None
    # if Manual/SISO's widgets have never rendered this session, or the
    # plant currently sitting in them doesn't parse.
    sections = []
    current_plant = _current_manual_plant_object()
    if current_plant is not None:
        sections.append(
            "<h2>Plant</h2>"
            f"<p>MATLAB form: <code>{report_html.e(current_plant.pretty())}</code></p>"
            f"<p>Symbolic form: {_symbolic_tf(current_plant)}</p>"
        )

    headers = ["Method", "Stable"] + TABLE_METRICS
    rows = []
    for e in active:
        row = e.mrow or {}
        stable = row.get("stable")
        cells = [e.label, "yes" if stable else "no"]
        cells += ["—" if not stable else _fmt_metric(row.get(m)) for m in TABLE_METRICS]
        rows.append(cells)
    sections.append(f"<h2>Compared methods</h2>{report_html.render_table(headers, rows)}")

    sim_active = [e for e in active if e.sim is not None]
    if sim_active:
        fig = _build_response_fig(sim_active)
        img = report_html.fig_to_data_uri(fig)
        sections.append(f'<h2>Step response</h2><img src="{img}" style="max-width:100%">')

    # Same rows() as the on-screen Heatmap/Radar views (_session_rows()),
    # reused rather than re-derived, so the report mirrors exactly what's
    # plotted there -- render_plots() already shows all three (Response/
    # Heatmap/Radar) stacked for every Mode, so a report that only had the
    # first of the three would be missing what's actually on screen.
    scv_rows = _session_rows()
    heatmap_html = scv.build_heatmap_html(scv_rows)
    if heatmap_html:
        sections.append(f"<h2>Heatmap</h2>{heatmap_html}")

    radar_fig = scv.build_radar_fig(scv_rows)
    if radar_fig is not None:
        radar_img = report_html.fig_to_data_uri(radar_fig)
        sections.append(f'<h2>Radar</h2><img src="{radar_img}" style="max-width:100%">')

    return subtitle, sections


def _build_report_html():
    """Manual mode's own SISO/PID report -- just build_entries_report_
    sections()'s sections, wrapped as a standalone document. See that
    function's docstring for what it actually contains."""
    subtitle, sections = build_entries_report_sections()
    return report_html.build_report("PIDTuner SISO/PID Report", subtitle, sections)


def _render_download_report_button():
    if not any(e.enabled for e in gs.get_by_kind("siso")):
        return
    st.download_button(
        "Download report", data=_build_report_html(),
        file_name=f"pidtuner-siso-report-{datetime.date.today().isoformat()}.html",
        mime="text/html", key="siso_download_report",
    )


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
# SISO session, so those get the same treatment separately, via
# _METHOD_ARG_KEYS inside _render_method_args itself.
_PROTECTED_KEYS = [
    "siso_plant_form", "siso_tf_expr", "siso_gain", "siso_num", "siso_den", "siso_L",
    "siso_method", "halve_gains",
    "sp_kind", "sp_amp", "sp_t_end", "u_min", "u_max", "N", "antiwindup", "ka_override",
]


def _render_llm_plant_carryover():
    """"LLM Supervisor last analyzed: <plant> — [Load this plant]", shown
    only when the most recent LLM-sourced entry ran against a plant this
    panel's own widgets don't currently hold -- so switching from Mode=
    LLM Supervisor back to Manual doesn't strand the plant just discussed
    in chat behind a re-typed expression. Compares the raw, reloadable
    plant_tf/plant_L (see gs.ControllerEntry's own note on why entry.plant
    itself -- a .pretty()-formatted display string -- can't be fed back
    into the siso_tf_expr/siso_L widgets).

    Must run before _render_plant_controls() instantiates those widgets:
    the click handler seeds session_state[...] directly, the same
    "pre-set session_state before creating the widget" trick
    gs.preserve_widget_state()/the "Select all" button already rely on.
    Unlike that button, this needs an explicit st.rerun() right after --
    the caption above is decided before the click can be seen, so without
    it the caption would still flash "last analyzed" once more on the
    very run that loads the plant, only clearing on whatever rerun
    happens next."""
    llm_entries = [e for e in gs.get_by_kind("siso") if e.source == "llm" and e.plant_tf]
    if not llm_entries:
        return
    last = llm_entries[-1]
    if (last.plant_tf == st.session_state.get("siso_tf_expr")
            and last.plant_L == st.session_state.get("siso_L")):
        return
    c1, c2 = st.columns([5, 1])
    c1.caption(f"🤖 LLM Supervisor last analyzed: {last.plant}")
    if c2.button("Load this plant", key="siso_load_llm_plant"):
        st.session_state["siso_tf_expr"] = last.plant_tf
        st.session_state["siso_L"] = last.plant_L
        st.rerun()


def render_controls():
    """The left-hand controls half — called by streamlit_unified_panel.py
    when Track=SISO/PID, Mode=Manual. Split from what used to be one
    render() (see git history) so the unified layout can put this in its
    own column and streamlit_llm_panel.py's chat can occupy the same slot
    for Mode=LLM Supervisor instead, without the two ever coexisting in
    the same script run — see streamlit_unified_panel.py's docstring for
    why that matters."""
    gs.preserve_widget_state(_PROTECTED_KEYS)
    _render_llm_plant_carryover()
    plant = _render_plant_controls()

    st.subheader("Compare all methods")
    if st.button("⊞  Compare all methods", key="siso_compare_all",
                disabled=plant is None):
        _do_compare_all(plant)

    st.subheader("Tune one method at a time")
    method = st.selectbox("Method", METHODS, key="siso_method")
    _render_method_args(method)
    # setdefault(), not value= -- see siso_plant_form's comment above.
    st.session_state.setdefault("halve_gains", False)
    st.checkbox("Halve gains (divide Kp, Ki, Kd by 2)", key="halve_gains")
    st.caption("Recommended for ZN-I/II when tracking setpoints.")
    if st.button("Tune & simulate", key="siso_tune", disabled=plant is None):
        _do_tune(plant, method)

    _render_sim_settings()
    gs.snapshot_widget_state(_PROTECTED_KEYS)
    _render_last_result()
    _render_download_report_button()


def render_plots():
    """The right-hand plots half — called by streamlit_unified_panel.py
    whenever Track=SISO/PID, regardless of Mode: entries in
    gs.get_by_kind("siso") come from either the manual controls above or
    streamlit_llm_panel.py's absorb_llm_rows(), tagged by entry.source,
    so this reads and draws the same session list either way.

    The session list itself renders here too, not in render_controls()
    -- it used to live there, but that meant it was invisible in Mode=
    LLM Supervisor (render_controls() shows the chat instead), so an
    LLM-triggered entry's checkbox/badge/Clear-all were only reachable
    by switching to Manual first. Reported live; moved here so it's
    visible and manageable regardless of Mode, same as the plots below
    it already were.

    The three plots are stacked vertically rather than switched via
    tabs/radio — an inner st.tabs nested inside the outer layout render
    unreliably in Streamlit's frontend (no error at the Python level,
    but the inner tab bar can end up invisible/non-interactive); showing
    all three at once sidesteps that entirely."""
    gs.assign_colors(gs.get_by_kind("siso"))
    _render_session_list()
    st.subheader("Response")
    _render_response_plot()
    st.subheader("Heatmap")
    scv.render_heatmap(_session_rows())
    st.subheader("Radar")
    scv.render_radar(_session_rows())
