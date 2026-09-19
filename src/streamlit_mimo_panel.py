"""Streamlit MIMO LQR/LQG panel — Build plan Step 4 (docs/gui_plan.md).

Same structure as streamlit_siso_panel.py (plant → method → args → tune
→ simulate → session list → plot), calling the same UI-agnostic backend
(lqg_design_methods.py, lqg_bryson.py, lqg_simulate.py, lqg_compare.py,
lqg_checks.py). Session state goes through streamlit_gui_state.py the
same way, using kind="mimo".

Per docs/lqg_plan.md "CLI vs. GUI", Q/R/N are exposed as scalar/broadcast
knobs (a scale on Qy=I, a diagonal broadcast, Bryson's x_max/u_max), never
raw matrix editors for the *preset* plants — a form full of spinboxes for
a plant with up to 15 states doesn't reduce cognitive load the way the
low-dimensional PID panels did. Custom-entered plants (below) are the
exception: there's no preset to spinbox-tune against, so the plant itself
is the thing the user types in.

Deliberately out of scope for this first pass (flagged, not silently
dropped — same spirit as the SISO panel's heatmap/radar gap early on):
  - output-feedback (Kalman-driven) simulation — only the state-feedback
    regulator response is simulated here, matching --sim state_feedback's
    default in cli_lqg.py.
  - a MIMO heatmap/radar comparison view — pid_compare.py's Ms/Mt-for-MIMO
    generalization is explicitly not done yet (docs/lqg_plan.md), so
    there's no tiered-metric data to draw one from.

Implicit/explicit model-following now has a GUI entry point after all —
see "4-curve comparison" below (2026-08-18 meeting notes item 5) — but only
through that bundled comparison, not as a standalone method in the
"Design one method at a time" list.
"""

from __future__ import annotations

import datetime

import numpy as np
import streamlit as st
from matplotlib.figure import Figure

from lqg_examples import list_examples, load_example, LQGExample
from lqg_design_methods import LQR, OutputWeightedLQR, LQG, add_reference_tracking
from lqg_bryson import BrysonLQR
from lqg_simulate import (
    simulate_state_feedback, simulate_per_channel_step, format_regulator_metrics, auto_t_end,
    auto_plot_window,
)
from lqg_compare import compare_regulator_methods, compare_bryson_output_modelfollowing
from lqg_checks import checks_for_result
from matrix_io import parse_matlab_literal, format_matlab_literal
from plant import StateSpacePlant

import report_html
import streamlit_gui_state as gs

METHODS = [
    "LQR (suggested Q/R)",
    "LQR (custom Q/R diagonal)",
    "Output-weighted LQR",
    "Bryson's rule",
    "LQG (Kalman filter)",
]

def _broadcast(text, n):
    """Parse a comma/space-separated numeric text field into an array of
    length n — one value broadcasts to all n, matching cli_lqg.py's
    --Q-diag/--x-max convention (one value or exactly n)."""
    text = text.strip()
    if not text:
        return None
    parts = [p for p in text.replace(",", " ").split() if p]
    values = [float(p) for p in parts]
    if len(values) == 1:
        return np.full(n, values[0])
    if len(values) != n:
        raise ValueError(f"needs 1 value (broadcast) or {n} values, got {len(values)}")
    return np.array(values, dtype=float)


# ── plant preset / custom entry ─────────────────────────────────────────
def _render_custom_plant_controls():
    st.caption("Enter each matrix in MATLAB literal syntax: rows separated by "
              "';', entries by spaces or commas, e.g. `[0 1; -2 -3]`.")
    # setdefault(), not value= -- see mimo_plant_source's comment above.
    st.session_state.setdefault("mimo_custom_name", "")
    st.text_input("Plant name (optional)", key="mimo_custom_name")
    st.session_state.setdefault("mimo_custom_A", "[0 1; -2 -3]")
    st.text_area("A (nx × nx)", key="mimo_custom_A")
    st.session_state.setdefault("mimo_custom_B", "[0; 1]")
    st.text_area("B (nx × nu)", key="mimo_custom_B")
    st.session_state.setdefault("mimo_custom_C", "[1 0]")
    st.text_area("C (ny × nx)", key="mimo_custom_C")
    st.session_state.setdefault("mimo_custom_D", "[0]")
    st.text_area("D (ny × nu)", key="mimo_custom_D")
    try:
        A = parse_matlab_literal(st.session_state["mimo_custom_A"])
        B = parse_matlab_literal(st.session_state["mimo_custom_B"])
        C = parse_matlab_literal(st.session_state["mimo_custom_C"])
        D = parse_matlab_literal(st.session_state["mimo_custom_D"])
        plant = StateSpacePlant(A=A, B=B, C=C, D=D,
                                name=st.session_state["mimo_custom_name"] or "Custom plant")
    except Exception as exc:
        st.error(f"Could not build plant: {exc}")
        return None
    ex = LQGExample(
        key="custom", name=plant.name, citation="user-entered", source_file="",
        plant=plant, suggested_Q_kind="identity", suggested_R_kind="identity",
        suggested_R_scale=1.0, notes="Custom-entered plant; no textbook suggested "
                                     "Q/R, using Q=R=I.")
    st.success(f"{ex.name}\n\nnx={plant.nx}  nu={plant.nu}  ny={plant.ny}")
    return ex


def current_manual_plant_name() -> str | None:
    """ex.name for the plant currently sitting in this panel's own plant
    widgets, read via their shadow state (gs.peek()) rather than
    requiring the widgets to have rendered this run -- see streamlit_
    siso_panel.current_manual_plant()'s twin docstring for why. Same
    name ControllerEntry.plant already carries for both a manual
    "Compare all methods" entry and an LLM-tagged one (see absorb_llm_
    rows()/_do_compare_all()), so the hint's caller can compare it
    against existing LLM entries directly. Returns None if Manual/MIMO
    hasn't rendered at least once this session (no shadow state yet) or
    its preset key no longer resolves."""
    plant = _current_manual_plant_object()
    return plant.name if plant is not None else None


def _current_manual_plant_object() -> StateSpacePlant | None:
    """The actual StateSpacePlant (real A/B/C/D matrices) currently
    sitting in this panel's own plant widgets, read via shadow state
    (gs.peek()) -- same resolution current_manual_plant_name() uses,
    split out so a caller needing the matrices themselves, not just the
    display name, can get them too -- see build_entries_report_
    sections()'s own "Plant" section, which needs this even for a preset
    (ControllerEntry.plant_A..D are only ever populated for a *custom*
    LLM-sourced plant, never a preset -- see that field's own docstring
    in streamlit_gui_state.py -- so there's nothing reliable to read off
    an entry for this; resolving fresh from the live widgets works
    regardless of preset vs. custom, and regardless of source)."""
    source = gs.peek("mimo_plant_source")
    if source is None:
        return None
    try:
        if source == "Custom (MATLAB matrix entry)":
            A = parse_matlab_literal(gs.peek("mimo_custom_A", ""))
            B = parse_matlab_literal(gs.peek("mimo_custom_B", ""))
            C = parse_matlab_literal(gs.peek("mimo_custom_C", ""))
            D = parse_matlab_literal(gs.peek("mimo_custom_D", ""))
            name = gs.peek("mimo_custom_name") or "Custom plant"
            return StateSpacePlant(A=A, B=B, C=C, D=D, name=name)
        return load_example(gs.peek("mimo_preset")).plant
    except Exception:
        return None


def _render_llm_plant_carryover():
    """"LLM Supervisor last analyzed: <plant> — [Load this plant]", the
    MIMO twin of streamlit_siso_panel._render_llm_plant_carryover() --
    see its docstring for the general shape and the st.rerun() note.
    Compares against the reloadable plant_preset/plant_A..plant_D
    identity (gs.ControllerEntry's own note on why), not the display
    name alone: two different custom plants can share a name, and
    plant_preset=="custom" always would.

    Must run before _render_plant_controls() instantiates the plant
    widgets, same reason as the SISO version."""
    llm_entries = [e for e in gs.get_by_kind("mimo") if e.source == "llm" and e.plant_preset]
    if not llm_entries:
        return
    last = llm_entries[-1]
    if last.plant_preset != "custom":
        if (gs.peek("mimo_plant_source") == "Preset"
                and gs.peek("mimo_preset") == last.plant_preset):
            return
    elif (gs.peek("mimo_plant_source") == "Custom (MATLAB matrix entry)"
            and gs.peek("mimo_custom_A") == last.plant_A
            and gs.peek("mimo_custom_B") == last.plant_B
            and gs.peek("mimo_custom_C") == last.plant_C
            and gs.peek("mimo_custom_D") == last.plant_D):
        return
    c1, c2 = st.columns([5, 1])
    c1.caption(f"🤖 LLM Supervisor last analyzed: {last.plant}")
    if c2.button("Load this plant", key="mimo_load_llm_plant"):
        if last.plant_preset != "custom":
            st.session_state["mimo_plant_source"] = "Preset"
            st.session_state["mimo_preset"] = last.plant_preset
        else:
            st.session_state["mimo_plant_source"] = "Custom (MATLAB matrix entry)"
            st.session_state["mimo_custom_name"] = last.plant
            st.session_state["mimo_custom_A"] = last.plant_A
            st.session_state["mimo_custom_B"] = last.plant_B
            st.session_state["mimo_custom_C"] = last.plant_C
            st.session_state["mimo_custom_D"] = last.plant_D
        st.rerun()


def _render_plant_controls():
    st.subheader("Plant")
    # setdefault(), not default= -- default= would trip Streamlit's "widget
    # created with a default value but also had its value set via the
    # Session State API" warning, since preserve_widget_state() (and the
    # "Load this plant" button above) already reseed this key directly
    # before this widget renders; setdefault() only seeds the very first
    # time this key has never existed.
    st.session_state.setdefault("mimo_plant_source", "Preset")
    st.segmented_control("Source", ["Preset", "Custom (MATLAB matrix entry)"],
                         key="mimo_plant_source", required=True)
    if st.session_state["mimo_plant_source"] == "Custom (MATLAB matrix entry)":
        return _render_custom_plant_controls()

    presets = list_examples()
    if not presets:
        st.error("No LQG example plants found (lqg_examples_json/ missing or empty).")
        return None
    st.selectbox("Preset", presets, key="mimo_preset")
    try:
        ex = load_example(st.session_state["mimo_preset"])
    except Exception as exc:
        st.error(str(exc))
        return None
    st.success(f"{ex.name}\n\nnx={ex.plant.nx}  nu={ex.plant.nu}  ny={ex.plant.ny}")
    st.caption(f"{ex.citation}" + (f" — {ex.notes}" if ex.notes else ""))
    return ex


# ── method-specific args ────────────────────────────────────────────────
# Every method's arg-widget keys -- only one method's args render at a time
# (_render_method_args dispatches on `method`), so without this a custom
# Q/R diagonal (say) typed in gets wiped by Streamlit the moment a
# different method is selected, same class of bug _PROTECTED_KEYS guards
# against for a Track/Mode switch. See streamlit_gui_state.
# preserve_widget_state's module note.
_METHOD_ARG_KEYS = [
    "mimo_Q_diag", "mimo_R_diag",
    "mimo_Qy_scale", "mimo_ow_R_scale",
    "mimo_x_max", "mimo_u_max",
    "mimo_Qw_scale", "mimo_Rv_scale",
]


def _render_method_args(method, ex):
    gs.preserve_widget_state(_METHOD_ARG_KEYS)
    if method == "LQR (custom Q/R diagonal)":
        # setdefault(), not value= -- see mimo_plant_source's comment above.
        st.session_state.setdefault("mimo_Q_diag", "1.0")
        st.text_input(f"Q diagonal ({ex.plant.nx} value(s), or 1 to broadcast)",
                      key="mimo_Q_diag")
        st.session_state.setdefault("mimo_R_diag", "1.0")
        st.text_input(f"R diagonal ({ex.plant.nu} value(s), or 1 to broadcast)",
                      key="mimo_R_diag")
    elif method == "Output-weighted LQR":
        st.session_state.setdefault("mimo_Qy_scale", 1.0)
        st.number_input("Qy scale (Qy = scale·I)", key="mimo_Qy_scale")
        st.session_state.setdefault("mimo_ow_R_scale", 1.0)
        st.number_input("R scale (R = scale·I)", key="mimo_ow_R_scale")
    elif method == "Bryson's rule":
        st.session_state.setdefault("mimo_x_max", "1.0")
        st.text_input(f"x_max ({ex.plant.nx} value(s), or 1 to broadcast)",
                      key="mimo_x_max")
        st.session_state.setdefault("mimo_u_max", "1.0")
        st.text_input(f"u_max ({ex.plant.nu} value(s), or 1 to broadcast)",
                      key="mimo_u_max")
    elif method == "LQG (Kalman filter)":
        st.session_state.setdefault("mimo_Qw_scale", 0.01)
        st.number_input("Qw scale (process-noise covariance = scale·I)",
                        key="mimo_Qw_scale")
        st.session_state.setdefault("mimo_Rv_scale", 0.1)
        st.number_input("Rv scale (measurement-noise covariance = scale·I)",
                        key="mimo_Rv_scale")
    gs.snapshot_widget_state(_METHOD_ARG_KEYS)


def _design_dispatch(method, ex):
    plant = ex.plant
    if method == "LQR (suggested Q/R)":
        Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
        return LQR(plant, Q=Q, R=R).design()

    if method == "LQR (custom Q/R diagonal)":
        Q_diag = _broadcast(st.session_state["mimo_Q_diag"], plant.nx)
        R_diag = _broadcast(st.session_state["mimo_R_diag"], plant.nu)
        if Q_diag is None or R_diag is None:
            raise ValueError("Both Q diagonal and R diagonal are required.")
        return LQR(plant, Q=np.diag(Q_diag), R=np.diag(R_diag)).design()

    if method == "Output-weighted LQR":
        Qy = st.session_state["mimo_Qy_scale"] * np.eye(plant.ny)
        R = st.session_state["mimo_ow_R_scale"] * np.eye(plant.nu)
        return OutputWeightedLQR(plant, Qy=Qy, R=R).design()

    if method == "Bryson's rule":
        x_max = _broadcast(st.session_state["mimo_x_max"], plant.nx)
        u_max = _broadcast(st.session_state["mimo_u_max"], plant.nu)
        if x_max is None or u_max is None:
            raise ValueError("Both x_max and u_max are required.")
        return BrysonLQR(plant, x_max=x_max, u_max=u_max).design()

    if method == "LQG (Kalman filter)":
        Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
        Qw = st.session_state["mimo_Qw_scale"] * np.eye(plant.nx)
        Rv = st.session_state["mimo_Rv_scale"] * np.eye(plant.ny)
        return LQG(plant, Q=Q, R=R, Qw=Qw, Rv=Rv).design()

    raise RuntimeError(f"unknown method {method}")


# ── simulation ───────────────────────────────────────────────────────────
def _render_sim_settings():
    st.subheader("Simulation")
    # setdefault(), not value= -- see mimo_plant_source's comment above.
    st.session_state.setdefault("mimo_t_end", "")
    st.text_input("t_end (blank = auto)", key="mimo_t_end")
    st.session_state.setdefault("mimo_dt", 0.01)
    st.number_input("dt", key="mimo_dt", format="%.4f")
    st.session_state.setdefault("mimo_ref_tracking", False)
    st.checkbox("Reference tracking", key="mimo_ref_tracking")
    st.session_state.setdefault("mimo_reference", "")
    st.text_input("reference (blank = all-ones, ny values or 1 to broadcast)",
                  key="mimo_reference")
    st.caption("Reference tracking requires a square plant (nu == ny) and adds "
               "the N̄ feedforward gain, then simulates tracking that constant "
               "reference instead of the default unit-perturbation regulator "
               "response.")


def _run_sim(result, ex):
    t_end_str = st.session_state["mimo_t_end"].strip()
    t_end = float(t_end_str) if t_end_str else auto_t_end(result.closed_loop_poles)
    dt = st.session_state["mimo_dt"]
    t = np.arange(0.0, t_end + dt, dt)

    if st.session_state["mimo_ref_tracking"]:
        if ex.plant.nu != ex.plant.ny:
            raise ValueError(
                f"Reference tracking requires a square plant (nu == ny); "
                f"{ex.name} has nu={ex.plant.nu}, ny={ex.plant.ny}")
        ref_str = st.session_state["mimo_reference"].strip()
        reference = _broadcast(ref_str, ex.plant.ny) if ref_str else np.ones(ex.plant.ny)
        result = add_reference_tracking(result)
        r_arr = np.tile(reference, (len(t), 1))
        sim = simulate_state_feedback(result, t, r=r_arr)
    else:
        sim = simulate_state_feedback(result, t)
    return result, sim


# ── actions ──────────────────────────────────────────────────────────────
def _next_label(method, ex):
    base = method
    mimo_entries = gs.get_by_kind("mimo")
    count = sum(1 for e in mimo_entries if e.label.split(" #")[0] == f"{base} ({ex.key})")
    label = f"{base} ({ex.key})"
    return label if count == 0 else f"{label} #{count + 1}"


def _do_design(method, ex):
    try:
        result = _design_dispatch(method, ex)
    except Exception as exc:
        st.error(f"Design failed: {exc}")
        return
    try:
        result, sim = _run_sim(result, ex)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return
    label = _next_label(method, ex)
    entry = gs.ControllerEntry(kind="mimo", label=label, params=result,
                               result=result, sim=sim, plant=ex.name)
    entry.checks = checks_for_result(result)
    gs.add_controller(entry)
    st.session_state["mimo_last_result"] = (result, sim, entry.checks)
    st.success(f"Designed: {label}")


def _do_per_channel_step():
    """Per-reference-channel step response, matching MATLAB's default
    step() grid behavior: steps each reference channel individually
    (others held at 0) and shows the response across all outputs.
    Operates on the last design from "Design one method at a time"
    (st.session_state["mimo_last_result"]) -- requires that design to have
    been run with Reference tracking checked (result.Nbar populated).
    Additive alongside the existing "Reference tracking" combined-step
    simulation (_run_sim), which stays exactly as it was -- this is a
    separate grid view, not a replacement. Reads the optional
    mimo_pcs_t_max/mimo_pcs_y_max text fields and stashes them alongside
    the sim results for _render_per_channel_step_plot to apply -- blank
    t_max auto-crops via auto_plot_window() (matching cli_lqg.py's
    --plot-t-max default), blank y_max stays fully auto-scaled per cell
    (matching cli_lqg.py's --plot-y-max, which has no auto-default)."""
    last = st.session_state.get("mimo_last_result")
    if last is None:
        st.error("Design a method first (with Reference tracking checked) "
                 "before requesting the per-channel step grid.")
        return
    result, sim, checks = last
    if result.Nbar is None:
        st.error("Per-channel step response requires Reference tracking to "
                 "be checked when designing.")
        return
    t_end_str = st.session_state["mimo_t_end"].strip()
    t_end = float(t_end_str) if t_end_str else auto_t_end(result.closed_loop_poles)
    dt = st.session_state["mimo_dt"]
    t = np.arange(0.0, t_end + dt, dt)
    try:
        per_channel_sims = simulate_per_channel_step(result, t)
    except Exception as exc:
        st.error(f"Per-channel step response failed: {exc}")
        return
    t_max_str = st.session_state["mimo_pcs_t_max"].strip()
    y_max_str = st.session_state["mimo_pcs_y_max"].strip()
    try:
        t_max = (float(t_max_str) if t_max_str
                 else auto_plot_window(t, *[s.y for s in per_channel_sims]))
        y_max = float(y_max_str) if y_max_str else None
    except ValueError:
        st.error("t_max/y_max must be numbers (or blank for auto-scaling).")
        return
    st.session_state["mimo_per_channel_step"] = (per_channel_sims, result.method, t_max, y_max)
    st.success("Per-channel step response ready — see grid plot below.")


def _build_per_channel_step_fig(cached):
    """The pure Figure-building half of _render_per_channel_step_plot() --
    split out so a caller that isn't rendering to the screen (report_html-
    based report generation) can embed the exact same figure without going
    through st.pyplot(). `cached` is st.session_state["mimo_per_channel_
    step"]'s own tuple shape -- this function has no opinion on session
    state, same split as _build_response_fig() above."""
    per_channel_sims, method_name, t_max, y_max = cached
    ny = len(per_channel_sims)
    fig = Figure(figsize=(3.2 * ny, 2.6 * ny), dpi=100)
    axes = fig.subplots(ny, ny, sharex=True)
    axes = np.atleast_2d(axes)
    for j, sim in enumerate(per_channel_sims):
        idx = _crop_idx(sim.t, t_max) if t_max is not None else len(sim.t)
        t_plot, y_plot = sim.t[:idx], sim.y[:idx]
        for i in range(ny):
            ax = axes[i, j]
            ax.axhline(1.0 if i == j else 0.0, color="k", linestyle="--", linewidth=1)
            ax.plot(t_plot, y_plot[:, i])
            if y_max is not None:
                ax.set_ylim(0, y_max)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"from r{j}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"y{i}(t)")
    fig.suptitle(f"{method_name} — per-channel step response")
    fig.tight_layout()
    return fig


def _render_per_channel_step_plot():
    cached = st.session_state.get("mimo_per_channel_step")
    if cached is None:
        return
    st.subheader("Per-channel step response (MATLAB step() grid)")
    fig = _build_per_channel_step_fig(cached)
    st.pyplot(fig)


def _do_four_curve(ex):
    """2026-08-18 meeting notes item 5: Bryson / Output-weighted /
    Implicit / Explicit model-following, one plot, per-output-channel y(t)
    against the (auto-derived) target model's own response — a different
    shape than the session-overlay ||x||/||u|| plot below, so it gets its
    own comparison button and its own plot rather than feeding into the
    session list."""
    try:
        rows, Am_used, (t, xm_ref) = compare_bryson_output_modelfollowing(
            ex, t_end=None, dt=st.session_state["mimo_dt"])
    except Exception as exc:
        st.error(f"4-curve comparison failed: {exc}")
        return
    st.session_state["mimo_four_curve"] = (rows, Am_used, t, xm_ref, ex.name)
    st.success("4-curve comparison ready (Bryson / Output-weighted / "
              "Implicit / Explicit) — see plot below. Am auto-derived from "
              "the plant; confirm with Prof Emami-Naeini whether this "
              "default is the right behavior.")


def _build_four_curve_fig(cached):
    """The pure Figure-building half of _render_four_curve_plot() -- split
    out so a caller that isn't rendering to the screen (report_html-based
    report generation) can embed the exact same figure without going
    through st.pyplot(). `cached` is st.session_state["mimo_four_curve"]'s
    own tuple shape."""
    rows, Am_used, t, xm_ref, plant_name = cached
    ny = xm_ref.shape[1]
    t_max = max([auto_plot_window(t, xm_ref)] +
               [auto_plot_window(row.sim.t, row.sim.y) for row in rows])
    idx_target = _crop_idx(t, t_max)
    fig = Figure(figsize=(9, 4 * ny), dpi=100)
    axes = fig.subplots(ny, 1, sharex=True)
    axes = np.atleast_1d(axes)
    for j in range(ny):
        ax = axes[j]
        ax.plot(t[:idx_target], xm_ref[:idx_target, j], "k--", linewidth=1.5, label="target model xm")
        for row, color in zip(rows, gs.PALETTE):
            idx_row = _crop_idx(row.sim.t, t_max)
            ax.plot(row.sim.t[:idx_row], row.sim.y[:idx_row, j], color=color, label=row.name, linewidth=1.3)
        ax.set_ylabel(f"y{j}(t)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].set_title(plant_name)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def _render_four_curve_plot():
    cached = st.session_state.get("mimo_four_curve")
    if cached is None:
        return
    _, Am_used, _, _, _ = cached
    st.subheader("4-curve comparison: Bryson / Output-weighted / Implicit / Explicit")
    st.caption("Am (target model, auto-derived) =\n" +
              np.array2string(Am_used, precision=4, separator=", "))
    fig = _build_four_curve_fig(cached)
    st.pyplot(fig)


def _do_compare_all(ex):
    try:
        rows = compare_regulator_methods(ex, t_end=None, dt=st.session_state["mimo_dt"])
    except Exception as exc:
        st.error(f"Comparison failed: {exc}")
        return
    # Only "you"-sourced entries -- see streamlit_siso_panel.py's own
    # _do_compare_all for why "Clear all" (below) stays the one place
    # that clears source="llm" entries too, unscoped and explicit.
    gs.clear_by_kind_and_source("mimo", "you")
    for row in rows:
        entry = gs.ControllerEntry(
            kind="mimo", label=f"{row.name} ({ex.key})",
            params=row.result, result=row.result, sim=row.sim, plant=ex.name)
        entry.checks = row.checks
        gs.add_controller(entry)
    st.success(f"Compared {len(rows)} regulator-family methods. Untick any below to declutter.")


def absorb_llm_rows(plant_id, rows, plant_preset="", custom_plant_literals=None,
                    four_curve=None):
    """Turn raw run_lqg_benchmark(return_sim=True) rows (ComparisonRow
    objects, .sim intact) into session entries — the same gs.
    ControllerEntry shape _do_compare_all builds for a manual "Compare
    all methods" click, so an LLM-triggered run lands in the same
    session list/plots as a manual one, tagged source="llm". Called by
    streamlit_llm_panel.py's _drain_plot_calls after each chat turn —
    see supervisor_session_lqg.LQGSession.plot_calls.

    plant_preset/custom_plant_literals carry the reloadable plant
    identity through to each entry (see gs.ControllerEntry's own note
    on why plant_id alone -- a display name -- isn't enough), for
    _render_llm_plant_carryover() below.

    four_curve (present only when the LLM's call supplied am_diag -- see
    run_lqg_benchmark's own note) seeds st.session_state["mimo_four_
    curve"] directly, the same shape _do_four_curve's manual button
    leaves it in, so _render_four_curve_plot() picks it up without a
    second, separate simulation -- its 4 rows are already among `rows`
    above (this doesn't skip adding them to the regular session list
    too, just reuses them for the extra view)."""
    literals = custom_plant_literals or {}
    n_ok = 0
    for row in rows:
        if row.sim is None:
            continue
        entry = gs.add_llm_entry(
            kind="mimo", label=row.name, params=row.result,
            result=row.result, sim=row.sim, plant=plant_id,
            plant_preset=plant_preset,
            plant_A=literals.get("A", ""), plant_B=literals.get("B", ""),
            plant_C=literals.get("C", ""), plant_D=literals.get("D", ""))
        entry.checks = row.checks
        n_ok += 1
    if four_curve is not None:
        st.session_state["mimo_four_curve"] = (
            four_curve["rows"], four_curve["Am"], four_curve["t"],
            four_curve["xm_ref"], plant_id)
    return n_ok


# ── session list ─────────────────────────────────────────────────────────
def _render_session_list():
    st.subheader("Designed controllers (session overlay)")
    mimo_entries = gs.get_by_kind("mimo")
    if not mimo_entries:
        st.caption("Design a method (or Compare all methods) to populate this list.")
        return

    cols = st.columns(5)
    if cols[0].button("Select all", key="mimo_select_all"):
        gs.set_all_enabled_by_kind("mimo", True)
        for e in mimo_entries:
            st.session_state[f"mimo_en_{e.id}"] = True
    if cols[1].button("Deselect all", key="mimo_deselect_all"):
        gs.set_all_enabled_by_kind("mimo", False)
        for e in mimo_entries:
            st.session_state[f"mimo_en_{e.id}"] = False
    if cols[2].button("Clear all", key="mimo_clear_all"):
        gs.clear_by_kind("mimo")
    if cols[3].button("Remove unchecked", key="mimo_remove_unchecked"):
        gs.remove_unchecked_by_kind("mimo")
    if cols[4].button("Clear LLM plots", key="mimo_clear_llm_plots"):
        # See streamlit_siso_panel.py's identical button/comment -- only
        # source="llm" entries, lives here (not in a chat Mode's own
        # controls) so it works the same regardless of which Mode is active.
        gs.clear_by_kind_and_source("mimo", "llm")
    mimo_entries = gs.get_by_kind("mimo")

    gs.assign_colors(mimo_entries)
    for i, entry in enumerate(mimo_entries):
        c1, c2, c3 = st.columns([1, 3, 4])
        checkbox_key = f"mimo_en_{entry.id}"
        st.session_state.setdefault(checkbox_key, entry.enabled)
        enabled = c1.checkbox("enabled", key=checkbox_key, label_visibility="collapsed")
        if enabled != entry.enabled:
            gs.set_enabled(entry.id, enabled)
        tag = "  :violet-badge[🤖 LLM]" if entry.source == "llm" else ""
        c2.markdown(f"{_circle_emoji(entry.color)} {entry.label}{tag}")
        stable = "stable" if entry.result.is_stable() else "UNSTABLE"
        checks_ok = all(c.passed for cs in (entry.checks or {}).values() for c in cs)
        plant_tag = f"  ·  {entry.plant}" if entry.plant else ""
        c3.caption(f"{stable}, checks {'PASS' if checks_ok else 'FAIL'}{plant_tag}")


def _palette_name(hex_color):
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
    """Colored-circle swatch for a session-list row -- see
    streamlit_siso_panel.py's own copy of this function for why this is
    a literal Unicode character, not a markdown :shortcode:."""
    return _CIRCLE_EMOJI[_palette_name(hex_color)]


def _crop_idx(t, t_max):
    """Index into a (possibly much longer) simulated t array that realizes
    an auto_plot_window()-style crop -- shared helper for every plot below
    that needs to slice per-entry arrays (which can have different lengths,
    since each design's own auto_t_end() can differ) down to one common,
    already-computed t_max."""
    return max(int(np.searchsorted(t, t_max, side="right")), 2)


# ── response plot ────────────────────────────────────────────────────────
def _build_response_fig(active):
    """The Figure-building half of _render_response_plot(), split out so
    a caller that isn't rendering to the screen (report_html-based report
    generation) can build the exact same figure without going through
    st.pyplot() -- mirrors streamlit_siso_panel.py's own split. `active`
    is already the enabled/sim-present filter the caller wants plotted."""
    fig = Figure(figsize=(9, 6), dpi=100)
    if not active:
        ax = fig.add_subplot(111)
        ax.set_title("No designed controllers shown — design a method or "
                     "tick one in the session list.")
        return fig

    if any(e.sim.tracking_metrics is not None for e in active):
        # Auto-cropped to whichever active design's own trajectory takes
        # longest to visually settle (auto_plot_window(), same mechanism as
        # cli_lqg.py) -- one shared window for the whole overlay rather than
        # auto_t_end()'s full (possibly very long) simulated duration.
        t_max = max(auto_plot_window(e.sim.t, e.sim.y) for e in active)
        ny = active[0].sim.y.shape[1]
        axes = fig.subplots(ny, 1, sharex=True)
        axes = np.atleast_1d(axes)
        for j in range(ny):
            ax = axes[j]
            for e in active:
                idx = _crop_idx(e.sim.t, t_max)
                ax.plot(e.sim.t[:idx], e.sim.y[:idx, j], color=e.color, label=e.label, linewidth=1.3)
            ax.set_ylabel(f"y{j}(t)")
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=7, loc="lower right")
        axes[-1].set_xlabel("time (s)")
    else:
        t_max = max(auto_plot_window(e.sim.t, e.sim.x, e.sim.u) for e in active)
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        for e in active:
            idx = _crop_idx(e.sim.t, t_max)
            x_norm = np.linalg.norm(e.sim.x[:idx], axis=1)
            u_norm = np.linalg.norm(e.sim.u[:idx], axis=1)
            label = e.label + ("  [UNSTABLE]" if e.sim.metrics.get("unstable") else "")
            ax1.plot(e.sim.t[:idx], x_norm, color=e.color, label=label, linewidth=1.3)
            ax2.plot(e.sim.t[:idx], u_norm, color=e.color, label=e.label, linewidth=1.3)
        ax1.set_ylabel("||x(t)||")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7, loc="upper right")
        ax2.set_ylabel("||u(t)||")
        ax2.set_xlabel("time (s)")
        ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _render_response_plot():
    active = [e for e in gs.get_by_kind("mimo") if e.enabled and e.sim is not None]
    fig = _build_response_fig(active)
    st.pyplot(fig)


def build_entries_report_sections():
    """(subtitle, [section_html, ...]) for every currently-enabled
    session-list entry in this Track (any source -- mirrors streamlit_
    siso_panel.py's own report scope, see its build_entries_report_
    sections() docstring): a checks/metrics table (reusing format_
    regulator_metrics(), the exact string already shown per-entry on
    screen -- not a new metric schema), the response plot, and -- when
    the user has actually run them -- the 4-curve comparison and the
    per-channel step response. The latter two are NOT session-list
    entries (see the comment below), so they're included independently,
    each conditional on its own cache being populated, not on `active`.
    Also a "Plant" section listing the currently-selected plant's real
    A/B/C/D matrices in MATLAB literal form -- including for a preset,
    which carries no matrices of its own to display otherwise (see where
    it's built for why this needs a fresh resolve, not an entry field).

    Public and parameter-free for the same reason as streamlit_siso_
    panel.py's own version -- streamlit_llm_panel.py/streamlit_judge_
    panel.py call this directly to embed the same section in their own
    reports."""
    # entry.color isn't a stored field -- render_plots() assigns it by
    # list position every run (gs.assign_colors()), but render_controls()
    # (where this is called from) runs *before* render_plots() in
    # streamlit_unified_panel.render()'s own column order -- see
    # streamlit_siso_panel.py's identical comment on its own report
    # builder for why this call is needed here too.
    gs.assign_colors(gs.get_by_kind("mimo"))
    active = [e for e in gs.get_by_kind("mimo") if e.enabled]
    plants = sorted({e.plant for e in active if e.plant})
    subtitle = "MIMO / LQR-LQG — " + ("; ".join(plants) if plants else "no plant selected")

    # The currently-selected plant's real matrices, preset or custom alike
    # -- see _current_manual_plant_object()'s own docstring for why this
    # has to resolve fresh from the live widgets rather than reading
    # anything off an entry (a preset's matrices are never stored on
    # ControllerEntry, by any creation path). Matches the common case (one
    # plant per report), same caveat as streamlit_siso_panel.py's twin
    # section: won't match every entry if more than one plant was compared
    # this session. None if Manual/MIMO's widgets have never rendered this
    # session, or the plant currently sitting in them doesn't resolve.
    sections = []
    current_plant = _current_manual_plant_object()
    if current_plant is not None:
        matrix_lines = "\n".join(
            f"{name} =\n{format_matlab_literal(getattr(current_plant, name))}"
            for name in ("A", "B", "C", "D")
        )
        sections.append(
            f"<h2>Plant</h2><p>{report_html.e(current_plant.name)}</p>"
            f"<pre><code>{report_html.e(matrix_lines)}</code></pre>"
        )

    headers = ["Method", "Checks", "Metrics"]
    rows = []
    for e in active:
        all_checks = [c for cs in (e.checks or {}).values() for c in cs]
        checks_str = "PASS" if all_checks and all(c.passed for c in all_checks) else "FAIL" if all_checks else "—"
        metrics_str = format_regulator_metrics(e.sim.metrics) if e.sim is not None else "—"
        rows.append([e.label, checks_str, metrics_str])
    sections.append(f"<h2>Compared methods</h2>{report_html.render_table(headers, rows)}")

    sim_active = [e for e in active if e.sim is not None]
    if sim_active:
        fig = _build_response_fig(sim_active)
        img = report_html.fig_to_data_uri(fig)
        sections.append(f'<h2>Response</h2><img src="{img}" style="max-width:100%">')

    # The 4-curve and per-channel-step comparisons are NOT session-list
    # entries -- "⊞ 4-curve comparison"/"⊞ Per-channel step response" each
    # store their own single cached result in st.session_state (see
    # _do_four_curve()/_do_per_channel_step()) rather than feeding
    # gs.ControllerEntry, so the `active` loop above never sees them. Both
    # render on screen whenever their cache is populated (see
    # _render_four_curve_plot()/_render_per_channel_step_plot()) --
    # included here on the same condition, so the report doesn't miss
    # comparisons the user has actually run.
    four_curve_cached = st.session_state.get("mimo_four_curve")
    if four_curve_cached is not None:
        fig = _build_four_curve_fig(four_curve_cached)
        img = report_html.fig_to_data_uri(fig)
        sections.append(
            '<h2>4-curve comparison: Bryson / Output-weighted / Implicit / Explicit</h2>'
            f'<img src="{img}" style="max-width:100%">')

    per_channel_cached = st.session_state.get("mimo_per_channel_step")
    if per_channel_cached is not None:
        fig = _build_per_channel_step_fig(per_channel_cached)
        img = report_html.fig_to_data_uri(fig)
        sections.append(f'<h2>Per-channel step response</h2><img src="{img}" style="max-width:100%">')

    return subtitle, sections


def _build_report_html():
    """Manual mode's own MIMO/LQG report -- just build_entries_report_
    sections()'s sections, wrapped as a standalone document."""
    subtitle, sections = build_entries_report_sections()
    return report_html.build_report("PIDTuner MIMO/LQG Report", subtitle, sections)


def _render_download_report_button():
    if not any(e.enabled for e in gs.get_by_kind("mimo")):
        return
    st.download_button(
        "Download report", data=_build_report_html(),
        file_name=f"pidtuner-mimo-report-{datetime.date.today().isoformat()}.html",
        mime="text/html", key="mimo_download_report",
    )


def _render_last_result():
    last = st.session_state.get("mimo_last_result")
    if last is None:
        return
    result, sim, checks = last
    parts = [f"**{result.method}**", "", result.notes]
    K_str = np.array2string(result.gains.K, precision=4, separator=', ')
    parts.append(f"\nK =\n```\n{K_str}\n```")
    poles_str = ", ".join(f"{p.real:.4g}{'+' if p.imag >= 0 else ''}{p.imag:.4g}j"
                          for p in result.closed_loop_poles)
    parts.append(f"\nClosed-loop poles: {poles_str}")
    parts.append(f"\nStable: {result.is_stable()}")
    all_checks = checks.get("pre", []) + checks.get("post", [])
    parts.append(f"\nChecks: {'PASS' if all(c.passed for c in all_checks) else 'FAIL'}")
    parts.append("\n\nMetrics: " + format_regulator_metrics(sim.metrics))
    st.info("\n".join(parts))


# ── entry point ──────────────────────────────────────────────────────────
# Every plant/sim-setting field this panel owns -- see streamlit_siso_
# panel.py's own _PROTECTED_KEYS comment for why these need listing (both
# plant sources' fields included even though only one renders at a time;
# method-specific args, e.g. mimo_Q_diag, not included here -- those get
# the same treatment separately, via _METHOD_ARG_KEYS inside
# _render_method_args itself, same as the SISO side).
_PROTECTED_KEYS = [
    "mimo_plant_source", "mimo_preset",
    "mimo_custom_name", "mimo_custom_A", "mimo_custom_B", "mimo_custom_C", "mimo_custom_D",
    "mimo_method",
    "mimo_t_end", "mimo_dt", "mimo_ref_tracking", "mimo_reference",
    "mimo_pcs_t_max", "mimo_pcs_y_max",
]


def render_controls():
    """The left-hand controls half — called by streamlit_unified_panel.py
    when Track=MIMO/LQR-LQG, Mode=Manual. Split from what used to be one
    render() (see git history), mirroring streamlit_siso_panel.py's own
    split — see its render_controls() docstring for why."""
    gs.preserve_widget_state(_PROTECTED_KEYS)
    _render_llm_plant_carryover()
    ex = _render_plant_controls()

    st.subheader("Compare all methods")
    if st.button("⊞  Compare all methods", key="mimo_compare_all",
                disabled=ex is None):
        _do_compare_all(ex)

    st.subheader("4-curve comparison")
    st.caption("Bryson's rule / Output-weighted LQR / Implicit / Explicit "
              "model-following, step response overlaid per output channel "
              "(2026-08-18 meeting notes item 5).")
    if st.button("⊞  4-curve comparison", key="mimo_four_curve_btn",
                disabled=ex is None):
        _do_four_curve(ex)

    st.subheader("Design one method at a time")
    method = st.selectbox("Method", METHODS, key="mimo_method")
    if ex is not None:
        _render_method_args(method, ex)
    if st.button("Design & simulate", key="mimo_design", disabled=ex is None):
        _do_design(method, ex)

    _render_sim_settings()

    st.subheader("Per-channel step response")
    st.caption("Steps each reference channel individually (others held at 0), "
              "reporting the response across all outputs — matching MATLAB's "
              "default step() grid, as opposed to the combined simultaneous "
              "step above. Requires the last design above to have been run "
              "with Reference tracking checked.")
    pcs_cols = st.columns(2)
    # setdefault(), not value= -- see mimo_plant_source's comment above.
    st.session_state.setdefault("mimo_pcs_t_max", "")
    pcs_cols[0].text_input("Grid t_max (blank = auto-cropped)", key="mimo_pcs_t_max")
    st.session_state.setdefault("mimo_pcs_y_max", "")
    pcs_cols[1].text_input("Grid y_max (blank = auto per cell)", key="mimo_pcs_y_max")
    if st.button("⊞  Per-channel step response", key="mimo_per_channel_step_btn"):
        _do_per_channel_step()

    gs.snapshot_widget_state(_PROTECTED_KEYS)
    _render_last_result()
    _render_download_report_button()


def render_plots():
    """The right-hand plots half — called by streamlit_unified_panel.py
    whenever Track=MIMO/LQR-LQG, regardless of Mode — see
    streamlit_siso_panel.py's render_plots() docstring for why entries
    from both Manual and LLM Supervisor mode show up here the same way,
    and why the session list itself renders here now too rather than in
    render_controls()."""
    gs.assign_colors(gs.get_by_kind("mimo"))
    _render_session_list()
    _render_response_plot()
    _render_four_curve_plot()
    _render_per_channel_step_plot()
