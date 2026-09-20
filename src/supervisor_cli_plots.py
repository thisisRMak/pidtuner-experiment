"""matplotlib figure-building + saving for the four cli_supervisor_*.py
REPL scripts. Split from supervisor_cli_output.py so a caller that only
needs filename/report/log helpers doesn't import matplotlib/
streamlit_mimo_panel/streamlit_gui_state just to resolve a --log-file
path.

SISO/PID (save_pid_turn_plots): mirrors cli_pid.py's own --plot figure
exactly -- same plt.subplots(2, 1, figsize=(10, 8), sharex=True) layout,
same setpoint axhline(1.0, ...) (every benchmark-produced row["sim"] is
always a unit-step response -- pid_compare.compare_all_methods always
calls simulate_closed_loop(..., setpoint=1.0, setpoint_kind="step", ...)
-- so this is never a guess), same titles/labels/grid (cli_pid.py:373-
396). The only difference from cli_pid.py's own code: this reads
row["sim"] directly (already computed by Session._wrap_benchmark's
return_sim=True request) instead of re-simulating from row["gains"] --
there's nothing to re-derive.

MIMO/LQG (save_lqg_turn_plots): reuses streamlit_mimo_panel.py's
_build_response_fig/_build_four_curve_fig verbatim -- both are pure
Figure-builders (no st.* calls in either body; confirmed by reading them
in full), fed streamlit_gui_state.ControllerEntry objects built directly
here -- NOT via gs.add_controller, so st.session_state is never touched
-- the same shape streamlit_mimo_panel.absorb_llm_rows() builds for the
GUI's own session list. LQGSession.plot_calls[i]["rows"] are raw
lqg_compare.ComparisonRow instances (.name/.result/.sim, attribute
access -- NOT dict keys, unlike the SISO/PID track's plot_calls rows,
which are plain dicts; this asymmetry is real, confirmed by reading
supervisor_tools_lqg.run_lqg_benchmark and supervisor_session_lqg.py, and
the two track's plot-savers below deliberately do not share one row-
adapter). plot_calls[i]["four_curve"], when not None (only on a turn
that supplied am_diag), is a dict with keys "rows"/"Am"/"t"/"xm_ref" --
no "plant_name" key; _build_four_curve_fig needs a 5-tuple, so the 5th
element (plant name) comes from the call's own "plant" field instead,
mirroring streamlit_mimo_panel.absorb_llm_rows's identical bridge.

Deliberately EXCLUDED: _build_per_channel_step_fig. Confirmed this is
NOT automatic even in the GUI -- it requires manually picking one
already-designed method after the fact ("before requesting the
per-channel step grid"), with no equivalent data sitting in plot_calls to
build it from. Out of scope for this pass.

Judge mode saves one file per candidate per turn, with no cross-
candidate dedup -- unlike the GUI's streamlit_judge_panel.
_drain_judge_plot_calls, which solves a different problem (an unbounded
*persistent* session list growing Nx for N identical candidate answers).
Here, each candidate's own turn-scoped PNG (candidate_label in the
filename) lets a user visually diff N candidates' output for the same
turn side by side.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit_gui_state as gs
import streamlit_mimo_panel as mimo_panel
from supervisor_cli_output import log_print


def save_pid_turn_plots(calls, run_paths, candidate_label=None, log_fh=None) -> list[str]:
    """`calls`: one turn's worth of Session.plot_calls (already drained
    by the caller). One PNG per entry with kind == "siso" and at least
    one row with row.get("stable") and row.get("sim") is not None -- the
    same guard cli_pid.py's own `if r.get("stable", False) and
    r.get("gains")` expresses, adapted to check for the already-computed
    sim instead of gains-to-resimulate-from. A call with nothing stable
    to plot is skipped (no file written). Returns the paths written."""
    paths = []
    for i, call in enumerate(calls, start=1):
        if call.get("kind") != "siso":
            continue
        stable_rows = [r for r in call.get("rows", [])
                       if r.get("stable") and r.get("sim") is not None]
        if not stable_rows:
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for row in stable_rows:
            sim = row["sim"]
            ax1.plot(sim.t, sim.y, label=row["name"])
            ax2.plot(sim.t, sim.u)
        ax1.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Setpoint")
        ax1.set_ylabel("Output y(t)")
        ax1.set_title("Step Response comparison")
        ax1.legend()
        ax1.grid(True)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Control Effort u(t)")
        ax2.grid(True)
        plt.tight_layout()

        path = f"{run_paths.plot_stem(i, call.get('plant'), candidate_label)}.png"
        plt.savefig(path)
        plt.close(fig)
        log_print(log_fh, f"saved plot: {path}")
        paths.append(path)
    return paths


def save_lqg_turn_plots(calls, run_paths, candidate_label=None, log_fh=None) -> list[str]:
    """`calls`: one turn's worth of LQGSession.plot_calls. Per call: a
    "-response.png" built from _build_response_fig, when at least one row
    has a non-None .sim; PLUS a "-fourcurve.png" from
    _build_four_curve_fig, only when call["four_curve"] is not None. A
    call with neither writes nothing. Returns the paths written."""
    paths = []
    for i, call in enumerate(calls, start=1):
        if call.get("kind") != "mimo":
            continue
        plant_id = call.get("plant") or "?"
        stem = run_paths.plot_stem(i, plant_id, candidate_label)

        active = [
            gs.ControllerEntry(kind="mimo", label=row.name, params=row.result,
                                result=row.result, sim=row.sim, plant=plant_id)
            for row in call.get("rows", []) if row.sim is not None
        ]
        if active:
            gs.assign_colors(active)
            fig = mimo_panel._build_response_fig(active)
            path = f"{stem}-response.png"
            fig.savefig(path)
            log_print(log_fh, f"saved plot: {path}")
            paths.append(path)

        four_curve = call.get("four_curve")
        if four_curve is not None:
            cached = (four_curve["rows"], four_curve["Am"], four_curve["t"],
                      four_curve["xm_ref"], plant_id)
            fig2 = mimo_panel._build_four_curve_fig(cached)
            path2 = f"{stem}-fourcurve.png"
            fig2.savefig(path2)
            log_print(log_fh, f"saved plot: {path2}")
            paths.append(path2)
    return paths
