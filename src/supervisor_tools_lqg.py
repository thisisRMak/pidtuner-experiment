"""The LQR/LQG supervisor's one benchmark tool: run every applicable
Phase-1 design method (LQR, output-weighted LQR, Bryson's rule, full LQG)
against one plant -- a professor-provided preset, or a user-supplied
custom A/B/C/D (wrapped into an LQGExample the same way
streamlit_mimo_panel.py's custom-entry path does, with Q=R=I standing in
for the missing textbook suggestion) -- and return metrics for each, plus
the two model-following techniques (implicit/explicit) when the caller
supplies a target model via am_diag.

One tool covering all six methods, deliberately -- not six separate tools.
Same shape as supervisor_tools_whitebox.run_whitebox_benchmark, which
bundles all 9 PID methods behind a single call via
compare.compare_all_methods; LQR and LQG are the same family here (LQG is
literally LQR plus a Kalman filter, sharing the same Q/R machinery), so
splitting them into separate supervisor tools would make "should I use LQR
or LQG" -- exactly the kind of question this supervisor should answer --
harder to ask in one turn.

Model-following (lqg_implicit.py/lqg_explicit.py) is opt-in via am_diag:
both need a target model (Am, Q1) as a per-design choice with no
"suggested" value per preset plant the way LQR/output-weighted/Bryson have
(see docs/lqg_testing.md "Model-following classes") -- there's no way to
auto-run them the way pole_cancellation auto-derives its poles in
pid_compare.py, so unlike the other four they're only included when the
caller (the LLM, on the user's behalf) actually supplies am_diag; never
guessed.

Uses lqg_compare.py (compare_regulator_methods / compare_model_following)
for the actual design/simulation work -- the same shared-core role
pid_compare.py plays for cli_pid.py and supervisor_tools_whitebox_pid.py -- and keeps
its own rounded/JSON-safe row serialization on top, independent of
cli_lqg.py's, following the precedent supervisor_tools_whitebox_pid.py already
set ("Independent of cli.serialize_row_json by design").

Imports lqg_examples.py/lqg_compare.py/lqg_explicit.py/plant.py -- this is
the only supervisor module allowed to import the lqg_* design/compare
modules or plant.py directly (mirrors supervisor_tools_whitebox_pid.py's
plant.py-only-here contract; plant.StateSpacePlant is what custom_plant
gets wrapped into, below).
There's no competing black-box LQG tool to isolate this from (see
supervisor_common_lqg.py's module docstring), so unlike
supervisor_tools_blackbox_pid.py there's nothing for an isolation test to
check here.
"""

from __future__ import annotations

import math

import numpy as np

from lqg_examples import list_examples, load_example, LQGExample
from lqg_explicit import ExplicitModelFollowingResult
from lqg_compare import compare_regulator_methods, compare_model_following
from plant import StateSpacePlant

RUN_LQG_BENCHMARK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_lqg_benchmark",
        "description": (
            "Run LQR (the plant's own suggested Q/R), output-weighted LQR, "
            "Bryson's rule, and full LQG (LQR + steady-state Kalman filter) "
            "against one plant -- either a named preset from the "
            "professor-provided catalog, or a user-supplied custom plant -- "
            "and return metrics + correctness checks for each. Also runs: "
            "implicit and explicit model-following IF am_diag is supplied; "
            "one 'Custom LQR' row IF Q_diag/R_diag are both supplied (refine "
            "one design -- propose weights, look at the result, propose "
            "different weights, call again); one 'Custom LQR N' row per "
            "pair IF Q_diag_list/R_diag_list are both supplied (compare "
            "several weightings in this one call instead of one call per "
            "weighting -- use this whenever comparing options, not the "
            "singular form repeated); reference-tracking metrics "
            "(Overshoot/Rise/Settling per output channel) on every regulator "
            "row IF reference is supplied, instead of the plain regulator "
            "response. Pass exactly one of plant_preset or custom_plant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plant_preset": {
                    "type": "string",
                    "enum": list_examples(),
                    "description": (
                        "Preset plant key, e.g. 'aircraft_hall'. Omit if "
                        "supplying custom_plant instead."
                    ),
                },
                "custom_plant": {
                    "type": "object",
                    "description": (
                        "A user-provided state-space plant, for a system that "
                        "isn't one of the presets -- e.g. a transfer function "
                        "the user converted to state-space themselves. A/B/C/D "
                        "as nested arrays (rows of numbers); D may be omitted "
                        "for an all-zero feedthrough. There's no textbook "
                        "suggested Q/R for a custom plant, so Q=R=I is used "
                        "for its LQR/output-weighted/LQG rows -- mention this "
                        "if the user asks why. Omit if supplying plant_preset "
                        "instead."
                    ),
                    "properties": {
                        "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                              "description": "nx x nx state matrix."},
                        "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                              "description": "nx x nu input matrix."},
                        "C": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                              "description": "ny x nx output matrix."},
                        "D": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                              "description": "ny x nu feedthrough matrix (default: all zeros)."},
                        "name": {"type": "string", "description": "optional label for the plant."},
                    },
                    "required": ["A", "B", "C"],
                },
                "x_max": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Bryson's rule: max desired deviation per state, length nx. "
                        "Defaults to all-ones (a neutral baseline) if omitted."
                    ),
                },
                "u_max": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Bryson's rule: max desired deviation per control input, "
                        "length nu. Defaults to all-ones if omitted."
                    ),
                },
                "Q_diag": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Custom LQR: diagonal Q weight per state, length nx -- how "
                        "heavily to penalize each state's deviation. Must be given "
                        "together with R_diag. Larger values on a given state push "
                        "the design to correct that state faster/more aggressively. "
                        "Use this to refine one design after seeing a result and "
                        "reacting to it -- don't just reason about the 'right' "
                        "direction abstractly, Q/R's effect on overshoot isn't "
                        "simple or monotonic in a coupled MIMO system, check "
                        "empirically. If you want to compare several weightings at "
                        "once (e.g. the user asks to see a trade-off across "
                        "options), use Q_diag_list/R_diag_list instead -- one call, "
                        "not several."
                    ),
                },
                "R_diag": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Custom LQR: diagonal R weight per control input, length nu "
                        "-- how heavily to penalize each input's effort. Larger "
                        "values make the design gentler/slower on that input. Must "
                        "be given together with Q_diag."
                    ),
                },
                "Q_diag_list": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": (
                        "Compare several Q/R weightings in ONE call instead of "
                        "calling this tool once per weighting -- use this whenever "
                        "the user wants to see a trade-off across multiple options "
                        "(e.g. 'try a few different weightings and show me how "
                        "overshoot and settling trade off'). Each entry is a "
                        "Q_diag array (length nx); paired positionally with "
                        "R_diag_list (same length). Adds one 'Custom LQR N' row "
                        "per pair, in order. Mutually exclusive with Q_diag/R_diag "
                        "-- use this form for comparing several, the singular form "
                        "for refining one at a time."
                    ),
                },
                "R_diag_list": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": (
                        "Paired with Q_diag_list, same length -- each entry is an "
                        "R_diag array (length nu) for the corresponding Q_diag_list "
                        "entry. Must be given together with Q_diag_list."
                    ),
                },
                "reference": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Constant reference command, length ny (one per output) -- "
                        "when given, every regulator-family row (LQR/output-weighted/"
                        "Bryson/LQG[/Custom]) is simulated tracking this reference "
                        "instead of the plain zero-reference regulator response, and "
                        "gains Overshoot/Rise/Settling metrics per output channel. "
                        "Only works for square plants (nu == ny) -- if the tool "
                        "returns an error about that, the plant can't be used with "
                        "reference at all, don't retry with different values. Ask "
                        "the user what target value(s) they want tracked; don't "
                        "invent them."
                    ),
                },
                "am_diag": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Model-following (implicit + explicit): desired model pole "
                        "magnitudes, positive numbers, length ny (one per output). "
                        "The target model is Am = diag(-am_diag). Only ask the user "
                        "for this if they want model-following compared -- don't "
                        "invent values. Omit to skip both model-following rows."
                    ),
                },
                "q1_scale": {
                    "type": "number",
                    "description": (
                        "Model-following: scalar multiplier on Q1=I, the "
                        "model-tracking-error weight (default: 1.0). Only used "
                        "when am_diag is given."
                    ),
                },
            },
        },
    },
}


def _sig_round(x, sig=4):
    """Same rounding contract as supervisor_tools_whitebox._sig_round --
    kept as an independent copy rather than a shared import, following the
    precedent that module already set (each entity keeps its own
    serializer)."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x):
            return None
        if x == 0.0:
            return 0.0
        d = sig - int(math.floor(math.log10(abs(x)))) - 1
        return round(x, d)
    return x


def _round_matrix(M):
    return [[_sig_round(v) for v in row] for row in np.atleast_2d(M).tolist()]


def _pole_margin(closed_loop_poles):
    """-max(Re(poles)): distance of the least-stable pole from the
    imaginary axis. A robustness/damping proxy standing in for the Ms/Mt
    metrics the PID track has and the LQG track doesn't compute yet (no
    MIMO Ms/Mt -- see docs/lqg_plan.md pid_compare.py entry). Larger = more
    margin before a perturbation could destabilize the design."""
    real_parts = np.real(closed_loop_poles)
    return float(-np.max(real_parts)) if len(real_parts) else float("nan")


def _serialize_row(row):
    """row: an lqg_compare.ComparisonRow (already has result/sim/checks
    computed by the shared core) -- this function just rounds it into an
    LLM-safe dict, the presentation-layer split supervisor_tools_whitebox_pid.py
    established for the PID side."""
    result, sim, checks = row.result, row.sim, row.checks
    all_checks = checks["pre"] + checks["post"]
    out = {
        "name": row.name,
        "stable": result.is_stable(),
        "all_checks_passed": all(c.passed for c in all_checks),
        "n_checks_failed": sum(1 for c in all_checks if not c.passed),
        "settling_2pct": _sig_round(sim.metrics.get("settling_2pct")),
        "ISU": _sig_round(sim.metrics.get("ISU")),
        "u_peak": _sig_round(sim.metrics.get("u_peak")),
        "final_state_norm": _sig_round(sim.metrics.get("final_state_norm")),
        "pole_margin": _sig_round(_pole_margin(result.closed_loop_poles)),
    }
    if sim.tracking_metrics is not None:
        out["tracking_metrics"] = [
            {k: _sig_round(v) for k, v in m.items()} for m in sim.tracking_metrics
        ]
    if isinstance(result, ExplicitModelFollowingResult):
        out["K1"] = _round_matrix(result.K1)
        out["K2"] = _round_matrix(result.K2)
    else:
        out["K"] = _round_matrix(result.gains.K)
        if result.kalman is not None:
            out["kalman_estimator_stable"] = bool(
                np.all(np.real(result.kalman.estimator_poles) < -1e-9))
    return out


def _build_custom_example(custom_plant: dict):
    """Wrap a user-supplied A/B/C/D into an LQGExample the same way
    streamlit_mimo_panel.py's _render_custom_plant_controls does for the
    GUI's custom-entry path -- Q=R=I, since there's no textbook suggested
    Q/R for a plant that isn't in the catalog."""
    if "A" not in custom_plant or "B" not in custom_plant or "C" not in custom_plant:
        raise ValueError("custom_plant needs at least A, B, and C")
    A = custom_plant["A"]
    B = custom_plant["B"]
    C = custom_plant["C"]
    D = custom_plant.get("D")
    if D is None:
        ny, nu = len(C), len(B[0])
        D = [[0.0] * nu for _ in range(ny)]
    plant = StateSpacePlant(A=A, B=B, C=C, D=D,
                            name=custom_plant.get("name") or "Custom plant")
    return LQGExample(
        key="custom", name=plant.name, citation="user-provided", source_file="",
        plant=plant, suggested_Q_kind="identity", suggested_R_kind="identity",
        suggested_R_scale=1.0, notes="Custom-entered plant; no textbook suggested "
                                     "Q/R, using Q=R=I.")


def run_lqg_benchmark(plant_preset: str = None, custom_plant: dict = None,
                      x_max=None, u_max=None,
                      Q_diag=None, R_diag=None,
                      Q_diag_list=None, R_diag_list=None, reference=None,
                      am_diag=None, q1_scale=1.0, return_sim: bool = False) -> dict:
    if (plant_preset is None) == (custom_plant is None):
        return {"ok": False, "error": "Pass exactly one of plant_preset or custom_plant."}
    try:
        ex = load_example(plant_preset) if plant_preset is not None \
            else _build_custom_example(custom_plant)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    plant = ex.plant

    try:
        raw_rows = compare_regulator_methods(
            ex, x_max=x_max, u_max=u_max, Q_diag=Q_diag, R_diag=R_diag,
            Q_diag_list=Q_diag_list, R_diag_list=R_diag_list, reference=reference)
        rows = [_serialize_row(r) for r in raw_rows]

        if am_diag is not None:
            am_diag_ = np.asarray(am_diag, dtype=float)
            if am_diag_.shape != (plant.ny,):
                return {"ok": False, "error": f"am_diag must have {plant.ny} entries "
                                              f"(one per output), got {len(am_diag_)}"}
            if np.any(am_diag_ <= 0):
                return {"ok": False, "error": "am_diag values must be strictly positive "
                                              "(they're pole magnitudes)"}
            Am = np.diag(-am_diag_)
            Q1 = q1_scale * np.eye(plant.ny)
            R = ex.build_suggested_R()
            mf_rows, _ = compare_model_following(plant, Am, Q1, R)
            raw_rows.extend(mf_rows)
            rows.extend(_serialize_row(r) for r in mf_rows)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the session
        return {"ok": False, "error": f"Benchmark failed: {exc}"}

    # plant_name (ex.name) alongside plant_preset (ex.key): key is a catalog
    # slug ("aircraft_hall") or the constant "custom" for any user-supplied
    # plant, so it collapses every custom plant to the same identity --
    # name is always the actual, distinguishing human-readable name (the
    # catalog's display name, or custom_plant["name"]/"Custom plant"; see
    # _build_custom_example). LQGSession._wrap_benchmark reads this to tag
    # plot_calls with something that actually identifies the plant.
    result = {"ok": True, "plant_preset": ex.key, "plant_name": ex.name, "nx": plant.nx,
              "nu": plant.nu, "ny": plant.ny, "citation": ex.citation, "rows": rows}
    if return_sim:
        # Raw ComparisonRow objects (each already carries .sim from the
        # shared core, unlike the PID whitebox tool these always have it)
        # for a caller that wants to plot the same traces -- see
        # supervisor_session_lqg.LQGSession, which strips this back off
        # before the result goes anywhere near json.dumps/the model.
        result["_sim_rows"] = raw_rows
    return result
