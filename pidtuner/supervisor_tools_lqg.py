"""The LQR/LQG supervisor's one benchmark tool: run every applicable
Phase-1 design method (LQR, output-weighted LQR, Bryson's rule, full LQG)
against one preset plant and return metrics for each, plus the two
model-following techniques (implicit/explicit) when the caller supplies a
target model via am_diag.

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
compare.py, so unlike the other four they're only included when the
caller (the LLM, on the user's behalf) actually supplies am_diag; never
guessed.

Uses lqg_compare.py (compare_regulator_methods / compare_model_following)
for the actual design/simulation work -- the same shared-core role
compare.py plays for cli.py and supervisor_tools_whitebox.py -- and keeps
its own rounded/JSON-safe row serialization on top, independent of
cli_lqg.py's, following the precedent supervisor_tools_whitebox.py already
set ("Independent of cli.serialize_row_json by design").

Imports lqg_examples.py/lqg_compare.py/lqg_explicit.py -- this is the only
supervisor module allowed to import the lqg_* design/compare modules
(mirrors supervisor_tools_whitebox.py's plant.py-only-here contract).
There's no competing black-box LQG tool to isolate this from (see
supervisor_common_lqg.py's module docstring), so unlike
supervisor_tools_blackbox.py there's nothing for an isolation test to
check here.
"""

from __future__ import annotations

import math

import numpy as np

from lqg_examples import list_examples, load_example
from lqg_explicit import ExplicitModelFollowingResult
from lqg_compare import compare_regulator_methods, compare_model_following

RUN_LQG_BENCHMARK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_lqg_benchmark",
        "description": (
            "Run LQR (the preset's own suggested Q/R), output-weighted LQR, "
            "Bryson's rule, and full LQG (LQR + steady-state Kalman filter) "
            "against one preset plant from the professor-provided catalog, "
            "and return metrics + correctness checks for each. Also runs: "
            "implicit and explicit model-following IF am_diag is supplied; "
            "a 5th 'Custom LQR' row IF Q_diag/R_diag are both supplied "
            "(the way to actually iterate on a design -- propose weights, "
            "look at the returned metrics, propose different weights based "
            "on what you saw, call again); reference-tracking metrics "
            "(Overshoot/Rise/Settling per output channel) on every regulator "
            "row IF reference is supplied, instead of the plain regulator "
            "response. Only preset plants are supported -- there's no way to "
            "hand this tool your own A/B/C/D matrices in conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plant_preset": {
                    "type": "string",
                    "enum": list_examples(),
                    "description": "Preset plant key, e.g. 'aircraft_hall'.",
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
                        "Use this to iterate: after seeing a benchmark result, "
                        "propose new Q_diag/R_diag based on what the user wants "
                        "changed and call again -- don't just reason about the "
                        "'right' direction abstractly, Q/R's effect on overshoot "
                        "isn't simple or monotonic in a coupled MIMO system, check "
                        "empirically."
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
            "required": ["plant_preset"],
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
    MIMO Ms/Mt -- see docs/lqg_plan.md compare.py entry). Larger = more
    margin before a perturbation could destabilize the design."""
    real_parts = np.real(closed_loop_poles)
    return float(-np.max(real_parts)) if len(real_parts) else float("nan")


def _serialize_row(row):
    """row: an lqg_compare.ComparisonRow (already has result/sim/checks
    computed by the shared core) -- this function just rounds it into an
    LLM-safe dict, the presentation-layer split supervisor_tools_whitebox.py
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


def run_lqg_benchmark(plant_preset: str, x_max=None, u_max=None,
                      Q_diag=None, R_diag=None, reference=None,
                      am_diag=None, q1_scale=1.0) -> dict:
    try:
        ex = load_example(plant_preset)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    plant = ex.plant

    try:
        rows = [_serialize_row(r) for r in compare_regulator_methods(
            ex, x_max=x_max, u_max=u_max, Q_diag=Q_diag, R_diag=R_diag, reference=reference)]

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
            rows.extend(_serialize_row(r) for r in mf_rows)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the session
        return {"ok": False, "error": f"Benchmark failed: {exc}"}

    return {"ok": True, "plant_preset": plant_preset, "nx": plant.nx,
            "nu": plant.nu, "ny": plant.ny, "citation": ex.citation, "rows": rows}
