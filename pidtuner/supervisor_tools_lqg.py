"""The LQR/LQG supervisor's one benchmark tool: run every applicable
Phase-1 design method (LQR, output-weighted LQR, Bryson's rule, full LQG)
against one preset plant and return metrics for each.

One tool covering all four methods, deliberately -- not four separate
tools. Same shape as supervisor_tools_whitebox.run_whitebox_benchmark,
which bundles all 9 PID methods behind a single call via
compare.compare_all_methods; LQR and LQG are the same family here (LQG is
literally LQR plus a Kalman filter, sharing the same Q/R machinery), so
splitting them into separate supervisor tools would make "should I use LQR
or LQG" -- exactly the kind of question this supervisor should answer --
harder to ask in one turn.

Model-following (lqg_implicit.py/lqg_explicit.py) is NOT included: both
need a target model (Am, Q1) as a per-design choice with no "suggested"
value per preset plant (see docs/lqg_testing.md "Model-following
classes") -- there's no way to auto-run them the way pole_cancellation
auto-derives its poles in compare.py. Revisit if a use case needs it.

Imports lqg_examples.py/lqg_design_methods.py/lqg_bryson.py/lqg_checks.py/
lqg_simulate.py -- this is the only supervisor module allowed to (mirrors
supervisor_tools_whitebox.py's plant.py-only-here contract). There's no
competing black-box LQG tool to isolate this from (see
supervisor_common_lqg.py's module docstring), so unlike
supervisor_tools_blackbox.py there's nothing for an isolation test to
check here.
"""

from __future__ import annotations

import math

import numpy as np

from lqg_examples import list_examples, load_example
from lqg_design_methods import LQR, OutputWeightedLQR, LQG
from lqg_bryson import BrysonLQR
from lqg_checks import checks_for_result
from lqg_simulate import simulate_state_feedback

RUN_LQG_BENCHMARK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_lqg_benchmark",
        "description": (
            "Run LQR (the preset's own suggested Q/R), output-weighted LQR, "
            "Bryson's rule, and full LQG (LQR + steady-state Kalman filter) "
            "against one preset plant from the professor-provided catalog, "
            "and return metrics + correctness checks for each. Only "
            "preset plants are supported -- there's no way to hand this "
            "tool your own A/B/C/D matrices in conversation."
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


def _auto_t_end(closed_loop_poles):
    """15x the slowest closed-loop time constant, floor 5s -- same heuristic
    as cli_lqg.py's _auto_t_end, duplicated rather than imported (this
    module avoids importing cli_lqg.py, matching the entity-isolation
    precedent of not reaching across CLI/tool boundaries)."""
    real_parts = np.real(closed_loop_poles)
    stable = real_parts[real_parts < -1e-9]
    if len(stable) == 0:
        return 20.0
    return max(15.0 / np.min(np.abs(stable)), 5.0)


def _pole_margin(closed_loop_poles):
    """-max(Re(poles)): distance of the least-stable pole from the
    imaginary axis. A robustness/damping proxy standing in for the Ms/Mt
    metrics the PID track has and the LQG track doesn't compute yet (no
    MIMO Ms/Mt -- see docs/lqg_plan.md compare.py entry). Larger = more
    margin before a perturbation could destabilize the design."""
    real_parts = np.real(closed_loop_poles)
    return float(-np.max(real_parts)) if len(real_parts) else float("nan")


def _serialize_row(name, result, plant):
    checks = checks_for_result(result)
    all_checks = checks["pre"] + checks["post"]
    t = np.arange(0.0, _auto_t_end(result.closed_loop_poles), 0.01)
    sim = simulate_state_feedback(result, t)
    metrics = sim.metrics
    row = {
        "name": name,
        "stable": result.is_stable(),
        "all_checks_passed": all(c.passed for c in all_checks),
        "n_checks_failed": sum(1 for c in all_checks if not c.passed),
        "settling_2pct": _sig_round(metrics.get("settling_2pct")),
        "ISU": _sig_round(metrics.get("ISU")),
        "u_peak": _sig_round(metrics.get("u_peak")),
        "final_state_norm": _sig_round(metrics.get("final_state_norm")),
        "pole_margin": _sig_round(_pole_margin(result.closed_loop_poles)),
        "K": _round_matrix(result.gains.K),
    }
    if result.kalman is not None:
        row["kalman_estimator_stable"] = bool(
            np.all(np.real(result.kalman.estimator_poles) < -1e-9))
    return row


def run_lqg_benchmark(plant_preset: str, x_max=None, u_max=None) -> dict:
    try:
        ex = load_example(plant_preset)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    plant = ex.plant

    rows = []
    try:
        Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
        rows.append(_serialize_row(
            "LQR (suggested Q/R)", LQR(plant, Q=Q, R=R).design(), plant))

        rows.append(_serialize_row(
            "Output-weighted LQR",
            OutputWeightedLQR(plant, Qy=np.eye(plant.ny), R=np.eye(plant.nu)).design(),
            plant))

        x_max_ = np.ones(plant.nx) if x_max is None else np.asarray(x_max, dtype=float)
        u_max_ = np.ones(plant.nu) if u_max is None else np.asarray(u_max, dtype=float)
        rows.append(_serialize_row(
            "Bryson's rule", BrysonLQR(plant, x_max=x_max_, u_max=u_max_).design(), plant))

        rows.append(_serialize_row(
            "LQG (Kalman filter)",
            LQG(plant, Q=Q, R=R, Qw=0.01 * np.eye(plant.nx), Rv=0.1 * np.eye(plant.ny)).design(),
            plant))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the session
        return {"ok": False, "error": f"Benchmark failed: {exc}"}

    return {"ok": True, "plant_preset": plant_preset, "nx": plant.nx,
            "nu": plant.nu, "ny": plant.ny, "citation": ex.citation, "rows": rows}
