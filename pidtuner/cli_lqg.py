#!/usr/bin/env python3
"""Command-line interface for the LQR/LQG design track (docs/lqg_plan.md
Phase 1). Same shape/conventions as cli.py — flat script, --json output,
--plot saves a matplotlib figure headlessly (Agg backend) — but over
StateSpacePlant/LQGDesignResult instead of TransferFunction/PIDGains.

No GUI companion by design — see docs/lqg_plan.md "CLI vs. GUI".
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lqg_examples import list_examples, load_example
from lqg_design_methods import LQR, OutputWeightedLQR, LQG, add_reference_tracking
from lqg_bryson import BrysonLQR
from lqg_simulate import (
    simulate_state_feedback, simulate_output_feedback, format_regulator_metrics,
)
from lqg_checks import checks_for_result, format_checks


def _array_json(a):
    return np.asarray(a, dtype=float).tolist()


def _checks_json(checks):
    return [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks]


def serialize_result_json(res, sim=None, checks=None):
    out = {
        "method": res.method,
        "K": _array_json(res.gains.K),
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in res.closed_loop_poles],
        "stable": res.is_stable(),
        "notes": res.notes,
    }
    if res.kalman is not None:
        out["kalman"] = {
            "Kf": _array_json(res.kalman.Kf),
            "estimator_poles": [[float(p.real), float(p.imag)]
                               for p in res.kalman.estimator_poles],
        }
    if res.Nbar is not None:
        out["Nbar"] = _array_json(res.Nbar)
    if sim is not None:
        m = dict(sim.metrics)
        for k, v in list(m.items()):
            if isinstance(v, (np.floating, float)) and not np.isfinite(v):
                m[k] = None
        out["sim"] = {"mode": sim.mode, "stable": sim.stable, "metrics": m}
    if checks is not None:
        out["checks"] = {k: _checks_json(v) for k, v in checks.items()}
        out["checks_all_passed"] = all(c["passed"] for cs in out["checks"].values() for c in cs)
    return out


def format_result_text(res, sim=None, checks=None):
    lines = [f"Method: {res.method}", res.notes]
    lines.append(f"K =\n{np.array2string(res.gains.K, precision=4, separator=', ')}")
    poles_str = ", ".join(f"{p.real:.4g}{'+' if p.imag >= 0 else ''}{p.imag:.4g}j"
                          for p in res.closed_loop_poles)
    lines.append(f"Closed-loop poles: {poles_str}")
    lines.append(f"Stable: {res.is_stable()}")
    if res.kalman is not None:
        lines.append(f"Kalman filter Kf =\n"
                     f"{np.array2string(res.kalman.Kf, precision=4, separator=', ')}")
        est_poles_str = ", ".join(f"{p.real:.4g}{'+' if p.imag >= 0 else ''}{p.imag:.4g}j"
                                  for p in res.kalman.estimator_poles)
        lines.append(f"Estimator poles: {est_poles_str}")
    if res.Nbar is not None:
        lines.append(f"N̄ (reference feedforward) =\n"
                     f"{np.array2string(res.Nbar, precision=4, separator=', ')}")
    if sim is not None:
        lines.append(f"Simulation ({sim.mode}): {format_regulator_metrics(sim.metrics)}")
    if checks is not None:
        lines.append("Pre-design checks (Q/R/N well-posedness, stabilizability, detectability):")
        lines.append(format_checks(checks["pre"]))
        lines.append("Post-design checks (Riccati residual, S properties, closed-loop stability):")
        lines.append(format_checks(checks["post"]))
        if "kalman_pre" in checks:
            lines.append("Kalman pre-design checks (Qw/Rv well-posedness, stabilizability, detectability):")
            lines.append(format_checks(checks["kalman_pre"]))
            lines.append("Kalman post-design checks (estimator Riccati residual, P properties, estimator stability):")
            lines.append(format_checks(checks["kalman_post"]))
        all_passed = all(c.passed for cs in checks.values() for c in cs)
        lines.append(f"All checks passed: {all_passed}")
    return "\n".join(lines)


def _auto_t_end(res):
    """15x the slowest closed-loop time constant, floor 5s — same spirit
    as plant.auto_dt()/simulate.py's t_end heuristic (duration scales with
    the actual dynamics rather than a fixed guess)."""
    real_parts = np.real(res.closed_loop_poles)
    stable = real_parts[real_parts < -1e-9]
    if len(stable) == 0:
        return 20.0
    return max(15.0 / np.min(np.abs(stable)), 5.0)


def _broadcast(values, n, name):
    if values is None:
        return None
    if len(values) == 1:
        return np.full(n, values[0])
    if len(values) != n:
        raise ValueError(f"--{name} needs 1 value (broadcast) or {n} values, got {len(values)}")
    return np.array(values, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="LQG design-track CLI. Designs an LQR/LQG controller for "
                    "a preset plant and prints/plots the closed-loop response."
    )
    parser.add_argument("--list-plants", action="store_true",
                       help="List the available preset plants and exit.")
    parser.add_argument("--plant-preset", choices=list_examples(),
                       help="Preset plant key, e.g. 'aircraft_hall'. See --list-plants.")
    parser.add_argument("--method", default="lqr",
                       choices=["lqr", "output_weighted", "bryson", "lqg"],
                       help="Design method (default: lqr, using the preset's "
                            "suggested Q/R).")
    parser.add_argument("--x-max", type=float, nargs="+", default=None,
                       help="Bryson's rule: max desired state deviation(s) "
                            "(one value broadcasts to all states).")
    parser.add_argument("--u-max", type=float, nargs="+", default=None,
                       help="Bryson's rule: max desired control deviation(s) "
                            "(one value broadcasts to all inputs).")
    parser.add_argument("--R-scale", type=float, default=1.0,
                       help="output_weighted: scalar multiplier on R=I (default: 1.0).")
    parser.add_argument("--Qy-scale", type=float, default=1.0,
                       help="output_weighted: scalar multiplier on Qy=I (default: 1.0).")
    parser.add_argument("--Qw-scale", type=float, default=0.01,
                       help="lqg: process-noise covariance Qw = scale·I (default: 0.01).")
    parser.add_argument("--Rv-scale", type=float, default=0.1,
                       help="lqg: measurement-noise covariance Rv = scale·I (default: 0.1).")
    parser.add_argument("--reference-tracking", action="store_true",
                       help="Add the N̄ feedforward gain (AILQG.pdf eq. 112-117) "
                            "after designing.")
    parser.add_argument("--sim", choices=["none", "state_feedback", "output_feedback"],
                       default="state_feedback",
                       help="Which closed-loop simulation to run and report "
                            "(default: state_feedback, a unit-perturbation regulator "
                            "response). 'output_feedback' requires --method lqg.")
    parser.add_argument("--t-end", type=float, default=None,
                       help="Simulation duration in seconds (default: auto, "
                            "15x the slowest closed-loop time constant).")
    parser.add_argument("--dt", type=float, default=0.01,
                       help="Simulation timestep in seconds (default: 0.01).")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument("--no-checks", action="store_true",
                       help="Skip the pre-/post-design correctness checks "
                            "(see docs/lqg_testing.md) — they run by default.")
    parser.add_argument("--plot", type=str,
                       help="Save state-trajectory / control-effort plot to this "
                            "filename (e.g. plot.png).")
    args = parser.parse_args()

    if args.list_plants:
        for key in list_examples():
            ex = load_example(key)
            print(f"{key:24s} nx={ex.plant.nx:<3d} nu={ex.plant.nu:<3d} "
                 f"ny={ex.plant.ny:<3d} {ex.name}  [{ex.citation}]")
        sys.exit(0)

    if not args.plant_preset:
        parser.error("--plant-preset is required (see --list-plants)")

    try:
        ex = load_example(args.plant_preset)
        plant = ex.plant

        if args.method == "lqr":
            Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
            res = LQR(plant, Q=Q, R=R).design()
        elif args.method == "output_weighted":
            res = OutputWeightedLQR(plant, Qy=args.Qy_scale * np.eye(plant.ny),
                                    R=args.R_scale * np.eye(plant.nu)).design()
        elif args.method == "bryson":
            if args.x_max is None or args.u_max is None:
                raise ValueError("--method bryson requires --x-max and --u-max")
            x_max = _broadcast(args.x_max, plant.nx, "x-max")
            u_max = _broadcast(args.u_max, plant.nu, "u-max")
            res = BrysonLQR(plant, x_max=x_max, u_max=u_max).design()
        elif args.method == "lqg":
            Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
            Qw = args.Qw_scale * np.eye(plant.nx)
            Rv = args.Rv_scale * np.eye(plant.ny)
            res = LQG(plant, Q=Q, R=R, Qw=Qw, Rv=Rv).design()
        else:
            raise ValueError(f"unknown method {args.method!r}")

        checks = None if args.no_checks else checks_for_result(res)

        if args.reference_tracking:
            res = add_reference_tracking(res)

        if args.sim == "output_feedback" and res.kalman is None:
            raise ValueError("--sim output_feedback requires --method lqg")

        sim = None
        t_end = args.t_end if args.t_end is not None else _auto_t_end(res)
        t = np.arange(0.0, t_end + args.dt, args.dt)
        if args.sim == "state_feedback":
            sim = simulate_state_feedback(res, t)
        elif args.sim == "output_feedback":
            sim = simulate_output_feedback(res, t, x0=np.ones(plant.nx))

        if args.json:
            print(json.dumps(serialize_result_json(res, sim, checks), indent=2))
        else:
            print(format_result_text(res, sim, checks))

        if args.plot:
            if sim is None:
                raise ValueError("--plot requires --sim to not be 'none'")
            nx, nu = plant.nx, plant.nu
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for i in range(nx):
                ax1.plot(sim.t, sim.x[:, i], label=f"x{i}")
            ax1.set_ylabel("State x(t)")
            ax1.set_title(f"{res.method} on {ex.name} ({sim.mode})")
            ax1.legend(fontsize="small", ncol=min(nx, 4))
            ax1.grid(True)
            for i in range(nu):
                ax2.plot(sim.t, sim.u[:, i], label=f"u{i}")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Control effort u(t)")
            ax2.legend(fontsize="small", ncol=min(nu, 4))
            ax2.grid(True)
            plt.tight_layout()
            plt.savefig(args.plot)
            plt.close()

    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
