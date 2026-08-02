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
from lqg_implicit import ImplicitModelFollowing
from lqg_explicit import ExplicitModelFollowing, ExplicitModelFollowingResult
from lqg_simulate import (
    simulate_state_feedback, simulate_output_feedback, simulate_explicit_model_following,
    format_regulator_metrics, format_tracking_metrics, auto_t_end,
)
from lqg_checks import checks_for_result, format_checks
from lqg_compare import compare_regulator_methods, compare_model_following


def _array_json(a):
    return np.asarray(a, dtype=float).tolist()


def _checks_json(checks):
    return [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks]


def serialize_result_json(res, sim=None, checks=None):
    if isinstance(res, ExplicitModelFollowingResult):
        out = {
            "method": res.method,
            "K1": _array_json(res.K1),
            "K2": _array_json(res.K2),
            "Am": _array_json(res.Am),
            "closed_loop_poles": [[float(p.real), float(p.imag)] for p in res.closed_loop_poles],
            "stable": res.is_stable(),
            "notes": res.notes,
        }
    else:
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
        if sim.tracking_metrics is not None:
            out["sim"]["tracking_metrics"] = sim.tracking_metrics
    if checks is not None:
        out["checks"] = {k: _checks_json(v) for k, v in checks.items()}
        out["checks_all_passed"] = all(c["passed"] for cs in out["checks"].values() for c in cs)
    return out


def format_result_text(res, sim=None, checks=None):
    lines = [f"Method: {res.method}", res.notes]
    if isinstance(res, ExplicitModelFollowingResult):
        lines.append(f"K1 (feedback) =\n{np.array2string(res.K1, precision=4, separator=', ')}")
        lines.append(f"K2 (feedforward on xm) =\n{np.array2string(res.K2, precision=4, separator=', ')}")
        poles_str = ", ".join(f"{p.real:.4g}{'+' if p.imag >= 0 else ''}{p.imag:.4g}j"
                              for p in res.closed_loop_poles)
        lines.append(f"Closed-loop poles (plant side, eig(A-B·K1)): {poles_str}")
        lines.append(f"Stable: {res.is_stable()}")
    else:
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
        if sim.tracking_metrics is not None:
            lines.append("Reference-tracking metrics (per output channel):")
            lines.append(format_tracking_metrics(sim.tracking_metrics))
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
    """Thin wrapper over lqg_simulate.auto_t_end: for explicit
    model-following, also considers Am's own poles — the model can be
    slower than the plant, and t_end should be long enough to see it (and
    the plant chasing it) settle."""
    extra = np.linalg.eigvals(res.Am) if isinstance(res, ExplicitModelFollowingResult) else None
    return auto_t_end(res.closed_loop_poles, extra_poles=extra)


def format_regulator_comparison_table(rows):
    """Text table for --method all: the regulator-family comparison
    (LQR/OutputWeightedLQR/BrysonLQR/LQG[/Custom]), same shape/columns as
    cli.py's --method all text output for the PID side. When
    reference-tracking is active, appends each row's per-channel
    Overshoot/Rise/Settling underneath the table (too many extra columns
    to fit inline for MIMO plants)."""
    header = f"{'Method':<28s} {'Stable':<7s} {'Checks':<6s} {'Settling(2%)':>13s} {'ISU':>10s} {'u_peak':>8s} {'pole_margin':>12s}"
    lines = [header, "-" * len(header)]
    for row in rows:
        all_checks = row.checks["pre"] + row.checks["post"]
        checks_str = "PASS" if all(c.passed for c in all_checks) else "FAIL"
        m = row.sim.metrics
        pole_margin = float(-np.max(np.real(row.result.closed_loop_poles)))
        lines.append(
            f"{row.name:<28s} {str(row.result.is_stable()):<7s} {checks_str:<6s} "
            f"{m['settling_2pct']:>13.3g} {m['ISU']:>10.3g} {m['u_peak']:>8.3g} {pole_margin:>12.4g}"
        )
    if rows and rows[0].sim.tracking_metrics is not None:
        lines.append("")
        lines.append("Reference-tracking metrics (per output channel):")
        for row in rows:
            lines.append(f"  {row.name}:")
            lines.append(format_tracking_metrics(row.sim.tracking_metrics))
    return "\n".join(lines)


def regulator_comparison_json(rows):
    out = []
    for row in rows:
        all_checks = row.checks["pre"] + row.checks["post"]
        entry = {
            "name": row.name,
            "stable": row.result.is_stable(),
            "all_checks_passed": all(c.passed for c in all_checks),
            "metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                       for k, v in row.sim.metrics.items()},
            "pole_margin": float(-np.max(np.real(row.result.closed_loop_poles))),
        }
        if row.sim.tracking_metrics is not None:
            entry["tracking_metrics"] = row.sim.tracking_metrics
        out.append(entry)
    return out


def format_model_following_comparison_table(rows, t, xm_ref):
    """Text table for --method model_following_all: implicit vs explicit,
    plus how close each one's final output landed relative to the target
    model's own final state (the thing they're both actually trying to
    match — see lqg_compare.py's module docstring for why this isn't the
    same comparison as the regulator family's)."""
    header = f"{'Method':<26s} {'Stable':<7s} {'Checks':<6s} {'Settling(2%)':>13s} {'ISU':>10s} {'final ||y-xm||':>15s}"
    lines = [header, "-" * len(header)]
    for row in rows:
        all_checks = row.checks["pre"] + row.checks["post"]
        checks_str = "PASS" if all(c.passed for c in all_checks) else "FAIL"
        m = row.sim.metrics
        final_gap = float(np.linalg.norm(row.sim.y[-1] - xm_ref[-1]))
        lines.append(
            f"{row.name:<26s} {str(row.result.is_stable()):<7s} {checks_str:<6s} "
            f"{m['settling_2pct']:>13.3g} {m['ISU']:>10.3g} {final_gap:>15.3g}"
        )
    return "\n".join(lines)


def model_following_comparison_json(rows, t, xm_ref):
    out = []
    for row in rows:
        all_checks = row.checks["pre"] + row.checks["post"]
        out.append({
            "name": row.name,
            "stable": row.result.is_stable(),
            "all_checks_passed": all(c.passed for c in all_checks),
            "metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                       for k, v in row.sim.metrics.items()},
            "final_gap_from_model": float(np.linalg.norm(row.sim.y[-1] - xm_ref[-1])),
        })
    return out


def plot_regulator_comparison(rows, ex, path, reference=None):
    """--method all: overlay ||x(t)|| and ||u(t)|| (state/control-effort
    norms, not individual channels — MIMO state counts vary a lot across
    the catalog, up to 15, so per-channel legends across 4/5 methods
    wouldn't stay readable) for all regulator-family methods on shared
    axes, the LQG-track analog of cli.py's --method all overlay plot.

    When --reference-tracking is active (reference given), the norm plot
    isn't useful (x doesn't go to 0), so this switches to one panel per
    output channel instead: each method's y(t) overlaid against the
    commanded reference (dashed) — directly comparable to
    plot_model_following_comparison's per-channel-vs-target-line shape."""
    if reference is not None:
        ny = len(reference)
        fig, axes = plt.subplots(ny, 1, figsize=(10, 4 * ny), sharex=True)
        axes = np.atleast_1d(axes)
        for j in range(ny):
            ax = axes[j]
            ax.axhline(reference[j], color="k", linestyle="--", linewidth=1.5, label="reference")
            for row in rows:
                ax.plot(row.sim.t, row.sim.y[:, j], label=row.name)
            ax.set_ylabel(f"y{j}(t)")
            ax.legend(fontsize="small")
            ax.grid(True)
        axes[0].set_title(f"Regulator-family reference-tracking comparison on {ex.name}")
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for row in rows:
        x_norm = np.linalg.norm(row.sim.x, axis=1)
        ax1.plot(row.sim.t, x_norm, label=row.name)
        u_norm = np.linalg.norm(row.sim.u, axis=1)
        ax2.plot(row.sim.t, u_norm, label=row.name)
    ax1.set_ylabel("||x(t)||")
    ax1.set_title(f"Regulator-family comparison on {ex.name}")
    ax1.legend(fontsize="small")
    ax1.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("||u(t)||")
    ax2.legend(fontsize="small")
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_model_following_comparison(rows, t, xm_ref, ex, path):
    """--method model_following_all: one panel per output channel, each
    overlaying the target model's own free response (dashed) against both
    implicit's and explicit's y(t) — the comparison that actually matters
    here (how closely does each one's output resemble the target
    dynamics), not a shared efficiency metric. See lqg_compare.py's module
    docstring."""
    ny = xm_ref.shape[1]
    fig, axes = plt.subplots(ny, 1, figsize=(10, 4 * ny), sharex=True)
    axes = np.atleast_1d(axes)
    for j in range(ny):
        ax = axes[j]
        ax.plot(t, xm_ref[:, j], "k--", label="target model xm", linewidth=1.5)
        for row in rows:
            ax.plot(row.sim.t, row.sim.y[:, j], label=row.name)
        ax.set_ylabel(f"y{j}(t)")
        ax.legend(fontsize="small")
        ax.grid(True)
    axes[0].set_title(f"Model-following comparison on {ex.name}")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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
                       choices=["lqr", "output_weighted", "bryson", "lqg", "implicit", "explicit",
                                "all", "model_following_all"],
                       help="Design method (default: lqr, using the preset's "
                            "suggested Q/R). 'implicit'/'explicit' are the two "
                            "model-following techniques (AILQG.pdf §4) and require "
                            "--am-diag. 'all' compares LQR/output_weighted/bryson/lqg "
                            "on one plot/table (the regulator family — same objective, "
                            "directly comparable); 'model_following_all' compares "
                            "implicit vs. explicit given the same --am-diag (a "
                            "different objective -- see docs/lqg_testing.md for why "
                            "these are two separate comparisons, not six-in-one).")
    parser.add_argument("--x-max", type=float, nargs="+", default=None,
                       help="Bryson's rule: max desired state deviation(s) "
                            "(one value broadcasts to all states).")
    parser.add_argument("--u-max", type=float, nargs="+", default=None,
                       help="Bryson's rule: max desired control deviation(s) "
                            "(one value broadcasts to all inputs).")
    parser.add_argument("--R-scale", type=float, default=1.0,
                       help="output_weighted/implicit/explicit: scalar multiplier "
                            "on R=I (default: 1.0).")
    parser.add_argument("--Qy-scale", type=float, default=1.0,
                       help="output_weighted: scalar multiplier on Qy=I (default: 1.0).")
    parser.add_argument("--Qw-scale", type=float, default=0.01,
                       help="lqg: process-noise covariance Qw = scale·I (default: 0.01).")
    parser.add_argument("--Rv-scale", type=float, default=0.1,
                       help="lqg: measurement-noise covariance Rv = scale·I (default: 0.1).")
    parser.add_argument("--Q-diag", type=float, nargs="+", default=None,
                       help="lqr/all: custom diagonal Q weight per state (length nx), "
                            "overriding the preset's suggested Q. Must be given "
                            "together with --R-diag. For 'lqr' this replaces the "
                            "suggested-Q design entirely; for 'all' it adds a 5th "
                            "'Custom LQR' row alongside the other four.")
    parser.add_argument("--R-diag", type=float, nargs="+", default=None,
                       help="lqr/all: custom diagonal R weight per input (length nu). "
                            "See --Q-diag.")
    parser.add_argument("--am-diag", type=float, nargs="+", default=None,
                       help="implicit/explicit: desired model pole magnitudes "
                            "(positive numbers; Am = diag(-values)), one per output "
                            "(one value broadcasts to all ny outputs). Required for "
                            "--method implicit/explicit -- there's no 'suggested' "
                            "target model the way there's a suggested Q/R.")
    parser.add_argument("--Q1-scale", type=float, default=1.0,
                       help="implicit/explicit: scalar multiplier on Q1=I, the "
                            "model-tracking-error weight (default: 1.0).")
    parser.add_argument("--reference-tracking", action="store_true",
                       help="Add the N̄ feedforward gain (AILQG.pdf eq. 112-117) "
                            "after designing, and (with --sim state_feedback) "
                            "actually simulate tracking --reference. Not compatible "
                            "with --method explicit, which already tracks a target "
                            "model via --am-diag.")
    parser.add_argument("--reference", type=float, nargs="+", default=None,
                       help="Constant reference command r for --reference-tracking's "
                            "simulation, one value per output (default: all-ones). "
                            "Only used with --sim state_feedback.")
    parser.add_argument("--sim", choices=["none", "state_feedback", "output_feedback", "model_following"],
                       default="state_feedback",
                       help="Which closed-loop simulation to run and report "
                            "(default: state_feedback, a unit-perturbation regulator "
                            "response, or a --reference step if --reference-tracking is "
                            "set). 'output_feedback' requires --method lqg. For "
                            "--method explicit, any value other than 'none' runs the "
                            "model-following simulation regardless (there's only one "
                            "simulation shape for that method).")
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

        if args.method == "all":
            x_max = _broadcast(args.x_max, plant.nx, "x-max") if args.x_max is not None else None
            u_max = _broadcast(args.u_max, plant.nu, "u-max") if args.u_max is not None else None
            if (args.Q_diag is None) != (args.R_diag is None):
                raise ValueError("--Q-diag and --R-diag must be given together")
            Q_diag = _broadcast(args.Q_diag, plant.nx, "Q-diag") if args.Q_diag is not None else None
            R_diag = _broadcast(args.R_diag, plant.nu, "R-diag") if args.R_diag is not None else None
            reference = None
            if args.reference_tracking:
                ref_vals = args.reference if args.reference is not None else [1.0]
                reference = _broadcast(ref_vals, plant.ny, "reference")
            rows = compare_regulator_methods(
                ex, x_max=x_max, u_max=u_max, Qy_scale=args.Qy_scale, R_scale=args.R_scale,
                Qw_scale=args.Qw_scale, Rv_scale=args.Rv_scale,
                Q_diag=Q_diag, R_diag=R_diag, reference=reference,
                t_end=args.t_end, dt=args.dt)
            if args.json:
                print(json.dumps(regulator_comparison_json(rows), indent=2))
            else:
                print(format_regulator_comparison_table(rows))
            if args.plot:
                plot_regulator_comparison(rows, ex, args.plot, reference=reference)
            return

        if args.method == "model_following_all":
            if args.am_diag is None:
                raise ValueError(
                    "--method model_following_all requires --am-diag (desired model "
                    f"pole magnitudes, {plant.ny} value(s) or 1 to broadcast)")
            am_diag = _broadcast(args.am_diag, plant.ny, "am-diag")
            if np.any(am_diag <= 0):
                raise ValueError(
                    "--am-diag values must be strictly positive (they're pole "
                    "magnitudes; Am = diag(-am_diag))")
            Am = np.diag(-am_diag)
            Q1 = args.Q1_scale * np.eye(plant.ny)
            R = args.R_scale * np.eye(plant.nu)
            rows, (t, xm_ref) = compare_model_following(plant, Am, Q1, R,
                                                         t_end=args.t_end, dt=args.dt)
            if args.json:
                print(json.dumps(model_following_comparison_json(rows, t, xm_ref), indent=2))
            else:
                print(format_model_following_comparison_table(rows, t, xm_ref))
            if args.plot:
                plot_model_following_comparison(rows, t, xm_ref, ex, args.plot)
            return

        if args.method == "lqr":
            if (args.Q_diag is None) != (args.R_diag is None):
                raise ValueError("--Q-diag and --R-diag must be given together")
            if args.Q_diag is not None:
                Q = np.diag(_broadcast(args.Q_diag, plant.nx, "Q-diag"))
                R = np.diag(_broadcast(args.R_diag, plant.nu, "R-diag"))
            else:
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
        elif args.method in ("implicit", "explicit"):
            if args.am_diag is None:
                raise ValueError(
                    f"--method {args.method} requires --am-diag (desired model "
                    f"pole magnitudes, {plant.ny} value(s) or 1 to broadcast)")
            am_diag = _broadcast(args.am_diag, plant.ny, "am-diag")
            if np.any(am_diag <= 0):
                raise ValueError(
                    "--am-diag values must be strictly positive (they're pole "
                    "magnitudes; Am = diag(-am_diag))")
            Am = np.diag(-am_diag)
            Q1 = args.Q1_scale * np.eye(plant.ny)
            R = args.R_scale * np.eye(plant.nu)
            if args.method == "implicit":
                res = ImplicitModelFollowing(plant, Am=Am, Q1=Q1, R=R).design()
            else:
                if args.reference_tracking:
                    raise ValueError(
                        "--reference-tracking is not compatible with --method "
                        "explicit -- explicit model-following already tracks a "
                        "target model (--am-diag); there's no separate "
                        "constant-reference feedforward for it.")
                res = ExplicitModelFollowing(plant, Am=Am, Q1=Q1, R=R).design()
        else:
            raise ValueError(f"unknown method {args.method!r}")

        checks = None if args.no_checks else checks_for_result(res)

        if args.reference_tracking:
            res = add_reference_tracking(res)

        is_explicit = isinstance(res, ExplicitModelFollowingResult)
        if args.sim == "output_feedback" and (is_explicit or res.kalman is None):
            raise ValueError("--sim output_feedback requires --method lqg")

        sim = None
        t_end = args.t_end if args.t_end is not None else _auto_t_end(res)
        t = np.arange(0.0, t_end + args.dt, args.dt)
        if is_explicit:
            if args.sim != "none":
                sim = simulate_explicit_model_following(res, t)
        elif args.sim == "state_feedback":
            r_arr = None
            if args.reference_tracking:
                reference = args.reference if args.reference is not None else [1.0]
                reference = _broadcast(reference, plant.ny, "reference")
                r_arr = np.tile(reference, (len(t), 1))
            sim = simulate_state_feedback(res, t, r=r_arr)
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
            n_panels = 3 if sim.xm is not None else 2
            fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)
            ax1, ax2 = axes[0], axes[1]
            for i in range(nx):
                ax1.plot(sim.t, sim.x[:, i], label=f"x{i}")
            ax1.set_ylabel("State x(t)")
            ax1.set_title(f"{res.method} on {ex.name} ({sim.mode})")
            ax1.legend(fontsize="small", ncol=min(nx, 4))
            ax1.grid(True)
            for i in range(nu):
                ax2.plot(sim.t, sim.u[:, i], label=f"u{i}")
            ax2.set_ylabel("Control effort u(t)")
            ax2.legend(fontsize="small", ncol=min(nu, 4))
            ax2.grid(True)
            if sim.xm is not None:
                ax3 = axes[2]
                nxm = sim.xm.shape[1]
                for i in range(nxm):
                    ax3.plot(sim.t, sim.xm[:, i], "--", label=f"xm{i}")
                ax3.set_ylabel("Model state xm(t)")
                ax3.legend(fontsize="small", ncol=min(nxm, 4))
                ax3.grid(True)
            axes[-1].set_xlabel("Time (s)")
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
