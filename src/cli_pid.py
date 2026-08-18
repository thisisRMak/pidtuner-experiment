#!/usr/bin/env python3
"""Command-line interface for PIDTuner.

Exposes the tuning methods and simulation metrics to the CLI, supporting structured JSON output.
"""

from __future__ import annotations

import argparse
import sys
import json
import numpy as np

# Force matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plant import TransferFunction
from pid_identify import run_step_test, find_ultimate_gain
from pid_tuning_methods import (
    PIDGains, TuningResult, halve_gains,
    StablePoleCancellation, ZieglerNicholsI, ZieglerNicholsII,
    Amigo, Simc, Boyd, CohenCoon, ChienHronesReswick, TyreusLuyben
)
from pid_compare import metric_row, compare_all_methods, select_slowest_stable_poles
from pid_simulate import simulate_closed_loop, format_metrics, saturation_mask
from signal_source import SignalGenerator
from signal_format import save_signal


def format_row_text(row: dict) -> str:
    """Format a metric row dict into a readable text format."""
    if not row.get("stable", False):
        return f"Method: {row['name']} -> Unstable/Error: {row.get('error', 'Unknown')}"

    gains = row["gains"]
    _, Ti, Td = gains.to_textbook()
    Ti_str = f"{Ti:.6g} s" if np.isfinite(Ti) else "∞"
    return (
        f"Method: {row['name']}\n"
        f"  Gains: Kp={gains.Kp:.6g}, Ki={gains.Ki:.6g}, Kd={gains.Kd:.6g}"
        f"  (Ti={Ti_str}, Td={Td:.6g} s)\n"
        f"  Metrics:\n"
        f"    Overshoot:          {row.get('OS%') if np.isfinite(row.get('OS%', float('nan'))) else 'N/A'}\n"
        f"    Settling Time (2%): {row.get('ts') if np.isfinite(row.get('ts', float('nan'))) else 'N/A'} s\n"
        f"    IAE (Setpoint):     {row.get('IAE') if np.isfinite(row.get('IAE', float('nan'))) else 'N/A'}\n"
        f"    IAE (Load):         {row.get('IAE_load') if np.isfinite(row.get('IAE_load', float('nan'))) else 'N/A'}\n"
        f"    Peak Sensitivity Ms:{row.get('Ms') if np.isfinite(row.get('Ms', float('nan'))) else 'N/A'}\n"
        f"    Peak Comp Sens Mt:  {row.get('Mt') if np.isfinite(row.get('Mt', float('nan'))) else 'N/A'}\n"
        f"    Gain Margin GM:     {row.get('GM_dB') if np.isfinite(row.get('GM_dB', float('nan'))) else 'N/A'} dB\n"
        f"    Phase Margin PM:    {row.get('PM_deg') if np.isfinite(row.get('PM_deg', float('nan'))) else 'N/A'} deg\n"
        f"    Control Effort TV:  {row.get('u_tv') if np.isfinite(row.get('u_tv', float('nan'))) else 'N/A'}\n"
    )


def serialize_row_json(row: dict) -> dict:
    """Convert numpy values and gains objects in a metric row to standard Python types for JSON."""
    gains = row.get("gains")
    if gains:
        _, Ti, Td = gains.to_textbook()
        gains_dict = {"Kp": gains.Kp, "Ki": gains.Ki, "Kd": gains.Kd,
                      "Ti": (float(Ti) if np.isfinite(Ti) else None), "Td": float(Td)}
    else:
        gains_dict = None

    out = {}
    for k, v in row.items():
        if k == "gains":
            out[k] = gains_dict
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def saturated_sim_info(plant, gains, args):
    """Simulate the step response under the requested actuator-saturation /
    anti-windup settings.

    Returns None when neither --u-min nor --u-max was given — in that case
    the actuator never saturates, so conditional vs. back_calc anti-windup
    would be indistinguishable and there's nothing informative to show.

    `Ka`/`Tt` are only populated when the actuator actually saturated in
    this specific simulation (see `saturated`) — back_calc's correction
    term is otherwise always zero, so reporting a derived Ka that never
    engaged would misleadingly suggest it did something.
    """
    if args.u_min is None and args.u_max is None:
        return None
    u_min = args.u_min if args.u_min is not None else -1e6
    u_max = args.u_max if args.u_max is not None else 1e6
    sim = simulate_closed_loop(plant, gains, setpoint=1.0, setpoint_kind="step",
                               u_min=u_min, u_max=u_max,
                               antiwindup=args.antiwindup, Ka=args.Ka)
    saturated = bool(np.any(saturation_mask(sim)))
    engaged = saturated and sim.antiwindup == "back_calc"
    return {
        "u_min": u_min, "u_max": u_max, "antiwindup": args.antiwindup,
        "saturated": saturated,
        "Ka": sim.Ka if engaged else None,
        "Tt": (sim.Tt if np.isfinite(sim.Tt) else None) if engaged else None,
        "metrics": sim.metrics, "sim": sim,
    }


def format_saturated_block(info):
    """Render a saturated_sim_info() dict as an indented text block."""
    ka_part = ""
    if info["Ka"] is not None:
        tt_str = f"{info['Tt']:.4g} s" if info["Tt"] is not None else "inf (no integral action)"
        ka_part = f", Ka={info['Ka']:.4g} [Tt={tt_str}]"
    header = (f"  Saturated-actuator simulation "
             f"(u_min={info['u_min']:g}, u_max={info['u_max']:g}, "
             f"saturated={'yes' if info['saturated'] else 'no'}, "
             f"antiwindup={info['antiwindup']}{ka_part}):")
    body = format_metrics(info["metrics"])
    indented = "\n".join("    " + line for line in body.splitlines())
    return header + "\n" + indented


def serialize_saturated_json(info):
    """JSON-safe version of saturated_sim_info() (drops the raw sim object)."""
    metrics_clean = {}
    for k, v in info["metrics"].items():
        if isinstance(v, (np.floating, float)):
            metrics_clean[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer, int)):
            metrics_clean[k] = int(v)
        else:
            metrics_clean[k] = v
    return {
        "u_min": info["u_min"], "u_max": info["u_max"],
        "antiwindup": info["antiwindup"], "saturated": info["saturated"],
        "Ka": info["Ka"], "Tt": info["Tt"],
        "metrics": metrics_clean,
    }


def main():
    parser = argparse.ArgumentParser(
        description="PIDTuner Command-Line Interface. Tunes controller for a plant and prints/plots metrics."
    )
    parser.add_argument(
        "--plant",
        required=True,
        help="Plant symbolic transfer function, e.g. '1000/((s+1)(10s+1))'"
    )
    parser.add_argument(
        "--L", "--delay",
        type=float,
        default=0.0,
        help="Plant dead time delay L (default: 0.0)"
    )
    parser.add_argument(
        "--method",
        default="all",
        choices=[
            "all", "pole_cancellation", "zn1", "zn2", "amigo",
            "simc", "boyd", "cohen_coon", "chr", "tyreus_luyben"
        ],
        help="Tuning method (default: all)"
    )
    parser.add_argument("--p1", type=float, help="Pole cancellation: pole 1")
    parser.add_argument("--p2", type=float, help="Pole cancellation: pole 2")
    parser.add_argument("--Kd", type=float, help="Pole cancellation: controller integrator gain Kd")
    parser.add_argument("--tau_c", type=float, help="SIMC: closed-loop time constant")
    parser.add_argument("--tau2", type=float, help="SIMC: second-order time constant")
    parser.add_argument("--Ms", type=float, default=1.4, help="Boyd: peak sensitivity bound (default: 1.4)")
    parser.add_argument("--Mt", type=float, default=1.4, help="Boyd: peak complementary sensitivity bound (default: 1.4)")
    parser.add_argument(
        "--response",
        default="setpoint",
        choices=["setpoint", "load"],
        help="CHR: setpoint or load response behavior (default: setpoint)"
    )
    parser.add_argument(
        "--overshoot",
        type=int,
        default=0,
        choices=[0, 20],
        help="CHR: overshoot percentage target (default: 0)"
    )
    parser.add_argument(
        "--integrating",
        action="store_true",
        help="AMIGO: flag for integrating plant variant"
    )
    parser.add_argument(
        "--no-derivative",
        action="store_true",
        help="Disable derivative action for Boyd, Tyreus-Luyben, etc."
    )
    parser.add_argument(
        "--halve",
        action="store_true",
        help="Halve all gains post-tuning"
    )
    parser.add_argument(
        "--u-min",
        type=float,
        default=None,
        help="Actuator saturation lower bound for the simulated response "
             "(default: unbounded, i.e. no saturation). Only meaningful "
             "together with --u-max and (optionally) --antiwindup."
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=None,
        help="Actuator saturation upper bound for the simulated response "
             "(default: unbounded, i.e. no saturation)."
    )
    parser.add_argument(
        "--antiwindup",
        choices=["conditional", "back_calc"],
        default="conditional",
        help="Anti-windup strategy for the simulated response when the "
             "actuator saturates (default: conditional-integration, the "
             "prior default behavior). 'back_calc' is Astrom & Hagglund's "
             "back-calculation method; has no effect without --u-min/--u-max."
    )
    parser.add_argument(
        "--Ka",
        type=float,
        default=None,
        help="Back-calculation anti-windup gain override (default: "
             "auto-derived per method from Ka=1/Tt, Tt=sqrt(Ti*Td)). "
             "Ignored unless --antiwindup back_calc."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="Save simulation step response plot to this filename (e.g. plot.png)"
    )
    parser.add_argument(
        "--gen-signal",
        choices=["step", "relay"],
        help="Instead of tuning, run this experiment against --plant and publish "
             "the resulting signal to --out-signal, then exit. This is entity A's "
             "CLI mode — the black-box tuner (cli_pid_blackbox.py) never sees --plant."
    )
    parser.add_argument("--out-signal", type=str, help="Output path for --gen-signal (.npz)")
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sample time override for --gen-signal (default: plant.auto_dt(), "
             "which gives only L/5 samples across the dead time). Denser "
             "sampling (smaller --dt) improves delay-sensitive black-box "
             "identification of L."
    )
    parser.add_argument("--step-amp", type=float, default=1.0, help="Step test: step amplitude")
    parser.add_argument("--noise-sigma", type=float, default=0.0, help="Step test: measurement noise std dev")
    parser.add_argument("--seed", type=int, default=None, help="Step test: noise RNG seed")
    parser.add_argument("--t-max", type=float, default=None, help="Signal generation: test duration (default: auto)")
    parser.add_argument("--h", type=float, default=1.0, help="Relay test: relay amplitude")
    parser.add_argument("--setpoint", type=float, default=0.0, help="Relay test: setpoint")
    parser.add_argument("--hysteresis", type=float, default=0.0, help="Relay test: hysteresis band")

    args = parser.parse_args()

    saturating = args.u_min is not None or args.u_max is not None
    if args.antiwindup == "back_calc" and not saturating:
        print("Note: --antiwindup back_calc has no effect without "
              "--u-min/--u-max (the actuator never saturates).", file=sys.stderr)
    if args.Ka is not None and args.antiwindup != "back_calc":
        print("Note: --Ka has no effect unless --antiwindup back_calc.",
              file=sys.stderr)

    # 1. Parse plant
    try:
        plant = TransferFunction.parse(args.plant, L=args.L)
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": f"Failed to parse plant transfer function: {exc}"}))
        else:
            print(f"Error: Failed to parse plant transfer function: {exc}", file=sys.stderr)
        sys.exit(1)

    # 1b. Signal-export mode (entity A): generate a signal and exit, no tuning.
    if args.gen_signal:
        if not args.out_signal:
            msg = "--gen-signal requires --out-signal <path.npz>"
            if args.json:
                print(json.dumps({"error": msg}))
            else:
                print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)
        try:
            gen = SignalGenerator(plant)
            if args.gen_signal == "step":
                sig = gen.step_test(step_amp=args.step_amp, t_max=args.t_max,
                                     noise_sigma=args.noise_sigma, seed=args.seed,
                                     dt=args.dt)
            else:
                sig = gen.relay_test(h=args.h, setpoint=args.setpoint,
                                      hysteresis=args.hysteresis,
                                      t_max=args.t_max if args.t_max is not None else 80.0,
                                      dt=args.dt)
            save_signal(sig, args.out_signal)
        except Exception as exc:
            if args.json:
                print(json.dumps({"error": f"Signal generation failed: {exc}"}))
            else:
                print(f"Error: Signal generation failed: {exc}", file=sys.stderr)
            sys.exit(1)

        info = {"experiment": sig.experiment, "n_samples": len(sig),
                "dt": sig.dt, "meta": sig.meta, "out": args.out_signal}
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"Wrote {sig.experiment} signal ({len(sig)} samples, dt={sig.dt:.4g}s) "
                  f"to {args.out_signal}")
        sys.exit(0)

    # 2. Run tuning
    results: list[tuple[str, TuningResult]] = []

    try:
        if args.method == "all":
            # Compare all methods
            rows = compare_all_methods(plant, include_variants=True)
            if args.halve:
                # Apply halving to results and refresh metrics
                for r in rows:
                    if r.get("stable", False) and r.get("gains"):
                        # Halve the gains and replace
                        new_gains = PIDGains(
                            Kp=r["gains"].Kp * 0.5,
                            Ki=r["gains"].Ki * 0.5,
                            Kd=r["gains"].Kd * 0.5
                        )
                        refreshed = metric_row(plant, r["name"] + " ½", new_gains)
                        r.update(refreshed)

            sat_infos = {}
            for r in rows:
                if r.get("stable", False) and r.get("gains"):
                    si = saturated_sim_info(plant, r["gains"], args)
                    if si is not None:
                        sat_infos[r["name"]] = si

            if args.json:
                serialized = []
                for r in rows:
                    row_json = serialize_row_json(r)
                    if r["name"] in sat_infos:
                        row_json["saturated_sim"] = serialize_saturated_json(sat_infos[r["name"]])
                    serialized.append(row_json)
                print(json.dumps(serialized, indent=2))
            else:
                print("=== PIDTuner Comparison Results ===")
                for r in rows:
                    print(format_row_text(r))
                    if r["name"] in sat_infos:
                        print(format_saturated_block(sat_infos[r["name"]]))
                    print("-" * 40)

            if args.plot:
                # Overlaid plot of step response for all stable methods
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                for r in rows:
                    if r.get("stable", False) and r.get("gains"):
                        sim = (sat_infos[r["name"]]["sim"] if r["name"] in sat_infos
                              else simulate_closed_loop(plant, r["gains"], setpoint=1.0, setpoint_kind="step"))
                        ax1.plot(sim.t, sim.y, label=r["name"])
                        ax2.plot(sim.t, sim.u)
                
                # Add setpoint line
                if rows and any(r.get("stable", False) for r in rows):
                    ax1.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Setpoint")
                
                ax1.set_ylabel("Output y(t)")
                ax1.set_title("Step Response comparison")
                ax1.legend()
                ax1.grid(True)
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Control Effort u(t)")
                ax2.grid(True)
                plt.tight_layout()
                plt.savefig(args.plot)
                plt.close()
            
            sys.exit(0)

        # Handle specific method
        fopdt = None
        Ku = Pu = None

        if args.method in ["zn1", "amigo", "simc", "cohen_coon", "chr", "boyd"]:
            try:
                _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0)
            except Exception as exc:
                if args.method != "boyd": # Boyd can run without a seed
                    raise ValueError(f"FOPDT identification failed: {exc}")

        if args.method in ["zn2", "tyreus_luyben"]:
            try:
                Ku, Pu, _ = find_ultimate_gain(plant)
            except Exception as exc:
                raise ValueError(f"Ultimate gain/period identification failed: {exc}")

        # Choose the right class
        if args.method == "pole_cancellation":
            p1 = args.p1
            p2 = args.p2
            if p1 is None or p2 is None:
                p1_auto, p2_auto = select_slowest_stable_poles(plant)
                if p1 is None:
                    p1 = p1_auto
                if p2 is None:
                    p2 = p2_auto
            
            Kd = args.Kd
            if Kd is None:
                Kd = 1.0 / abs(plant.dc_gain()) if abs(plant.dc_gain()) > 1e-9 else 1.0
            
            tuner = StablePoleCancellation(plant, p1, p2, Kd)
            res = tuner.tune()

        elif args.method == "zn1":
            tuner = ZieglerNicholsI(fopdt)
            res = tuner.tune()

        elif args.method == "zn2":
            tuner = ZieglerNicholsII(Ku, Pu)
            res = tuner.tune()

        elif args.method == "amigo":
            tuner = Amigo(fopdt, integrating=args.integrating)
            res = tuner.tune()

        elif args.method == "simc":
            tuner = Simc(fopdt, tau_c=args.tau_c, tau2=args.tau2)
            res = tuner.tune()

        elif args.method == "boyd":
            seed = None
            if fopdt is not None:
                try:
                    seed = Simc(fopdt).tune().gains
                except Exception:
                    seed = None
            tuner = Boyd(
                plant, Ms=args.Ms, Mt=args.Mt, seed_gains=seed,
                use_derivative=not args.no_derivative
            )
            res = tuner.tune()

        elif args.method == "cohen_coon":
            tuner = CohenCoon(fopdt)
            res = tuner.tune()

        elif args.method == "chr":
            tuner = ChienHronesReswick(fopdt, response=args.response, overshoot=args.overshoot)
            res = tuner.tune()

        elif args.method == "tyreus_luyben":
            tuner = TyreusLuyben(Ku, Pu, use_derivative=not args.no_derivative)
            res = tuner.tune()

        else:
            raise ValueError(f"Unknown method {args.method}")

        if args.halve:
            res = halve_gains(res)

        # 3. Simulate and gather metrics
        mrow = metric_row(plant, res.method, res.gains)
        sat_info = saturated_sim_info(plant, res.gains, args)
        if (sat_info is not None and args.antiwindup == "back_calc"
                and not sat_info["saturated"]):
            print("Note: the actuator never reached --u-min/--u-max in this "
                  "simulation — back_calc had no effect (Ka not reported).",
                  file=sys.stderr)

        # 4. Print / serialize output
        if args.json:
            out = serialize_row_json(mrow)
            if sat_info is not None:
                out["saturated_sim"] = serialize_saturated_json(sat_info)
            print(json.dumps(out, indent=2))
        else:
            print(format_row_text(mrow))
            if sat_info is not None:
                print(format_saturated_block(sat_info))

        # 5. Plot
        if args.plot:
            sim = (sat_info["sim"] if sat_info is not None else
                  simulate_closed_loop(plant, res.gains, setpoint=1.0, setpoint_kind="step"))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1.plot(sim.t, sim.y, label=res.method)
            ax1.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Setpoint")
            ax1.set_ylabel("Output y(t)")
            ax1.set_title(f"Step Response: {res.method}")
            ax1.legend()
            ax1.grid(True)
            ax2.plot(sim.t, sim.u)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Control Effort u(t)")
            ax2.grid(True)
            plt.tight_layout()
            plt.savefig(args.plot)
            plt.close()

    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error during tuning: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
