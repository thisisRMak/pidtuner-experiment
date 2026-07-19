#!/usr/bin/env python3
"""Command-line interface for the black-box PID tuner (entity C).

Deliberately has no --plant/--L flag: this script can only ever consume a
previously-published Signal (via cli.py --gen-signal, or any other producer
of the signal_format.Signal .npz format), never the ground-truth transfer
function. This is a process-boundary reinforcement of blackbox.py's
isolation contract, not just a code-level one.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from signal_format import load_signal
from blackbox import BlackBoxTuner


def _fmt(v):
    if isinstance(v, (int, float, np.floating, np.integer)):
        return f"{float(v):.6g}"
    return str(v)


def format_row_text(row) -> str:
    if not row.available:
        return f"Method: {row.name} -> Unavailable: {row.reason}"
    g = row.result.gains
    return (
        f"Method: {row.name}\n"
        f"  Gains: Kp={g.Kp:.6g}, Ki={g.Ki:.6g}, Kd={g.Kd:.6g}\n"
        f"  {row.result.notes}\n"
    )


def serialize_row_json(row) -> dict:
    if not row.available:
        return {"name": row.name, "available": False, "reason": row.reason}
    g = row.result.gains
    return {
        "name": row.name, "available": True,
        "gains": {"Kp": g.Kp, "Ki": g.Ki, "Kd": g.Kd},
        "black_box": row.result.black_box,
        "notes": row.result.notes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Black-box PID tuner CLI: identifies a model and attempts "
                     "all 9 tuning methods purely from published signal files "
                     "(no access to the ground-truth transfer function)."
    )
    parser.add_argument("--in-signal-step", type=str,
                         help="Path to a step-test Signal .npz (e.g. produced by "
                              "cli.py --gen-signal step)")
    parser.add_argument("--in-signal-relay", type=str,
                         help="Path to a relay-test Signal .npz (e.g. produced by "
                              "cli.py --gen-signal relay). Enables ZN-II/Tyreus-Luyben "
                              "via direct empirical Ku,Pu measurement.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if not args.in_signal_step and not args.in_signal_relay:
        msg = "at least one of --in-signal-step / --in-signal-relay is required"
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)

    try:
        step_signal = load_signal(args.in_signal_step) if args.in_signal_step else None
        relay_signal = load_signal(args.in_signal_relay) if args.in_signal_relay else None
    except Exception as exc:
        msg = f"Failed to load signal file: {exc}"
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)

    tuner = BlackBoxTuner(step_signal=step_signal, relay_signal=relay_signal)
    model = tuner.identify()
    rows = tuner.tune_all()

    if args.json:
        out = {
            "model": {
                "fopdt": (model.fopdt.pretty() if model.fopdt else None),
                "fopdt_reason": model.fopdt_reason,
                "delay_detected": (model.fopdt.delay_detected if model.fopdt else None),
                "delay_reason": (model.fopdt.delay_reason if model.fopdt else None),
                "sopdt": (model.sopdt.pretty() if model.sopdt else None),
                "sopdt_reason": model.sopdt_reason,
                "Ku": model.Ku, "Pu": model.Pu, "ku_pu_source": model.ku_pu_source,
                "ku_pu_reason": model.ku_pu_reason,
            },
            "rows": [serialize_row_json(r) for r in rows],
        }
        print(json.dumps(out, indent=2))
    else:
        print("=== Black-box identification ===")
        print(f"FOPDT: {model.fopdt.pretty() if model.fopdt else 'unavailable — ' + str(model.fopdt_reason)}")
        if model.fopdt is not None:
            verdict = "YES" if model.fopdt.delay_detected else "no"
            print(f"Time delay detected: {verdict} — {model.fopdt.delay_reason}")
        print(f"SOPDT: {model.sopdt.pretty() if model.sopdt else 'unavailable — ' + str(model.sopdt_reason)}")
        if model.Ku is not None:
            print(f"Ku, Pu: {model.Ku:.6g}, {model.Pu:.6g}  (source: {model.ku_pu_source})")
        else:
            print(f"Ku, Pu: unavailable — {model.ku_pu_reason}")
        print()
        print("=== Black-box tuning results ===")
        for r in rows:
            print(format_row_text(r))
            print("-" * 40)


if __name__ == "__main__":
    main()
