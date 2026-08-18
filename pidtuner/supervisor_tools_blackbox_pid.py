"""Entity C's supervisor tool: identify a model purely from published
signals and benchmark all applicable tuning methods against it.

Isolation contract (same as pid_blackbox.py/cli_pid_blackbox.py, and CI-checked the
same way -- see test_supervisor_pid.py's test_no_plant_import): this module must
NEVER import plant.py, directly or otherwise. It only ever consumes
signal_format.Signal objects loaded from disk.
"""

from __future__ import annotations

import math

import numpy as np

from pid_blackbox import BlackBoxTuner
from signal_format import load_signal

RUN_BLACKBOX_BENCHMARK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_blackbox_benchmark",
        "description": (
            "Identify a process model purely from published step/relay test "
            "signal files (no transfer function) and run all applicable PID "
            "tuning methods against it. Provide at least one signal path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step_signal_path": {
                    "type": "string",
                    "description": "path to a step-test Signal .npz (e.g. from `python cli_pid.py --gen-signal step`)",
                },
                "relay_signal_path": {
                    "type": "string",
                    "description": "path to a relay-test Signal .npz",
                },
            },
        },
    },
}


def _sig_round(x, sig=4):
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


def _serialize_row(row) -> dict:
    if not row.available:
        return {"name": row.name, "available": False, "reason": row.reason}
    g = row.result.gains
    return {
        "name": row.name,
        "available": True,
        "gains": {"Kp": _sig_round(g.Kp), "Ki": _sig_round(g.Ki), "Kd": _sig_round(g.Kd)},
        "black_box": row.result.black_box,
        "notes": row.result.notes,
    }


def _serialize_model(model) -> dict:
    return {
        "fopdt": (model.fopdt.pretty() if model.fopdt else None),
        "fopdt_reason": model.fopdt_reason,
        "delay_detected": (model.fopdt.delay_detected if model.fopdt else None),
        "delay_reason": (model.fopdt.delay_reason if model.fopdt else None),
        "sopdt": (model.sopdt.pretty() if model.sopdt else None),
        "sopdt_reason": model.sopdt_reason,
        "Ku": _sig_round(model.Ku), "Pu": _sig_round(model.Pu),
        "ku_pu_source": model.ku_pu_source, "ku_pu_reason": model.ku_pu_reason,
    }


def run_blackbox_benchmark(step_signal_path: str = None, relay_signal_path: str = None) -> dict:
    if not step_signal_path and not relay_signal_path:
        return {"ok": False, "error": "at least one of step_signal_path / relay_signal_path is required"}
    try:
        step_signal = load_signal(step_signal_path) if step_signal_path else None
        relay_signal = load_signal(relay_signal_path) if relay_signal_path else None
    except Exception as exc:  # noqa: BLE001 - report, don't crash the session
        return {"ok": False, "error": f"Failed to load signal file: {exc}"}
    try:
        tuner = BlackBoxTuner(step_signal=step_signal, relay_signal=relay_signal)
        model = tuner.identify()
        rows = tuner.tune_all()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Black-box identification/tuning failed: {exc}"}
    return {
        "ok": True,
        "model": _serialize_model(model),
        "rows": [_serialize_row(r) for r in rows],
    }
