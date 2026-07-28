"""Entity B's supervisor tool: benchmark all 9 tuning methods against a known
transfer function.

Imports plant.TransferFunction — this module is the ONLY supervisor tool
module allowed to. supervisor_tools_blackbox.py must never import from here
or from plant.py directly (enforced by test_supervisor.py's
test_no_plant_import, mirroring test_pid_tuner.py's existing isolation
tests for signal_format.py/blackbox.py).
"""

from __future__ import annotations

import math

import numpy as np

from compare import compare_all_methods
from plant import TransferFunction

RUN_WHITEBOX_BENCHMARK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_whitebox_benchmark",
        "description": (
            "Run all 9 PID tuning methods (plus CHR variants) against a known "
            "plant transfer function and return metrics for each. Only call "
            "once tf_known=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plant_tf": {
                    "type": "string",
                    "description": "e.g. '1000/((s+1)(10s+1))', same syntax as PIDTuner's --plant flag",
                },
                "delay": {
                    "type": "number",
                    "description": "additional dead time L in seconds, default 0.0",
                },
            },
            "required": ["plant_tf"],
        },
    },
}


def _sig_round(x, sig=4):
    """Round a value to `sig` significant figures for a token-lean, still
    LLM-readable tool result. Non-finite floats become None (JSON has no
    inf/nan) rather than silently stringified."""
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


def _serialize_row(row: dict) -> dict:
    """Compact, rounded, JSON-safe row for LLM consumption. Independent of
    cli.serialize_row_json by design -- see plan: each entity keeps its own
    serializer, following the precedent cli.py/cli_blackbox.py already set."""
    gains = row.get("gains")
    out = {}
    for k, v in row.items():
        if k == "gains":
            out[k] = ({"Kp": _sig_round(gains.Kp), "Ki": _sig_round(gains.Ki),
                        "Kd": _sig_round(gains.Kd)} if gains else None)
        else:
            out[k] = _sig_round(v)
    return out


def run_whitebox_benchmark(plant_tf: str, delay: float = 0.0) -> dict:
    try:
        plant = TransferFunction.parse(plant_tf, L=delay)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the session
        return {"ok": False, "error": f"Failed to parse plant_tf: {exc}"}
    try:
        rows = compare_all_methods(plant, include_variants=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Benchmark failed: {exc}"}
    return {"ok": True, "rows": [_serialize_row(r) for r in rows]}
