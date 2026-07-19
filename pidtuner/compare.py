"""Cross-method comparison: robustness metrics + a driver that tunes the
plant with every applicable method and assembles one row of metrics each.

This is the "data layer" behind the Compare-all-methods table, the radar
chart, and (eventually) the Pareto plot. Nothing here touches Tkinter, so it
is unit-testable in isolation.

Metrics per method
------------------
Tracking (unit setpoint step):
    OS%      overshoot
    ts       2% settling time
    IAE      integral |error|
Disturbance rejection (unit load step at the plant input, setpoint = 0):
    IAE_load integral |output deviation|
Robustness (loop frequency response L = C·P):
    Ms       max sensitivity  = max_w |1/(1+L)|   (lower = more robust)
    Mt       max comp. sens.  = max_w |L/(1+L)|
    GM_dB    gain margin
    PM_deg   phase margin
Effort / smoothness (from the tracking sim):
    u_tv     total variation of u(t)  (lower = smoother)
    u_peak   peak |u|

Lower is better for every metric above, which makes a uniform red→green
heatmap and a "bigger polygon = better" radar both straightforward.
"""

from __future__ import annotations

import numpy as np

from identify import run_step_test, find_ultimate_gain
from simulate import simulate_closed_loop
from tune import select_slowest_stable_poles
from tuning_methods import (
    StablePoleCancellation, ZieglerNicholsI, ZieglerNicholsII,
    Amigo, Simc, Boyd, CohenCoon, ChienHronesReswick, TyreusLuyben,
)


# ─────────────────────────────────────────────────────────────────────────────
# Robustness from the loop frequency response
# ─────────────────────────────────────────────────────────────────────────────

def _controller_response(gains, omega):
    """C(jω) = Kp + Ki/(jω) + Kd·(jω) for the parallel-form PID."""
    jw = 1j * np.asarray(omega, dtype=float)
    # Guard ω = 0 (integrator blows up); the grid below starts above 0.
    return gains.Kp + gains.Ki / jw + gains.Kd * jw


def _robustness_grid(plant, n=2000):
    """A wide log-spaced frequency grid covering the plant's features."""
    feats = []
    for arr in (plant.poles(), plant.zeros()):
        if len(arr):
            feats.extend(np.abs(arr[np.abs(arr) > 1e-9]))
    if plant.L > 0:
        feats.append(1.0 / plant.L)
    if feats:
        lo = max(min(feats) * 1e-3, 1e-9)
        hi = max(feats) * 1e3
    else:
        lo, hi = 1e-3, 1e3
    return np.logspace(np.log10(lo), np.log10(hi), n)


def robustness_metrics(plant, gains):
    """Return {Ms, Mt, GM_dB, PM_deg} from L(jω) = C(jω)·P(jω)."""
    omega = _robustness_grid(plant)
    L = _controller_response(gains, omega) * plant.freq_response(omega)
    one_plus = 1.0 + L
    # Avoid division warnings at near-encirclement points
    with np.errstate(divide="ignore", invalid="ignore"):
        S = 1.0 / one_plus
        T = L / one_plus
    Ms = float(np.nanmax(np.abs(S)))
    Mt = float(np.nanmax(np.abs(T)))

    # Gain margin: where phase crosses -180°, GM = 1/|L|
    phase = np.unwrap(np.angle(L))
    mag = np.abs(L)
    GM_dB = float("inf")
    for i in range(len(omega) - 1):
        # phase crossing of -180° (=-π)
        if (phase[i] + np.pi) * (phase[i + 1] + np.pi) < 0:
            # linear interp for |L| at the crossing
            f = (-np.pi - phase[i]) / (phase[i + 1] - phase[i])
            mag_x = mag[i] + f * (mag[i + 1] - mag[i])
            if mag_x > 1e-12:
                GM_dB = float(-20.0 * np.log10(mag_x))
            break

    # Phase margin: where |L| crosses 1, PM = 180° + phase
    PM_deg = float("inf")
    for i in range(len(omega) - 1):
        if (mag[i] - 1.0) * (mag[i + 1] - 1.0) < 0:
            f = (1.0 - mag[i]) / (mag[i + 1] - mag[i])
            ph_x = phase[i] + f * (phase[i + 1] - phase[i])
            PM_deg = float(180.0 + np.degrees(ph_x))
            break

    return {"Ms": Ms, "Mt": Mt, "GM_dB": GM_dB, "PM_deg": PM_deg}


# ─────────────────────────────────────────────────────────────────────────────
# Load-disturbance rejection
# ─────────────────────────────────────────────────────────────────────────────

def load_rejection_metrics(plant, gains, t_end=None):
    """Unit load step at the plant input, setpoint = 0.

    Returns {IAE_load, peak_dev}. Lower is better for both.
    """
    sim = simulate_closed_loop(plant, gains, t_end=t_end,
                               setpoint=0.0, setpoint_kind="step",
                               load_step=1.0, load_step_time=0.0,
                               u_min=-1e6, u_max=1e6)
    if sim.metrics.get("unstable", True):
        return {"IAE_load": float("inf"), "peak_dev": float("inf")}
    iae_load = float(np.trapezoid(np.abs(sim.y), sim.t))
    peak_dev = float(np.max(np.abs(sim.y)))
    return {"IAE_load": iae_load, "peak_dev": peak_dev}


# ─────────────────────────────────────────────────────────────────────────────
# The driver: tune with every method, collect one metric row each
# ─────────────────────────────────────────────────────────────────────────────

# Direction of "good" for each metric: -1 means lower is better (all of ours).
METRIC_DIRECTION = {
    "OS%": -1, "ts": -1, "IAE": -1, "IAE_load": -1,
    "Ms": -1, "Mt": -1, "u_tv": -1, "u_peak": -1,
}
# The metric columns shown in the comparison table, in order.
TABLE_METRICS = ["OS%", "ts", "IAE", "IAE_load", "Ms", "Mt", "u_tv"]


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as exc:               # noqa: BLE001 - report, don't crash
        return ("__error__", str(exc))


def metric_row(plant, name, gains):
    """Compute one comparison row for an arbitrary (name, gains) pair.

    Returns a dict shaped exactly like a row of compare_all_methods:
        {name, gains, stable, error, OS%, ts, IAE, IAE_load, Ms, Mt,
         GM_dB, PM_deg, u_tv, u_peak}
    Unstable / failed tunings come back with stable=False and an error string
    so the UI can grey them out.
    """
    if gains is None:
        return {"name": name, "gains": None, "stable": False,
                "error": "no gains"}
    try:
        track = simulate_closed_loop(plant, gains, setpoint=1.0,
                                     setpoint_kind="step",
                                     u_min=-1e6, u_max=1e6)
    except Exception as exc:               # noqa: BLE001
        return {"name": name, "gains": gains, "stable": False,
                "error": f"sim failed: {exc}"}
    if track.metrics.get("unstable", True) or not track.stable:
        return {"name": name, "gains": gains, "stable": False,
                "error": "closed loop unstable"}
    load = load_rejection_metrics(plant, gains)
    rob = robustness_metrics(plant, gains)
    return {
        "name": name, "gains": gains, "stable": True, "error": None,
        "OS%": track.metrics.get("Overshoot", float("nan")),
        "ts": track.metrics.get("Settling", float("nan")),
        "IAE": track.metrics.get("IAE", float("nan")),
        "IAE_load": load["IAE_load"],
        "Ms": rob["Ms"], "Mt": rob["Mt"],
        "GM_dB": rob["GM_dB"], "PM_deg": rob["PM_deg"],
        "u_tv": track.metrics.get("u_tv", float("nan")),
        "u_peak": track.metrics.get("u_peak", float("nan")),
    }


def compare_all_methods(plant, include_variants=True):
    """Tune `plant` with every applicable method and return a list of rows.

    Each row is a dict: {name, gains, OS%, ts, IAE, IAE_load, Ms, Mt, u_tv,
    u_peak, stable, error}. Methods that fail (e.g. ZN-II on a plant with no
    -180° crossing) are returned with stable=False and an error string, so the
    UI can grey them out rather than vanish.
    """
    # Shared identification (compute once)
    fopdt = None
    try:
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0)
    except Exception:
        fopdt = None
    try:
        Ku, Pu, _ = find_ultimate_gain(plant)
    except Exception:
        Ku = Pu = None

    seed = None
    if fopdt is not None:
        seed = _safe(lambda: Simc(fopdt).tune().gains)
        if isinstance(seed, tuple):
            seed = None

    # Build the (name, tuning-result-or-error) list
    entries = []

    # 1. Pole cancellation (auto, two slowest stable poles)
    def _pole():
        p1, p2 = select_slowest_stable_poles(plant)
        Kd = 1.0 / abs(plant.dc_gain()) if abs(plant.dc_gain()) > 1e-9 else 1.0
        return StablePoleCancellation(plant, p1, p2, Kd=Kd).tune()
    entries.append(("Pole cancellation", _safe(_pole)))

    if fopdt is not None:
        entries.append(("ZN-I", _safe(lambda: ZieglerNicholsI(fopdt).tune())))
    if Ku is not None:
        entries.append(("ZN-II", _safe(lambda: ZieglerNicholsII(Ku, Pu).tune())))
    if fopdt is not None:
        entries.append(("AMIGO", _safe(lambda: Amigo(fopdt).tune())))
        entries.append(("SIMC", _safe(lambda: Simc(fopdt).tune())))
    entries.append(("Boyd", _safe(lambda: Boyd(plant, 1.4, 1.4, seed_gains=seed).tune())))
    if fopdt is not None:
        entries.append(("Cohen–Coon", _safe(lambda: CohenCoon(fopdt).tune())))
        if include_variants:
            entries.append(("CHR set 0%", _safe(lambda: ChienHronesReswick(fopdt, "setpoint", 0).tune())))
            entries.append(("CHR set 20%", _safe(lambda: ChienHronesReswick(fopdt, "setpoint", 20).tune())))
            entries.append(("CHR load 0%", _safe(lambda: ChienHronesReswick(fopdt, "load", 0).tune())))
            entries.append(("CHR load 20%", _safe(lambda: ChienHronesReswick(fopdt, "load", 20).tune())))
        else:
            entries.append(("CHR set 0%", _safe(lambda: ChienHronesReswick(fopdt, "setpoint", 0).tune())))
    if Ku is not None:
        entries.append(("Tyreus–Luyben", _safe(lambda: TyreusLuyben(Ku, Pu).tune())))

    rows = []
    for name, res in entries:
        if isinstance(res, tuple) and res and res[0] == "__error__":
            rows.append({"name": name, "stable": False, "error": res[1],
                         "gains": None})
            continue
        rows.append(metric_row(plant, name, res.gains))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers for heatmap colouring and radar plotting
# ─────────────────────────────────────────────────────────────────────────────

def normalize_column(values, direction=-1):
    """Map a metric column to [0,1] where 1 = best.

    `values` may contain inf/nan (unstable methods); those map to 0 (worst).
    direction = -1 → lower is better; +1 → higher is better.
    """
    v = np.array([x if np.isfinite(x) else np.nan for x in values], dtype=float)
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return np.zeros(len(values))
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi - lo < 1e-15:
        # All equal → everyone is "best"
        out = np.where(np.isfinite(v), 1.0, 0.0)
        return out
    scaled = (v - lo) / (hi - lo)          # 0 at best-low … 1 at worst-high
    if direction == -1:
        good = 1.0 - scaled                # invert: 1 = best (lowest)
    else:
        good = scaled
    good[~np.isfinite(good)] = 0.0
    return good
