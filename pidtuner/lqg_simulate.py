"""Closed-loop simulation for the LQR/LQG design track — the LQG-track
analog of simulate.py.

Three simulation modes, matching the shapes an LQGDesignResult/
ExplicitModelFollowingResult can take:
  - simulate_state_feedback: full-state-feedback regulator ẋ = (A-BK)x, or
    (with add_reference_tracking's N̄) reference tracking u = -Kx + N̄r.
    Uses scipy.signal.lsim directly — the same primitive the professor's own
    AIKreindlerRothschildModelFollowingN.m uses via MATLAB's `lsim`.
  - simulate_output_feedback: the estimator (not the true state) drives u,
    simulated in the (x, x̃) error coordinates of AILQG.pdf eq. 106 so the
    block-triangular structure — and hence the separation principle — is
    exact by construction rather than approximately true after integration.
  - simulate_explicit_model_following: the augmented [x; xm] closed loop for
    an ExplicitModelFollowingResult (u = -K1·x - K2·xm), xm evolving
    autonomously under Am (eq. 48 — no exogenous input to the model).

Actuator saturation is a plain clip (no back-calculation-style anti-windup —
that machinery is PID-specific, see simulate.py's module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import StateSpace, lsim

from lqg_design_methods import LQGDesignResult
from lqg_explicit import ExplicitModelFollowingResult


@dataclass
class LQGSimResult:
    t: np.ndarray
    x: np.ndarray            # (n, nx) true state trajectory
    y: np.ndarray            # (n, ny) output trajectory
    u: np.ndarray            # (n, nu) control effort
    stable: bool
    metrics: dict
    x_hat: Optional[np.ndarray] = None  # (n, nx) estimated state — None unless output_feedback
    xm: Optional[np.ndarray] = None     # (n, nxm) model-state trajectory — None unless
                                         # mode == "model_following"
    tracking_metrics: Optional[list] = None  # per-output-channel step metrics — None
                                              # unless simulate_state_feedback was given r
    mode: str = "state_feedback"        # "state_feedback" | "output_feedback" | "model_following"


# ─────────────────────────────────────────────────────────────────────────────
# Shared sizing helper — single source of truth (previously duplicated in
# cli_lqg.py and supervisor_tools_lqg.py; centralized here since it's pure
# poles-in-duration-out math, no presentation/rounding concerns like the
# rest of what those two modules keep as their own separate copies)
# ─────────────────────────────────────────────────────────────────────────────

def auto_t_end(closed_loop_poles, extra_poles=None):
    """15x the slowest closed-loop time constant, floor 5s — same spirit as
    plant.auto_dt()/simulate.py's t_end heuristic (duration scales with the
    actual dynamics rather than a fixed guess). extra_poles lets a second
    pole set (e.g. explicit model-following's Am, which can be slower than
    the plant) factor into the duration too."""
    real_parts = np.real(closed_loop_poles)
    if extra_poles is not None:
        real_parts = np.concatenate([real_parts, np.real(extra_poles)])
    stable = real_parts[real_parts < -1e-9]
    if len(stable) == 0:
        return 20.0
    return max(15.0 / np.min(np.abs(stable)), 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — the LQ-cost-native analog of simulate.py's compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_regulator_metrics(t, x, u):
    """||x||-based settling time (2% of the initial norm) + control-effort
    metrics. ISU = ∫ uᵀu dt is literally the LQ cost's control term (AILQG.pdf
    eq. 10), so it's a more native fit here than it was as a borrowed metric
    on the PID side."""
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
        return {"unstable": True, "ISU": float("inf"), "u_peak": float("inf"),
                "settling_2pct": float("inf"), "final_state_norm": float("inf")}
    x_norm = np.linalg.norm(x, axis=1)
    ref_norm = x_norm[0] if x_norm[0] > 1e-9 else max(float(np.max(x_norm)), 1.0)
    band = 0.02 * ref_norm
    out_of_band = np.where(x_norm > band)[0]
    settling = float(t[out_of_band[-1]]) if len(out_of_band) else float(t[0])
    return {
        "unstable": False,
        "settling_2pct": settling,
        "ISU": float(np.trapezoid(np.sum(u ** 2, axis=1), t)),
        "u_peak": float(np.max(np.abs(u))) if u.size else 0.0,
        "final_state_norm": float(x_norm[-1]),
    }


def format_regulator_metrics(m):
    if m.get("unstable"):
        return "UNSTABLE (diverging response)"
    return (f"settling (2%) = {m['settling_2pct']:.3g} s,  "
            f"ISU (control energy) = {m['ISU']:.3g},  "
            f"|u|_peak = {m['u_peak']:.3g},  "
            f"final ||x|| = {m['final_state_norm']:.3g}")


def _step_channel_metrics(t, y_col, final):
    """One output channel's step-response metrics (Overshoot%, Rise 10-90%,
    Settling 2%) — same definitions as simulate.py's compute_metrics, but
    sign-aware: simulate.py's version silently assumes a positive-going
    step (`np.max(y)` for overshoot), which breaks for a negative reference
    (the LQG track's own reference-tracking test uses r=[1.0, -0.5] — a
    mixed-sign command, exactly the case that would misfire). Overshoot is
    "how far past final, in the commanded direction" — for a negative
    final, that means how far below final the response dips, using
    np.min(y) rather than np.max(y)."""
    if abs(final) < 1e-9:
        return {"Overshoot": 0.0, "Rise": float("nan"), "Settling": float(t[0])}
    sign = 1.0 if final > 0 else -1.0
    extreme = float(np.max(y_col)) if sign > 0 else float(np.min(y_col))
    overshoot = max(0.0, sign * (extreme - final) / abs(final) * 100.0)

    progress = sign * y_col / abs(final)  # 0 at start-ish, 1 at final, sign-normalized
    i10_mask = progress >= 0.10
    i90_mask = progress >= 0.90
    i10 = int(np.argmax(i10_mask)) if i10_mask.any() else -1
    i90 = int(np.argmax(i90_mask)) if i90_mask.any() else -1
    rise = float(t[i90] - t[i10]) if (i10 >= 0 and i90 > i10) else float("nan")

    band = 0.02 * abs(final)
    out_of_band = np.where(np.abs(y_col - final) > band)[0]
    settling = float(t[out_of_band[-1]]) if len(out_of_band) else float(t[0])
    return {"Overshoot": overshoot, "Rise": rise, "Settling": settling}


def compute_tracking_metrics(t, y, r):
    """Per-output-channel step-response metrics (Overshoot%, Rise, Settling)
    for a reference-tracking simulation — the MIMO/LQG-track analog of
    simulate.py's step-response branch of compute_metrics. Only meaningful
    when y is actually being driven toward a nonzero constant r (see
    simulate_state_feedback's r parameter) — there's no equivalent for the
    plain regulator case (driving x to 0 has no "final value" to overshoot
    past in the same sense; settling_2pct in compute_regulator_metrics is
    the metric that already covers that case)."""
    y = np.atleast_2d(y)
    r = np.atleast_2d(r)
    ny = y.shape[1]
    return [_step_channel_metrics(t, y[:, j], float(r[-1, j])) for j in range(ny)]


def format_tracking_metrics(metrics_list):
    lines = []
    for j, m in enumerate(metrics_list):
        lines.append(f"  y{j}: Overshoot = {m['Overshoot']:.3g}%,  "
                     f"Rise (10-90%) = {m['Rise']:.3g} s,  "
                     f"Settling (2%) = {m['Settling']:.3g} s")
    return "\n".join(lines)


def _as_time_major(arr, n):
    """lsim's output orientation varies with nx==1 edge cases; normalize to
    (n, ncols)."""
    arr = np.atleast_2d(arr)
    if arr.shape[0] != n and arr.shape[1] == n:
        arr = arr.T
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Full state feedback
# ─────────────────────────────────────────────────────────────────────────────

def simulate_state_feedback(result: LQGDesignResult, t, x0=None, r=None,
                            u_min=-1e9, u_max=1e9) -> LQGSimResult:
    """Simulate ẋ = (A-BK)x (regulator, x0 != 0) or, if `r` is given,
    ẋ = (A-BK)x + B·N̄·r (reference tracking — requires
    add_reference_tracking() to have set result.Nbar first).

    x0 defaults to a generic unit perturbation (np.ones(nx)) when neither
    x0 nor r is given, so a bare call still produces a meaningful regulator
    response to plot.
    """
    plant = result.plant
    nx = plant.nx
    K = result.gains.K
    Acl = plant.A - plant.B @ K
    t = np.asarray(t, dtype=float)
    n = len(t)

    if r is None:
        x0_ = np.ones(nx) if x0 is None else np.asarray(x0, dtype=float)
        Bcl = np.zeros((nx, 1))
        U = np.zeros((n, 1))
        r_arr = None
    else:
        if result.Nbar is None:
            raise ValueError(
                "r was given but result.Nbar is None — call "
                "add_reference_tracking(result) first"
            )
        x0_ = np.zeros(nx) if x0 is None else np.asarray(x0, dtype=float)
        Bcl = plant.B @ result.Nbar
        r_arr = _as_time_major(np.asarray(r, dtype=float), n)
        U = r_arr

    sys = StateSpace(Acl, Bcl, np.eye(nx), np.zeros((nx, Bcl.shape[1])))
    _, _, x_out = lsim(sys, U=U, T=t, X0=x0_)
    x_out = _as_time_major(x_out, n)

    u = -(x_out @ K.T)
    if r_arr is not None:
        u = u + r_arr @ result.Nbar.T
    u_clipped = np.clip(u, u_min, u_max)
    y = x_out @ plant.C.T + u_clipped @ plant.D.T

    stable = result.is_stable() and bool(np.all(np.isfinite(x_out))) \
        and bool(np.max(np.abs(x_out)) < 1e9 if x_out.size else True)
    metrics = compute_regulator_metrics(t, x_out, u_clipped)
    tracking_metrics = compute_tracking_metrics(t, y, r_arr) if r_arr is not None else None
    return LQGSimResult(t=t, x=x_out, y=y, u=u_clipped, stable=stable,
                        metrics=metrics, x_hat=None,
                        tracking_metrics=tracking_metrics, mode="state_feedback")


# ─────────────────────────────────────────────────────────────────────────────
# Output feedback (Kalman-filtered)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_output_feedback(result: LQGDesignResult, t, x0=None, x0_hat=None,
                             process_noise_std=0.0, measurement_noise_std=0.0,
                             seed=None, u_min=-1e9, u_max=1e9) -> LQGSimResult:
    """Simulate the LQG-controlled loop with u = -K·x̂ (the estimator, not
    the true state, drives the control law), in the (x, x̃) error
    coordinates of AILQG.pdf eq. 106:

        [ẋ]   [A-BK      BK   ] [x]
        [x̃̇] = [ 0      A-KfC ] [x̃]     x̃ = x - x̂

    This block-triangular construction makes the separation principle exact
    by construction rather than approximately true after numeric
    integration — the same matrix lqg_design_methods.lqg_full_closed_loop_poles
    builds for the eigenvalue check.

    Noise injection is a demonstration approximation (piecewise-constant
    Gaussian forcing at the simulation grid, not a rigorous SDE/Ito
    integration): process noise `w` enters the true-state block additively,
    measurement noise `v` enters the error block as `-Kf·v` (AILQG.pdf
    eq. 103's `-Lv` term). Both default to 0 (deterministic estimator
    convergence from a wrong initial guess).
    """
    if result.kalman is None:
        raise ValueError(
            "result has no Kalman filter — use simulate_state_feedback for "
            "a full-state-feedback design"
        )
    plant = result.plant
    nx, ny = plant.nx, plant.ny
    K, Kf = result.gains.K, result.kalman.Kf
    A, B, C = plant.A, plant.B, plant.C

    top = np.hstack([A - B @ K, B @ K])
    bottom = np.hstack([np.zeros((nx, nx)), A - Kf @ C])
    Acl = np.vstack([top, bottom])

    t = np.asarray(t, dtype=float)
    n = len(t)
    x0_ = np.zeros(nx) if x0 is None else np.asarray(x0, dtype=float)
    x0_hat_ = np.zeros(nx) if x0_hat is None else np.asarray(x0_hat, dtype=float)
    z0 = np.concatenate([x0_, x0_ - x0_hat_])

    rng = np.random.default_rng(seed)
    w = (rng.normal(0.0, process_noise_std, size=(n, nx))
         if process_noise_std > 0 else np.zeros((n, nx)))
    v = (rng.normal(0.0, measurement_noise_std, size=(n, ny))
         if measurement_noise_std > 0 else np.zeros((n, ny)))
    Bforce = np.block([[np.eye(nx), np.zeros((nx, ny))],
                       [np.zeros((nx, nx)), -Kf]])
    U = np.hstack([w, v])

    sys = StateSpace(Acl, Bforce, np.eye(2 * nx), np.zeros((2 * nx, nx + ny)))
    _, _, z_out = lsim(sys, U=U, T=t, X0=z0)
    z_out = _as_time_major(z_out, n)
    x_out = z_out[:, :nx]
    x_tilde = z_out[:, nx:]
    x_hat = x_out - x_tilde

    u = -(x_hat @ K.T)
    u_clipped = np.clip(u, u_min, u_max)
    y = x_out @ C.T + u_clipped @ plant.D.T

    stable = result.is_stable() and bool(np.all(np.real(result.kalman.estimator_poles) < -1e-9)) \
        and bool(np.all(np.isfinite(x_out)))
    metrics = compute_regulator_metrics(t, x_out, u_clipped)
    return LQGSimResult(t=t, x=x_out, y=y, u=u_clipped, stable=stable,
                        metrics=metrics, x_hat=x_hat, mode="output_feedback")


# ─────────────────────────────────────────────────────────────────────────────
# Explicit model-following (u = -K1·x - K2·xm, xm autonomous under Am)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_explicit_model_following(result: ExplicitModelFollowingResult, t,
                                      x0=None, xm0=None,
                                      u_min=-1e9, u_max=1e9) -> LQGSimResult:
    """Simulate the augmented closed loop for an ExplicitModelFollowing
    design: ẋm = Am·xm (eq. 48, autonomous — the model has no exogenous
    input) and ẋ = (A - B·K1)x - B·K2·xm, i.e. the block system

        [ẋ ]   [A-B·K1   -B·K2] [x ]
        [ẋm] = [  0        Am ] [xm]

    x0/xm0 default to 0/ones(nxm) rather than both-ones like
    simulate_state_feedback's regulator default: starting the plant at rest
    and the model away from it is what actually demonstrates tracking (the
    plant should visibly chase the model's decay), whereas a shared unit
    perturbation on both wouldn't show the two trajectories diverging and
    reconverging the way AILQG.pdf's Figures 2-3 do.
    """
    if not isinstance(result, ExplicitModelFollowingResult):
        raise ValueError(
            "simulate_explicit_model_following requires an "
            "ExplicitModelFollowingResult (from ExplicitModelFollowing.design())"
        )
    plant = result.plant
    nx = plant.nx
    nxm = result.Am.shape[0]
    K1, K2, Am = result.K1, result.K2, result.Am
    A, B, C = plant.A, plant.B, plant.C

    top = np.hstack([A - B @ K1, -B @ K2])
    bottom = np.hstack([np.zeros((nxm, nx)), Am])
    Aaug = np.vstack([top, bottom])

    t = np.asarray(t, dtype=float)
    n = len(t)
    x0_ = np.zeros(nx) if x0 is None else np.asarray(x0, dtype=float)
    xm0_ = np.ones(nxm) if xm0 is None else np.asarray(xm0, dtype=float)
    z0 = np.concatenate([x0_, xm0_])

    sys = StateSpace(Aaug, np.zeros((nx + nxm, 1)), np.eye(nx + nxm), np.zeros((nx + nxm, 1)))
    _, _, z_out = lsim(sys, U=np.zeros((n, 1)), T=t, X0=z0)
    z_out = _as_time_major(z_out, n)
    x_out = z_out[:, :nx]
    xm_out = z_out[:, nx:]

    u = -(x_out @ K1.T) - (xm_out @ K2.T)
    u_clipped = np.clip(u, u_min, u_max)
    y = x_out @ C.T + u_clipped @ plant.D.T

    am_stable = bool(np.all(np.real(np.linalg.eigvals(Am)) < -1e-9))
    stable = (result.is_stable() and am_stable
             and bool(np.all(np.isfinite(x_out))) and bool(np.all(np.isfinite(xm_out))))
    metrics = compute_regulator_metrics(t, x_out, u_clipped)
    return LQGSimResult(t=t, x=x_out, y=y, u=u_clipped, stable=stable,
                        metrics=metrics, xm=xm_out, mode="model_following")
