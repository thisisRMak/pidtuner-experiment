"""Closed-loop simulation and performance metrics.

The PID is implemented in parallel form with:
  - Selectable anti-windup: "conditional" (integral only accumulates when
    the unsaturated command is in [u_min, u_max]) — the default, matching
    prior behavior — or "back_calc" (back-calculation/tracking: the
    integral keeps accumulating but is corrected by Ka*(u_sat - u_unsat),
    per Astrom & Hagglund). See ANTI-WINDUP section below for the
    rationale and formula. Both are no-ops when the actuator never
    saturates (u_min/u_max wide enough that u_sat == u_unsat always).
  - First-order derivative filter with bandwidth N/Td (default N = 10):
    the pure D term is replaced by Kd*s / (1 + (Kd/(N*Kp))*s). Without
    this, a step setpoint hits the controller with an infinite derivative
    and the response is dominated by the actuator spike rather than the
    PID design.
  - Closed-loop stability check via the characteristic equation roots
    of 1 + C(s)*G(s) (for plants without dead time; for plants with
    dead time we approximate L with a 2nd-order Pade and check that).
    C(s) includes the same derivative filter pole the simulation ran
    with (closed_loop_poles(..., N=N_eff)), so the "stable" flag matches
    what was actually simulated rather than the ideal Kd*s design model.

─────────────────────────────────────────────────────────────────────────────
ANTI-WINDUP: conditional-integration vs. back-calculation
─────────────────────────────────────────────────────────────────────────────
Both address the same failure mode: when the actuator saturates (u_unsat
outside [u_min, u_max]), a plain integrator keeps accumulating error it
can't act on. When the error finally reverses sign, the controller has to
"unwind" that excess integral before it starts correcting the right way —
producing large overshoot and a sluggish recovery ("integrator windup").

Conditional-integration (this module's original/default method): freeze
the integral outright while saturated. Simple, cheap, no extra parameter.
Recovery is limited by how much integral had already accumulated *before*
saturation was detected each step — it doesn't actively unwind anything,
it just stops making things worse.

Back-calculation (a.k.a. "tracking anti-windup", Astrom & Hagglund):
instead of freezing, keep integrating but add a correction term
proportional to the saturation error itself:

    dI/dt = e + Ka*(u_sat - u_unsat)

When not saturated, u_sat == u_unsat, so the correction vanishes and this
is ordinary integration. When saturated, the term actively pulls the
integral back toward consistency with what the actuator can actually
deliver, at a rate set by Ka — so it unwinds faster than conditional
integration rather than merely pausing. Astrom & Hagglund size Ka via a
tracking time constant Tt: Ka = 1/Tt, with a stated rule of thumb (Advanced
PID Control, Ch. 3 "Integrator Windup" / "Back-Calculation and Tracking"):

    Tt = sqrt(Ti * Td)   (full PID — this is the book's rule of thumb)

For a pure PI controller (Td = 0) that formula degenerates to Tt = 0
(Ka = infinity), which is useless, so this module falls back to:

    Tt = Ti               (PI-only, Td = 0 — NOT from the textbook)

This fallback is this codebase's own choice, not Astrom & Hagglund's. The
book only ever mentions Tt = Ti in an unrelated context (an alternate
"interacting form" implementation) and there explicitly warns it's "often
too large" — so treat this branch as a reasonable stopgap, not a cited
recommendation.

Both branches are derived here from whatever gains were already computed
by one of the 9 tuning methods (Ti = Kp/Ki, Td = Kd/Kp — see
PIDGains.to_textbook()) via compute_back_calc_Ka().

Neither mode does anything unless the actuator actually saturates, i.e.
u_min/u_max are set tighter than the natural command range — with the
library defaults (u_min=-1e6, u_max=1e6) both modes are identical to no
anti-windup at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plant import TransferFunction, poly_mul, poly_add
from pid_tuning_methods import PIDGains


# ─────────────────────────────────────────────────────────────────────────────
# PID controller object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PIDState:
    integral: float = 0.0
    d_filt: float = 0.0
    prev_pv: float = 0.0


def compute_back_calc_Ka(gains, Ka=None):
    """Back-calculation anti-windup gain.

    Ka = 1/Tt, with Tt = sqrt(Ti*Td) for full PID — Astrom & Hagglund's
    stated rule of thumb (Advanced PID Control, Ch. 3) — or Tt = Ti when
    Td = 0 (PI-only). The PI-only branch is NOT from the textbook: it's
    this module's own fallback to avoid the degenerate Tt = sqrt(Ti*0) = 0
    (Ka = infinity) that the PID formula would otherwise give a pure PI
    controller. See the module docstring's ANTI-WINDUP section for the
    full attribution.

    Derived from the gains' own (Ti, Td) — see PIDGains.to_textbook(). An
    explicit `Ka` override bypasses the derivation entirely (Tt reported
    back as 1/Ka for display only).

    Returns (Ka, Tt). If there's no integral action (Ki == 0, Ti = inf)
    there's nothing for an integrator to wind up, so Ka = 0, Tt = inf.
    """
    if Ka is not None:
        return float(Ka), (1.0 / Ka if Ka > 0 else float("inf"))
    _, Ti, Td = gains.to_textbook()
    if not np.isfinite(Ti) or Ti <= 0:
        return 0.0, float("inf")
    Tt = np.sqrt(Ti * Td) if Td > 1e-12 else Ti
    if Tt <= 0:
        return 0.0, float("inf")
    return float(1.0 / Tt), float(Tt)


def pid_step(gains, state, sp, pv, dt, u_min, u_max, N=10.0,
             antiwindup="conditional", Ka=0.0):
    """One PID timestep with anti-windup + derivative filter.

    Derivative is applied to -pv (not to error) to avoid the "derivative
    kick" on setpoint changes — a standard practice in process control.

    `antiwindup` selects the integrator-saturation strategy:
      - "conditional" (default): freeze the integral while saturated.
      - "back_calc": keep integrating, corrected by Ka*(u_sat - u_unsat)
        (see module docstring). `Ka` is ignored in "conditional" mode.
    """
    e = sp - pv
    # Filtered derivative of -pv (avoids derivative kick on SP step)
    # tau_d = Td/N is the filter time constant; with parallel-form Kd = Kp*Td,
    # tau_d = Kd/(N*Kp). Guard against Kp=0.
    if abs(gains.Kp) > 1e-12 and abs(gains.Kd) > 1e-12 and N > 0:
        tau_d = gains.Kd / (N * abs(gains.Kp))
    else:
        tau_d = 0.0
    if tau_d > 0:
        alpha = dt / (tau_d + dt)  # first-order LPF coefficient
        dpv = (pv - state.prev_pv) / dt if dt > 0 else 0.0
        state.d_filt = state.d_filt + alpha * (dpv - state.d_filt)
        d_term = -gains.Kd * state.d_filt
    else:
        # Unfiltered: derivative of error (still kicks on SP step)
        de = (e - (sp - state.prev_pv)) / dt if dt > 0 else 0.0
        d_term = gains.Kd * de

    trial_int = state.integral + e * dt
    u_unsat = gains.Kp * e + gains.Ki * trial_int + d_term
    u = float(np.clip(u_unsat, u_min, u_max))
    if antiwindup == "back_calc":
        # Always integrate, corrected by the saturation error itself —
        # vanishes (reduces to plain integration) when u == u_unsat.
        state.integral = state.integral + (e + Ka * (u - u_unsat)) * dt
    else:
        # Conditional-integration: freeze while saturated.
        if u_min <= u_unsat <= u_max:
            state.integral = trial_int
    state.prev_pv = pv
    return u


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop stability check
# ─────────────────────────────────────────────────────────────────────────────
# Uses a 2nd-order Pade approximation for the dead time to form a finite-
# dimensional characteristic polynomial. The check is conservative —
# Pade can fail to flag instabilities at very high frequencies — but in
# practice it catches the cases that matter for this app.

def _pade_2nd(L):
    """2nd-order Pade for exp(-Ls): (12 - 6Ls + L²s²) / (12 + 6Ls + L²s²)."""
    if L <= 0:
        return np.array([1.0]), np.array([1.0])
    num = np.array([L * L, -6.0 * L, 12.0])
    den = np.array([L * L, 6.0 * L, 12.0])
    return num, den


def _filter_tau_d(gains, N):
    """τ_d = Kd/(N·|Kp|), the derivative filter time constant used by
    pid_step()/simulate_closed_loop(). Returns 0.0 (filter term disabled,
    reducing C(s)'s D-term to the ideal Kd*s) under the same fallback
    conditions as pid_step: Kp≈0, Kd≈0, or N<=0."""
    if N > 0 and abs(gains.Kp) > 1e-12 and abs(gains.Kd) > 1e-12:
        return gains.Kd / (N * abs(gains.Kp))
    return 0.0


def closed_loop_poles(plant, gains, N=10.0):
    """Roots of the closed-loop characteristic polynomial, with dead time
    approximated by 2nd-order Pade and the derivative low-pass filter
    folded into C(s) so this matches what pid_step()/simulate_closed_loop()
    actually simulate (see docs/derivative_filter.md):

        C(s) = Kp + Ki/s + Kd*s/(1 + τ_d*s),   τ_d = Kd/(N*|Kp|)

    Written over a common denominator, C(s) = nC(s)/dC(s) with:
        nC(s) = (Kp*τ_d + Kd)*s² + (Kp + Ki*τ_d)*s + Ki
        dC(s) = τ_d*s² + s

    Pass N=0 to recover the older *ideal*/unfiltered characteristic
    polynomial (nC = Kd*s² + Kp*s + Ki, dC = s) that the tuning methods
    themselves (Boyd, pole placement, ZN, ...) are derived against — useful
    for comparing "as-designed" vs. "as-simulated" stability.

    The characteristic polynomial of the unity-feedback loop is:
        1 + C(s) * G(s) * e^(-Ls) = 0
    Multiplied out:
        dC(s) * den_G * den_pade + nC(s) * num_G * num_pade = 0
    """
    tau_d = _filter_tau_d(gains, N)
    if tau_d > 0:
        nC = np.array([gains.Kp * tau_d + gains.Kd,
                        gains.Kp + gains.Ki * tau_d,
                        gains.Ki])
        dC = np.array([tau_d, 1.0, 0.0])
    else:
        nC = np.array([gains.Kd, gains.Kp, gains.Ki])  # Kd*s² + Kp*s + Ki
        dC = np.array([1.0, 0.0])                       # s
    nP, dP = _pade_2nd(plant.L)
    # 1 + (nC/dC) * (num/den) * (nP/dP) = 0
    # => dC*den*dP + nC*num*nP = 0
    char = poly_add(
        poly_mul(poly_mul(dC, plant.den), dP),
        poly_mul(poly_mul(nC, plant.num), nP),
    )
    return np.roots(char)


def is_closed_loop_stable(plant, gains, N=10.0):
    poles = closed_loop_poles(plant, gains, N=N)
    if len(poles) == 0:
        return True
    return bool(np.all(np.real(poles) < -1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClosedLoopResult:
    t: np.ndarray
    sp: np.ndarray
    y: np.ndarray
    u: np.ndarray
    e: np.ndarray
    stable: bool
    metrics: dict
    sp_kind: str = "step"
    antiwindup: str = "conditional"
    Ka: float = 0.0     # back_calc gain actually used (0 in "conditional" mode)
    Tt: float = float("inf")  # 1/Ka, for display — inf when Ka == 0
    u_min: float = -1e6  # actuator bounds used, so callers can identify
    u_max: float = 1e6   # exactly which samples of u(t) were saturated


# ─────────────────────────────────────────────────────────────────────────────
# Setpoint waveform generators
# ─────────────────────────────────────────────────────────────────────────────
# Step:    sp(t) = A for t > 0
# Ramp:    sp(t) = (A/t_end) * t — finishes at amplitude A at the end
# Pulse:   sp(t) = A for t in [t_end/4, t_end/2], 0 elsewhere
# Custom:  pass your own sp array to simulate_closed_loop directly

def make_setpoint(t, kind, amplitude=1.0):
    kind = kind.lower()
    sp = np.zeros_like(t, dtype=float)
    if kind == "step":
        sp[:] = amplitude
        sp[0] = 0.0  # discrete jump at t=0+
    elif kind == "ramp":
        # Linear ramp from 0 to amplitude over the full duration
        t_end = float(t[-1]) if len(t) else 1.0
        if t_end > 0:
            sp[:] = amplitude * (t / t_end)
        else:
            sp[:] = 0.0
    elif kind == "pulse":
        t_end = float(t[-1]) if len(t) else 1.0
        mask = (t >= 0.25 * t_end) & (t <= 0.50 * t_end)
        sp[mask] = amplitude
    else:
        raise ValueError(f"unknown setpoint kind: {kind!r}")
    return sp


def simulate_closed_loop(plant, gains, t_end=None, setpoint=1.0,
                         setpoint_kind="step", sp_array=None,
                         u_min=-1e6, u_max=1e6, N=10.0, use_d_filter=True,
                         load_step=None, load_step_time=0.0,
                         antiwindup="conditional", Ka=None):
    """Simulate the unity-feedback loop with PID controller.

    Parameters
    ----------
    plant         : TransferFunction
    gains         : PIDGains
    t_end         : final time (None → auto from plant dynamics)
    setpoint      : amplitude for step/ramp/pulse generators
    setpoint_kind : 'step' | 'ramp' | 'pulse'  (ignored if sp_array given)
    sp_array      : if provided, override the generator and use this array
    load_step     : if not None, a constant disturbance of this amplitude is
                    injected at the *plant input* for t >= load_step_time.
                    Use with setpoint=0 for a pure load-rejection test.
    load_step_time: time at which the load disturbance switches on.
    antiwindup    : 'conditional' (default) | 'back_calc' — see module
                    docstring. Only affects behavior when the actuator
                    actually saturates (u_min/u_max tighter than the
                    command range).
    Ka            : back-calculation gain override; None (default) derives
                    it from `gains` via compute_back_calc_Ka(). Ignored
                    when antiwindup='conditional'.
    """
    dt = plant.auto_dt()
    if t_end is None:
        # Pick duration from the slower of (open-loop slow mode, dead time)
        poles = plant.poles()
        real_neg = np.real(poles)[np.real(poles) < -1e-9]
        slow = 1.0 / np.min(np.abs(real_neg)) if len(real_neg) else 10.0
        t_end = max(15.0 * slow + 5.0 * plant.L, 20.0)

    t = np.arange(0.0, t_end + dt, dt)

    if sp_array is not None:
        sp = np.asarray(sp_array, dtype=float)
        if len(sp) != len(t):
            raise ValueError(
                f"sp_array length {len(sp)} doesn't match time vector "
                f"length {len(t)} (dt={dt:g}, t_end={t_end:g})"
            )
        kind = "custom"
    else:
        sp = make_setpoint(t, setpoint_kind, amplitude=float(setpoint))
        kind = setpoint_kind

    # Discretize plant
    Ad, Bd, C, D = plant.discretize(dt)
    nx = Ad.shape[0]
    Bd_flat = Bd.flatten() if nx else np.zeros(0)
    d_scalar = float(np.atleast_2d(D).flatten()[0])
    delay = int(round(plant.L / dt))

    # PID state
    state = PIDState()
    if not use_d_filter:
        N_eff = 0.0
    else:
        N_eff = N

    x = np.zeros(nx)
    y = np.zeros(len(t))
    u = np.zeros(len(t))

    Ka_used, Tt_used = ((0.0, float("inf")) if antiwindup != "back_calc"
                        else compute_back_calc_Ka(gains, Ka))

    for i in range(1, len(t)):
        u[i] = pid_step(gains, state, sp[i], y[i - 1], dt, u_min, u_max,
                        N=N_eff, antiwindup=antiwindup, Ka=Ka_used)
        j = i - 1 - delay
        u_eff = u[j] if j >= 0 else 0.0
        # Load disturbance enters at the plant input (after the controller).
        if load_step is not None and t[i] >= load_step_time:
            u_eff = u_eff + float(load_step)
        if nx:
            x = Ad @ x + Bd_flat * u_eff
            y[i] = float((C @ x).item()) + d_scalar * u_eff
        else:
            y[i] = d_scalar * u_eff

    e = sp - y

    # N_eff mirrors whatever was actually just simulated (0.0 if the
    # derivative filter was disabled), so "stable" reflects the sim, not
    # an idealization of it.
    stable = is_closed_loop_stable(plant, gains, N=N_eff)
    # Also flag clearly divergent simulations
    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e6:
        stable = False

    m = compute_metrics(t, sp, y, e, u, sp_kind=kind)
    return ClosedLoopResult(t=t, sp=sp, y=y, u=u, e=e,
                            stable=stable, metrics=m, sp_kind=kind,
                            antiwindup=antiwindup, Ka=Ka_used, Tt=Tt_used,
                            u_min=u_min, u_max=u_max)


def saturation_mask(sim):
    """Boolean array over sim.t: True where sim.u sits exactly at u_min/u_max.

    This is the only place anti-windup mode can have made a difference —
    back_calc's correction term is zero everywhere else, so `sim.Ka` is
    inert unless `saturation_mask(sim).any()`. Shared by the CLI's
    saturated-sim reporting, the GUI's legend tagging, and the response
    plot's per-sample marking so all three agree on the same definition.
    """
    return np.isclose(sim.u, sim.u_min) | np.isclose(sim.u, sim.u_max)


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(t, sp, y, e, u, sp_kind="step"):
    """Compute classic step-response metrics, plus kind-aware extras.

    Always computed:
      - IAE          : ∫|e| dt   (lower = better)
      - ITAE         : ∫ t·|e| dt   (penalizes late error)
      - |u|_peak, u_rms : control effort
      - ISU          : ∫ u² dt   (control effort / energy, P1 metric)

    Step (sp tends to a constant final value):
      - Overshoot %, Rise (10–90%), Settling (2%)

    Ramp (sp = const·t):
      - ss_error     : tracking error e(t) approached at large t.
                       For a type-1 (one integrator) loop this is finite
                       and equals (ramp slope)/Kv. For type-2+ it tends
                       to zero.

    Pulse (returns to 0 after a window):
      - peak_error   : max |e| during the pulse (analog of overshoot)
    """
    base = {
        "IAE": float("inf"), "ITAE": float("inf"),
        "u_peak": float("inf"), "u_rms": float("inf"), "u_tv": float("inf"),
        "ISU": float("inf"),
        "unstable": True, "sp_kind": sp_kind,
    }
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
        return base

    iae = float(np.trapezoid(np.abs(e), t))
    itae = float(np.trapezoid(np.abs(e) * t, t))
    out = {
        "IAE": iae, "ITAE": itae,
        "u_peak": float(np.max(np.abs(u))),
        "u_rms": float(np.sqrt(np.mean(u ** 2))),
        "u_tv": float(np.sum(np.abs(np.diff(u)))),  # total variation (smoothness)
        "ISU": float(np.trapezoid(u ** 2, t)),       # control effort/energy
        "unstable": False,
        "sp_kind": sp_kind,
    }

    if sp_kind == "step":
        final = float(sp[-1])
        if abs(final) > 1e-9:
            out["Overshoot"] = max(0.0,
                                   (float(np.max(y)) - final) / abs(final) * 100.0)
        else:
            out["Overshoot"] = 0.0

        if abs(final) > 1e-9:
            try:
                t10_mask = (np.sign(final) * y) >= (0.10 * final)
                t90_mask = (np.sign(final) * y) >= (0.90 * final)
                i10 = int(np.argmax(t10_mask)) if t10_mask.any() else -1
                i90 = int(np.argmax(t90_mask)) if t90_mask.any() else -1
                out["Rise"] = (float(t[i90] - t[i10])
                                if i10 >= 0 and i90 > i10 else float("nan"))
            except Exception:
                out["Rise"] = float("nan")
        else:
            out["Rise"] = float("nan")

        band = 0.02 * abs(final) if abs(final) > 1e-9 else 0.02
        out_of_band = np.where(np.abs(y - final) > band)[0]
        out["Settling"] = float(t[out_of_band[-1]]) if len(out_of_band) else float(t[0])

    elif sp_kind == "ramp":
        # Steady-state error: average over the last 10% of the trace.
        n_tail = max(1, int(0.1 * len(e)))
        out["ss_error"] = float(np.mean(e[-n_tail:]))
        out["max_error"] = float(np.max(np.abs(e)))

    elif sp_kind == "pulse":
        out["peak_error"] = float(np.max(np.abs(e)))
        # Residual after pulse ends — how well controller returns to 0
        n_tail = max(1, int(0.1 * len(y)))
        out["final_residual"] = float(np.mean(np.abs(y[-n_tail:])))

    return out


def format_metrics(m):
    if m.get("unstable"):
        return "UNSTABLE (diverging response)"
    common = (f"IAE = {m['IAE']:.3g},  ITAE = {m['ITAE']:.3g}\n"
              f"|u|_peak = {m['u_peak']:.3g},  u_rms = {m['u_rms']:.3g},  "
              f"ISU (control effort) = {m['ISU']:.3g}")
    kind = m.get("sp_kind", "step")
    if kind == "step":
        rise_str = (f"{m['Rise']:.3g}" if np.isfinite(m.get("Rise", np.nan)) else "—")
        return (f"Overshoot = {m['Overshoot']:.2f} %,  "
                f"Rise (10→90%) = {rise_str} s,  "
                f"Settle (2%) = {m['Settling']:.3g} s\n" + common)
    if kind == "ramp":
        return (f"ss tracking error = {m['ss_error']:.4g},  "
                f"max |e| = {m['max_error']:.4g}\n" + common)
    if kind == "pulse":
        return (f"peak |e| = {m['peak_error']:.4g},  "
                f"residual after pulse = {m['final_residual']:.4g}\n" + common)
    return common
