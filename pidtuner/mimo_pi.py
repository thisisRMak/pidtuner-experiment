"""MIMO PI control with multivariable integral anti-windup — Windup_AEN 7.pdf
§9.3 (pp. 278-283), the multivariable analog of simulate.py's SISO PID
anti-windup.

The controller is the matrix-PI structure the book's Fig 9.31/9.32 actually
draw (not the full-state-feedback LQR/LQG track's u = -Kx):

    Uc = KP·E + KI·xI,      E = r - y,      xI = ∫E dt

KP, KI are (nu, ny) gain matrices (square plants only: nu == ny here, same
restriction the book's Method II states outright — "the plant be square").
This module doesn't derive KP/KI (that's a tuning-method concern, out of
scope here); it takes them as given and focuses on what saturation does to
xI, which is genuinely a different problem in MIMO than in SISO: a single
saturated actuator doesn't correspond to a single output channel once KP/KI
have off-diagonal terms, so freezing/reset decisions are made for the whole
xI vector at once rather than per-channel. That's the chapter's opening
warning ("it would be naive to think that it suffices to just implement
SISO integrator windup for each integrator and ignore the coupling").

Reuses lqg_simulate.py's auto_t_end/compute_tracking_metrics/
format_tracking_metrics rather than reimplementing them — they're pole- and
(t, y, r)-generic, nothing LQR/LQG-specific in their bodies (auto_t_end's
own docstring already calls this out: "previously duplicated ... centralized
here"). Borrowing them from lqg_simulate.py rather than promoting them to a
new shared module is a deliberate Phase-1 scope call — that promotion would
touch simulate.py and lqg_simulate.py, both outside this module's diff.

─────────────────────────────────────────────────────────────────────────────
ANTI-WINDUP MODES
─────────────────────────────────────────────────────────────────────────────
"conditional" (default): freeze the whole xI vector whenever ANY actuator
channel saturates. Direct MIMO lift of simulate.py's conditional-integration
— simplest, no invertibility requirement, but (like its SISO counterpart)
only stops things from getting worse; it doesn't actively unwind.

"resettable" — Method II (eq. 9.121-9.123): while any channel saturates,
continually reset xI = KI⁻¹(u_sat - KP·E), so Uc lands exactly on u_sat (eq.
9.123). No extra dynamics are added — the book's own point of comparison
against Method I. Requires KI invertible; the book itself flags using a
pseudo-inverse here as an unstudied fallback ("the ramifications of it
should be studied carefully") — this module allows it but reports
`pinv_used=True` on the result so callers don't mistake it for exact.

"hanus" — Method I (eq. 9.117-9.120): an always-on correction, the MIMO
analog of simulate.py's back_calc —

    xI(t+dt) = xI(t) + [E + KP⁻¹·(u_sat - u_unsat)] dt

— which vanishes when u_sat == u_unsat (no saturation) and otherwise pulls
xI toward consistency with what the actuators can deliver, with first-order
lag dynamics set by KI·KP⁻¹ (eq. 9.120). Requires KP invertible, same
pinv-fallback/flag convention as above.

Method III (§9.4's directional/SVD-based crush/msat, for p > m sensor-rich
systems) is out of scope here — it needs a genuinely different, direction-
preserving saturation function plus an SVD of the open-loop DC gain, not an
anti-windup mode on this controller. Not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lqg_simulate import auto_t_end, compute_tracking_metrics, format_tracking_metrics
from simulate import saturation_mask  # noqa: F401 — re-exported, see saturation_mask note below


# ─────────────────────────────────────────────────────────────────────────────
# Controller object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MIMOPIGains:
    KP: np.ndarray   # (nu, ny) proportional gain, acting on E = r - y
    KI: np.ndarray   # (nu, ny) integral gain, acting on xI = integral(E) dt

    def __post_init__(self):
        self.KP = np.atleast_2d(np.asarray(self.KP, dtype=float))
        self.KI = np.atleast_2d(np.asarray(self.KI, dtype=float))
        if self.KP.shape != self.KI.shape:
            raise ValueError(
                f"KP and KI must have the same shape, got {self.KP.shape} "
                f"and {self.KI.shape}"
            )

    @property
    def nu(self):
        return self.KP.shape[0]

    @property
    def ny(self):
        return self.KP.shape[1]


def _validate_plant_gains(plant, gains):
    """Shared invariant for every entry point that assumes a square plant
    with gains sized to match — mimo_pi_closed_loop_poles and
    simulate_mimo_pi both need this, so it lives here rather than only on
    whichever one happened to be written first (a mismatched pair fed
    straight to mimo_pi_closed_loop_poles would otherwise fail with an
    opaque shape error deep in np.block instead of a clear message)."""
    if plant.nu != plant.ny:
        raise ValueError(
            f"MIMO-PI requires a square plant (nu == ny), got "
            f"nu={plant.nu}, ny={plant.ny}"
        )
    if gains.ny != plant.ny or gains.nu != plant.nu:
        raise ValueError(
            f"gains shape {gains.KP.shape} doesn't match plant "
            f"(nu={plant.nu}, ny={plant.ny})"
        )


def _inv_or_pinv(M):
    """Exact inverse when square and well-conditioned; Moore-Penrose
    pseudo-inverse otherwise, with a `used_pinv` flag. Windup_AEN 7.pdf is
    explicit that substituting a pseudo-inverse for KI⁻¹ (Method II) is a
    documented-but-unstudied fallback, not an exact equivalent — the flag
    lets callers surface that rather than silently trusting the numbers.
    """
    M = np.asarray(M, dtype=float)
    if M.shape[0] != M.shape[1]:
        return np.linalg.pinv(M), True
    try:
        cond = np.linalg.cond(M)
        if not np.isfinite(cond) or cond > 1e10:
            return np.linalg.pinv(M), True
        return np.linalg.inv(M), False
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M), True


def mimo_pi_step(gains, integral, r, y, dt, u_min, u_max,
                 antiwindup="conditional", inv=None):
    """One MIMO-PI timestep with anti-windup. Returns (u, new_integral) —
    functional rather than mutate-in-place, so callers thread `integral`
    through the loop explicitly (see simulate_mimo_pi).

    `inv` is whichever precomputed inverse the chosen mode needs (KP⁻¹ for
    "hanus", KI⁻¹ for "resettable", unused for "conditional") — computed
    once per simulation via _inv_or_pinv, not every step. See module
    docstring for the three modes' formulas (eq. 9.117-9.123).
    """
    e = np.asarray(r, dtype=float) - np.asarray(y, dtype=float)
    trial_int = integral + e * dt
    u_unsat = gains.KP @ e + gains.KI @ trial_int
    u = np.clip(u_unsat, u_min, u_max)

    if antiwindup == "hanus":
        # eq. 9.117-9.120 — always-on, vanishes when u == u_unsat.
        new_integral = trial_int + inv @ (u - u_unsat) * dt
    elif antiwindup == "resettable":
        if np.any(u != u_unsat):
            new_integral = inv @ (u - gains.KP @ e)  # eq. 9.121 -> Uc == u_sat (eq. 9.123)
        else:
            new_integral = trial_int
    else:  # "conditional"
        new_integral = trial_int if np.all(u == u_unsat) else integral  # freeze whole vector
    return u, new_integral


# ─────────────────────────────────────────────────────────────────────────────
# Linearized (unsaturated) closed-loop poles — the "as designed" analog of
# simulate.py's closed_loop_poles() / lqg_simulate's is_stable(), used for
# the stability flag and for auto-sizing t_end. Ignores saturation, same
# split as both existing tracks between "as designed" and "as simulated".
# Named mimo_pi_closed_loop_poles (not closed_loop_poles) so it can't be
# mistaken for simulate.py's same-named-but-different-signature SISO version
# if both ever end up imported into the same file.
# ─────────────────────────────────────────────────────────────────────────────

def mimo_pi_closed_loop_poles(plant, gains):
    """Eigenvalues of the augmented linear system [x; xI] under
    Uc = KP·(r - y) + KI·xI with r = 0 (homogeneous), ignoring saturation:

        u = M (-KP·C·x + KI·xI),   M = (I + KP·D)⁻¹
        ẋ  = A x + B u
        ẋI = -(C x + D u)
    """
    _validate_plant_gains(plant, gains)
    A, B, C, D = plant.A, plant.B, plant.C, plant.D
    nu = plant.nu
    M, _ = _inv_or_pinv(np.eye(nu) + gains.KP @ D)
    Au = -M @ gains.KP @ C   # du/dx
    Bu = M @ gains.KI        # du/dxI
    Acl = np.block([
        [A + B @ Au,        B @ Bu],
        [-(C + D @ Au),    -(D @ Bu)],
    ])
    return np.linalg.eigvals(Acl)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mimo_pi_metrics(t, r, y, u):
    """IAE/ITAE on the tracking error (L1 norm across channels, matching
    simulate.py's scalar IAE/ITAE), ISU/u_peak on control effort (same
    definitions as lqg_simulate.compute_regulator_metrics — kept as a local
    two-line computation rather than imported, since factoring it out would
    mean editing lqg_simulate.py's existing metrics function), plus
    per-channel step-response metrics via lqg_simulate.compute_tracking_metrics
    — reused rather than reimplemented since it's already MIMO-general and
    sign-aware.
    """
    base = {"unstable": True, "IAE": float("inf"), "ITAE": float("inf"),
            "ISU": float("inf"), "u_peak": float("inf"), "channels": []}
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
        return base
    e = r - y
    e_norm = np.sum(np.abs(e), axis=1)
    return {
        "unstable": False,
        "IAE": float(np.trapezoid(e_norm, t)),
        "ITAE": float(np.trapezoid(e_norm * t, t)),
        "ISU": float(np.trapezoid(np.sum(u ** 2, axis=1), t)),
        "u_peak": float(np.max(np.abs(u))) if u.size else 0.0,
        "channels": compute_tracking_metrics(t, y, r),
    }


def format_mimo_pi_metrics(m):
    if m.get("unstable"):
        return "UNSTABLE (diverging response)"
    head = (f"IAE = {m['IAE']:.3g},  ITAE = {m['ITAE']:.3g},  "
            f"ISU (control effort) = {m['ISU']:.3g},  |u|_peak = {m['u_peak']:.3g}")
    return head + "\n" + format_tracking_metrics(m["channels"])


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MIMOPIResult:
    t: np.ndarray
    r: np.ndarray
    y: np.ndarray
    u: np.ndarray
    e: np.ndarray
    stable: bool
    metrics: dict
    u_min: np.ndarray   # always populated by simulate_mimo_pi (its only
    u_max: np.ndarray   # constructor) -> required, not Optional
    antiwindup: str = "conditional"
    pinv_used: bool = False  # True if the mode's required inverse (KP or
                              # KI, per antiwindup) fell back to a pinv


def simulate_mimo_pi(plant, gains, t=None, t_end=None, r=None, x0=None,
                     u_min=None, u_max=None,
                     antiwindup="conditional") -> MIMOPIResult:
    """Simulate the MIMO unity-feedback loop Uc = KP·(r-y) + KI·∫(r-y)dt
    against a square StateSpacePlant, with actuator saturation and one of
    three anti-windup strategies — see module docstring for the formulas.

    plant       : StateSpacePlant, must be square (nu == ny)
    gains       : MIMOPIGains(KP, KI), both (nu, ny)
    t           : explicit time vector; if None, built from plant.auto_dt()
                  and t_end (t_end itself auto-picked from the linearized/
                  unsaturated closed-loop poles via mimo_pi_closed_loop_poles
                  + lqg_simulate.auto_t_end when not given)
    r           : reference — (ny,) for a step to that vector (r[0] forced
                  to 0, matching simulate.py's step generator), or (n, ny)
                  for a custom trace. Defaults to a unit step on every
                  channel. For a pure regulator test (x0 perturbed, no
                  tracking), pass r=np.zeros(ny) explicitly.
    x0          : initial plant state, defaults to zeros(nx)
    u_min/u_max : (nu,) actuator bounds; default ±1e6 (effectively
                  unsaturated, matching the PID track's defaults)
    antiwindup  : "conditional" (default) | "resettable" | "hanus"
    """
    _validate_plant_gains(plant, gains)
    nx, nu, ny = plant.nx, plant.nu, plant.ny

    u_min = np.full(nu, -1e6) if u_min is None else np.asarray(u_min, dtype=float)
    u_max = np.full(nu, 1e6) if u_max is None else np.asarray(u_max, dtype=float)

    poles = None  # computed at most once, reused for t_end sizing and the stability check
    if t is None:
        dt = plant.auto_dt()
        if t_end is None:
            poles = mimo_pi_closed_loop_poles(plant, gains)
            t_end = auto_t_end(poles)
        t = np.arange(0.0, t_end + dt, dt)
    else:
        t = np.asarray(t, dtype=float)
        dt = float(t[1] - t[0]) if len(t) > 1 else plant.auto_dt()
    n = len(t)

    if r is None:
        r = np.ones(ny)
    r = np.atleast_2d(np.asarray(r, dtype=float))
    if r.shape[0] == n and r.shape[1] == ny:
        r_arr = r.copy()
    elif r.size == ny:
        r_arr = np.tile(r.reshape(1, ny), (n, 1))
        r_arr[0] = 0.0  # discrete step at t=0+, matching simulate.py's make_setpoint
    else:
        raise ValueError(
            f"r shape {r.shape} doesn't match ny={ny} (as a step) or "
            f"(n={n}, ny={ny}) (as a custom trace)"
        )

    Ad, Bd, C, D = plant.discretize(dt)
    x = np.zeros(nx) if x0 is None else np.asarray(x0, dtype=float)

    y = np.zeros((n, ny))
    u = np.zeros((n, nu))
    y[0] = C @ x  # D @ u[0] is always 0 here (u[0] is the zero row above)

    integral = np.zeros(ny)
    inv, pinv_used = None, False
    if antiwindup == "hanus":
        inv, pinv_used = _inv_or_pinv(gains.KP)
    elif antiwindup == "resettable":
        inv, pinv_used = _inv_or_pinv(gains.KI)

    for i in range(1, n):
        u[i], integral = mimo_pi_step(gains, integral, r_arr[i], y[i - 1], dt,
                                      u_min, u_max, antiwindup=antiwindup, inv=inv)
        x = Ad @ x + Bd @ u[i]
        y[i] = C @ x + D @ u[i]

    e = r_arr - y

    if poles is None:
        poles = mimo_pi_closed_loop_poles(plant, gains)
    stable = bool(np.all(np.real(poles) < -1e-9))
    if not np.all(np.isfinite(y)) or (y.size and np.max(np.abs(y)) > 1e6):
        stable = False

    metrics = compute_mimo_pi_metrics(t, r_arr, y, u)
    return MIMOPIResult(t=t, r=r_arr, y=y, u=u, e=e, stable=stable,
                        metrics=metrics, u_min=u_min, u_max=u_max,
                        antiwindup=antiwindup, pinv_used=pinv_used)
