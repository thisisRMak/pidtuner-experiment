"""Six PID tuning methods, each returning PIDGains in parallel form.

Internal controller form throughout:
    D(s) = Kp + Ki/s + Kd*s
which relates to textbook form (Kp, Ti, Td) via Ki = Kp/Ti, Kd = Kp*Td.

Each tuning function returns a TuningResult with the gains plus enough
metadata for the UI to display the identified parameters that drove them
(e.g. K, tau, L for FOPDT methods; Ku, Pu for ZN-II; cancelled poles for
pole-cancellation).

Methods implemented:
  1. Stable pole cancellation  (course quick-and-dirty method, HO PS4)
  2. Ziegler–Nichols Method I  (reaction curve, FOPDT-based)
  3. Ziegler–Nichols Method II (ultimate gain, Ku/Pu-based)
  4. AMIGO                     (Aström–Hägglund 2006, HO13)
  5. SIMC                      (Skogestad, HO14)
  6. Boyd                      (convex-concave ECC 2013)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from plant import TransferFunction, poly_mul
from identify import FOPDT


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PIDGains:
    """Parallel form: D(s) = Kp + Ki/s + Kd*s."""

    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0

    @classmethod
    def from_textbook(cls, Kp, Ti=None, Td=None):
        """Convert (Kp, Ti, Td) textbook form to (Kp, Ki, Kd) parallel form."""
        Ki = Kp / Ti if Ti and np.isfinite(Ti) and Ti > 0 else 0.0
        Kd = Kp * Td if Td else 0.0
        return cls(Kp=float(Kp), Ki=float(Ki), Kd=float(Kd))

    def to_textbook(self):
        Ti = self.Kp / self.Ki if abs(self.Ki) > 1e-12 else float("inf")
        Td = self.Kd / self.Kp if abs(self.Kp) > 1e-12 else 0.0
        return self.Kp, Ti, Td

    def pretty(self):
        Ti = self.Kp / self.Ki if abs(self.Ki) > 1e-12 else float("inf")
        Td = self.Kd / self.Kp if abs(self.Kp) > 1e-12 else 0.0
        Ti_str = "∞" if not np.isfinite(Ti) else f"{Ti:.4g} s"
        return (f"Kp = {self.Kp:.4g},  Ki = {self.Ki:.4g},  Kd = {self.Kd:.4g}\n"
                f"   (Ti = {Ti_str},  Td = {Td:.4g} s)")


@dataclass
class TuningResult:
    method: str
    gains: PIDGains
    # Optional context to display in the UI
    fopdt: Optional[FOPDT] = None
    Ku: Optional[float] = None
    Pu: Optional[float] = None
    omega_180: Optional[float] = None
    cancelled_poles: list = field(default_factory=list)
    free_param: dict = field(default_factory=dict)  # e.g. {'tau_c': 0.5}
    notes: str = ""


def halve_gains(result):
    """Return a new TuningResult with all gains halved.

    The Emami / Aström recommendation is to halve Kp specifically (ZN's
    rules were tuned for disturbance rejection and overshoot heavily on
    setpoint tracking). Some practitioners halve all three gains — same
    spirit, slightly different effect on derivative action. We halve all
    three for simplicity; the override is a free knob anyway.
    """
    new_gains = PIDGains(
        Kp=result.gains.Kp * 0.5,
        Ki=result.gains.Ki * 0.5,
        Kd=result.gains.Kd * 0.5,
    )
    return TuningResult(
        method=result.method + " ½",
        gains=new_gains,
        fopdt=result.fopdt,
        Ku=result.Ku,
        Pu=result.Pu,
        omega_180=result.omega_180,
        cancelled_poles=list(result.cancelled_poles),
        free_param=dict(result.free_param),
        notes=(result.notes + "\nAll gains halved post-tuning.").strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stable pole cancellation
# ─────────────────────────────────────────────────────────────────────────────
# For plant G(s) = N(s) / [(s+p1)(s+p2) * R(s)], the PID controller
#     C(s) = Kd * (s+p1)(s+p2) / s
# cancels the two chosen plant poles, leaving an open-loop transfer
# function of  Kd * N(s) / [s * R(s)]. The Kp, Ki gains are then read
# off by expanding:
#   C(s) = Kd*s + (p1+p2)*Kd + (p1*p2)*Kd/s
# so   Kp = (p1+p2)*Kd,  Ki = p1*p2*Kd.
# Kd is the user's free parameter and sets the closed-loop speed.
#
# RHP poles must never be cancelled — imperfect cancellation leaves a
# hidden unstable mode (covered explicitly in lecture 4/23/26 and in
# HO PS4).

def tune_pole_cancellation(plant, p1, p2, Kd=1.0):
    """Cancel two stable plant poles at -p1, -p2.

    Plant pole at s = -p1 means the corresponding factor is (s + p1).
    `p1` and `p2` here are the *positive* values (e.g. for a pole at
    s = -2, pass p1 = 2).
    """
    for p, name in ((p1, "p1"), (p2, "p2")):
        if p <= 0:
            raise ValueError(
                f"Pole-cancellation requires a stable pole, but {name}={p} "
                f"corresponds to an RHP or imaginary-axis pole. "
                f"Refusing — see lecture: 'never, ever do RHP cancellations.'"
            )
    Kp = (p1 + p2) * Kd
    Ki = p1 * p2 * Kd
    return TuningResult(
        method="Stable pole cancellation",
        gains=PIDGains(Kp=Kp, Ki=Ki, Kd=Kd),
        cancelled_poles=[-float(p1), -float(p2)],
        free_param={"Kd": float(Kd)},
        notes=(f"Controller zeros placed at s = -{p1:g}, s = -{p2:g} to "
               f"cancel chosen plant poles. Kd is free and sets the "
               f"closed-loop integrator gain (open loop reduces to "
               f"K·Kd/s after cancellation)."),
    )


def select_slowest_stable_poles(plant):
    """Pick the two slowest stable real-part poles for default cancellation.

    Returns (p1, p2) as positive values (so the pole is at s = -p1).
    For complex-conjugate pairs we cancel the pair together (using both
    members of the pair), so we look for real-pole pairs first.
    """
    poles = plant.poles()
    if len(poles) == 0:
        raise ValueError("Plant has no finite poles — nothing to cancel.")

    stable = poles[np.real(poles) < -1e-9]
    if len(stable) < 2:
        raise ValueError(
            f"Plant has only {len(stable)} stable pole(s); pole-cancellation "
            "PID needs at least 2. Try a different method."
        )

    # Match conjugate pairs together (the controller zeros must be real
    # or complex-conjugate to give real gains).
    sorted_by_speed = sorted(stable, key=lambda p: abs(np.real(p)))

    # Look for a complex conjugate pair in the slowest few
    for i, p in enumerate(sorted_by_speed):
        if abs(np.imag(p)) > 1e-9:
            # find its conjugate
            for j, q in enumerate(sorted_by_speed):
                if j != i and abs(p - np.conj(q)) < 1e-6:
                    # cancel this pair
                    # (s + p1)(s + p2) where p1 = -p, p2 = -conj(p)
                    # -> s^2 - 2*Re(p)*s + |p|^2
                    # For our API we return (p1, p2) such that
                    # we'll need to handle complex pair specially.
                    return float(-np.real(p) + abs(np.imag(p)) * 1j), \
                           float(-np.real(p) - abs(np.imag(p)) * 1j)

    # Pick two slowest *real* poles
    real_only = [float(-np.real(p)) for p in sorted_by_speed
                 if abs(np.imag(p)) < 1e-9]
    if len(real_only) >= 2:
        # Cancel the two SLOWEST poles (smallest |real part|), per lecture
        # discussion: cancelling the slow ones gives the fastest response.
        return real_only[0], real_only[1]
    # Fallback: cancel two slowest poles, even if complex
    return float(-np.real(sorted_by_speed[0])), float(-np.real(sorted_by_speed[1]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ziegler–Nichols Method I  (reaction curve / FOPDT)
# ─────────────────────────────────────────────────────────────────────────────
# Plant fit to A * exp(-L*s) / (tau*s + 1), with reaction rate R = A/tau.
# Ziegler & Nichols (1942) table for PID:
#   Kp = 1.2 / (R*L) = 1.2 * tau / (A * L)
#   Ti = 2 * L
#   Td = 0.5 * L

def tune_zn_method_1(fopdt):
    """ZN-I from a fitted FOPDT."""
    L = max(fopdt.L, 1e-6)
    if fopdt.K == 0 or fopdt.tau == 0:
        raise ValueError("ZN-I needs a non-zero K and tau from the FOPDT fit.")
    Kp = 1.2 * fopdt.tau / (fopdt.K * L)
    Ti = 2.0 * L
    Td = 0.5 * L
    gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
    return TuningResult(
        method="Ziegler–Nichols I",
        gains=gains,
        fopdt=fopdt,
        notes=("ZN-I reaction-curve rules (1942). Designed for "
               "disturbance rejection, tends to overshoot on setpoint "
               "tracking. Apply the 'Halve gains' toggle for a more "
               "conservative tracking response."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ziegler–Nichols Method II  (ultimate-gain)
# ─────────────────────────────────────────────────────────────────────────────
#   Kp = 0.6 * Ku
#   Ti = 0.5 * Pu
#   Td = 0.125 * Pu

def tune_zn_method_2(Ku, Pu):
    Kp = 0.6 * Ku
    Ti = 0.5 * Pu
    Td = 0.125 * Pu
    gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
    return TuningResult(
        method="Ziegler–Nichols II",
        gains=gains,
        Ku=float(Ku),
        Pu=float(Pu),
        notes=(f"Ultimate-gain rules. Ku = {Ku:.4g}, Pu = {Pu:.4g} s. "
               f"Like ZN-I, designed for disturbance rejection; apply "
               f"the 'Halve gains' toggle for tracking."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. AMIGO  (HO13)
# ─────────────────────────────────────────────────────────────────────────────
# For G(s) = A * exp(-Ls) / (tau*s + 1):
#   Kp = (1/A) * (0.2 + 0.45 * tau/L)
#   Ti = ((0.4*L + 0.8*tau) / (L + 0.1*tau)) * L
#   Td = (0.5*tau / (0.3*L + tau)) * L
# For integrating G(s) = A*exp(-Ls)/s:
#   Kp = 0.45/A,  Ti = 8L,  Td = 0.5L

def tune_amigo(fopdt, integrating=False):
    L = max(fopdt.L, 1e-6)
    A = fopdt.K
    if abs(A) < 1e-12:
        raise ValueError("AMIGO needs non-zero gain A.")
    if integrating:
        Kp = 0.45 / A
        Ti = 8.0 * L
        Td = 0.5 * L
        notes = "AMIGO rules for integrating + dead-time process."
    else:
        tau = max(fopdt.tau, 1e-9)
        Kp = (1.0 / A) * (0.2 + 0.45 * tau / L)
        Ti = ((0.4 * L + 0.8 * tau) / (L + 0.1 * tau)) * L
        Td = (0.5 * tau / (0.3 * L + tau)) * L
        notes = ("AMIGO rules (Aström & Hägglund 2006). Targets ~20% "
                 "overshoot with a robustness constraint on the "
                 "sensitivity function — more conservative than ZN-I.")
    gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
    return TuningResult(
        method="AMIGO" + (" (integrating)" if integrating else ""),
        gains=gains, fopdt=fopdt, notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIMC  (HO14, Skogestad)
# ─────────────────────────────────────────────────────────────────────────────
# For FOPDT G(s) = A*exp(-Ls)/(tau*s + 1):
#   Kp = (1/A) * tau / (tau_c + L)
#   Ti = min(tau, 4*(tau_c + L))
# For SOPDT G(s) = A*exp(-Ls)/[(tau1*s+1)(tau2*s+1)]  (SIMC PID form):
#   Kp = (1/A) * tau1 / (tau_c + L)
#   Ti = min(tau1, 4*(tau_c + L))
#   Td = tau2
# The single tuning knob is tau_c; default tau_c = L is "fast and robust."

def tune_simc(fopdt, tau_c=None, tau2=None):
    L = max(fopdt.L, 1e-6)
    if abs(fopdt.K) < 1e-12:
        raise ValueError("SIMC needs non-zero K.")
    if tau_c is None:
        tau_c = L

    tau = max(fopdt.tau, 1e-9)
    Kp = (1.0 / fopdt.K) * tau / (tau_c + L)
    Ti = min(tau, 4.0 * (tau_c + L))

    if tau2 is not None and tau2 > 0:
        Td = float(tau2)
        notes = (f"SIMC (Skogestad) for second-order plant, τ₁={tau:.3g} s, "
                 f"τ₂={tau2:.3g} s, τc={tau_c:.3g} s.")
    else:
        # FOPDT case — Skogestad does not require derivative action.
        # Include it with Td = 0 for compatibility with the unified UI.
        Td = 0.0
        notes = (f"SIMC (Skogestad) PI from FOPDT, τc={tau_c:.3g} s. "
                 f"For PID, supply a τ₂ from a second-order fit.")
    gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
    return TuningResult(
        method="SIMC", gains=gains, fopdt=fopdt,
        free_param={"tau_c": float(tau_c)}, notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Boyd convex-concave PID design
# ─────────────────────────────────────────────────────────────────────────────
# Hast, Aström, Bernhardsson & Boyd, "PID Design by Convex-Concave
# Optimization" (ECC 2013, Stanford). Maximize integral gain Ki subject
# to robustness constraints |1 + L(jω)| >= 1/Ms (sensitivity bound) and
# |L(jω) - c_T| >= r_T (complementary-sensitivity bound) over a frequency
# grid. L(jω) is affine in (Kp, Ki, Kd) at each ω, so linearizing the
# concave |L - c| at the current iterate yields a linear program. Iterate
# to convergence.

def _boyd_omega_grid(plant, n_freq):
    poles = plant.poles()
    zeros = plant.zeros()
    feats = np.concatenate([np.abs(poles), np.abs(zeros)])
    feats = feats[feats > 1e-9]
    if len(feats) == 0:
        omega_lo, omega_hi = 1e-3, 1e3
    else:
        omega_lo = max(np.min(feats) * 1e-3, 1e-9)
        omega_hi = max(np.max(feats) * 1e3, omega_lo * 100)
    if plant.L > 0:
        omega_hi = max(omega_hi, 100.0 / plant.L)
    return np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_freq)


def tune_boyd(plant, Ms=1.4, Mt=1.4, seed_gains=None,
              use_derivative=True, n_freq=300, max_iter=40, tol=1e-4):
    """Convex-concave PID design against the true plant frequency response.

    Ms : peak sensitivity bound (typically 1.2–2.0; smaller = more robust)
    Mt : peak complementary-sensitivity bound (typically 1.2–2.0)
    seed_gains : optional PIDGains used to seed/scale; if None, defaults to
                 (1, 0.1, 0) modulated by the plant DC gain.
    """
    from scipy.optimize import linprog

    # Sign-flip for negative-DC plants so the loop has the right phase.
    sign = 1.0 if plant.dc_gain() >= 0 else -1.0
    plant_eff = TransferFunction(num=plant.num * sign, den=plant.den, L=plant.L)

    omega = _boyd_omega_grid(plant_eff, n_freq)
    P = plant_eff.freq_response(omega)

    # L(jω) = (Kp + Ki/(jω) + Kd*jω) * P(jω) is affine in (Kp, Ki, Kd)
    cols = [P, P / (1j * omega)]
    if use_derivative:
        cols.append(P * (1j * omega))
    basis = np.stack(cols, axis=1)
    n_var = basis.shape[1]

    Mt2 = Mt * Mt
    # Robustness circles in the Nyquist plane: |L(jω) - c| >= r
    #   Sensitivity:  c = -1, r = 1/Ms
    #   Comp. sens.:  c = -Mt²/(Mt²-1), r = Mt/(Mt²-1)
    circles = [(-1.0 + 0j, 1.0 / Ms),
               (-Mt2 / (Mt2 - 1.0) + 0j, Mt / (Mt2 - 1.0))]

    # Seed: caller may supply seed (e.g. from SIMC). Otherwise crude default.
    if seed_gains is None:
        # Crude default: gain of order 1/|dc_gain|, integral much smaller
        K_dc = max(abs(plant.dc_gain()), 1e-3)
        seed_vec = np.array([1.0 / K_dc, 0.1 / K_dc, 0.0])[:n_var]
    else:
        seed_vec = np.array([seed_gains.Kp, seed_gains.Ki, seed_gains.Kd])[:n_var]
    seed_vec = np.abs(seed_vec)  # work in positive-gain space; sign re-applied below
    seed_abs = np.maximum(seed_vec, 1e-6)
    bounds = [(0.0, float(b)) for b in seed_abs * 100.0]

    obj = np.zeros(n_var)
    obj[1] = -1.0  # maximize Ki

    alpha = np.maximum(seed_vec, 0.0)
    last_ki = alpha[1] if alpha[1] > 0 else -np.inf

    for _ in range(max_iter):
        L_k = basis @ alpha
        rows_A, rows_b = [], []
        for c, r in circles:
            d_k = L_k - c
            mag = np.maximum(np.abs(d_k), 1e-12)
            unit = np.conj(d_k) / mag
            # Linearize |L - c| at alpha: |L - c| ≈ Re(unit·(L - c)).
            # Constraint Re(unit·(L - c)) >= r  ⇒  -Re(unit·L) <= -r + Re(unit·c)
            coef = (unit[:, None] * basis).real
            rhs = r + (unit * c).real
            rows_A.append(-coef)
            rows_b.append(-rhs)

        res = linprog(obj, A_ub=np.vstack(rows_A),
                      b_ub=np.concatenate(rows_b),
                      bounds=bounds, method="highs")
        if not res.success:
            break
        alpha_new = res.x
        ki_new = alpha_new[1]
        if last_ki > 0 and abs(ki_new - last_ki) / max(last_ki, 1e-9) < tol:
            alpha = alpha_new
            break
        alpha, last_ki = alpha_new, ki_new

    Kp = float(alpha[0]) * sign
    Ki = float(alpha[1]) * sign
    Kd = float(alpha[2]) * sign if use_derivative else 0.0
    return TuningResult(
        method="Boyd convex-concave",
        gains=PIDGains(Kp=Kp, Ki=Ki, Kd=Kd),
        free_param={"Ms": float(Ms), "Mt": float(Mt)},
        notes=(f"Maximized Ki subject to |S|∞ ≤ {Ms:g} and "
               f"|T|∞ ≤ {Mt:g} via linearized convex-concave iteration. "
               f"Operates on the true plant frequency response."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cohen–Coon  (1953)
# ─────────────────────────────────────────────────────────────────────────────
# Open-loop reaction-curve rules, like ZN-I but using both tau and L explicitly
# (ZN-I collapses them into the single slope R = A/tau). Derived to give a
# quarter-amplitude-decay load response, with an explicit dead-time correction
# that makes them noticeably better than ZN-I on dead-time-dominant plants
# (large L/tau). For FOPDT G(s) = A*exp(-Ls)/(tau*s + 1), the PID row is
#   Kp = (1/A)(tau/L)(4/3 + L/(4 tau))
#   Ti = L (32 + 6 L/tau) / (13 + 8 L/tau)
#   Td = 4 L / (11 + 2 L/tau)
# Reference: Cohen & Coon, Trans. ASME 75, 827-834 (1953).

def tune_cohen_coon(fopdt):
    """Cohen–Coon PID from a fitted FOPDT."""
    L = max(fopdt.L, 1e-6)
    A = fopdt.K
    if abs(A) < 1e-12 or fopdt.tau == 0:
        raise ValueError("Cohen–Coon needs non-zero A and tau from the FOPDT fit.")
    tau = max(fopdt.tau, 1e-9)
    r = L / tau  # fractional dead time

    Kp = (1.0 / A) * (tau / L) * (4.0 / 3.0 + r / 4.0)
    Ti = L * (32.0 + 6.0 * r) / (13.0 + 8.0 * r)
    Td = 4.0 * L / (11.0 + 2.0 * r)
    gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
    return TuningResult(
        method="Cohen–Coon",
        gains=gains,
        fopdt=fopdt,
        notes=("Cohen–Coon reaction-curve rules (1953). Like ZN-I but uses "
               "tau and L separately, with a dead-time correction that helps "
               "on delay-dominant plants (large L/tau). Targets quarter-decay "
               "load response; can be aggressive on setpoint tracking."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Chien–Hrones–Reswick  (CHR, 1952)
# ─────────────────────────────────────────────────────────────────────────────
# A refinement of ZN-I that lets the user choose (a) whether the loop is tuned
# for setpoint tracking ("servo") or load-disturbance rejection ("regulator"),
# and (b) a "quickest response with 0% overshoot" vs "quickest with 20%
# overshoot" damping target. Four PID rows result, all driven by the FOPDT fit.
# Reference: Chien, Hrones & Reswick, Trans. ASME 74, 175-185 (1952).
#
# PID rows (K, tau, L from the FOPDT fit):
#   setpoint  0%:  Kp = 0.60 tau/(K L),  Ti = tau,    Td = 0.50 L
#   setpoint 20%:  Kp = 0.95 tau/(K L),  Ti = 1.40 tau, Td = 0.47 L
#   load      0%:  Kp = 0.95 tau/(K L),  Ti = 2.40 L,  Td = 0.42 L
#   load     20%:  Kp = 1.20 tau/(K L),  Ti = 2.00 L,  Td = 0.42 L

_CHR_PID_TABLE = {
    # (response, overshoot_pct): (Kp_coef, Ti_expr, Td_coef)
    #   Kp = Kp_coef * tau/(K L);  Ti per expr;  Td = Td_coef * L
    ("setpoint", 0):  (0.60, ("tau", 1.00), 0.50),
    ("setpoint", 20): (0.95, ("tau", 1.40), 0.47),
    ("load", 0):      (0.95, ("L",   2.40), 0.42),
    ("load", 20):     (1.20, ("L",   2.00), 0.42),
}


def tune_chr(fopdt, response="setpoint", overshoot=0):
    """Chien–Hrones–Reswick PID from a fitted FOPDT.

    response  : "setpoint" (servo / tracking) or "load" (regulator / rejection)
    overshoot : 0 or 20  (the design's damping target, in percent)
    """
    response = response.lower()
    overshoot = int(overshoot)
    key = (response, overshoot)
    if key not in _CHR_PID_TABLE:
        raise ValueError(
            f"CHR: unknown variant {key}. response must be 'setpoint' or "
            f"'load'; overshoot must be 0 or 20.")
    L = max(fopdt.L, 1e-6)
    A = fopdt.K
    if abs(A) < 1e-12 or fopdt.tau == 0:
        raise ValueError("CHR needs non-zero A and tau from the FOPDT fit.")
    tau = max(fopdt.tau, 1e-9)

    Kp_coef, (Ti_base, Ti_coef), Td_coef = _CHR_PID_TABLE[key]
    Kp = Kp_coef * tau / (A * L)
    Ti = Ti_coef * (tau if Ti_base == "tau" else L)
    Td = Td_coef * L
    gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
    pretty_resp = "setpoint tracking" if response == "setpoint" else "load rejection"
    return TuningResult(
        method=f"CHR ({response} {overshoot}%)",
        gains=gains,
        fopdt=fopdt,
        free_param={"response": response, "overshoot": overshoot},
        notes=(f"Chien–Hrones–Reswick (1952), tuned for {pretty_resp} with a "
               f"{overshoot}% overshoot target. Unlike ZN-I, CHR uses tau and "
               f"L separately and lets you pick servo vs. regulator behaviour."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Tyreus–Luyben  (Luyben & Luyben, 1997)
# ─────────────────────────────────────────────────────────────────────────────
# The ultimate-gain rule popularized in Luyben & Luyben, "Essentials of Process
# Control" (McGraw-Hill, 1997). Like ZN-II it works from (Ku, Pu), but it is
# deliberately more conservative: a smaller proportional gain and a much longer
# integral time give larger stability margins and far less oscillation than ZN,
# at the cost of slower response. Well-suited to the long-dead-time / integrating
# loops common in process control.
#   PID:  Kp = Ku/2.2,  Ti = 2.2 Pu,  Td = Pu/6.3
#   PI :  Kp = Ku/3.2,  Ti = 2.2 Pu
# Reference: Tyreus & Luyben, Ind. Eng. Chem. Res. 31, 2625-2628 (1992); also
# Luyben & Luyben (1997).

def tune_tyreus_luyben(Ku, Pu, use_derivative=True):
    """Tyreus–Luyben from the ultimate gain/period."""
    if use_derivative:
        Kp = Ku / 2.2
        Ti = 2.2 * Pu
        Td = Pu / 6.3
        struct = "PID"
    else:
        Kp = Ku / 3.2
        Ti = 2.2 * Pu
        Td = 0.0
        struct = "PI"
    gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
    return TuningResult(
        method="Tyreus–Luyben" + ("" if use_derivative else " (PI)"),
        gains=gains,
        Ku=float(Ku),
        Pu=float(Pu),
        notes=(f"Tyreus–Luyben {struct} (Luyben & Luyben, 1997). "
               f"Ku = {Ku:.4g}, Pu = {Pu:.4g} s. A conservative ultimate-gain "
               f"rule: larger margins and far less overshoot than ZN-II, but "
               f"slower. No 'halve gains' needed — it is already detuned."),
    )
