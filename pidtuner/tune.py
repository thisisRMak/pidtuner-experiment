"""Nine PID tuning methods, each returning PIDGains in parallel form.

Internal controller form throughout:
    D(s) = Kp + Ki/s + Kd*s
which relates to textbook form (Kp, Ti, Td) via Ki = Kp/Ti, Kd = Kp*Td.

Each tuning function returns a TuningResult with the gains plus enough
metadata for the UI to display the identified parameters that drove them
(e.g. K, tau, L for FOPDT methods; Ku, Pu for ZN-II; cancelled poles for
pole-cancellation).

Each function is a thin wrapper instantiating the corresponding class in
tuning_methods.py and calling .tune(). app.py, cli.py, and compare.py now
call the tuning_methods classes directly; these wrappers remain only
because test_pid_tuner.py's ~72 assertions call them by name.

TODO(future): consider deleting these wrapper functions and rewriting
test_pid_tuner.py to instantiate tuning_methods classes directly instead
(e.g. Simc(fopdt, tau_c=...).tune() rather than tune_simc(fopdt, tau_c=...)),
then relocate select_slowest_stable_poles into tuning_methods.py (e.g. as a
StablePoleCancellation staticmethod) so this module can shrink or go away
entirely. Deferred because it touches every test call site and requires
explicitly relaxing the original invariant that tune.py's public functions
must keep working unchanged for existing callers.

Methods implemented:
  1. Stable pole cancellation  (course quick-and-dirty method, HO PS4)
  2. Ziegler–Nichols Method I  (reaction curve, FOPDT-based)
  3. Ziegler–Nichols Method II (ultimate gain, Ku/Pu-based)
  4. AMIGO                     (Aström–Hägglund 2006, HO13)
  5. SIMC                      (Skogestad, HO14)
  6. Boyd                      (convex-concave ECC 2013)
  7. Cohen–Coon                (reaction curve, 1953)
  8. Chien–Hrones–Reswick      (reaction curve, 1952)
  9. Tyreus–Luyben             (ultimate gain, 1997)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from plant import TransferFunction, poly_mul
from identify import FOPDT
from tuning_methods import (
    PIDGains, TuningResult, halve_gains,
    StablePoleCancellation, ZieglerNicholsI, ZieglerNicholsII,
    Amigo, Simc, Boyd, CohenCoon, ChienHronesReswick, TyreusLuyben
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
    return StablePoleCancellation(plant, p1, p2, Kd).tune()


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


def tune_zn_method_1(fopdt):
    """ZN-I from a fitted FOPDT."""
    return ZieglerNicholsI(fopdt).tune()


def tune_zn_method_2(Ku, Pu):
    return ZieglerNicholsII(Ku, Pu).tune()


def tune_amigo(fopdt, integrating=False):
    return Amigo(fopdt, integrating).tune()


def tune_simc(fopdt, tau_c=None, tau2=None):
    return Simc(fopdt, tau_c, tau2).tune()


def tune_boyd(plant, Ms=1.4, Mt=1.4, seed_gains=None,
              use_derivative=True, n_freq=300, max_iter=40, tol=1e-4):
    """Convex-concave PID design against the true plant frequency response.

    Ms : peak sensitivity bound (typically 1.2–2.0; smaller = more robust)
    Mt : peak complementary-sensitivity bound (typically 1.2–2.0)
    seed_gains : optional PIDGains used to seed/scale; if None, defaults to
                 (1, 0.1, 0) modulated by the plant DC gain.
    """
    return Boyd(plant, Ms, Mt, seed_gains, use_derivative, n_freq, max_iter, tol).tune()


def tune_cohen_coon(fopdt):
    """Cohen–Coon PID from a fitted FOPDT."""
    return CohenCoon(fopdt).tune()


def tune_chr(fopdt, response="setpoint", overshoot=0):
    """Chien–Hrones–Reswick PID from a fitted FOPDT.

    response  : "setpoint" (servo / tracking) or "load" (regulator / rejection)
    overshoot : 0 or 20  (the design's damping target, in percent)
    """
    return ChienHronesReswick(fopdt, response, overshoot).tune()


def tune_tyreus_luyben(Ku, Pu, use_derivative=True):
    """Tyreus–Luyben from the ultimate gain/period."""
    return TyreusLuyben(Ku, Pu, use_derivative).tune()
