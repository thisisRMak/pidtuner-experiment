"""Identification: extract FOPDT model and ultimate gain/period from a plant.

Three identification primitives:
  1. FOPDT step fit (28.3% / 63.2% method) — used by ZN-I, AMIGO, SIMC.
     The plant is simulated against a step input, then K, tau, L are
     fitted from the response. This matches what a real plant test
     would yield, and so its output is realistic for the tuning rules
     that assume a process step response was the input.
  2. Relay-feedback test (Aström–Hägglund) — drives the plant into a
     limit cycle via a hysteretic on/off controller, extracts Ku and Pu
     from the cycle amplitude and period.
  3. Bode-based ultimate gain — for plants where you have the model in
     closed form, solve phase(G(jω)) = -180° analytically and read off
     Ku = 1/|G(jω₁₈₀)|. Faster and more accurate than the relay test,
     but requires the model (which the user provided anyway).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from plant import TransferFunction
from signal_source import SignalGenerator


# ─────────────────────────────────────────────────────────────────────────────
# FOPDT model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FOPDT:
    """First-order plus dead time:  G(s) = K * exp(-Ls) / (tau*s + 1)."""

    K: float
    tau: float
    L: float

    def to_tf(self):
        return TransferFunction.fopdt(self.K, self.tau, self.L)

    def pretty(self):
        return f"K={self.K:.4g}, τ={self.tau:.4g} s, L={self.L:.4g} s"


# ─────────────────────────────────────────────────────────────────────────────
# Step-test FOPDT fit  (28.3% / 63.2% method, Sundaresan & Krishnaswamy)
# ─────────────────────────────────────────────────────────────────────────────
# Why these two percentages: for a true FOPDT response,
#   y(t) = K*Δu * [1 - exp(-(t-L)/τ)]   for t >= L,
# so y reaches 28.3% of its final change at t = L + τ/3,
# and 63.2% at t = L + τ. Solving for τ and L from those two crossing
# times is more robust to noise than fitting a tangent at the inflection
# point (the Ziegler-Nichols I "tangent" construction), because the
# tangent slope depends on a numerical derivative.

def fit_fopdt_from_step(t, y, u_step):
    """Fit K, tau, L to a step response.

    Parameters
    ----------
    t       : time vector (uniform spacing)
    y       : measured plant output
    u_step  : amplitude of the step input
    """
    y0 = float(y[0])
    n_tail = max(1, int(0.05 * len(y)))
    y_ss = float(np.mean(y[-n_tail:]))
    delta = y_ss - y0
    K = delta / u_step if u_step != 0 else 0.0

    if abs(delta) < 1e-9:
        # No discernible response — fall back to a placeholder
        return FOPDT(K=0.0, tau=1.0, L=0.0)

    def first_crossing(frac):
        target = y0 + frac * delta
        sign = np.sign(delta)
        mask = sign * (y - target) >= 0
        if not mask.any():
            return float(t[-1])
        idx = int(np.argmax(mask))
        if idx == 0:
            return float(t[0])
        a, b = y[idx - 1], y[idx]
        if b == a:
            return float(t[idx])
        # linear interpolation between samples
        frac_step = (target - a) / (b - a)
        return float(t[idx - 1] + frac_step * (t[idx] - t[idx - 1]))

    t_283 = first_crossing(0.283)
    t_632 = first_crossing(0.632)
    tau = max(1.5 * (t_632 - t_283), 1e-3)
    L = max(t_632 - tau, 0.0)
    return FOPDT(K=K, tau=tau, L=L)


def run_step_test(plant, t_max=None, step_amp=1.0, noise_sigma=0.0, seed=None):
    """Run an open-loop step test on the plant. Returns (t, u, y_true, y_meas, fopdt).

    Generation is delegated to signal_source.SignalGenerator (the same code
    entity A uses); this function additionally keeps y_true and runs the
    FOPDT fit, for backward compatibility with existing callers.
    """
    t, u, y_true, y_meas, dt = SignalGenerator(plant)._simulate_step_response(
        step_amp=step_amp, t_max=t_max, noise_sigma=noise_sigma, seed=seed)
    fopdt = fit_fopdt_from_step(t, y_meas, u_step=step_amp)
    return t, u, y_true, y_meas, fopdt


# ─────────────────────────────────────────────────────────────────────────────
# Step-test SOPDT fit  (second-order + dead time, two real poles)
# ─────────────────────────────────────────────────────────────────────────────
# Used as a black-box surrogate model wherever a method needs more than a
# single pole (pole-cancellation, Boyd's frequency response) but no ground-
# truth plant is available. Reparametrized as (tau_dom, frac, L) with
# tau2 = frac*tau_dom, frac in (0,1] to remove the tau1<->tau2 permutation
# symmetry a naive (tau1, tau2, L) fit would have two mirror-image optima
# for. Seeded from the (already robust) FOPDT fit and solved with bounded
# nonlinear least squares (scipy.optimize.least_squares, method="trf").

@dataclass
class SOPDT:
    """Second-order plus dead time: G(s) = K*exp(-Ls) / ((tau1*s+1)(tau2*s+1)).

    tau1 is always the larger (dominant) time constant; both are real by
    construction — this fit form can never produce a complex pole pair.
    """

    K: float
    tau1: float
    tau2: float
    L: float

    def to_tf(self):
        den = np.polymul([self.tau1, 1.0], [self.tau2, 1.0])
        return TransferFunction(num=[self.K], den=den, L=self.L)

    def pretty(self):
        return f"K={self.K:.4g}, τ1={self.tau1:.4g} s, τ2={self.tau2:.4g} s, L={self.L:.4g} s"


def _fopdt_step_response(t, K, tau, L, u_step, y0):
    shifted = t - L
    resp = np.zeros_like(t)
    mask = shifted >= 0
    resp[mask] = 1.0 - np.exp(-shifted[mask] / max(tau, 1e-9))
    return y0 + K * u_step * resp


def _sopdt_step_response(t, K, tau1, tau2, L, u_step, y0):
    shifted = t - L
    resp = np.zeros_like(t)
    mask = shifted >= 0
    ts = shifted[mask]
    if abs(tau1 - tau2) < 1e-6 * max(tau1, tau2, 1e-9):
        tau = 0.5 * (tau1 + tau2)
        resp[mask] = 1.0 - (1.0 + ts / tau) * np.exp(-ts / tau)
    else:
        resp[mask] = 1.0 - (tau1 * np.exp(-ts / tau1) - tau2 * np.exp(-ts / tau2)) / (tau1 - tau2)
    return y0 + K * u_step * resp


def _check_monotonic_step(y, y_ss, delta):
    """Reject responses that overshoot/undershoot their steady state — a
    signature of complex (underdamped) poles the real-pole SOPDT form
    cannot represent. Smoothed to avoid tripping on measurement noise."""
    n = len(y)
    w = max(1, n // 100)
    y_smooth = np.convolve(y, np.ones(w) / w, mode="same") if w > 1 else y
    tol = 0.03 * abs(delta)
    overshoot = (np.max(y_smooth) - y_ss) if delta >= 0 else (y_ss - np.min(y_smooth))
    return overshoot <= tol


def fit_sopdt_from_step(t, y, u_step, fopdt_hint: FOPDT | None = None) -> SOPDT:
    """Fit K, tau1, tau2, L to a step response. Raises RuntimeError (with a
    specific reason) rather than a degenerate/invalid fit — callers should
    catch this and fall back to the FOPDT surrogate."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    n_tail = max(1, int(0.05 * len(y)))
    y_ss = float(np.mean(y[-n_tail:]))
    delta = y_ss - y0
    K = delta / u_step if u_step != 0 else 0.0

    if abs(delta) < 1e-9:
        raise RuntimeError("SOPDT fit: no discernible response.")

    if not _check_monotonic_step(y, y_ss, delta):
        raise RuntimeError(
            "SOPDT fit: response overshoots/undershoots its steady state, "
            "indicating complex (underdamped) poles. The two-real-pole SOPDT "
            "form cannot represent that — declining rather than mis-fitting."
        )

    if fopdt_hint is None:
        fopdt_hint = fit_fopdt_from_step(t, y, u_step)

    # Bounding L tightly around the FOPDT fit's own L estimate is a trap: the
    # 28.3%/63.2% method (and any simple "onset" threshold) systematically
    # biases L upward on multi-time-constant plants, since a two-real-pole
    # response has zero initial slope right after the dead time (unlike
    # FOPDT's immediate nonzero slope) — so the search must be allowed to
    # explore well below that biased estimate. Bounding L to [0, tau_dom0]
    # (rather than around any point estimate) reliably finds the true
    # minimum regardless of the L seed, verified against the benchmark plant.
    tau_dom0 = max(fopdt_hint.tau, 1e-3)
    L0 = min(max(fopdt_hint.L, 0.0), tau_dom0)
    lb = [0.05 * tau_dom0, 0.005, 0.0]
    ub = [20.0 * tau_dom0, 1.0, tau_dom0]
    x0 = np.clip([tau_dom0, 0.2, L0], lb, ub)

    def residuals(x):
        tau1, frac, L = x
        return _sopdt_step_response(t, K, tau1, frac * tau1, L, u_step, y0) - y

    res = least_squares(residuals, x0, bounds=(lb, ub), method="trf")
    if not res.success:
        raise RuntimeError(f"SOPDT fit did not converge: {res.message}")

    tau1, frac, L = res.x
    tau2 = frac * tau1

    sse_sopdt = float(np.sum(res.fun ** 2))
    fopdt_resid = _fopdt_step_response(t, fopdt_hint.K, fopdt_hint.tau, fopdt_hint.L, u_step, y0) - y
    sse_fopdt = float(np.sum(fopdt_resid ** 2))
    if sse_fopdt > 1e-12 and sse_sopdt / sse_fopdt > 0.9:
        raise RuntimeError(
            "SOPDT fit offers no real improvement over the FOPDT fit — "
            "treating as degenerate (data doesn't support a second time constant)."
        )
    if frac > 0.98:
        raise RuntimeError(
            "SOPDT fit found no distinguishable second time constant "
            "(tau2 ≈ tau1) — treating as degenerate."
        )

    return SOPDT(K=float(K), tau1=float(tau1), tau2=float(tau2), L=float(L))


# ─────────────────────────────────────────────────────────────────────────────
# Relay-feedback test  (Aström–Hägglund 1984)
# ─────────────────────────────────────────────────────────────────────────────
# Drive the plant through a hysteretic relay in closed loop. The describing-
# function approximation gives:
#       Ku = 4*h / (pi * a)
#       Pu = period of the steady-state limit cycle
# where h is the relay amplitude and a is the peak-to-peak/2 of the output.

def run_relay_test(plant, t_max=80.0, h=1.0, setpoint=0.0, hysteresis=0.0):
    """Run a relay-feedback test. Returns (Ku, Pu, t, u, y).

    Generation is delegated to signal_source.SignalGenerator; the Ku, Pu
    extraction is delegated to identify_ultimate_gain_from_relay, which is
    signal-only and reusable without a plant reference.
    """
    t, u, y, dt = SignalGenerator(plant)._simulate_relay_response(
        h=h, setpoint=setpoint, hysteresis=hysteresis, t_max=t_max)
    Ku, Pu = identify_ultimate_gain_from_relay(t, u, y, h, setpoint=setpoint, hysteresis=hysteresis)
    return Ku, Pu, t, u, y


def identify_ultimate_gain_from_relay(t, u, y, h, setpoint=0.0, hysteresis=0.0):
    """Signal-only extraction of Ku, Pu from a relay-test oscillation.
    Needs only the (t, u, y) arrays and the relay parameters used to
    produce them — no plant reference."""
    err = setpoint - y
    crosses = np.where(np.diff(np.signbit(err).astype(int)) != 0)[0]
    if len(crosses) < 4:
        raise RuntimeError(
            "Relay test produced too few oscillations. "
            "Try increasing test duration, or use the Bode-based "
            "method instead — some plants don't cycle under a pure relay."
        )
    # Use the last few half-periods (steady-state cycles)
    use = crosses[-min(8, len(crosses)):]
    half_periods = np.diff(t[use])
    Pu = float(2.0 * np.mean(half_periods))
    tail_start = crosses[-min(6, len(crosses))]
    a = float((np.max(y[tail_start:]) - np.min(y[tail_start:])) / 2.0)
    if a <= 0:
        raise RuntimeError("Relay test amplitude is zero — cannot compute Ku.")
    Ku = 4.0 * h / (np.pi * a)
    return float(Ku), Pu


# ─────────────────────────────────────────────────────────────────────────────
# Bode-based ultimate gain  (analytical)
# ─────────────────────────────────────────────────────────────────────────────
# Solve phase(G(jω)) = -180° for the smallest positive ω₁₈₀, then
# Ku = 1/|G(jω₁₈₀)| and Pu = 2π/ω₁₈₀.
# This requires that such an ω exists — i.e., the phase of G crosses -180°.
# For plants where it doesn't (e.g., a pure first-order system), there is
# no finite Ku, and ZN-II is not applicable.

def _phase_unwrapped(plant, omega):
    """Continuous phase of G(jω) over a frequency grid."""
    H = plant.freq_response(omega)
    return np.unwrap(np.angle(H))


def find_ultimate_gain(plant, omega_lo=None, omega_hi=None, n_freq=4000):
    """Find Ku and Pu by locating the -180° phase crossover frequency."""
    poles = plant.poles()
    zeros = plant.zeros()
    feats = np.concatenate([np.abs(poles), np.abs(zeros)])
    feats = feats[feats > 1e-9]

    if omega_lo is None:
        omega_lo = max(np.min(feats) * 1e-3, 1e-6) if len(feats) else 1e-3
    if omega_hi is None:
        omega_hi = max(np.max(feats) * 1e3, omega_lo * 1e4) if len(feats) else 1e3
        if plant.L > 0:
            omega_hi = max(omega_hi, 50.0 / plant.L)

    omega = np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_freq)
    phase = _phase_unwrapped(plant, omega)

    # Look for crossings of phase = -π
    target = -np.pi
    crossings = []
    for i in range(len(omega) - 1):
        if (phase[i] - target) * (phase[i + 1] - target) < 0:
            crossings.append((omega[i], omega[i + 1]))

    if not crossings:
        raise RuntimeError(
            "No -180° phase crossover found — plant has no finite Ku, "
            "ZN Method II is not applicable. Try ZN-I or another method."
        )

    # Use the lowest-frequency crossing (the one that bounds stability margin).
    # The unwrapped phase is locally monotonic across each crossing, so linear
    # interpolation in log-ω is accurate to better than a frequency-grid step.
    # We refine the bracket with a few iterations of bisection on a continuous
    # phase function constructed by anchoring to the LHS bracket endpoint's
    # unwrapped value (this avoids the (-π, π] wraparound that bites brentq).
    w_lo, w_hi = crossings[0]
    idx_lo = int(np.argmin(np.abs(omega - w_lo)))
    phase_lo_unwrapped = float(phase[idx_lo])
    # Refine by bisection — recompute phase as a continuous extension from idx_lo
    H_ref = plant.freq_response(np.array([w_lo]))[0]
    angle_ref = float(np.angle(H_ref))
    # the unwrap shift between np.angle and the unwrapped grid value
    shift = phase_lo_unwrapped - angle_ref

    def phase_minus_target(w):
        H = plant.freq_response(np.array([w]))[0]
        return float(np.angle(H)) + shift - target

    try:
        w180 = brentq(phase_minus_target, w_lo, w_hi, xtol=1e-9, rtol=1e-9)
    except ValueError:
        # fall back to linear interp on the grid (in log-ω for accuracy)
        idx_hi = idx_lo + 1
        frac = (target - phase[idx_lo]) / (phase[idx_hi] - phase[idx_lo])
        log_w = np.log(omega[idx_lo]) + frac * (np.log(omega[idx_hi]) -
                                                np.log(omega[idx_lo]))
        w180 = float(np.exp(log_w))

    H180 = plant.freq_response(np.array([w180]))[0]
    Ku = 1.0 / abs(H180)
    Pu = 2.0 * np.pi / w180
    return float(Ku), float(Pu), float(w180)
