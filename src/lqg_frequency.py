"""Sensitivity S(jw) and complementary sensitivity T(jw) for the LQR/LQG
design track — nothing in the repo computed this before; see docs/lqg_plan.md
for why it wasn't part of Phase 1/2.

Loop is broken at the PLANT INPUT u, per an explicit scope decision (2026-08-18):
for full-state-feedback designs this is the textbook LQR loop transfer
L(s) = K(sI-A)^-1 B, for which the classical LQR guarantees (Ms<=2, >=60deg
phase margin, >=-6dB/inf gain margin) are known to hold. For output-feedback
LQG designs (result.kalman is not None), L(s) is the observer-based
compensator K_c(s) = K(sI-(A-BK-KfC))^-1 Kf in series with the plant
G(s) = C(sI-A)^-1 B + D — the classical single-loop LQR guarantees do NOT
carry over to this loop transfer (that's the whole point of LQG loop-transfer
recovery being a separate, non-automatic step). Reporting S/T at the plant
input for LQG plants anyway is a deliberate simplification for a first pass;
flag it in any report generated from this module as an open question to be
verified, not a settled result — plant-output loop-breaking (arguably the more
standard place to judge MIMO robustness) is not computed here.

S = (I + L)^-1,  T = I - S = L(I+L)^-1.  Reported metric is the peak
structured-singular-value-free bound: Ms = max_w sigma_max(S(jw)),
Mt = max_w sigma_max(T(jw)) — the MIMO generalization of the SISO
peak-sensitivity number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lqg_design_methods import LQGDesignResult


@dataclass
class SensitivityResult:
    omega: np.ndarray            # (nw,) rad/s, log-spaced
    sigma_S: np.ndarray          # (nw,) sigma_max(S(jw)) at each frequency
    sigma_T: np.ndarray          # (nw,) sigma_max(T(jw)) at each frequency
    Ms: float                    # peak sensitivity = max(sigma_S)
    Mt: float                    # peak complementary sensitivity = max(sigma_T)
    loop_point: str = "plant_input"  # documents the scope decision above


def loop_transfer_at_input(result: LQGDesignResult, s: complex) -> np.ndarray:
    """L(s), evaluated at the plant input, shape (nu, nu). s is a single
    complex frequency (typically jw)."""
    plant = result.plant
    A, B, C, D = plant.A, plant.B, plant.C, plant.D
    nx = plant.nx
    K = result.gains.K
    I_nx = np.eye(nx)

    if result.kalman is None:
        return K @ np.linalg.solve(s * I_nx - A, B)

    Kf = result.kalman.Kf
    Ac = A - B @ K - Kf @ C
    Kc = K @ np.linalg.solve(s * I_nx - Ac, Kf)          # compensator, y -> u
    G = C @ np.linalg.solve(s * I_nx - A, B) + D          # plant, u -> y
    return Kc @ G


def sensitivity_complementary(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S = (I+L)^-1, T = I-S, at one frequency."""
    n = L.shape[0]
    S = np.linalg.inv(np.eye(n) + L)
    T = np.eye(n) - S
    return S, T


def compute_sensitivity(result: LQGDesignResult, omega=None) -> SensitivityResult:
    """Sweeps omega (default: 300 log-spaced points, 1e-3 to 1e3 rad/s,
    reasonable default for the plant time constants seen across the 12-plant
    catalog — callers with faster/slower plants should pass their own
    omega). Returns peak (Ms, Mt) plus the full sweep for Bode-magnitude
    plotting."""
    if omega is None:
        omega = np.logspace(-3, 3, 300)
    omega = np.asarray(omega, dtype=float)
    sigma_S = np.empty(len(omega))
    sigma_T = np.empty(len(omega))
    for i, w in enumerate(omega):
        L = loop_transfer_at_input(result, 1j * w)
        S, T = sensitivity_complementary(L)
        sigma_S[i] = np.linalg.svd(S, compute_uv=False)[0]
        sigma_T[i] = np.linalg.svd(T, compute_uv=False)[0]
    return SensitivityResult(omega=omega, sigma_S=sigma_S, sigma_T=sigma_T,
                             Ms=float(np.max(sigma_S)), Mt=float(np.max(sigma_T)))
