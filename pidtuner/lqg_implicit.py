"""Implicit model-following — one of three "how do I pick Q/R" methods
split into its own file (alongside lqg_bryson.py, lqg_explicit.py)
specifically so the three can be read/diffed side by side; the core
Phase-1 methods (LQR, OutputWeightedLQR, LQG) stay together in
lqg_design_methods.py. All four share the same interface
(BaseControlDesignMethod.design()), defined there.

AILQG.pdf §4.2, eq. 55-60. See lqg_explicit.py's module docstring for how
this compares to the explicit approach.
"""

from __future__ import annotations

import numpy as np

from plant import StateSpacePlant
from lqg_design_methods import BaseControlDesignMethod, LQGDesignResult, StateFeedbackGains, _lqr_core


class ImplicitModelFollowing(BaseControlDesignMethod):
    """Implicit model-following: shape the plant's own feedback (u = -Kx, no
    augmentation) so its closed-loop dynamics approach a desired model
    ẋm = Am·xm, compared through the output y = Cx. AILQG.pdf §4.2, eq. 55-60.

    The performance index J = ∫(ẏ - Am·y)ᵀQ1(ẏ - Am·y) + uᵀRu dt (eq. 56) is
    algebraically equivalent to a standard LQR problem with a cross-term
    (eq. 57), built here as:
        Q̂ = (CA - AmC)ᵀQ1(CA - AmC)      (eq. 58)
        N̂ = BᵀCᵀQ1(CA - AmC)              (eq. 59, shape (nu, nx))
        R̂ = R + BᵀCᵀQ1CB                  (eq. 60)
    and solved with the same `_lqr_core` used everywhere else (N̂ᵀ is what
    `_lqr_core`'s `N` parameter expects — see its docstring's MATLAB-N
    convention).

    Matches AIKreindlerRothschildModelFollowingN.m's own
    `Qhat=(C*A-Am*C)'*Qi*(C*A-Am*C); N=B'*C'*Qi*(C*A-Am*C); Rhat=R+B'*C'*Qi*C*B`
    — i.e. the professor's own code builds exactly these three matrices
    before calling `lqr`. **Validated against AILQG.pdf Example 3** (an
    electro-mechanical system): `S`, `K`, and the closed-loop poles all
    match the PDF's printed values — see test_lqg.py."""

    name = "Implicit model-following"

    def __init__(self, plant: StateSpacePlant, Am, Q1, R):
        self.plant = plant
        self.Am = np.atleast_2d(np.asarray(Am, dtype=float))
        self.Q1 = np.atleast_2d(np.asarray(Q1, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        if self.Am.shape != (plant.ny, plant.ny):
            raise ValueError(
                f"Am must be (ny, ny) = ({plant.ny}, {plant.ny}) to compare "
                f"against y=Cx, got shape {self.Am.shape}")
        if self.Q1.shape != (plant.ny, plant.ny):
            raise ValueError(
                f"Q1 must be (ny, ny) = ({plant.ny}, {plant.ny}), got shape "
                f"{self.Q1.shape}")

    def design(self) -> LQGDesignResult:
        A, B, C = self.plant.A, self.plant.B, self.plant.C
        diff = C @ A - self.Am @ C           # (ny, nx)
        Qhat = diff.T @ self.Q1 @ diff        # (nx, nx)
        Nhat = B.T @ C.T @ self.Q1 @ diff     # (nu, nx)
        Rhat = self.R + B.T @ C.T @ self.Q1 @ C @ B  # (nu, nu)
        K, S, clp = _lqr_core(A, B, Qhat, Rhat, N=Nhat.T)
        return LQGDesignResult(
            method=self.name, plant=self.plant, gains=StateFeedbackGains(K=K),
            S=S, closed_loop_poles=clp, Q=Qhat, R=Rhat, N=Nhat.T,
            notes="Q̂/N̂/R̂ built from (Am, Q1, R) per AILQG.pdf eq. 58-60 "
                 "(implicit model-following); u = -Kx.",
        )
