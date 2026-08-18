"""Explicit model-following — one of three "how do I pick Q/R" methods
split into its own file (alongside lqg_bryson.py, lqg_implicit.py)
specifically so the three can be read/diffed side by side; the core
Phase-1 methods (LQR, OutputWeightedLQR, LQG) stay together in
lqg_design_methods.py.

Unlike the other two (and unlike every Phase-1 method), this one's control
law is u = -K1·x - K2·xm — two gains on two signals, not a single u = -Kx —
so it can't return a plain LQGDesignResult and gets its own
ExplicitModelFollowingResult below. It still implements
BaseControlDesignMethod.design() like everything else, just with a
different result type; see lqg_checks.checks_for_result() for how the
checks framework dispatches on that.

AILQG.pdf §4.1, eq. 50-54. See lqg_implicit.py's module docstring for how
this compares to the implicit approach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plant import StateSpacePlant
from lqg_design_methods import BaseControlDesignMethod, _lqr_core


@dataclass
class ExplicitModelFollowingResult:
    """Result of ExplicitModelFollowing — distinct from LQGDesignResult
    because the control law has two gains on two different signals
    (u = -K1·x - K2·xm, AILQG.pdf eq. 52), not a single u = -Kx."""

    method: str
    plant: StateSpacePlant
    Am: np.ndarray               # desired model dynamics, shape (nxm, nxm)
    K1: np.ndarray                # feedback gain on the true state, shape (nu, nx)
    K2: np.ndarray                # feedforward gain on the model state, shape (nu, nxm)
    S: np.ndarray                  # augmented Riccati solution, shape (nx+nxm, nx+nxm)
    closed_loop_poles: np.ndarray  # eig(A - B·K1) — the plant-side poles (xm's own
                                    # dynamics are unaffected/autonomous, eq. 48)
    Q1: np.ndarray
    R: np.ndarray
    notes: str = ""

    def is_stable(self):
        return bool(np.all(np.real(self.closed_loop_poles) < -1e-9))

    def pretty(self):
        return (f"K1 =\n{np.array2string(self.K1, precision=4, separator=', ')}\n"
                f"K2 =\n{np.array2string(self.K2, precision=4, separator=', ')}")


class ExplicitModelFollowing(BaseControlDesignMethod):
    """Explicit model-following: feed the desired model's state xm forward
    directly (u = -K1·x - K2·xm), rather than shaping the plant's own
    feedback to approximate it. AILQG.pdf §4.1, eq. 50-54.

    The augmented state [x; xm] evolves under a block-diagonal [A 0; 0 Am]
    (eq. 51 — xm has autonomous dynamics ẋm = Am·xm, no exogenous input),
    and the cost xᵀQ̂x + uᵀRu (eq. 53) uses
        Q̂ = [[CᵀQ1C, -CᵀQ1], [-Q1C, Q1]]      (eq. 54)
    — a standard (no-cross-term) LQR problem on the augmented system, whose
    gain splits into K1 (feedback, first nx columns) and K2 (feedforward on
    xm, remaining nxm columns).

    **Validated structurally, not against AILQG.pdf Example 4's printed S**:
    reusing Example 3's plant/Am/Q1/R, this class's S satisfies the
    continuous-time ARE to a residual of ~1e-12, while Example 4's own
    printed S (eq. 67) fails to satisfy that same ARE (residual ~14) — the
    PDF table appears to have a transcription/rounding error, not this
    implementation. See test_lqg.py and docs/lqg_testing.md for the residual
    check and the closed-form derivation this class is checked against
    instead."""

    name = "Explicit model-following"

    def __init__(self, plant: StateSpacePlant, Am, Q1, R):
        self.plant = plant
        self.Am = np.atleast_2d(np.asarray(Am, dtype=float))
        self.Q1 = np.atleast_2d(np.asarray(Q1, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        if self.Am.shape[0] != self.Am.shape[1]:
            raise ValueError(f"Am must be square, got shape {self.Am.shape}")
        if self.Q1.shape != (plant.ny, plant.ny):
            raise ValueError(
                f"Q1 must be (ny, ny) = ({plant.ny}, {plant.ny}), got shape "
                f"{self.Q1.shape}")
        if self.Am.shape != self.Q1.shape:
            raise ValueError(
                f"Am must be (ny, ny) = {self.Q1.shape} to match Q1 (Q̂'s "
                f"block construction, eq. 54, requires xm's dimension to "
                f"equal Q1's), got shape {self.Am.shape}")
        if not np.all(np.real(np.linalg.eigvals(self.Am)) < 0):
            raise ValueError(
                f"Am must be Hurwitz-stable (a target model with an unstable "
                f"or marginal mode isn't a sensible thing to track), got "
                f"eigenvalues {np.linalg.eigvals(self.Am)}. Besides being "
                f"conceptually nonsensical, xm is uncontrollable inside the "
                f"augmented [x;xm] system (eq. 51 has no B for the xm block), "
                f"so an unstable Am also makes the augmented Riccati problem "
                f"unsolvable -- scipy would raise an opaque LinAlgError "
                f"instead of this message.")

    def design(self) -> ExplicitModelFollowingResult:
        A, B, C = self.plant.A, self.plant.B, self.plant.C
        nx, nu = self.plant.nx, self.plant.nu
        nxm = self.Am.shape[0]
        Aaug = np.block([[A, np.zeros((nx, nxm))],
                         [np.zeros((nxm, nx)), self.Am]])
        Baug = np.vstack([B, np.zeros((nxm, nu))])
        Qhat = np.block([[C.T @ self.Q1 @ C, -C.T @ self.Q1],
                         [-self.Q1 @ C, self.Q1]])
        Kaug, S, _ = _lqr_core(Aaug, Baug, Qhat, self.R)
        K1, K2 = Kaug[:, :nx], Kaug[:, nx:]
        clp = np.linalg.eigvals(A - B @ K1)
        return ExplicitModelFollowingResult(
            method=self.name, plant=self.plant, Am=self.Am, K1=K1, K2=K2,
            S=S, closed_loop_poles=clp, Q1=self.Q1, R=self.R,
            notes="Q̂ built from (Am, Q1) per AILQG.pdf eq. 54 on the "
                 "augmented state [x; xm]; u = -K1·x - K2·xm.",
        )
