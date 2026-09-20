"""Loop transfer recovery (LTR) — recovery at the plant input, one of the
"how do I pick Q/R (and Qw/Rv)" methods split into its own file (alongside
lqg_bryson.py, lqg_implicit.py, lqg_explicit.py) for the same reason: easy
to read/diff on its own. Shares the interface every method here does
(BaseControlDesignMethod.design()), defined in lqg_design_methods.py.

AI Agents for Control Design_v3.pdf, Sec III.B item 5 ("loop transfer
recovery") is the gap this fills. Matches LTR.m's worked example (repo
root) rather than A_Tutorial_on_the_LQG_LTR_Method.pdf's own narrative —
those are dual variants (recovery at the plant input vs. at the plant
output); LTR.m's is the concrete, numeric one, so it's what's implemented
here. See the class docstring below for the construction and its scope.
"""

from __future__ import annotations

import numpy as np

from plant import StateSpacePlant
from lqg_design_methods import (
    BaseControlDesignMethod, LQGDesignResult, StateFeedbackGains,
    KalmanFilterResult, _lqr_core, _kalman_core,
)


class LoopTransferRecovery(BaseControlDesignMethod):
    """Loop transfer recovery at the plant input (Doyle-Stein), matching
    LTR.m's worked example exactly:

        K  = lqr(A, B, Q, R)               -- fixed once, Q=C'C, R=I by default
        Qw = (q*B)(q*B)'                    -- n x n, fed straight into the
        Rv = I                              -- Kalman-filter Riccati equation
        Kf = kalman(A, C, Qw, Rv)

    As the scalar recovery parameter q -> infinity, the resulting
    observer-based compensator's loop transfer at the plant input,
    Kc(s)G(s), approaches the full-state-feedback target loop
    K(sI-A)^-1 B (lqg_frequency.loop_transfer_at_input computes both,
    selecting on whether result.kalman is None) -- see
    TestLoopTransferRecovery.test_recovery_gap_shrinks_as_q_increases in
    test_lqg.py for the numeric check of this property, on LTR.m's own
    plant.

    Larger q means better recovery, at the cost of a higher-bandwidth
    (more aggressive, noisier in practice) filter -- same fundamental
    tradeoff paid by cheap-control LQR, just expressed on the estimator
    side instead.

    Per A_Tutorial_on_the_LQG_LTR_Method.pdf Sec 7: exact recovery only
    holds for minimum-phase plants (no right-half-plane transmission
    zeros). That is not checked or enforced here -- a non-minimum-phase
    plant will silently fail to fully recover as q grows, same as any
    other method in this codebase that notes its own inapplicability
    conditions rather than guarding against them.
    """

    name = "Loop transfer recovery (LTR)"

    def __init__(self, plant: StateSpacePlant, q, Q=None, R=None):
        self.plant = plant
        q = float(q)
        if q <= 0:
            raise ValueError(f"LTR requires q > 0 (recovery parameter), got {q}")
        self.q = q
        self.Q = (plant.C.T @ plant.C) if Q is None else np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.eye(plant.nu) if R is None else np.atleast_2d(np.asarray(R, dtype=float))

    def design(self) -> LQGDesignResult:
        K, S, clp = _lqr_core(self.plant.A, self.plant.B, self.Q, self.R)
        gamma = self.q * self.plant.B
        Qw = gamma @ gamma.T
        Rv = np.eye(self.plant.ny)
        Kf, P, est_poles = _kalman_core(self.plant.A, self.plant.C, Qw, Rv)
        kalman = KalmanFilterResult(Kf=Kf, P=P, estimator_poles=est_poles, Qw=Qw, Rv=Rv)
        return LQGDesignResult(
            method=self.name, plant=self.plant, gains=StateFeedbackGains(K=K),
            S=S, closed_loop_poles=clp, Q=self.Q, R=self.R, kalman=kalman,
            notes=(f"Loop transfer recovery at the plant input (q={self.q:g}): "
                   "K fixed via LQR (Q=C'C, R=I by default), Kf swept via "
                   "Qw=(qB)(qB)', Rv=I so the observer-based compensator's loop "
                   "recovers K(sI-A)^-1 B as q -> infinity. Exact recovery holds "
                   "only for minimum-phase plants (not checked/enforced here)."),
        )
