"""Bryson's rule — one of three "how do I pick Q/R" methods split into its
own file (alongside lqg_implicit.py, lqg_explicit.py) specifically so the
three can be read/diffed side by side; the core Phase-1 methods (LQR,
OutputWeightedLQR, LQG) stay together in lqg_design_methods.py. All four
share the same interface (BaseControlDesignMethod.design()), defined there.

AILQG.pdf §3.1, eq. 27.
"""

from __future__ import annotations

import numpy as np

from plant import StateSpacePlant
from lqg_design_methods import BaseControlDesignMethod, LQGDesignResult, StateFeedbackGains, _lqr_core


class BrysonLQR(BaseControlDesignMethod):
    """Bryson's rule: Qii = 1/x_max², Rii = 1/u_max² — diagonal weights
    sized from the maximum desired deviation of each state/control.
    AILQG.pdf §3.1, eq. 27. No direct .m example, but it's the PDF's
    headline "give me sane defaults" heuristic — the LQG analog of
    AMIGO/SIMC's role on the PID side."""

    name = "Bryson's rule"

    def __init__(self, plant: StateSpacePlant, x_max, u_max):
        self.plant = plant
        x_max = np.asarray(x_max, dtype=float)
        u_max = np.asarray(u_max, dtype=float)
        if x_max.shape != (plant.nx,):
            raise ValueError(f"x_max must have {plant.nx} entries, got shape {x_max.shape}")
        if u_max.shape != (plant.nu,):
            raise ValueError(f"u_max must have {plant.nu} entries, got shape {u_max.shape}")
        if np.any(x_max <= 0) or np.any(u_max <= 0):
            raise ValueError("Bryson's rule requires strictly positive x_max/u_max")
        self.x_max = x_max
        self.u_max = u_max

    def design(self) -> LQGDesignResult:
        Q = np.diag(1.0 / self.x_max ** 2)
        R = np.diag(1.0 / self.u_max ** 2)
        K, S, clp = _lqr_core(self.plant.A, self.plant.B, Q, R)
        return LQGDesignResult(
            method=self.name, plant=self.plant, gains=StateFeedbackGains(K=K),
            S=S, closed_loop_poles=clp, Q=Q, R=R,
            notes="Q/R diagonal, sized from x_max/u_max (Bryson's rule).",
        )
